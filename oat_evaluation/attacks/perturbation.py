import math
import torch
import tqdm
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe
from oat_evaluation.utils import print_mem_usage, print_timey, calculate_forward_flops, calculate_backward_flops


class PerturbationAttack(Attack):
    def __init__(
        self,
        num_epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        adam_epsilon: float = 1e-8,
        epsilon: float = 5.0,
        probe_loss_weight: float = 1.0,
        debug_mode: bool = False,
        chunk_size: int = 4,
        max_steps: int = -1
    ):
        if chunk_size > batch_size:
             print(f"Warning: chunk_size ({chunk_size}) > batch_size ({batch_size}). Setting chunk_size = batch_size.")
             chunk_size = batch_size
             
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.probe_loss_weight = probe_loss_weight
        self.debug_mode = debug_mode
        self.chunk_size = chunk_size
        self.adam_epsilon = adam_epsilon
        self.max_steps = max_steps
        print(f"PerturbationAttack initialized with num_epochs {num_epochs}, learning_rate {learning_rate}, batch_size {batch_size}, epsilon {epsilon}, probe_loss_weight {probe_loss_weight}, debug_mode {debug_mode}, chunk_size {chunk_size}")

    def clip_perturbation(self, perturbation: torch.Tensor, epsilon: float) -> None:
        """Clips the perturbation's total L2 norm in-place to be <= epsilon."""
        with torch.no_grad():
            # Compute the total L2 norm of the entire perturbation
            norm = perturbation.norm(p=2)
            if norm > epsilon:
                perturbation *= (epsilon / norm)

    def clip_perturbation_per_row(self, perturbation: torch.Tensor, epsilon: float) -> None:
        """Clips each row of `perturbation` to be within L2 = epsilon."""
        with torch.no_grad():
            row_norms = perturbation.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            scale = (row_norms / epsilon).clamp(min=1.0)
            perturbation.div_(scale)  # divides each row by scale if norm > epsilon

    @property
    def is_slow(self) -> bool:
        return False

    def run_attack(
        self,
        llm: LLM,
        prompts: List[str],
        harmful_responses: List[str] = None,
        probe_to_obfuscate: Optional[Probe] = None,
        probe_refusal_threshold: Optional[float] = None,
        generate_final_responses: bool = True,
        callback_steps: List[int] = None,
        callbacks: List[Callable[[AttackDetails], Any]] = None
    ) -> Tuple[LLMResponses, AttackDetails]:

        torch.enable_grad()

        batch_size = min(self.batch_size, len(prompts))
        
        if callback_steps is None:
            callback_steps = []
        if callbacks is None:
            callbacks = []
        
        if harmful_responses is None or len(harmful_responses) == 0:
            raise ValueError("PerturbationAttack requires harmful_responses.")
        if len(prompts) != len(harmful_responses):
             # Replicate harmful responses if only one is provided for multiple prompts
            if len(harmful_responses) == 1 and len(prompts) > 1:
                print(f"Warning: Only one harmful_response provided for {len(prompts)} prompts. Replicating the harmful response.")
                harmful_responses = harmful_responses * len(prompts)
            else:
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of harmful_responses ({len(harmful_responses)}).")

        # Precompute embeddings to find max length
        print("Precomputing embeddings...")
        prompt_embeddings = [llm.string_to_embedding(p).detach() for p in tqdm.tqdm(prompts)]
        harmful_embeddings = [llm.string_to_embedding(r).detach() for r in tqdm.tqdm(harmful_responses)]
        harmful_token_ids = [llm.string_to_token_ids(r).long() for r in harmful_responses]
        harmful_token_ids_including_ending = [llm.string_to_token_ids(r, add_response_ending=True).long().to(llm.device) for r in harmful_responses]
        print("Embeddings precomputed.")

        # Find max input length
        max_input_length = max(p.shape[0] for p in prompt_embeddings)

        # Create perturbation parameter of max input length
        perturbation = torch.zeros(
            max_input_length,
            llm.embedding_size,
            device=llm.device,
            dtype=llm.dtype,
            requires_grad=True
        )

        # Apply Kaiming uniform initialization in-place
        torch.nn.init.kaiming_uniform_(perturbation)

        # Immediately clip it to epsilon
        self.clip_perturbation_per_row(perturbation, self.epsilon)

        # Ensure right dtype
        perturbation = perturbation.to(llm.dtype)
        self.perturbation = perturbation # to save it later

        if self.debug_mode:
            print("-" * 20)
            print(f"Initial perturbation requires_grad: {perturbation.requires_grad}")
            if prompt_embeddings:
                print(f"Prompt embedding requires_grad: {prompt_embeddings[0].requires_grad}")
            if harmful_embeddings:
                print(f"Harmful embedding requires_grad: {harmful_embeddings[0].requires_grad}")
            print(f"Pad embedding requires_grad: {llm.pad_embedding.requires_grad}")
            print("-" * 20)

        # Ensure pad_embedding was detached during LLM init
        if llm.pad_embedding.requires_grad:
            raise ValueError("LLM's pad_embedding requires grad! Detach it in AutoLLM.__init__.")
        if any(p.requires_grad for p in prompt_embeddings):
            raise ValueError("Prompt embeddings require grad! Detach them.")
        if any(h.requires_grad for h in harmful_embeddings):
            raise ValueError("Harmful embeddings require grad! Detach them.")

        if self.debug_mode:
            print("Max harmful token ID:", max(r.max().item() for r in harmful_token_ids_including_ending))
            print("Vocab size:", llm.vocab_size)
            print(f"Harmful token ID dtypes: {[r.dtype for r in harmful_token_ids_including_ending]}")
            print(f"Harmful token ID devices: {[r.device for r in harmful_token_ids_including_ending]}")

        def apply_perturbation(prompt_embeddings_batch: List[torch.Tensor], 
                               perturbation_to_apply: torch.Tensor) -> List[torch.Tensor]:
            """
            Applies the given perturbation_to_apply to a batch of prompt_embeddings.
            Handles cases where prompt embeddings are shorter or longer than the perturbation.
            """
            perturbed_embeddings_list = []
            L_pert = perturbation_to_apply.shape[0]  # Length of the perturbation
            pert_device = perturbation_to_apply.device

            for p_orig_embedding in prompt_embeddings_batch:
                p_embedding = p_orig_embedding.to(pert_device) # Ensure same device
                L_p = p_embedding.shape[0]  # Length of the current prompt embedding

                if L_p >= L_pert:
                    # Prompt embedding is longer than or equal to the perturbation length.
                    # Perturb the prefix of p_embedding (of length L_pert) with the full perturbation_to_apply.
                    prefix_to_be_perturbed = p_embedding[:L_pert]
                    perturbed_prefix = prefix_to_be_perturbed + perturbation_to_apply
                    
                    if L_p > L_pert:
                        # If prompt is longer, concatenate the unperturbed tail.
                        unperturbed_suffix = p_embedding[L_pert:]
                        final_perturbed_embedding = torch.cat((perturbed_prefix, unperturbed_suffix), dim=0)
                    else: # L_p == L_pert
                        final_perturbed_embedding = perturbed_prefix
                else: # L_p < L_pert
                    # Prompt embedding is shorter than the perturbation length.
                    # Perturb the full prompt_embedding with a prefix of perturbation_to_apply (of length L_p).
                    perturbation_slice = perturbation_to_apply[:L_p]
                    final_perturbed_embedding = p_embedding + perturbation_slice
                
                perturbed_embeddings_list.append(final_perturbed_embedding)
            return perturbed_embeddings_list


        # Helper function to generate responses with the perturbation
        def generate_responses_with_perturbation(
            prompt_embeddings_batch: List[torch.Tensor],
            perturbation: torch.Tensor,
            response_embeddings_batch: List[torch.Tensor]
        ) -> Tuple[LLMResponses, List[torch.Tensor]]:
            if self.debug_mode:
                print(f"Generating responses with perturbation of shape {perturbation.shape}...")
            # Create new prompt embeddings by adding perturbation
            new_prompt_embeddings_batch = apply_perturbation(prompt_embeddings_batch, perturbation)

            new_responses = llm.generate_responses_forced(
                new_prompt_embeddings_batch,
                response_embeddings_batch,
                probe_to_obfuscate.activation_extraction_request if probe_to_obfuscate is not None else None,
                add_response_ending=True
            )
            return new_responses, new_prompt_embeddings_batch

        # Train the attack!
        optimizer = torch.optim.Adam([perturbation], lr=self.learning_rate, eps=self.adam_epsilon)
        num_prompts = len(prompts)

        print(f"Starting training for {self.num_epochs} epochs with batch size {batch_size}, chunk size {self.chunk_size}, probe_loss_weight {self.probe_loss_weight}")

        total_steps = 0
        flop_count = 0
        # --- Epoch Loop ---
        for epoch in range(self.num_epochs):
            print_timey(f"\n--- Starting Epoch {epoch+1}/{self.num_epochs} ---")
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_probe_loss = 0.0

            batch_pbar = tqdm.tqdm(range(0, num_prompts, batch_size), desc=f"Epoch {epoch+1} Batches")
            for batch_start in batch_pbar:
                batch_end = min(batch_start + batch_size, num_prompts)
                actual_batch_size = batch_end - batch_start

                prompt_batch = prompt_embeddings[batch_start:batch_end]
                response_batch = harmful_embeddings[batch_start:batch_end]
                target_ids_batch = harmful_token_ids_including_ending[batch_start:batch_end]

                optimizer.zero_grad()

                batch_total_loss_accumulator = 0.0
                batch_ce_loss_accumulator = 0.0
                batch_probe_loss_accumulator = 0.0

                num_chunks_in_batch = math.ceil(actual_batch_size / self.chunk_size)
                for chunk_start_in_batch in range(0, actual_batch_size, self.chunk_size):
                    chunk_end_in_batch = min(chunk_start_in_batch + self.chunk_size, actual_batch_size)
                    current_chunk_size = chunk_end_in_batch - chunk_start_in_batch

                    prompt_chunk = prompt_batch[chunk_start_in_batch:chunk_end_in_batch]
                    response_chunk = response_batch[chunk_start_in_batch:chunk_end_in_batch]
                    target_ids_chunk = target_ids_batch[chunk_start_in_batch:chunk_end_in_batch]

                    print_timey(f"Performing forward pass for chunk number {1 + (chunk_start_in_batch / self.chunk_size)} of {num_chunks_in_batch}...")
                    new_responses, _ = generate_responses_with_perturbation(prompt_chunk, perturbation, response_chunk)
                    if llm.debug_mode:
                        print_timey(f"Forward pass for chunk number {1 + (chunk_start_in_batch / self.chunk_size)} of {num_chunks_in_batch} complete.")
                    new_logits = new_responses.responses_logits
                    for resp in new_logits:
                        flop_count += calculate_forward_flops(llm.num_params, resp.shape[0])
                    for pr in prompt_chunk:
                        flop_count += calculate_forward_flops(llm.num_params, pr.shape[0])
                    if probe_to_obfuscate is not None:
                        assert new_responses.activation_layers[0][0].requires_grad, "Activation layers should require grad"

                    # --- Calculate CE Loss for the chunk ---
                    ce_loss = 0.0
                    valid_indices = 0
                    for i in range(len(new_logits)):
                        logits_i = new_logits[i]
                        target_ids_i = target_ids_chunk[i]
                        target_ids_i = target_ids_i.to(logits_i.device)

                        if logits_i.shape[0] != target_ids_i.shape[0]:
                            print(f"Warning: Mismatch in sequence length for item {i} in chunk! Logits shape {logits_i.shape}, Target IDs shape {target_ids_i.shape}. Skipping this item.")
                            continue

                        if logits_i.shape[0] == 0:
                            print(f"Warning: Zero length sequence for item {i} in chunk! Skipping.")
                            continue

                        if target_ids_chunk[i].max() >= llm.vocab_size or target_ids_chunk[i].min() < 0:
                             print(f"Warning: Target token ID out of bounds for item {i} in chunk. Skipping.")
                             continue
                        if new_logits[i].shape[0] != target_ids_chunk[i].shape[0] or new_logits[i].shape[0] == 0:
                             print(f"Warning: Mismatch or zero length for item {i}. Logits {new_logits[i].shape}, Targets {target_ids_chunk[i].shape}. Skipping.")
                             continue

                        ce_loss += torch.nn.functional.cross_entropy(
                            new_logits[i],
                            target_ids_chunk[i].to(new_logits[i].device)
                        )
                        valid_indices += 1

                    if valid_indices > 0:
                        ce_loss /= valid_indices
                    else:
                        print("Warning: No valid items found in chunk to compute CE loss.")
                        ce_loss = torch.tensor(0.0, device=perturbation.device, requires_grad=True)

                    # --- Calculate Probe Loss for the chunk ---
                    probe_loss_term = 0.0
                    if probe_to_obfuscate is not None and valid_indices > 0:
                        probe_scores = probe_to_obfuscate.compute_scores(new_responses)
                        if probe_scores and len(probe_scores) == valid_indices:
                            if isinstance(probe_scores[0], torch.Tensor):
                                probe_loss_term = sum(s.detach() if s.requires_grad else s for s in probe_scores) / valid_indices
                                probe_loss_for_opt = sum(probe_scores) / valid_indices
                            else:
                                probe_loss_term = sum(probe_scores) / valid_indices
                                probe_loss_for_opt = torch.tensor(probe_loss_term, device=perturbation.device, requires_grad=False)
                        elif probe_scores:
                             print(f"Warning: Number of probe scores ({len(probe_scores)}) doesn't match valid indices ({valid_indices}). Skipping probe loss term for chunk.")
                             probe_loss_term = 0.0
                             probe_loss_for_opt = torch.tensor(0.0, device=perturbation.device, requires_grad=False)
                        else:
                             probe_loss_for_opt = torch.tensor(0.0, device=perturbation.device, requires_grad=False)

                    # --- Combine Losses with Weighting ---
                    total_loss = ce_loss
                    if probe_to_obfuscate is not None and valid_indices > 0:
                        total_loss = total_loss + self.probe_loss_weight * probe_loss_for_opt

                    # --- Accumulate Gradients ---
                    scaled_loss = total_loss / num_chunks_in_batch

                    # --- Accumulate Loss Values for Reporting ---
                    batch_total_loss_accumulator += scaled_loss.item() * current_chunk_size
                    batch_ce_loss_accumulator += (ce_loss.item() / num_chunks_in_batch) * current_chunk_size
                    batch_probe_loss_accumulator += (probe_loss_term / num_chunks_in_batch) * self.probe_loss_weight * current_chunk_size

                    print_timey(f"About to call .backward() for chunk number {1 + (chunk_start_in_batch / self.chunk_size)} of {num_chunks_in_batch}...")
                    scaled_loss.backward()
                    for resp in new_logits:
                        flop_count += calculate_backward_flops(llm.num_params, resp.shape[0])
                    for pr in prompt_chunk:
                        flop_count += calculate_backward_flops(llm.num_params, pr.shape[0])

                    if llm.debug_mode:
                        print_timey(f".backward() for chunk number {1 + (chunk_start_in_batch / self.chunk_size)} of {num_chunks_in_batch} complete.")

                    if self.debug_mode:
                        grad_norm_chunk = perturbation.grad.norm().item() if perturbation.grad is not None else 0
                        probe_val = probe_loss_term
                        print(f"  Chunk Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Probe: {probe_val:.4f}, Weighted Probe: {self.probe_loss_weight * probe_val:.4f}), Scaled Loss: {scaled_loss.item():.4f}, Grad Norm (chunk): {grad_norm_chunk:.4f}")

                print_timey(f"Chunks complete! About to call optimizer.step()...")
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
                optimizer.step()
                print_timey(f"Just finished optimizer.step() for batch at index {batch_start}!")
                print_mem_usage()
                total_steps += 1
                self.clip_perturbation_per_row(perturbation, self.epsilon)

                avg_batch_total_loss = batch_total_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0
                avg_batch_ce_loss = batch_ce_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0
                avg_batch_probe_loss = batch_probe_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0

                batch_pbar.set_postfix({
                    "Total Loss": f"{avg_batch_total_loss:.4f}",
                    "Probe Loss (Weighted)": f"{avg_batch_probe_loss:.4f}",
                    "CE Loss": f"{avg_batch_ce_loss:.4f}",
                    "Grad Norm": f"{grad_norm_before_clip.item():.4f}",
                    "Perturbation Norm": f"{perturbation.norm().item():.4f}"
                })
                epoch_loss += batch_total_loss_accumulator
                epoch_ce_loss += batch_ce_loss_accumulator
                epoch_probe_loss += batch_probe_loss_accumulator

                if total_steps in callback_steps:
                    attack_details = AttackDetails(
                        flop_cost=flop_count,
                        generated_embedding_prompts=apply_perturbation(prompt_embeddings, perturbation),
                        generated_embedding_attack_function=lambda ps: apply_perturbation(ps, perturbation),
                        steps_trained=total_steps
                    )
                    for callback in callbacks:
                        print_timey(f"Executing callback {callback.__name__} for step {total_steps}...")
                        callback(attack_details)

                if (total_steps) % 128 == 0:
                    print(f"total_steps % 128 == 0! Decreasing LR by 25% (from {self.learning_rate} to {self.learning_rate * 0.75:.4f})")
                    self.learning_rate *= 0.75
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.75
                
                if self.max_steps > 0 and total_steps >= self.max_steps:
                    print(f"Reached max steps ({self.max_steps}). Stopping training.")
                    break

            if self.max_steps > 0 and total_steps >= self.max_steps:
                print(f"Reached max steps ({self.max_steps}). Stopping training.")
                break

            avg_epoch_loss = epoch_loss / num_prompts if num_prompts > 0 else 0
            avg_epoch_ce_loss = epoch_ce_loss / num_prompts if num_prompts > 0 else 0
            avg_epoch_probe_loss = epoch_probe_loss / num_prompts if num_prompts > 0 else 0
            print_timey(f"--- Epoch {epoch+1} Finished --- Avg Epoch Loss: {avg_epoch_loss:.4f} (CE: {avg_epoch_ce_loss:.4f}, Weighted Probe: {avg_epoch_probe_loss:.4f}) ---")

        if generate_final_responses:
            print_timey("Training finished. Generating final responses with the optimized perturbation...")
            final_responses, final_prompt_embeddings = generate_responses_with_perturbation(
                prompt_embeddings,
                perturbation,
                harmful_embeddings
            )
            print_timey("Final response generation complete.")
        else:
            final_responses = None
            final_prompt_embeddings = None

        attack_details = AttackDetails(
            flop_cost=flop_count,
            generated_embedding_prompts=final_prompt_embeddings,
            generated_embedding_attack_function=lambda ps: apply_perturbation(ps, perturbation),
            steps_trained=total_steps
        )
        return final_responses, attack_details

    @property
    def is_universal(self) -> bool:
        """Whether this attack is universal or per-prompt."""
        return True

    @property
    def box_requirement(self) -> BoxRequirement:
        """Whether this attack is white-box (activations) / black-box (tokens) / gray-box (logits)."""
        return BoxRequirement.WHITE_BOX

    @property
    def can_perform_obfuscation(self) -> bool:
        """Whether this attack can perform probe-obfuscation."""
        return True

    def __str__(self) -> str:
        return f"PerturbationAttack(num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, batch_size={self.batch_size}, epsilon={self.epsilon}, probe_loss_weight={self.probe_loss_weight}, debug_mode={self.debug_mode}, chunk_size={self.chunk_size}, max_steps={self.max_steps})"