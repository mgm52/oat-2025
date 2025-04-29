import math
import torch
import tqdm
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe

def print_mem_usage():
    print(f"    [Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB]")
    print(f"    [Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB]")


class SoftSuffixAttack(Attack):
    def __init__(
        self,
        suffix_length: int,
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
             
        self.suffix_length = suffix_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.probe_loss_weight = probe_loss_weight
        self.debug_mode = debug_mode
        self.chunk_size = chunk_size
        self.adam_epsilon = adam_epsilon
        self.max_steps = max_steps
        print(f"SoftSuffixAttack initialized with suffix_length {suffix_length}, num_epochs {num_epochs}, learning_rate {learning_rate}, batch_size {batch_size}, epsilon {epsilon}, probe_loss_weight {probe_loss_weight}, debug_mode {debug_mode}, chunk_size {chunk_size}")

    def clip_suffix(self, suffix: torch.Tensor, epsilon: float) -> None:
        """Clips the suffix's total L2 norm in-place to be <= epsilon."""
        with torch.no_grad():
            # Compute the total L2 norm of the entire suffix
            norm = suffix.norm(p=2)
            if norm > epsilon:
                suffix *= (epsilon / norm)

    def clip_suffix_per_row(self, suffix: torch.Tensor, epsilon: float) -> None:
        """Clips each row of `suffix` to be within L2 = epsilon."""
        with torch.no_grad():
            row_norms = suffix.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            scale = (row_norms / epsilon).clamp(min=1.0)
            suffix.div_(scale)  # divides each row by scale if norm > epsilon

    def run_attack(
        self,
        llm: LLM,
        prompts: List[str],
        harmful_responses: List[str] = None,
        probe_to_obfuscate: Optional[Probe] = None,
        run_final_attack: bool = True
    ) -> Tuple[LLMResponses, AttackDetails]:

        torch.enable_grad()

        batch_size = min(self.batch_size, len(prompts))

        if harmful_responses is None or len(harmful_responses) == 0:
            raise ValueError("SoftSuffixAttack requires harmful_responses.")
        if len(prompts) != len(harmful_responses):
             # Replicate harmful responses if only one is provided for multiple prompts
            if len(harmful_responses) == 1 and len(prompts) > 1:
                print(f"Warning: Only one harmful_response provided for {len(prompts)} prompts. Replicating the harmful response.")
                harmful_responses = harmful_responses * len(prompts)
            else:
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of harmful_responses ({len(harmful_responses)}).")

        # Start with a random suffix (shape (suffix_length, embedding_size))
        # 1. Create a suffix parameter
        suffix = torch.zeros(
            self.suffix_length,
            llm.embedding_size,
            device=llm.device,
            dtype=llm.dtype,
            requires_grad=True
        )

        # 2. Apply Kaiming uniform initialization in-place
        torch.nn.init.kaiming_uniform_(suffix)

        # 3. Immediately clip it to epsilon
        self.clip_suffix_per_row(suffix, self.epsilon)

        # Ensure right dtype
        suffix = suffix.to(llm.dtype)
        self.suffix = suffix # to save it later

        # --- Precompute embeddings and token IDs outside the loop ---
        print("Precomputing embeddings...")
        prompt_embeddings = [llm.string_to_embedding(p).detach() for p in tqdm.tqdm(prompts)]
        harmful_embeddings = [llm.string_to_embedding(r).detach() for r in tqdm.tqdm(harmful_responses)]
        harmful_token_ids = [llm.string_to_token_ids(r).long() for r in harmful_responses] # Targets for CE loss
        harmful_token_ids_including_ending = [llm.string_to_token_ids(r, add_response_ending=True).long().to(llm.device) for r in harmful_responses] # Targets for CE loss
        print("Embeddings precomputed.")
        # --- End precomputation ---

        if self.debug_mode:
            print("-" * 20)
            print(f"Initial suffix requires_grad: {suffix.requires_grad}") # Should be True
            if prompt_embeddings:
                print(f"Prompt embedding requires_grad: {prompt_embeddings[0].requires_grad}") # Should be False
            if harmful_embeddings:
                print(f"Harmful embedding requires_grad: {harmful_embeddings[0].requires_grad}") # Should be False
            print(f"Pad embedding requires_grad: {llm.pad_embedding.requires_grad}") # Should be False
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

        def apply_suffix(prompt_embeddings_batch: List[torch.Tensor], suffix: torch.Tensor) -> List[torch.Tensor]:
            return [torch.cat([p.to(suffix.device), suffix], dim=0) for p in prompt_embeddings_batch]

        # Helper function to generate responses with the suffix
        def generate_responses_with_suffix(
            prompt_embeddings_batch: List[torch.Tensor],
            suffix: torch.Tensor,
            response_embeddings_batch: List[torch.Tensor]
        ) -> Tuple[LLMResponses, List[torch.Tensor]]:
            if self.debug_mode:
                print(f"Generating responses with suffix of shape {suffix.shape}...")
                # print(f"Suffix value sample: {suffix[0, :5]}") # Avoid printing huge tensor
            # Create new prompt embeddings by concatenating suffix
            new_prompt_embeddings_batch = apply_suffix(prompt_embeddings_batch, suffix)

            new_responses = llm.generate_responses_forced(
                new_prompt_embeddings_batch,
                response_embeddings_batch, # These are the target *embeddings* used for forcing generation length/structure
                probe_to_obfuscate.activation_extraction_request if probe_to_obfuscate is not None else None,
                add_response_ending=True # Important: Ensure target length matches harmful_token_ids_including_ending
            )
            return new_responses, new_prompt_embeddings_batch # Return modified prompts too

        # Train the attack!
        optimizer = torch.optim.Adam([suffix], lr=self.learning_rate, eps=self.adam_epsilon)
        num_prompts = len(prompts)

        print(f"Starting training for {self.num_epochs} epochs with batch size {batch_size}, chunk size {self.chunk_size}, probe_loss_weight {self.probe_loss_weight}") # Added weight info

        total_steps = 0
        # --- Epoch Loop ---
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            epoch_loss = 0.0
            epoch_ce_loss = 0.0 # Accumulator for CE loss part
            epoch_probe_loss = 0.0 # Accumulator for probe loss part

            batch_pbar = tqdm.tqdm(range(0, num_prompts, batch_size), desc=f"Epoch {epoch+1} Batches")
            for batch_start in batch_pbar:
                batch_end = min(batch_start + batch_size, num_prompts)
                actual_batch_size = batch_end - batch_start

                prompt_batch = prompt_embeddings[batch_start:batch_end]
                response_batch = harmful_embeddings[batch_start:batch_end]
                target_ids_batch = harmful_token_ids_including_ending[batch_start:batch_end]

                optimizer.zero_grad()

                batch_total_loss_accumulator = 0.0 # Accumulate total loss values for reporting
                batch_ce_loss_accumulator = 0.0    # Accumulate CE loss values for reporting
                batch_probe_loss_accumulator = 0.0 # Accumulate probe loss values for reporting

                num_chunks_in_batch = math.ceil(actual_batch_size / self.chunk_size)
                for chunk_start_in_batch in range(0, actual_batch_size, self.chunk_size):
                    chunk_end_in_batch = min(chunk_start_in_batch + self.chunk_size, actual_batch_size)
                    current_chunk_size = chunk_end_in_batch - chunk_start_in_batch

                    prompt_chunk = prompt_batch[chunk_start_in_batch:chunk_end_in_batch]
                    response_chunk = response_batch[chunk_start_in_batch:chunk_end_in_batch]
                    target_ids_chunk = target_ids_batch[chunk_start_in_batch:chunk_end_in_batch]

                    print(f"Generating responses for chunk number {chunk_start_in_batch / self.chunk_size}...")
                    new_responses, _ = generate_responses_with_suffix(prompt_chunk, suffix, response_chunk)
                    new_logits = new_responses.responses_logits
                    assert new_responses.activation_layers[0][0].requires_grad, "Activation layers should require grad"

                    # --- Calculate CE Loss for the chunk ---
                    ce_loss = 0.0
                    valid_indices = 0
                    for i in range(len(new_logits)):
                        logits_i = new_logits[i]
                        target_ids_i = target_ids_chunk[i]
                        # Ensure target_ids are on the same device as logits
                        target_ids_i = target_ids_i.to(logits_i.device)

                        if logits_i.shape[0] != target_ids_i.shape[0]:
                            print(f"Warning: Mismatch in sequence length for item {i} in chunk! Logits shape {logits_i.shape}, Target IDs shape {target_ids_i.shape}. Skipping this item.")
                            continue

                        if logits_i.shape[0] == 0:
                            print(f"Warning: Zero length sequence for item {i} in chunk! Skipping.")
                            continue

                        # Ensure target IDs are within vocab size check
                        if target_ids_chunk[i].max() >= llm.vocab_size or target_ids_chunk[i].min() < 0:
                             print(f"Warning: Target token ID out of bounds for item {i} in chunk. Skipping.")
                             continue
                        # Length check
                        if new_logits[i].shape[0] != target_ids_chunk[i].shape[0] or new_logits[i].shape[0] == 0:
                             print(f"Warning: Mismatch or zero length for item {i}. Logits {new_logits[i].shape}, Targets {target_ids_chunk[i].shape}. Skipping.")
                             continue

                        ce_loss += torch.nn.functional.cross_entropy(
                            new_logits[i],
                            target_ids_chunk[i].to(new_logits[i].device)
                        )
                        valid_indices += 1

                    if valid_indices > 0:
                        ce_loss /= valid_indices # Average CE loss over valid items in the chunk
                    else:
                        print("Warning: No valid items found in chunk to compute CE loss.")
                        ce_loss = torch.tensor(0.0, device=suffix.device, requires_grad=True)

                    # --- Calculate Probe Loss for the chunk ---
                    probe_loss_term = 0.0
                    if probe_to_obfuscate is not None and valid_indices > 0: # Only compute if CE loss was possible
                        probe_scores = probe_to_obfuscate.compute_scores(new_responses) # Assuming probe handles potential errors / filters internally if needed
                        if probe_scores and len(probe_scores) == valid_indices: # Ensure scores match valid items
                            # Check if scores require grad, detach if necessary for accumulation
                            if isinstance(probe_scores[0], torch.Tensor):
                                probe_loss_term = sum(s.detach() if s.requires_grad else s for s in probe_scores) / valid_indices # Average probe score
                                probe_loss_for_opt = sum(probe_scores) / valid_indices # Keep grad for optimization step
                            else: # Handle non-tensor scores (e.g., floats)
                                probe_loss_term = sum(probe_scores) / valid_indices
                                probe_loss_for_opt = torch.tensor(probe_loss_term, device=suffix.device, requires_grad=False) # No grad if input wasn't tensor

                        elif probe_scores:
                             print(f"Warning: Number of probe scores ({len(probe_scores)}) doesn't match valid indices ({valid_indices}). Skipping probe loss term for chunk.")
                             probe_loss_term = 0.0
                             probe_loss_for_opt = torch.tensor(0.0, device=suffix.device, requires_grad=False)
                        else:
                             probe_loss_for_opt = torch.tensor(0.0, device=suffix.device, requires_grad=False)

                    # --- Combine Losses with Weighting ---
                    total_loss = ce_loss
                    if probe_to_obfuscate is not None and valid_indices > 0:
                        # Ensure probe_loss_for_opt requires grad if probe computation involved differentiable steps
                        # If probe_loss_for_opt came from tensors that required grad, it should have grad_fn
                        # If not, adding it won't contribute to gradients anyway.
                        total_loss = total_loss + self.probe_loss_weight * probe_loss_for_opt

                    # --- Accumulate Gradients ---
                    scaled_loss = total_loss / num_chunks_in_batch

                    # --- Accumulate Loss Values for Reporting ---
                    batch_total_loss_accumulator += scaled_loss.item() * current_chunk_size # Store scaled loss for reporting average total loss
                    batch_ce_loss_accumulator += (ce_loss.item() / num_chunks_in_batch) * current_chunk_size # Store scaled CE loss part
                    # probe_loss_term is already detached average score, scale it
                    batch_probe_loss_accumulator += (probe_loss_term / num_chunks_in_batch) * self.probe_loss_weight * current_chunk_size # Store scaled & weighted probe loss part

                    print(f"About to call .backward()...")
                    scaled_loss.backward() # Accumulate gradients on the combined, scaled loss

                    if self.debug_mode:
                        grad_norm_chunk = suffix.grad.norm().item() if suffix.grad is not None else 0
                        probe_val = probe_loss_term # Use the detached value for printing
                        print(f"  Chunk Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Probe: {probe_val:.4f}, Weighted Probe: {self.probe_loss_weight * probe_val:.4f}), Scaled Loss: {scaled_loss.item():.4f}, Grad Norm (chunk): {grad_norm_chunk:.4f}")


                # --- End of Chunk Loop ---

                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_([suffix], max_norm=1.0)
                optimizer.step()
                print(f"Just finished optimizer.step() for batch at index {batch_start}...")
                print_mem_usage()
                total_steps += 1
                self.clip_suffix_per_row(suffix, self.epsilon)

                # Calculate average losses for the batch for reporting
                avg_batch_total_loss = batch_total_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0
                avg_batch_ce_loss = batch_ce_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0
                avg_batch_probe_loss = batch_probe_loss_accumulator / actual_batch_size if actual_batch_size > 0 else 0 # This is the weighted average

                batch_pbar.set_postfix({
                    "Total Loss": f"{avg_batch_total_loss:.4f}", # Report average scaled loss
                    "Probe Loss (Weighted)": f"{avg_batch_probe_loss:.4f}", # Report avg scaled weighted probe loss
                    "CE Loss": f"{avg_batch_ce_loss:.4f}",
                    "Grad Norm": f"{grad_norm_before_clip.item():.4f}",
                    "Suffix Norm": f"{suffix.norm().item():.4f}"
                })
                epoch_loss += batch_total_loss_accumulator # Accumulate scaled loss for epoch average
                epoch_ce_loss += batch_ce_loss_accumulator
                epoch_probe_loss += batch_probe_loss_accumulator

                if (total_steps) % 128 == 0:
                    print(f"total_steps % 128 == 0! Decreasing LR by 25% (from {self.learning_rate} to {self.learning_rate * 0.75:.4f})")
                    self.learning_rate *= 0.75
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.75  # halve the learning rate
                
                if self.max_steps > 0 and total_steps >= self.max_steps:
                    print(f"Reached max steps ({self.max_steps}). Stopping training.")
                    break

            if self.max_steps > 0 and total_steps >= self.max_steps:
                print(f"Reached max steps ({self.max_steps}). Stopping training.")
                break

            # --- End of Batch Loop ---
            avg_epoch_loss = epoch_loss / num_prompts if num_prompts > 0 else 0
            avg_epoch_ce_loss = epoch_ce_loss / num_prompts if num_prompts > 0 else 0
            avg_epoch_probe_loss = epoch_probe_loss / num_prompts if num_prompts > 0 else 0
            print(f"--- Epoch {epoch+1} Finished --- Avg Epoch Loss: {avg_epoch_loss:.4f} (CE: {avg_epoch_ce_loss:.4f}, Weighted Probe: {avg_epoch_probe_loss:.4f}) ---")

        # --- End of Epoch Loop ---

        if run_final_attack:
            print("Training finished. Generating final responses with the optimized suffix...")
            # Return the final responses using the optimized suffix on the original (full) data
            final_responses, final_prompt_embeddings = generate_responses_with_suffix(
                prompt_embeddings, # Use all original prompt embeddings
                suffix,
                harmful_embeddings # Use all original harmful embeddings for forcing
            )
            print("Final response generation complete.")
        else:
            final_responses = None
            final_prompt_embeddings = None

        return final_responses, AttackDetails(generated_embedding_prompts=final_prompt_embeddings, generated_embedding_attack_function=lambda ps: apply_suffix(ps, suffix)) 

    @property
    def is_universal(self) -> bool:
        """Whether this attack is universal or per-prompt."""
        return True

    @property
    def box_requirement(self) -> BoxRequirement:
        """Whether this attack is white-box (activations) / black-box (tokens) / gray-box (logits)."""
        # Requires access to embeddings and gradients (implicitly via backward())
        return BoxRequirement.WHITE_BOX

    @property
    def can_perform_obfuscation(self) -> bool:
        """Whether this attack can perform probe-obfuscation."""
        # Can include probe loss in the optimization objective
        return True