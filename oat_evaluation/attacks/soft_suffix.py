from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import tqdm
from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe

class SoftSuffixAttack(Attack):
    def __init__(
        self,
        suffix_length: int,
        training_steps: int,
        learning_rate: float,
        epsilon: float = 5.0, # I think default 50.0 in original abhay code
        debug_mode: bool = False
    ):
        self.suffix_length = suffix_length
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.debug_mode = debug_mode

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
        probe_to_obfuscate: Optional[Probe] = None
    ) -> Tuple[LLMResponses, AttackDetails]:
        
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

        prompt_embeddings = [llm.string_to_embedding(p).detach() for p in prompts]
        harmful_embeddings = [llm.string_to_embedding(r).detach() for r in harmful_responses]
        harmful_token_ids = [llm.string_to_token_ids(r).long() for r in harmful_responses]
        harmful_token_ids_including_ending = [llm.string_to_token_ids(r, add_response_ending=True).long() for r in harmful_responses]

        if self.debug_mode:
            print("-" * 20)
            print(f"Initial suffix requires_grad: {suffix.requires_grad}") # Should be True
            if prompt_embeddings:
                print(f"Prompt embedding requires_grad: {prompt_embeddings[0].requires_grad}") # Should be False
            if harmful_embeddings:
                print(f"Harmful embedding requires_grad: {harmful_embeddings[0].requires_grad}") # Should be False
            print(f"Pad embedding requires_grad: {llm.pad_embedding.requires_grad}") # Should be False
            print("-" * 20)

        # Ensure pad_embedding was detached during LLM init (verify this in AutoLLM)
        if llm.pad_embedding.requires_grad:
            raise ValueError("LLM's pad_embedding requires grad! Detach it in AutoLLM.__init__.")
        if any(p.requires_grad for p in prompt_embeddings):
            raise ValueError("Prompt embeddings require grad! Detach them.")
        if any(h.requires_grad for h in harmful_embeddings):
            raise ValueError("Harmful embeddings require grad! Detach them.")

        if self.debug_mode:
            print("Max harmful token ID:", max(r.max().item() for r in harmful_token_ids))
            print("Vocab size:", llm.vocab_size)
            print(f"Harmful token ID dtypes: {[r.dtype for r in harmful_token_ids]}")
        
        # Helper function to generate responses with the suffix
        def generate_responses_with_suffix(prompt_embeddings: List[torch.Tensor], suffix: torch.Tensor, response_embeddings: List[torch.Tensor]) -> Tuple[LLMResponses, List[torch.Tensor]]:
            if self.debug_mode:
                print(f"About to generate responses with suffix of shape {suffix.shape}: {suffix}...")
            new_prompt_embeddings = [torch.cat([p, suffix], dim=0) for p in prompt_embeddings]
            new_responses = llm.generate_responses_forced(
                new_prompt_embeddings,
                response_embeddings,
                probe_to_obfuscate.activation_extraction_request if probe_to_obfuscate is not None else None,
                add_response_ending=True # so the model can learn to stop after harmful response
            )
            return new_responses, new_prompt_embeddings

        # Train the attack!
        optimizer = torch.optim.Adam([suffix], lr=self.learning_rate)
        for _ in tqdm.tqdm(range(self.training_steps)):
            optimizer.zero_grad()

            # Generate the responses with the suffix
            new_responses, _ = generate_responses_with_suffix(prompt_embeddings, suffix, harmful_embeddings)
            new_logits = new_responses.responses_logits

            if self.debug_mode:
                with torch.no_grad():
                    print(f"Actual next-token predictions were {new_responses.responses_strings}")
                    print(f"New logits are len {len(new_logits)}, shapes {[l.shape for l in new_logits]}")
                    print(f"And we're about to compare them to the target harmful embeddings of shapes {[r.shape for r in harmful_token_ids]}, dtypes {[r.dtype for r in harmful_token_ids]}")
            
            # Get the loss (CE between the logits and the target harmful embeddings)
            assert len(new_logits) == len(harmful_token_ids_including_ending)
            assert new_logits[0].shape[0] == len(harmful_token_ids_including_ending[0])

            loss = sum([
                torch.nn.functional.cross_entropy(l, r)
                for l, r in zip(new_logits, harmful_token_ids_including_ending)
            ])

            if probe_to_obfuscate is not None:
                probe_scores = probe_to_obfuscate.compute_scores(new_responses)
                loss += sum(probe_scores) / len(probe_scores)

            if self.debug_mode:
                print(f"Loss is {loss}")

            # Inside the training loop in run_attack
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([suffix], max_norm=1.0) # Clip and get norm before clipping

            if self.debug_mode:
                # --- Debugging ---
                print(f"Gradient norm (before clipping): {grad_norm.item()}")
                if suffix.grad is None:
                    print("WARNING: suffix.grad is None!")
                elif torch.isnan(suffix.grad).any():
                    print("WARNING: NaNs detected in suffix.grad!")
                    # Optionally: optimizer.zero_grad() # Skip update if grad is bad
                    # Optionally: continue # Skip rest of loop iteration
                elif torch.isinf(suffix.grad).any():
                    print("WARNING: Infs detected in suffix.grad!")
                    # Optionally: optimizer.zero_grad()
                    # Optionally: continue
                # --- End Debugging ---

            optimizer.step()

            # Re-clip suffix to ensure it stays within epsilon
            self.clip_suffix_per_row(suffix, self.epsilon)

            if self.debug_mode:
                # --- Debugging ---
                print(f"Suffix norm after step/clip: {suffix.norm().item()}")
                # --- End Debugging ---
        
        # Return the final responses
        final_responses, final_prompt_embeddings = generate_responses_with_suffix(prompt_embeddings, suffix, harmful_embeddings)
        return final_responses, AttackDetails(generated_embedding_prompts=final_prompt_embeddings)

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