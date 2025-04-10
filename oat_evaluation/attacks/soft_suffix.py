from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from oat_evaluation.attacks.attack import Attack, AttackDetails
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe

class SoftSuffixAttack(Attack):
    def __init__(self, suffix_length: int, training_steps: int, learning_rate: float):
        self.suffix_length = suffix_length
        self.training_steps = training_steps
        self.learning_rate = learning_rate

    def run_attack(
        self,
        llm: LLM,
        prompts: List[str],
        harmful_responses: List[str] = None,
        probe_to_obfuscate: Optional[Probe] = None
    ) -> Tuple[LLMResponses, AttackDetails]:
        
        # Start with a random suffix (shape (suffix_length, vocab_size))
        suffix = torch.randn(self.suffix_length, llm.vocab_size)
        prompt_embeddings = [llm.string_to_embedding(p) for p in prompts]
        harmful_embeddings = [llm.string_to_embedding(r) for r in harmful_responses]

        # Helper function to generate responses with the suffix
        def generate_responses_with_suffix(prompt_embeddings: List[torch.Tensor], suffix: torch.Tensor) -> Tuple[LLMResponses, List[torch.Tensor]]:
            new_prompt_embeddings = [torch.cat([p, suffix]) for p in prompt_embeddings]
            new_responses = llm.generate_responses(new_prompt_embeddings)
            return new_responses, new_prompt_embeddings

        # Train the attack!
        optimizer = torch.optim.Adam(suffix.flatten(), lr=self.learning_rate)
        for _ in range(self.training_steps):
            optimizer.zero_grad()

            # Generate the responses with the suffix
            new_responses, _ = generate_responses_with_suffix(prompt_embeddings, suffix)
            new_logits = new_responses.responses_logits
            new_probs = [torch.softmax(l, dim=-1) for l in new_logits]

            # Get the loss (CE between the probabilities and the target harmful embeddings)
            loss = sum([torch.nn.functional.cross_entropy(p, r) for p, r in zip(new_probs, harmful_embeddings)])

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

        # Return the final responses
        final_responses, final_prompt_embeddings = generate_responses_with_suffix(prompt_embeddings, suffix)
        return final_responses, AttackDetails(generated_embedding_prompts=final_prompt_embeddings)
