from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe

class BoxRequirement(Enum):
    WHITE_BOX = "white_box" # uses model activations
    GRAY_BOX = "gray_box" # uses model output logits
    BLACK_BOX = "black_box" # uses model output tokens

class AttackDetails():
    def __init__(
        self,
        flop_cost: Optional[int] = None,
        generated_str_prompts: Optional[List[str]] = None,
        generated_embedding_prompts: Optional[List[torch.Tensor]] = None,
        generated_embedding_attack_function: Optional[Callable] = None
    ):
        self.flop_cost = flop_cost
        self.generated_str_prompts = generated_str_prompts
        self.generated_embedding_prompts = generated_embedding_prompts
        self.generated_embedding_attack_function = generated_embedding_attack_function

class Attack(ABC):
    @abstractmethod
    def run_attack(
        self,
        llm: LLM,
        prompts: List[str],
        harmful_responses: List[str] = None,
        probe_to_obfuscate: Optional[Probe] = None,
        generate_final_responses: bool = True,
        callback_steps: List[int] = None,
        callbacks: List[Callable[[AttackDetails], Any]] = None
    ) -> Tuple[LLMResponses, AttackDetails]:
        """
        Generate harmful responses for the given prompts by attacking the LLM.
        
        Args:
            llm: The LLM to attack
            prompts: The original prompts to attack
            harmful_responses: Optional target responses, e.g. for soft-suffix attacks
            probe_to_obfuscate: Optional probe to obfuscate against
            generate_final_responses: Whether to run the final attack or just return the attack function
            callback_steps: List of steps at which to execute callbacks
            callbacks: List of callback functions that take AttackDetails as input
            
        Returns:
            Tuple containing LLMResponses and AttackDetails.
        """
        pass

    @property
    @abstractmethod
    def is_universal(self) -> bool:
        """Whether this attack is universal or per-prompt."""
        pass

    @property
    @abstractmethod
    def box_requirement(self) -> BoxRequirement:
        """Whether this attack is white-box (activations) / black-box (tokens) / gray-box (logits)."""
        pass

    @property
    @abstractmethod
    def can_perform_obfuscation(self) -> bool:
        """Whether this attack can perform probe-obfuscation."""
        pass

    @property
    def name(self) -> str:
        """The name of the attack"""
        return self.__class__.__name__
