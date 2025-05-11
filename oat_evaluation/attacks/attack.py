from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Any, Optional, Tuple

import torch
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe


class BoxRequirement(Enum):
    WHITE_BOX = "white_box" # uses model activations
    GRAY_BOX = "gray_box" # uses model output logits
    BLACK_BOX = "black_box" # uses model output tokens


# 624 bfloat16 TFLOPS for A100 SXM: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
flops_per_second = 624 * 10**12
# Based on runpod pricing
usd_per_second = 1.89 / 3600
USD_PER_FLOP = usd_per_second / flops_per_second


def usd_to_flop(usd: float) -> float:
    return int(usd / USD_PER_FLOP)


def flop_to_usd(flop: float) -> float:
    return flop * USD_PER_FLOP


@dataclass
class AttackDetails:
    equivalent_flop_cost: Optional[int] = 0  # Obtained by converting the dollars to FLOPs too, corresponds to num_dollars_cost
    generated_str_prompts: Optional[List[str]] = None
    generated_embedding_prompts: Optional[List[torch.Tensor]] = None
    generated_embedding_attack_function: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None
    generated_str_attack_function: Optional[Callable[[List[str]], List[str]]] = None
    
    equivalent_usd_cost: Optional[float] = 0.  # Obtained by converting the FLOPs to dollars too, corresponds to flop_cost
    local_flop_cost: Optional[int] = 0
    api_usd_cost: Optional[float] = 0.
    
    used_api_llm: bool = False  # Whether API calls were used
    num_total_tokens: Optional[int] = 0
    num_api_calls: Optional[int] = 0
    
    @property
    def flop_cost(self) -> Optional[int]:
        return self.equivalent_flop_cost

    def __add__(self, other: 'AttackDetails') -> 'AttackDetails':
        if not isinstance(other, AttackDetails):
            raise NotImplementedError(f"Expected type AttackDetails, got type {type(other)}")
        return AttackDetails(equivalent_flop_cost=(self.equivalent_flop_cost or 0) + (other.equivalent_flop_cost or 0),
                             generated_str_prompts=(self.generated_str_prompts or []) + (other.generated_str_prompts or []),
                             generated_embedding_prompts=(self.generated_embedding_prompts or []) + (other.generated_embedding_prompts or []),
                             generated_embedding_attack_function=(self.generated_embedding_attack_function or other.generated_embedding_attack_function),
                             generated_str_attack_function=(self.generated_str_attack_function or other.generated_str_attack_function),
                             equivalent_usd_cost=(self.equivalent_usd_cost or 0.) + (other.equivalent_usd_cost or 0.),
                             local_flop_cost=(self.local_flop_cost or 0) + (other.local_flop_cost or 0),
                             api_usd_cost=(self.api_usd_cost or 0.) + (other.api_usd_cost or 0.),
                             used_api_llm=self.used_api_llm or other.used_api_llm,
                             num_total_tokens=self.num_total_tokens + other.num_total_tokens,
                             num_api_calls=self.num_api_calls + other.num_api_calls,
        )


class Attack(ABC):
    @abstractmethod
    def run_attack(
        self,
        llm: LLM,
        prompts: List[str],
        harmful_responses: Optional[List[str]] = None,
        probe_to_obfuscate: Optional[Probe] = None,
        probe_refusal_threshold: Optional[float] = None,
        generate_final_responses: bool = True,
        callback_steps: Optional[List[int]] = None,
        callbacks: Optional[List[Callable[[AttackDetails], Any]]] = None
    ) -> Tuple[LLMResponses, AttackDetails]:
        """
        Generate harmful responses for the given prompts by attacking the LLM.
        
        Args:
            llm: The LLM to attack
            prompts: The original prompts to attack
            harmful_responses: Optional target responses, e.g. for soft-suffix attacks
            probe_to_obfuscate: Optional probe to obfuscate against -- should pair use this??
            generate_final_responses: Whether to run the final attack or just return the attack function -- when is this needed??
            callback_steps: List of steps at which to execute callbacks -- what are steps??
            callbacks: List of callback functions that take AttackDetails as input
            
        Returns:
            Tuple containing LLMResponses and AttackDetails.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_universal(self) -> bool:
        """Whether this attack is universal or per-prompt."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_cross_model_transferrable(self) -> bool:
        """Whether this attack can transfer across models. (e.g. GCG)"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def box_requirement(self) -> BoxRequirement:
        """Whether this attack is white-box (activations) / black-box (tokens) / gray-box (logits)."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def can_perform_obfuscation(self) -> bool:
        """Whether this attack can perform probe-obfuscation."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The name of the attack"""
        return self.__class__.__name__

    # String representation
    def __str__(self) -> str:
        return self.name
