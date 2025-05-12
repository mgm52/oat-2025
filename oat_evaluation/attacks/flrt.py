from typing import Any, List, Optional, Callable
import torch

import nanoflrt

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.llms.llm import LLMResponses
from oat_evaluation.probes.probe import Probe
from oat_evaluation.utils import calculate_forward_flops


class FLRTAttack(Attack):
    def __init__(
        self,
        activation_loss_weight: float = 1.0,
        k1: int = 16,
        k2: int = 64,
        num_steps: int = 500,
        init_len: int = 10,
        buffer_size: int = 10,
        p_add: float = 0.5,
        p_swap: float = 0.25,
        p_del: float = 0.25,
        seed: int | None = 42,
        verbose: bool = False,
    ):
        self.k1 = k1
        self.k2 = k2
        self.num_steps = num_steps
        self.init_len = init_len
        self.buffer_size = buffer_size
        self.p_add = p_add
        self.p_swap = p_swap
        self.p_del = p_del
        self.seed = seed
        self.verbose = verbose
        self.activation_loss_weight = activation_loss_weight

    @property
    def uses_api_llm(self) -> bool:
        """Whether this attack uses an API-based LLM."""
        return False

    def run_attack(
        self,
        llm: AutoLLM,
        prompts: List[str],
        harmful_responses: List[str] = None,
        probe_to_obfuscate: Optional[Probe] = None,
        probe_refusal_threshold: Optional[float] = None,
        generate_final_responses: bool = True,
        callback_steps: Optional[List[int]] = None,
        callbacks: Optional[List[Callable[[AttackDetails], Any]]] = None,
    ) -> tuple[LLMResponses, AttackDetails]:
        """
        Generate harmful responses for the given prompts by attacking the LLM.

        Args:
            llm: The LLM to attack
            prompts: The original prompts to attack
            harmful_responses: Optional target responses, e.g. for soft-suffix attacks
            probe_to_obfuscate: Optional probe to obfuscate against
            probe_refusal_threshold: Optional threshold for probe refusal
            generate_final_responses: Whether to run the final attack or just return the attack function
            callback_steps: List of steps at which to execute callbacks
            callbacks: List of callback functions that take AttackDetails as input

        Returns:
            Tuple containing LLMResponses and AttackDetails.
        """
        assert harmful_responses is not None, (
            "Harmful responses must be provided for FLRT attack"
        )
        assert len(prompts) == len(harmful_responses), (
            "Number of prompts must match number of harmful responses for FLRT attack"
        )
        if probe_to_obfuscate is not None:
            raise NotImplementedError("FLRT attack does not support obfuscation")

        use_kv_cache = False if "google" in llm._model.name_or_path else True
        attack_config = nanoflrt.FLRTConfig(
            k1=self.k1,
            k2=self.k2,
            num_steps=self.num_steps,
            init_len=self.init_len,
            buffer_size=self.buffer_size,
            p_add=self.p_add,
            p_swap=self.p_swap,
            p_del=self.p_del,
            seed=self.seed,
            verbose=self.verbose,
            activation_loss_weight=self.activation_loss_weight,
            use_kv_cache=use_kv_cache,
        )

        prompt_to_attacked_prompt_map = {}
        total_flop_cost = 0
        all_attacked_prompts_for_final_response = []

        for prompt, response in zip(prompts, harmful_responses):
            # Estimate FLOPs for this prompt's FLRT run
            # FLRT is a search algorithm that primarily involves forward passes
            # Rough estimate: num_steps * k2 (top candidates considered) * forward_flops
            num_tokens_in_prefix_plus_init = llm.get_num_tokens_in_str(prompt) + self.init_len
            approx_seq_len = num_tokens_in_prefix_plus_init + llm.get_num_tokens_in_str(response)
            forward_flops_one_pass = calculate_forward_flops(llm.num_params, approx_seq_len)
            # Rough average passes per step
            passes_per_step = self.k1 + self.p_add + self.p_swap
            current_prompt_flops = self.num_steps * passes_per_step * forward_flops_one_pass
            total_flop_cost += current_prompt_flops

            result = nanoflrt.run(
                model=llm._model,
                tokenizer=llm._tokenizer,
                messages=prompt,
                target=response,
                config=attack_config,
                activation_loss_fn=None,  # TODO: implement obfuscation loss
            )
            suffix = result.best_string
            prompt_with_suffix = prompt + suffix
            prompt_to_attacked_prompt_map[prompt] = prompt_with_suffix
            all_attacked_prompts_for_final_response.append(prompt_with_suffix)

        def _flrt_attack_fn(input_prompts: List[str]) -> List[str]:
            return [prompt_to_attacked_prompt_map.get(p, p) for p in input_prompts]

        final_attack_details = AttackDetails(
            flop_cost=total_flop_cost,
            generated_str_attack_function=_flrt_attack_fn,
            generated_str_prompts=[prompt_to_attacked_prompt_map[p] for p in prompts] if prompt_to_attacked_prompt_map else None,
            steps_trained=self.num_steps
        )

        # If generate_final_responses is True, generate responses for all attacked prompts
        if generate_final_responses:
            final_llm_responses = llm.generate_responses(all_attacked_prompts_for_final_response)
        else:
            final_llm_responses = LLMResponses([], [], [])

        return final_llm_responses, final_attack_details

    @property
    def is_universal(self) -> bool:
        return False

    @property
    def box_requirement(self) -> BoxRequirement:
        return BoxRequirement.GRAY_BOX

    @property
    def can_perform_obfuscation(self) -> bool:
        # TODO: Implement obfuscation, should only need to pass in activation_loss_fn
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_slow(self) -> bool:
        return True
