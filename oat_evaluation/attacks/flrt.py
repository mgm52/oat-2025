from typing import Any

import nanoflrt

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.llms.llm import LLMResponses
from oat_evaluation.probes.probe import Probe


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

    def run_attack(
        self,
        llm: AutoLLM,
        prompts: list[str],
        harmful_responses: list[str] | None = None,
        probe_to_obfuscate: Probe | None = None,
        run_final_attack: bool = True,
        **generate_kwargs: Any,
    ) -> tuple[LLMResponses, AttackDetails]:
        """
        Generate harmful responses for the given prompts by attacking the LLM.

        Args:
            llm: The LLM to attack
            prompts: The original prompts to attack
            harmful_responses: Optional target responses, e.g. for soft-suffix attacks
            probe_to_obfuscate: Optional probe to obfuscate against
            run_final_attack: Whether to run the final attack or just return the attack function

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
            # TODO: implement probe obfuscation
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

        responses_collection = LLMResponses(
            responses_strings=[],
            responses_logits=[],
            activation_layers=[],
        )
        for prompt, response in zip(prompts, harmful_responses):
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
            response = llm.generate_responses(
                prompts=[prompt_with_suffix],
                **generate_kwargs,
            )
            responses_collection.responses_strings.extend(response.responses_strings)
            responses_collection.responses_logits.extend(response.responses_logits)
            if response.activation_layers is not None:
                responses_collection.activation_layers.extend(
                    response.activation_layers
                )

        # TODO: add attack details
        return responses_collection, AttackDetails()

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
