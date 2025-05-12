from typing import Any

import nanogcg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.llms.llm import LLMResponses
from oat_evaluation.probes.probe import Probe


class GCGAttack(Attack):
    """GCG attack. Uses GraySwan's nanoGCG library."""

    def __init__(
        self,
        num_steps: int = 250,
        optim_str_init: str | list[str] = "x x x x x x x x x x x x x x x x x x x x",
        search_width: int = 512,
        batch_size: int = 128,
        use_probe_sampling: bool = False,
        topk: int = 256,
        n_replace: int = 1,
        buffer_size: int = 0,
        use_mellowmax: bool = False,
        mellowmax_alpha: float = 1.0,
        early_stop: bool = False,
        use_prefix_cache: bool = True,
        allow_non_ascii: bool = False,
        filter_ids: bool = True,
        add_space_before_target: bool = False,
        seed: int | None = None,
        verbosity: str = "WARNING",
    ):
        self.num_steps = num_steps
        self.optim_str_init = optim_str_init
        self.search_width = search_width
        self.batch_size = batch_size
        self.use_probe_sampling = use_probe_sampling
        self.topk = topk
        self.n_replace = n_replace
        self.buffer_size = buffer_size
        self.use_mellowmax = use_mellowmax
        self.mellowmax_alpha = mellowmax_alpha
        self.early_stop = early_stop
        self.use_prefix_cache = use_prefix_cache
        self.allow_non_ascii = allow_non_ascii
        self.filter_ids = filter_ids
        self.add_space_before_target = add_space_before_target
        self.seed = seed
        self.verbosity = verbosity

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
            raise NotImplementedError("GCG attack does not support obfuscation")

        probe_sampling_config = None
        if self.use_probe_sampling:
            draft_model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2", torch_dtype=llm._model.dtype
            ).to(llm.device)
            draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            probe_sampling_config = nanogcg.ProbeSamplingConfig(
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,  # type: ignore
            )

        use_cache = (
            False if "google" in llm._model.name_or_path else self.use_prefix_cache
        )
        attack_config = nanogcg.GCGConfig(
            num_steps=self.num_steps,
            optim_str_init=self.optim_str_init,
            search_width=self.search_width,
            batch_size=self.batch_size,
            topk=self.topk,
            n_replace=self.n_replace,
            buffer_size=self.buffer_size,
            use_mellowmax=self.use_mellowmax,
            mellowmax_alpha=self.mellowmax_alpha,
            early_stop=self.early_stop,
            use_prefix_cache=use_cache,
            allow_non_ascii=self.allow_non_ascii,
            filter_ids=self.filter_ids,
            add_space_before_target=self.add_space_before_target,
            seed=self.seed,
            verbosity=self.verbosity,
            probe_sampling_config=probe_sampling_config,
        )

        responses_collection = LLMResponses(
            responses_strings=[],
            responses_logits=[],
            activation_layers=[],
        )
        for prompt, response in zip(prompts, harmful_responses):
            result = nanogcg.run(
                model=llm._model,
                tokenizer=llm._tokenizer,  # type: ignore
                messages=prompt,
                target=response,
                config=attack_config,
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
        return BoxRequirement.WHITE_BOX

    @property
    def can_perform_obfuscation(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__
