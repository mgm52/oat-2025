from typing import Any, List, Optional, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.llms.llm import LLMResponses
from oat_evaluation.probes.probe import Probe
from oat_evaluation.utils import calculate_forward_flops, calculate_backward_flops, print_timey, print_mem_usage

import nanogcg

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
            "Harmful responses must be provided for GCG attack"
        )
        assert len(prompts) == len(harmful_responses), (
            "Number of prompts must match number of harmful responses for GCG attack"
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

        prompt_to_attacked_prompt_map = {}
        total_flop_cost = 0
        all_attacked_prompts_for_final_response = []

        for original_prompt, target_response in zip(prompts, harmful_responses):
            # --- Start of new FLOP calculation logic ---

            # 1. Determine the optim_str to use for length calculations
            #    If self.optim_str_init is a list, GCG typically uses it to initialize a buffer.
            #    For FLOPs, we need a representative string. Let's use the first one if it's a list,
            #    or the string itself. GCG expects all strings in the list to tokenize to the same length.
            # optim_str_for_len_calc = self.optim_str_init
            # if isinstance(self.optim_str_init, list):
            #     if not self.optim_str_init: # Handle empty list case
            #         optim_str_for_len_calc = " ".join(["x"] * 20) # Default GCG-like init
            #     else:
            #         optim_str_for_len_calc = self.optim_str_init[0]

            # # 2. More accurate count of optimized tokens for backward pass
            # num_optim_tokens = llm.get_num_tokens_in_str(optim_str_for_len_calc)

            # # 3. Estimate sequence length for FORWARD passes on the TARGET model
            # #    This estimates the full sequence length GCG works with.
            # #    It's an overestimate if use_prefix_cache=True, but simpler than precise calculation.
            # approx_seq_len_target_fwd = llm.get_num_tokens_in_str(original_prompt + optim_str_for_len_calc) + \
            #                         llm.get_num_tokens_in_str(target_response)

            # # 4. Calculate FLOPs for a single forward and backward pass on the TARGET model
            # flops_fwd_pass_target = calculate_forward_flops(llm.num_params, approx_seq_len_target_fwd)
            # flops_bwd_pass_target = calculate_backward_flops(llm.num_params, num_optim_tokens)

            # # 5. Cost of ONE gradient calculation step (1 fwd + 1 bwd) on TARGET model
            # cost_one_grad_calc_step_target = flops_fwd_pass_target + flops_bwd_pass_target

            # # 6. Cost of evaluating all `search_width` candidates on TARGET model (search_width fwd passes)
            # cost_eval_candidates_target = self.search_width * flops_fwd_pass_target

            # # 7. Calculate FLOPs per GCG step based on whether probe sampling is used
            # if not self.use_probe_sampling:
            #     flops_per_gcg_step = cost_one_grad_calc_step_target + cost_eval_candidates_target
            # else:
            #     # Probe sampling has additional/different components
            #     psc = attack_config.probe_sampling_config # attack_config is already defined in your method
            #     assert psc is not None, "ProbeSamplingConfig is needed for FLOP calculation"

            #     # Cost for probe set evaluation on TARGET model
            #     probe_size = self.search_width // psc.sampling_factor
            #     cost_probe_eval_target = probe_size * flops_fwd_pass_target

            #     # Cost for filtered set evaluation on TARGET model (estimate)
            #     # psc.r is typically 8. This estimates (1-alpha)*B/R with alpha=0 for max computation.
            #     estimated_filtered_size = max(1, self.search_width // psc.r)
            #     cost_filtered_eval_target = estimated_filtered_size * flops_fwd_pass_target

            #     # Cost for DRAFT model evaluations (all search_width candidates)
            #     draft_model_num_params = sum(p.numel() for p in psc.draft_model.parameters())

            #     # Simplified: Use target model's approx_seq_len for draft model as a rough proxy.
            #     # This avoids complex re-tokenization for FLOP estimate here.
            #     # A more accurate way would be to tokenize with draft_tokenizer, but this is simpler.
            #     approx_seq_len_draft_fwd_proxy = approx_seq_len_target_fwd
            #     flops_fwd_pass_draft = calculate_forward_flops(draft_model_num_params, approx_seq_len_draft_fwd_proxy)
            #     cost_draft_eval_all = self.search_width * flops_fwd_pass_draft

            #     flops_per_gcg_step = (
            #         cost_one_grad_calc_step_target +  # Grad calc on target
            #         cost_probe_eval_target +          # Probe eval on target
            #         cost_filtered_eval_target +       # Filtered eval on target
            #         cost_draft_eval_all               # All candidates eval on draft
            #     )

            # # 8. Total FLOPs for the current prompt
            # current_prompt_flops = self.num_steps * flops_per_gcg_step

            # total_flop_cost += current_prompt_flops
            # # --- End of new FLOP calculation logic ---

            print_timey(f"About to start GCG attack on {original_prompt} -> {target_response}")
            print_mem_usage()

            result = nanogcg.run(
                model=llm._model,
                tokenizer=llm._tokenizer,  # type: ignore
                messages=original_prompt,
                target=target_response,
                config=attack_config,
            )

            print_timey(f"Completed GCG attack on {original_prompt} -> {target_response}")
            print_mem_usage()

            suffix = result.best_string
            attacked_prompt = original_prompt + suffix
            prompt_to_attacked_prompt_map[original_prompt] = attacked_prompt
            all_attacked_prompts_for_final_response.append(attacked_prompt)

        def _gcg_attack_fn(input_prompts: List[str]) -> List[str]:
            return [prompt_to_attacked_prompt_map.get(p, p) for p in input_prompts]

        final_attack_details = AttackDetails(
            flop_cost=total_flop_cost,
            generated_str_attack_function=_gcg_attack_fn,
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
        return BoxRequirement.WHITE_BOX

    @property
    def can_perform_obfuscation(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_slow(self) -> bool:
        return True

    @property
    def uses_api_llm(self) -> bool:
        """Whether this attack uses an API-based LLM."""
        return False
