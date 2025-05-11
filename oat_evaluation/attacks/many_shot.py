from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import gc
import json
import os
import random
import re
import sys
import time
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
import rapidfuzz


from oat_evaluation.llms.api_llm import ApiLLM
from oat_evaluation.llms.autollm import AutoLLM
from oat_evaluation.utils import FlopCounter, TokenCounter, calculate_forward_flops, print_timey
import numpy as np
import openai
import torch
from torch import nn
from tqdm import tqdm
from oat_evaluation.attacks.attack import Attack, AttackDetails, BoxRequirement, flop_to_usd, usd_to_flop
from oat_evaluation.llms.llm import LLM, LLMResponses
from oat_evaluation.probes.probe import Probe


@dataclass
class HarmfulExample:
    prompt: str
    completion: str

@dataclass
class HarmfulExamples:
    examples: List[HarmfulExample]

    def __init__(self, examples):
        # Convert list of dicts to list of HarmfulExample objects if necessary
        self.examples = []
        for example in examples:
            if isinstance(example, dict):
                self.examples.append(HarmfulExample(**example))
            elif isinstance(example, HarmfulExample):
                self.examples.append(example)
            else:
                raise ValueError("Each example must be a HarmfulExample object or a dict.")

    def __iter__(self):
        for example in self.examples:
            yield example

@dataclass
class ManyShotAttack(Attack):
    # Many-Shot Jailbreaking specific fields    
    harmful_examples_to_use: HarmfulExamples
    num_examples: int = 8
    similarity_threshold_warning: float = 0.7

    is_cross_model_transferrable: bool = True  # Whether this attack can be transferred across models (e.g. GCG).
    is_universal: bool = True  # Whether this attack is universal or per-prompt.
    box_requirement: BoxRequirement = BoxRequirement.BLACK_BOX  # Whether this attack is white-box (activations) / black-box (tokens) / gray-box (logits).
    can_perform_obfuscation: bool = False  # Whether this attack can perform probe-obfuscation.

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
            probe_to_obfuscate: Optional probe to obfuscate against
            probe_refusal_threshold: Threshold for probe refusal detection
            generate_final_responses: Whether to run the final attack or just return the attack function
            callback_steps: List of steps at which to execute callbacks
            callbacks: List of callback functions that take AttackDetails as input
            
        Returns:
            Tuple containing LLMResponses and AttackDetails.
        """
        if callback_steps is None:
            callback_steps = []
        if callbacks is None:
            callbacks = []
            
        # Get examples from the HarmfulExamples object
        examples = list(self.harmful_examples_to_use)
        
        # Ensure we have the correct number of prompts and responses
        assert harmful_responses is not None, "Harmful responses must be provided"
        assert len(prompts) == len(harmful_responses), "Number of prompts and harmful responses must match"
        
        updated_prompts = []
        for prompt, harmful_response in zip(prompts, harmful_responses):
            # Randomly shuffle all examples
            all_examples = examples.copy()
            random.shuffle(all_examples)
            
            # Sample examples, skipping those that are too similar to the prompt
            selected_examples = []
            for example in all_examples:
                if len(selected_examples) >= self.num_examples:
                    break
                    
                harmful_example_prompt = example.prompt
                similarity_score = rapidfuzz.fuzz.QRatio(prompt, harmful_example_prompt, processor=rapidfuzz.utils.default_process) / 100.
                
                if similarity_score <= self.similarity_threshold_warning:
                    selected_examples.append(example)
                else:
                    print(f"Skipping similar example: Similarity {similarity_score:.2f}")
                    print(f"Prompt       : {prompt}")
                    print(f"Demonstration: {harmful_example_prompt}")
            
            # Create conversation format with selected examples
            shuffled_conversation = []
            for example in selected_examples:
                shuffled_conversation.append({"role": "user", "content": example.prompt})
                shuffled_conversation.append({"role": "assistant", "content": example.completion})
            
            # Add the current prompt to the end
            shuffled_conversation.append({"role": "user", "content": prompt})
            
            updated_prompts.append(shuffled_conversation)
        
        llm_responses = llm.generate_responses(updated_prompts)
        attack_details = AttackDetails(flop_cost=0)
        
        return llm_responses, attack_details

    def __str__(self) -> str:
        return f"{self.name}(num_examples={self.num_examples}, similarity_threshold_warning={self.similarity_threshold_warning})"
