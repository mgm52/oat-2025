from abc import ABC, abstractmethod
import collections
import contextlib
from enum import Enum
import time
import logging
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

from prompt_toolkit import prompt
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from oat_evaluation.llms.llm import LLM, ExposedActivationsRequest, LLMResponses, TokenSelectionMethod
from oat_evaluation.utils import FlopCounter, print_timey
from contextlib import contextmanager

from oat_training.src.utils import calculate_flops

@contextmanager
def add_fwd_hooks(module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]]):
    handles = []
    try:
        for module, hook_fn in module_forward_pre_hooks:
            handle = module.register_forward_pre_hook(hook_fn)
            handles.append(handle)
        yield
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()

class AutoLLM(LLM):

    def __init__(self, model_path, dtype=torch.bfloat16, debug_mode=False):
        self.debug_mode = debug_mode
        if self.debug_mode:
            torch._logging.set_logs(dynamo=True,
                                    aot=True,
                                    graph=True,
                                    inductor=True,
                                    )
        else:
            torch._logging.set_logs(dynamo=False,
                                    aot=False,
                                    graph=False,
                                    inductor=False,
                                    )

        print_timey(f"Loading model from {model_path}...")

        #torch._dynamo.config.cache_size_limit = 64  # Increase cache size
        # --- Speed Improvement Suggestion ---
        # Try adding attn_implementation if supported by model and hardware
        # Requires: pip install flash-attn
        # Check model documentation/Hugging Face page for compatibility.
        try:
             self._model = AutoModelForCausalLM.from_pretrained(
                 model_path,
                 device_map="auto", # Use "auto" for better distribution if possible
                 torch_dtype=dtype,
                 trust_remote_code=True,
                 attn_implementation="flash_attention_2" # Use Flash Attention 2
             )
             print_timey("Model loaded with Flash Attention 2.")
        except ImportError:
             print_timey("Flash Attention 2 not available or not installed. Falling back.")
             self._model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )
        except Exception as e: # Catch other potential errors like model incompatibility
             print_timey(f"Failed to load with Flash Attention 2 ({e}). Falling back.")
             self._model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )

        print_timey("Model loaded. Loading tokenizer...")
        # Override LAT llama tokenizer (not sure why it breaks)
        is_lat_llama = "lat-llama" in model_path or "robust_llama" in model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path if not is_lat_llama else "meta-llama/Meta-Llama-3-8B-Instruct")
        print_timey("Tokenizer loaded.")

        self.prepare_model()

    @property
    def num_params(self) -> int:
        return self._num_params

    def set_offsets(self, target_layer_offset: int = 0, target_token_start_offset: int = 0, target_token_end_offset: int = 0):
        self.target_layer_offset = target_layer_offset
        self.target_token_start_offset = target_token_start_offset
        self.target_token_end_offset = target_token_end_offset
        print(f"Set offsets: target_layer_offset = {self.target_layer_offset}, target_token_start_offset = {self.target_token_start_offset}, target_token_end_offset = {self.target_token_end_offset}")

    def prepare_model(self):
        print(f"Preparing model...")

        self._num_params = self._model.num_parameters()

        # Consider disabling grad here, until an attack is run...?
        # --- Speed Improvement Suggestion ---
        # Try compiling the model (requires PyTorch 2.0+)
        # This happens *after* loading but before use.
        # Mode can be tuned: None, "default", "reduce-overhead", "max-autotune"
        # Start with default or reduce-overhead. max-autotune takes longer initially.
        try:
            # Ensure model is fully on device before compiling if using device_map
            self._model.to(self.device) # May not be needed with device_map="auto"
            # TODO: Reintroduce torch.compile(). Doesn't seem to work with full_eval_test.py as torch.compile() returns a function, so self._model.generation_config (below) will not work
            # print_timey("Attempting to compile model with torch.compile...")
            # # Note: Compilation might fail for some models or require specific PyTorch/CUDA versions.
            # if self.reduce_compile_overhead:
            #     self._model = torch.compile(self._model, 
            #                 mode="reduce-overhead", 
            #                 dynamic=True, # dynamic=True might help with variable sequence lengths
            #     )
            # else:
            #     self._model = torch.compile(self._model, 
            #                 dynamic=True, # dynamic=True might help with variable sequence lengths
            #                 options={
            #                     # "triton.cudagraphs": True,  # Enable CUDA graphs for static workloads. Could try enabling if we pad with static batch sizes and sequence lengths
            #                     "dynamic_shapes": True,
            #                     "epilogue_fusion": True,    # Fuse epilogue operations
            #                     "max_autotune": True,        # Autotune kernel configurations
            #                     "verbose_inductor": False,
            #                 }
            #     )
            # print_timey("Model compiled successfully.")
        except Exception as e:
            print_timey(f"torch.compile failed: {e}. Proceeding without compilation.")

        # Pad from left, in case we run a soft-suffix attack...
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token:
            pass
        elif self._tokenizer.unk_token:
            self._tokenizer.pad_token_id = self._tokenizer.unk_token_id
        elif self._tokenizer.eos_token:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        else:
            self._tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        # Transformer blocks in a list; useful for extracting activations
        self._model_block_modules = self._get_block_modules()
        self._model_embedding_layer = self._model.get_input_embeddings()

        # Get pad token in other forms
        self.pad_token_id = torch.tensor(self._tokenizer.pad_token_id, device=self.device).unsqueeze(0).unsqueeze(0)
        self.pad_embedding = self._token_ids_to_embeddings(self.pad_token_id).to(self.dtype).detach()

        self._figure_out_chat_function()

        if self.debug_mode:
            print(f"Loaded model with left-padding token: {self._tokenizer.pad_token}")

    # TODO: write unit tests to check that size of logits etc is consistent across different input types...
    # And check that the first "prediction" logit corresponds to the first response token...
    def generate_responses(
        self,
        prompts: Union[List[str], List[torch.Tensor], list[list[dict]]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        max_new_tokens: int = 64,
        requires_grad: bool = True,
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model.
        
        Args:
            prompts: The prompts to generate responses for. Accepted formats include:
                list[str]: list of prompt strings
                list[torch.Tensor]: list of tensors, each representing a prompt
                list[list[dict]]: list of conversations, e.g. [[{"role": "user", "content": ...}, ...]]
            exposed_activations_request: Request specifying which activation layers to extract
            max_new_tokens: Maximum number of new tokens to generate.
            requires_grad: If True, gradients will be computed for the forward pass. Defaults to True.
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """
        # Check if we need to split into batches
        batch_size = 16
        if len(prompts) > batch_size:
            # Split prompts into batches and process each batch
            all_responses = []
            all_logits = []
            all_activations = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_result = self.generate_responses(
                    batch_prompts,
                    exposed_activations_request=exposed_activations_request,
                    max_new_tokens=max_new_tokens,
                    requires_grad=requires_grad,
                )
                
                all_responses.extend(batch_result.responses_strings)
                all_logits.extend(batch_result.responses_logits)
                
                if batch_result.activation_layers is not None:
                    all_activations.extend(batch_result.activation_layers)
                
                del batch_result
                if not requires_grad:
                    torch.cuda.empty_cache()
                
            return LLMResponses(
                responses_strings=all_responses,
                responses_logits=all_logits,
                activation_layers=all_activations if exposed_activations_request else None
            )

        # Choose context based on requires_grad
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()

        with context:
            if isinstance(prompts[0], str) or isinstance(prompts[0], list):
                if isinstance(prompts[0], list):
                    # Conversation histories
                    assert isinstance(prompts[0][0], dict)
                    messages = prompts
                else:
                    # Add special token chat template & padding!
                    messages = [
                        [{"role": "user", "content": prompt}]
                        for prompt in prompts
                    ]
                tokenized_chat = self._tokenize_chat(messages, add_generation_prompt=True)

                if self.debug_mode:
                    print_timey(f"About to generate with tokenized_chat: {tokenized_chat}")
                    print_timey(f"About to generate with tokenized_chat.input_ids.shape: {tokenized_chat['input_ids'].shape}")
                    print_timey(f"About to generate with tokenized_chat.attention_mask.shape: {tokenized_chat['attention_mask'].shape}")
                
                outputs = self._model.generate(**tokenized_chat, 
                                               return_dict_in_generate=True,
                                               max_new_tokens=max_new_tokens,
                                               use_cache=True,
                                               )
                if self.debug_mode:
                    print_timey("Model generation complete. Decoding...")
                start_length = tokenized_chat["input_ids"].shape[1]

                decoded_responses = [
                    self._tokenizer.decode(seq[start_length:], skip_special_tokens=True)
                    for seq in outputs.sequences
                ]

                if self.debug_mode:
                    print_timey(f"Outputs.sequences: {outputs.sequences}")
                    print_timey(f"Decoded responses (len {len(decoded_responses)}): {decoded_responses}")

                sequences = outputs.sequences
                del outputs

                # TODO: Update generate_responses_forced() to also handle full conversations
                #   list[list[dict]]: list of conversations, e.g. [[{"role": "user", "content": ...}, ...]]
                if isinstance(prompts[0], str):
                    if self.debug_mode:
                        print_timey("Force-forwarding responses to get logits & activations...")
                    forced_responses = self.generate_responses_forced(
                        prompts,
                        decoded_responses,
                        exposed_activations_request=exposed_activations_request,
                        requires_grad=requires_grad,
                    )
                    if self.debug_mode:
                        print_timey("Force-forwarding complete. Returning LLMResponses!")

                    return LLMResponses(
                        responses_strings=decoded_responses,
                        responses_logits=forced_responses.responses_logits,
                        activation_layers=forced_responses.activation_layers if exposed_activations_request else None
                    )
                else:
                    return LLMResponses(decoded_responses, None, None)
            else:
                responses_embeddings = []

                gen_embeddings = [self._embeddings_to_gen_embeddings(prompt_embedding.unsqueeze(0)) for prompt_embedding in prompts]
                if self.debug_mode: print(f"Generated gen embeddings of shapes {[e.shape for e in gen_embeddings]}...")
                # now perform left-padding
                gen_embeddings_tensor, attention_masks = self._left_pad_embeddings(gen_embeddings)
                if self.debug_mode: print_timey(f"Turned into padded tensor of shape {gen_embeddings_tensor.shape}, with attention masks of shape {attention_masks.shape}... About to generate!")

                outputs = self._model.generate(inputs_embeds=gen_embeddings_tensor,
                                               attention_mask=attention_masks, 
                                               return_dict_in_generate=True, 
                                               max_new_tokens=max_new_tokens,
                                               use_cache=True,
                                               )
                start_length = gen_embeddings_tensor.shape[1]

                sequences = outputs.sequences
                if self.debug_mode: print_timey(f"Generation complete. Outputs.sequences (len {len(sequences)}), shapes {[s.shape for s in sequences]}: {sequences}")

                decoded_responses = [
                    self._tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in sequences
                ]
                del outputs
                if not requires_grad:
                    torch.cuda.empty_cache()

                if self.debug_mode: print(f"Decoded responses (len {len(decoded_responses)}): {decoded_responses}")

                # Convert to embeddings
                responses_embeddings = [self.string_to_embedding(response) for response in decoded_responses]
                if self.debug_mode: print(f"Responses embeddings (len {len(responses_embeddings)}) shapes {[e.shape for e in responses_embeddings]}...")

                forced_responses = self.generate_responses_forced(
                    prompts,
                    responses_embeddings,
                    exposed_activations_request=exposed_activations_request,
                    requires_grad=requires_grad,
                )

                return LLMResponses(
                    responses_strings=decoded_responses,
                    responses_logits=forced_responses.responses_logits,
                    activation_layers=forced_responses.activation_layers if exposed_activations_request else None
                )

    @property
    def device(self) -> torch.device:
        return self._model.device

    def _left_pad_embeddings(
        self,
        embeddings: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Left-pad a list of embeddings (each of shape (1, seq_len_i, embedding_size))
        to the same max sequence length using the given pad_embedding (1, 1, embedding_size).

        Returns:
            A tensor of shape (batch_size, max_seq_len, embedding_size)
        """
        # Ensure pad_embedding is the correct shape
        assert self.pad_embedding.dim() == 3 and self.pad_embedding.size(0) == 1 and self.pad_embedding.size(1) == 1

        embedding_size = self.pad_embedding.size(-1)
        seq_lens = [emb.size(1) for emb in embeddings]
        max_seq_len = max(seq_lens)

        padded_embeddings = []
        attention_masks = []
        for emb in embeddings:
            seq_len = emb.size(1)
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                # Repeat self.pad_embedding to match pad_len
                padding = self.pad_embedding.expand(1, pad_len, embedding_size)
                padded = torch.cat([padding, emb], dim=1)
                attention_mask = torch.cat([torch.zeros(1, pad_len), torch.ones(1, seq_len)], dim=1)
            else:
                padded = emb
                attention_mask = torch.ones(1, seq_len)
            padded_embeddings.append(padded)
            attention_masks.append(attention_mask)

        embeddings = torch.cat(padded_embeddings, dim=0).to(self._model.device)  # (batch_size, max_seq_len, embedding_size)
        attention_masks = torch.cat(attention_masks, dim=0).to(self._model.device)  # (batch_size, max_seq_len)
        return embeddings, attention_masks


    def _figure_out_chat_function(self):
        # Let's try to figure out how to turn embeddings into chat-template embeddings!
        # First, let's establish what "chat-template" embed we actually want.
        # i.e. let's embed pre and post chat and compare them...

        prompt = "How to bake?"
        response = "This is how."

        # Returns (batch_size, seq_len, vocab_size)
        prompt_token_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_embeddings = self._model_embedding_layer(prompt_token_ids["input_ids"])
        response_token_ids = self._tokenizer(response, return_tensors="pt", add_special_tokens=False).to("cuda")
        response_embeddings = self._model_embedding_layer(response_token_ids["input_ids"])

        messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        ]
        chat_token_ids = self._tokenize_chat(messages, add_generation_prompt=False)
        chat_embeddings = self._model_embedding_layer(chat_token_ids["input_ids"])

        prompt_insertion_index = -1
        response_insertion_index = -1
        for i in range(chat_embeddings.shape[1]):
            if chat_embeddings[0, i].equal(prompt_embeddings[0, 0]) and prompt_insertion_index == -1:
                # Insertion here!
                if self.debug_mode:
                    print(f"Found chat template intro length {i}, outro length {chat_embeddings.shape[1] - i - prompt_embeddings.shape[1]}")
                prompt_insertion_index = i
            if chat_embeddings[0, i].equal(response_embeddings[0, 0]):
                response_insertion_index = i
                break

        if prompt_insertion_index == -1 or response_insertion_index == -1:
            raise ValueError("Failed to find insertion index for prompt or response")

        self._chat_intro = chat_embeddings[0, :prompt_insertion_index].unsqueeze(0).detach() # shape (1, intro_len)
        self._chat_middle = chat_embeddings[0, prompt_insertion_index+prompt_embeddings.shape[1]:response_insertion_index].unsqueeze(0).detach() # shape (1, middle_len)
        self._chat_outro = chat_embeddings[0, response_insertion_index+response_embeddings.shape[1]:].unsqueeze(0).detach() # shape (1, outro_len)
        self._chat_outro_token_ids = chat_token_ids["input_ids"][0, response_insertion_index+response_embeddings.shape[1]:].detach() # shape (outro_len)

        # Expects prompt and response to be of shape (batch_size, seq_len, ...)
        self._embeddings_to_chat_embeddings = lambda prompt, response: torch.cat((self._chat_intro, prompt, self._chat_middle, response, self._chat_outro), dim=1)
        self._embeddings_to_gen_embeddings = lambda prompt: torch.cat((self._chat_intro, prompt, self._chat_middle), dim=1)
        
    def string_to_token_ids(self, input_string, add_response_ending=False):
        """Output shape (seq_len)"""
        #print_timey(f"String to token ids: input string of length {len(input_string)}...")
        input_tokens = self._tokenizer(input_string, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self._model.device)
        if add_response_ending:
            if self.debug_mode:
                print_timey(f"Adding response ending of length {self._chat_outro_token_ids.shape[0]} (specifically: {self._chat_outro_token_ids}) to input tokens of length {input_tokens.shape[0]}...")
            return torch.cat((input_tokens, self._chat_outro_token_ids), dim=0)
        else:
            if self.debug_mode:
                print_timey(f"String to token ids: returning tokens of length {input_tokens.shape[0]}...")
            return input_tokens

    def _token_ids_to_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Expects input shape (batch_size, seq_len). Outputs shape (batch_size, seq_len, embedding_size)."""
        return self._model_embedding_layer(token_ids)

    def generate_responses_forced(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor]],
        target_responses_or_embeddings: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        add_response_ending: bool = False,
        requires_grad: bool = True,
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model, while forcing the outputs.
        This function is useful for extracting activations & logits for a target response, e.g. for soft-suffix attacks.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or of naked embeddings (i.e. no special tokens or padding). Each of shape (seq_len (varying), embedding_size), or a string.
            target_responses_or_embeddings: The target responses to force the model to generate. Each of shape (seq_len (varying), embedding_size), or a string.
            exposed_activations_request: Request specifying which activation layers to extract
            add_response_ending: Whether to include response ending tokens in logits/activations.
            requires_grad: If True, gradients will be computed for the forward pass. Defaults to True.
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """

        # Check if we need to split into batches
        batch_size = 16
        if len(prompts_or_embeddings) > batch_size:
            # Split prompts and responses into batches and process each batch
            all_responses = []
            all_logits = []
            all_activations = []
            
            for i in range(0, len(prompts_or_embeddings), batch_size):
                batch_prompts = prompts_or_embeddings[i:i+batch_size]
                batch_responses = target_responses_or_embeddings[i:i+batch_size]
                batch_result = self.generate_responses_forced(
                    batch_prompts,
                    batch_responses,
                    exposed_activations_request=exposed_activations_request,
                    add_response_ending=add_response_ending,
                    requires_grad=requires_grad
                )
                
                all_responses.extend(batch_result.responses_strings)
                all_logits.extend(batch_result.responses_logits)
                
                if batch_result.activation_layers is not None:
                    all_activations.extend(batch_result.activation_layers)
                
                # Clear batch results to free memory
                del batch_result
                if not requires_grad:
                    torch.cuda.empty_cache()
                
            return LLMResponses(
                responses_strings=all_responses,
                responses_logits=all_logits,
                activation_layers=all_activations if exposed_activations_request else None
            )
        
        if self.debug_mode:
            print_timey(f"Generating responses forced for {len(prompts_or_embeddings)} prompts... requires_grad={requires_grad}")

        assert len(prompts_or_embeddings) == len(target_responses_or_embeddings)
        # Choose context based on requires_grad
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()

        with context:
            if isinstance(prompts_or_embeddings[0], torch.Tensor):
                assert isinstance(target_responses_or_embeddings[0], torch.Tensor)
                assert len(prompts_or_embeddings[0].shape) == 2
                assert len(target_responses_or_embeddings[0].shape) == 2

                # check dtypes
                assert prompts_or_embeddings[0].dtype == target_responses_or_embeddings[0].dtype, f"Prompts and target responses must have the same dtype, but got {prompts_or_embeddings[0].dtype} and {target_responses_or_embeddings[0].dtype}"
                assert prompts_or_embeddings[0].dtype == self.dtype, f"Prompts and target responses must have the same dtype as the model, but got {prompts_or_embeddings[0].dtype} and {self.dtype}"

                # add batch dimension
                prompts_or_embeddings = [prompt.unsqueeze(0) for prompt in prompts_or_embeddings]
                target_responses_or_embeddings = [response.unsqueeze(0) for response in target_responses_or_embeddings]
            else:
                assert isinstance(prompts_or_embeddings[0], str)
                assert isinstance(target_responses_or_embeddings[0], str)

            # Set up hooks for extracting activations
            raw_activations_list = [] # Will store list[tensor(batch, seq, hidden)]
            fwd_extraction_hooks = []
            if exposed_activations_request:
                target_layers_raw = [li + self.target_layer_offset for li in exposed_activations_request.extract_layers_indices]
                target_layers = [li for li in target_layers_raw if li >= 0 and li < len(self._model_block_modules)]
                if len(target_layers_raw) != len(target_layers):
                    print(f"WARNING: Some target layers were out of range, so we ignored them. Target layers raw length: {len(target_layers_raw)}, target layers filtered length: {len(target_layers)}")
                
                # Make raw_activations_list the right size initially
                raw_activations_list = [None] * len(target_layers)

                # Modified hook to place tensor at the correct index
                def activation_extraction_hook_simple(destination: List[Optional[torch.Tensor]], index: int, debug_mode: bool = False):
                    def hook_fn(module, input):
                        # Ensure list is mutable if needed (though pre-sized should be ok)
                        if debug_mode:
                            print(f"Hook for layer index {index} triggered with input[0] shape: {input[0].shape}")
                        # We shouldn't detach here, because we want to backprop through the activations
                        destination[index] = input[0]
                    return hook_fn

                fwd_extraction_hooks = [(
                        self._model_block_modules[layer],
                        activation_extraction_hook_simple(destination=raw_activations_list, index=i, debug_mode=self.debug_mode)
                    ) for i, layer in enumerate(target_layers)
                ]

            if isinstance(prompts_or_embeddings[0], str) or isinstance(prompts_or_embeddings[0], list):
                # Add special token chat template & padding!
                if isinstance(prompts_or_embeddings[0], str):
                    messages = [
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": target_response}
                        ]
                        for prompt, target_response in zip(prompts_or_embeddings, target_responses_or_embeddings)
                    ]
                else:
                    messages = prompts_or_embeddings
                    for conv, target_response in zip(prompts_or_embeddings, target_responses_or_embeddings):
                        conv.append({"role": "assistant", "content": target_response})
                tokenized_chat = self._tokenize_chat(messages, add_generation_prompt=False)

                if self.debug_mode:
                    print(f"Tokenized chat: {tokenized_chat}")
                    print(f"About to forward with tokenized_chat, with input_ids.shape: {tokenized_chat['input_ids'].shape}")
                if self.debug_mode:
                    print_timey(f"About to force-forward string inputs...")
                with add_fwd_hooks(fwd_extraction_hooks):
                    outputs = self._model.forward(**tokenized_chat)
                if self.debug_mode:
                    print_timey(f"Force-forwarding complete. Now turning strings into token ids...")
                original_response_lengths = [self.string_to_token_ids(response).shape[0] for response in target_responses_or_embeddings]
            else:
                chat_embeddings = [self._embeddings_to_chat_embeddings(prompt, response) for prompt, response in zip(prompts_or_embeddings, target_responses_or_embeddings)]
                if self.debug_mode:
                    print(f"Chat embeddings: {chat_embeddings}")
                chat_embeddings_tensor, attention_masks = self._left_pad_embeddings(chat_embeddings)

                if self.debug_mode:
                    print(f"Chat embeddings tensor: {chat_embeddings_tensor}")
                    print(f"Attention masks: {attention_masks}")
                    print(f"About to forward with embeddings tensor shape: {chat_embeddings_tensor.shape}")
                if self.debug_mode:
                    print_timey(f"About to force-forward embedding inputs...")
                with add_fwd_hooks(fwd_extraction_hooks):
                    outputs = self._model.forward(
                        inputs_embeds=chat_embeddings_tensor,
                        attention_mask=attention_masks,
                        #use_cache=False # test to try preventing graph issues...
                    )
                if self.debug_mode:
                    print_timey(f"Force-forwarding complete.")
                original_response_lengths = [response.shape[1] for response in target_responses_or_embeddings]

            if self.debug_mode:
                print(f"Outputs: {outputs}")
                print(f"Outputs.logits.shape: {outputs.logits.shape}") # shape (batch_size, seq_len, vocab_size)

            # Now we just need to trim the logits down to the response prediction only...
            if self.debug_mode:
                print(f"Original response lengths: {original_response_lengths}")
            

            trimmed_logits = []
            for i, response_length in enumerate(original_response_lengths):
                response_start = outputs.logits.shape[1]-self._chat_outro.shape[1]-response_length-1
                response_end = outputs.logits.shape[1]-self._chat_outro.shape[1]-1

                if add_response_ending:
                    if self.debug_mode:
                        print(f"Althrough ordinarily we'd trim to {response_start}:{response_end}, we're adding the response ending back on, so including all of end except final...")
                    response_logits = outputs.logits[i, response_start:-1]
                    if self.debug_mode:
                        print(f"We got response start {response_start} for response {i} of length {response_length}, including chat outro of length {self._chat_outro.shape[1]}... So appending logits of shape {response_logits.shape}")
                else:
                    response_logits = outputs.logits[i, response_start:response_end]
                    if self.debug_mode:
                        print(f"We got response start {response_start} and end {response_end} for response {i} of length {response_length}... So appending logits of shape {response_logits.shape}")
                trimmed_logits.append(response_logits)
            
            decoded_responses = [self._logits_to_strings(logit.unsqueeze(0))[0] for logit in trimmed_logits]

            if self.debug_mode:
                print(f"Decoded responses: {decoded_responses}")

            # --- Process Activations ---
            final_activation_layers = None
            if exposed_activations_request and raw_activations_list and all(t is not None for t in raw_activations_list):
                # Desired output structure: list[list[tensor(num_tokens, hidden)]]
                # Outer list: batch_size
                # Inner list: num_req_layers
                final_activation_layers = [[] for _ in range(len(prompts_or_embeddings))]

                if self.debug_mode:
                    # map[batch_idx][layer_idx] = (start_idx, end_idx)
                    slice_map = [
                        [None] * len(raw_activations_list)
                        for _ in range(len(prompts_or_embeddings))
                    ]

                # raw_activations_list contains tensors of shape (batch, seq, hidden)
                for layer_idx, full_layer_activation in enumerate(raw_activations_list):
                    # full_layer_activation shape: (batch_size, activation_seq_len, hidden_size)
                    activation_seq_len = full_layer_activation.shape[1] # Use actual seq len from activation

                    for batch_idx in range(len(prompts_or_embeddings)):
                        response_length = original_response_lengths[batch_idx]
                        # Recalculate start/end based on *activation* sequence length
                        # NOTE: Activation seq len might differ slightly from logit seq len depending on model/hook timing,
                        # but often they are the same for pre-hooks. Use the activation tensor's shape.
                        act_slice_start = activation_seq_len - self._chat_outro.shape[1] - response_length - 1 + self.target_token_start_offset
                        act_slice_end = activation_seq_len - self._chat_outro.shape[1] - 1 + self.target_token_end_offset

                        if add_response_ending:
                            act_slice_end = activation_seq_len - 1 # Include tokens for outro

                        if self.debug_mode and slice_map is not None and len(slice_map) > batch_idx and slice_map[batch_idx] is not None and len(slice_map[batch_idx]) > layer_idx:
                            slice_map[batch_idx][layer_idx] = (act_slice_start, act_slice_end)

                        # Basic validation for activation slice indices
                        if act_slice_start < 0 or act_slice_end > activation_seq_len or act_slice_start >= act_slice_end:
                            print(f"Warning: Invalid activation slice for layer {layer_idx}, batch item {batch_idx}. Start: {act_slice_start}, End: {act_slice_end}, ActSeqLen: {activation_seq_len}, RespLen: {response_length}. Adjusting slice indices...")
                            # Getting hidden size correctly:
                            act_slice_start = max(0, act_slice_start)
                            act_slice_end = min(activation_seq_len, act_slice_end)
                            print(f"Adjusted slice indices: Start: {act_slice_start}, End: {act_slice_end}")

                        # Select the slice for this batch item and this layer
                        token_activations = full_layer_activation[batch_idx, act_slice_start:act_slice_end, :]
                        # token_activations shape: (num_req_tokens, hidden_size)

                        if False and self.debug_mode and final_activation_layers:
                            for batch_idx, logit in enumerate(trimmed_logits):
                                # 1) token strings
                                pred_ids     = logit.argmax(dim=-1)
                                pred_tokens  = self._tokenizer.convert_ids_to_tokens(pred_ids.tolist())
                                print(f"[Debug] Response {batch_idx} tokens: {list(enumerate(pred_tokens))}")

                                # 2) activations
                                for layer_idx, acts in enumerate(final_activation_layers[batch_idx]):
                                    if slice_map is not None and len(slice_map) > batch_idx and slice_map[batch_idx] is not None and len(slice_map[batch_idx]) > layer_idx and slice_map[batch_idx][layer_idx] is not None:
                                        start, end = slice_map[batch_idx][layer_idx]
                                        full_len   = raw_activations_list[layer_idx].shape[1]

                                        kept = end - start
                                        trimmed = full_len - kept
                                        print(f"[Debug]  Layer {layer_idx}: full_seq={full_len}, kept_range={start}:{end} "
                                            f"({kept} tokens kept, {trimmed} trimmed), acts.shape={tuple(acts.shape)}")

                                        # per‐index status
                                        status = [(i, 'kept' if start <= i < end else 'trimmed')
                                                for i in range(full_len)]
                                        print(f"[Debug]    indices: {status}")

                                        # sanity check
                                        assert acts.shape[0] == len(pred_tokens), (
                                            f"Mismatch on batch {batch_idx}, layer {layer_idx}: "
                                            f"{acts.shape[0]} activations vs {len(pred_tokens)} tokens"
                                        )

                        # Append to the correct place in the final structure
                        final_activation_layers[batch_idx].append(token_activations)

                if self.debug_mode:
                    print(f"Processed activation structure: {len(final_activation_layers)} batch items.")
                    if final_activation_layers:
                        print(f"First batch item has {len(final_activation_layers[0])} layers.")
                        if final_activation_layers[0]:
                            print(f"First layer tensor shape for first batch item: {final_activation_layers[0][0].shape}")

            # --- Cleanup --- (Outside context block)
            # Clear potentially large tensors. Only clear GPU cache if no grads were needed.
            del outputs 
            del raw_activations_list 
            if not requires_grad:
                torch.cuda.empty_cache()

            if self.debug_mode:
                print_timey(f"Returning LLMResponses from forced forward!")

            # Return logits that may or may not have gradients attached based on requires_grad
            return LLMResponses(
                responses_strings=decoded_responses,
                responses_logits=trimmed_logits, # list length (batch_size), each element shape (response_length (varying), vocab_size)
                activation_layers=final_activation_layers # list (batch_size) -> list (num_req_layers) -> tensor (num_req_tokens, hidden_size)
            )

    def string_to_embedding(self, string: str) -> torch.Tensor:
        """Converts a prompt/response string to a "naked" embedding tensor. i.e. Does not add any special tokens or padding. Returns shape (seq_len, embedding_size)"""

        return self._token_ids_to_embeddings(self.string_to_token_ids(string).unsqueeze(0))[0]

    def _get_block_modules(self) -> List[torch.nn.Module]:
        """Get the transformer block modules for hooking."""
        blocks = []
        for name, module in self._model.named_modules():
            # For Gemma models, the transformer blocks are in model.layers
            if isinstance(module, torch.nn.Module) and hasattr(module, 'self_attn'):
                blocks.append(module)
        return blocks

    def get_num_tokens_in_chat(self, messages: list[list[dict]]) -> int:
        tokenized_chat = self._tokenize_chat(messages)
        return tokenized_chat["input_ids"].numel()
    
    def get_num_tokens_in_str(self, string: str) -> int:
        tokenized = self._tokenizer(string, return_tensors="pt")
        return tokenized["input_ids"].numel()
    
    def _tokenize_chat(self, messages, **kwargs):
        return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=kwargs.get("tokenize", True),
                    add_generation_prompt=kwargs.get("add_generation_prompt", False),
                    padding=kwargs.get("padding", True),
                    return_tensors=kwargs.get("return_tensors", "pt"),
                    return_dict=kwargs.get("return_dict", True)
                ).to(self._model.device)

    def _logits_to_strings(self, logits: torch.Tensor) -> List[str]:
        """Converts a logits tensor to a list of strings. Expects shape (batch_size, seq_len, vocab_size)."""
        return self._tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=False)

    @property
    def num_layers(self) -> int:
        """The number of transformer blocks in the model (i.e. the max number of activation layers to extract)"""
        return len(self._model_block_modules)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary of the model (number of possible tokens, i.e. number of columns in the logits tensor)"""
        return len(self._tokenizer)

    @property
    def embedding_size(self) -> int:
        """The number of columns in embedding tensors"""
        return self._model_embedding_layer.embedding_dim

    @property
    def name(self) -> str:
        """The name of the model"""
        return self.__class__.__name__

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the model"""
        return self._model.dtype

    # TODO: consider how to support attacks injected into activation space...
    # We'll need to be able to extract activations, give them to an arbitrary attack function, then continue generation....
    # I expect we can just add a sort of custom hooks feature into ActivationRequest.

if __name__ == "__main__":
    llm = AutoLLM("/workspace/gemma_2_9b_instruct", debug_mode=True)
    prompts = ["How to bake?", "What is 2+2?"]
    responses = ["First, mix the ingredients.", "4"]

    print("\nSTRING SPACE FORCE:\n")

    resp1 = llm.generate_responses_forced(prompts, responses)
    print(resp1.responses_strings)

    print("\nEMBEDDING SPACE FORCE:\n")

    prompt_embeds = [llm.string_to_embedding(prompt) for prompt in prompts]
    resp_embeds = [llm.string_to_embedding(response) for response in responses]
    resp2 = llm.generate_responses_forced(prompt_embeds, resp_embeds)
    print(resp2.responses_strings)

    print("\nSTRING SPACE GENERATION:\n")

    output = llm.generate_responses(["Hello, how are you?"], ExposedActivationsRequest(extract_layers_indices=[0], token_selection_method=TokenSelectionMethod.ALL_RESPONSE))
    print(output.responses_strings)
    print(output.responses_logits)
    print(len(output.responses_logits))
    print(output.responses_logits[0].shape)

    # TODO: check activation shape...
    print(output.activation_layers)
    print(len(output.activation_layers))
    print(len(output.activation_layers[0]))
    print(output.activation_layers[0][0].shape)

    print("\nEMBEDDING SPACE GENERATION:\n")

    output = llm.generate_responses([llm.string_to_embedding("Hello, how are you?")], ExposedActivationsRequest(extract_layers_indices=[0], token_selection_method=TokenSelectionMethod.ALL_RESPONSE))
    print(output.responses_strings)
    print(output.responses_logits)
    print(len(output.responses_logits))
    print(output.responses_logits[0].shape)

    # TODO: check activation shape...
    print(output.activation_layers)
    print(len(output.activation_layers))
    print(len(output.activation_layers[0]))
    print(output.activation_layers[0][0].shape)
