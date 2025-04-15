from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from oat_evaluation.llms.llm import LLM, ExposedActivationsRequest, LLMResponses, TokenSelectionMethod
from contextlib import contextmanager

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

def activation_extraction_hook(destination: List[List[torch.Tensor]], index: int, debug_mode: bool = False):
    def hook_fn(module, input):
        if debug_mode:
            print(f"Hook number {index} triggered with input: {input}")
        destination[index].append(input[0].detach().clone())
    return hook_fn

class AutoLLM(LLM):

    def __init__(self, model_path, dtype=torch.float32, debug_mode=False):
        self.debug_mode = debug_mode
        print(f"Loading model from {model_path}...")

        self._model = AutoModelForCausalLM.from_pretrained(
            # float16 was the default for obfuscated-activations...
            # but I find that float32 is more stable for attacks...
            model_path, device_map="cuda", torch_dtype=dtype
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Consider disabling grad here, until an attack is run...?

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
        self.pad_token_id = torch.tensor(self._tokenizer.pad_token_id, device='cuda').unsqueeze(0).unsqueeze(0)
        self.pad_embedding = self._token_ids_to_embeddings(self.pad_token_id).detach()

        self._figure_out_chat_function()

        if self.debug_mode:
            print(f"Loaded model with left-padding token: {self._tokenizer.pad_token}")

    # TODO: write unit tests to check that size of logits etc is consistent across different input types...
    # And check that the first "prediction" logit corresponds to the first response token...
    def generate_responses(
        self,
        prompts: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        max_new_tokens: int = 64
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model.
        
        Args:
            prompts: The prompts to generate responses for, as a list of strings
            exposed_activations_request: Request specifying which activation layers to extract
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """

        # Set up hooks for extracting activations
        if exposed_activations_request:
            # TODO: filter to exposed_activations_request extraction token type...
            target_layers = exposed_activations_request.extract_layers_indices
            activations_list = [[] for _ in target_layers]
            fwd_extraction_hooks = [(
                    self._model_block_modules[layer],
                    activation_extraction_hook(destination=activations_list, index=i, debug_mode=self.debug_mode)
                ) for i, layer in enumerate(target_layers)
            ]
        else:
            fwd_extraction_hooks = []

        if isinstance(prompts[0], str):
            # Add special token chat template & padding!
            messages = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]
            tokenized_chat = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            ).to(self._model.device)

            if self.debug_mode:
                print(f"About to forward with tokenized_chat: {tokenized_chat}")
                print(f"About to forward with tokenized_chat.input_ids.shape: {tokenized_chat['input_ids'].shape}")
            with add_fwd_hooks(fwd_extraction_hooks):
                outputs = self._model.generate(**tokenized_chat, output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens)

            start_length = tokenized_chat["input_ids"].shape[1]
            decoded_responses = [
                self._tokenizer.decode(seq[start_length:], skip_special_tokens=True)
                for seq in outputs.sequences
            ]

            # TODO: consider how to handle padding (where even is it...?)
            if isinstance(outputs.scores, tuple):
                if self.debug_mode:
                    print(f"Outputs.scores is a tuple of length {len(outputs.scores)}, with shapes {[s.shape for s in outputs.scores]}...")
                # right now, len(scores) = seq_len, and each element is a tensor of shape (batch_size, vocab_size)...
                # let's convert it to a list of tensors of shape (seq_len, vocab_size) for each batch item
                batch_size = outputs.scores[0].shape[0]
                responses_scores = []
                for batch_idx in range(batch_size):
                    # Extract scores for this batch item across all sequence positions
                    batch_scores = torch.stack([score[batch_idx] for score in outputs.scores], dim=0)
                    responses_scores.append(batch_scores)
                if self.debug_mode:
                    print(f"Responses scores are now len {len(responses_scores)}, shapes {[s.shape for s in responses_scores]}")
            else:
                raise ValueError("Outputs.scores is not a tuple")

            return LLMResponses(
                responses_strings=decoded_responses,
                responses_logits=responses_scores,
                activation_layers=activations_list if exposed_activations_request else None
            )
        else:
            # Prompt is an embedding tensor! Manual generation. This will be slow...
            # We'll generate the whole thing first, then get activations & logits using the others method!
            responses_embeddings = []

            for prompt_embedding in prompts:
                responses_embeddings.append(None)
                prompt_embedding = prompt_embedding.unsqueeze(0)
                if self.debug_mode:
                    print(f"About to handle prompt embedding of shape {prompt_embedding.shape}...")
                gen_embedding = self._embeddings_to_gen_embeddings(prompt_embedding)
                if self.debug_mode:
                    print(f"Converted to gen/chat embedding of shape {gen_embedding.shape}...")
                attention_mask = torch.ones(
                    (1, gen_embedding.size(1)), dtype=torch.long, device=self._model.device
                )
                if self.debug_mode:
                    print(f"Generated attention mask of shape {attention_mask.shape}...")

                found_eos = False
                for step_idx in range(max_new_tokens):
                    if self.debug_mode:
                        print(f"\nGeneration step {step_idx} of {max_new_tokens}...")
                    if responses_embeddings[-1] is None:
                        inputs_embeds = gen_embedding
                    else:
                        inputs_embeds = torch.cat([gen_embedding, responses_embeddings[-1]], dim=1)

                    if self.debug_mode:
                        print(f"About to forward with inputs_embeds of shape {inputs_embeds.shape}...")
                    outputs = self._model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                    if self.debug_mode:
                        print(f"Got logits of shape {outputs.logits.shape}...")
                        all_token_ids = outputs.logits.argmax(dim=-1)
                        print(f"All token ids: {all_token_ids}")
                        print(f"All token ids shape: {all_token_ids.shape}")
                        all_string_tokens = self._tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
                        print(f"All string tokens: {all_string_tokens}")
                    next_token_logits = outputs.logits[:, -1, :]
                    if self.debug_mode:
                        print(f"Next token logits shape: {next_token_logits.shape}")
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    if self.debug_mode:
                        special_tokens_output_all = self._tokenizer.get_special_tokens_mask(all_token_ids[0].tolist(), already_has_special_tokens=True)
                        print(f"Special tokens output all: {special_tokens_output_all}")
                        print(f"Got next_token_id of shape {next_token_id.shape}...")
                    
                    special_tokens_output_next = self._tokenizer.get_special_tokens_mask([next_token_id], already_has_special_tokens=True)
                    if special_tokens_output_next[0] == 1:
                        found_eos = True
                        if self.debug_mode: print(f"Found special token at step {step_idx}!!! Will break soon")

                    next_token_emb = self._token_ids_to_embeddings(next_token_id.unsqueeze(0))
                    if self.debug_mode:
                        print(f"Converted to next_token_emb of shape {next_token_emb.shape}...")
                    # next_token_emb => shape (1, 1, embedding_size)

                    # Expand input embedding
                    if not found_eos: # we don't want to include eos / eot token in the response
                        if responses_embeddings[-1] is None:
                            responses_embeddings[-1] = next_token_emb
                        else:
                            responses_embeddings[-1] = torch.cat([responses_embeddings[-1], next_token_emb], dim=1)
                    # Expand attention mask
                    am_extra = torch.ones((1, 1), dtype=torch.long, device=self._model.device)
                    attention_mask = torch.cat([attention_mask, am_extra], dim=1)
                    if self.debug_mode:
                        print(f"Updated gen embedding to shape {gen_embedding.shape}, attention mask to shape {attention_mask.shape}...")

                    if found_eos:
                        break
            
            # squeeze first dim of all response embeddings
            responses_embeddings = [e.squeeze(0) for e in responses_embeddings]
            if self.debug_mode:
                print(f"\nFinished collecting response embeddings! They have shapes {[e.shape for e in responses_embeddings]}")

            # TODO next: i need to trim the embeddings down to remove special tokens...
            # I also need to alter the "responses" string to include the first character...
            # Also need to think about whether "generate"'s function should include first-character logits or not...
            
            return self.generate_responses_forced(
                prompts,
                responses_embeddings,
                exposed_activations_request=exposed_activations_request
            )

    @property
    def device(self) -> torch.device:
        return self._model.device

    def _left_pad_embeddings(
        self,
        embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
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

        embeddings = torch.cat(padded_embeddings, dim=0)  # (batch_size, max_seq_len, embedding_size)
        attention_masks = torch.cat(attention_masks, dim=0)  # (batch_size, max_seq_len)
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
        chat_token_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
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
        input_tokens = self._tokenizer(input_string, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self._model.device)
        if add_response_ending:
            if self.debug_mode:
                print(f"Adding response ending of length {self._chat_outro_token_ids.shape[0]} (specifically: {self._chat_outro_token_ids}) to input tokens of length {input_tokens.shape[0]}...")
            return torch.cat((input_tokens, self._chat_outro_token_ids), dim=0)
        else:
            return input_tokens

    def _token_ids_to_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Expects input shape (batch_size, seq_len). Outputs shape (batch_size, seq_len, embedding_size)."""
        return self._model_embedding_layer(token_ids)

    def generate_responses_forced(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor]],
        target_responses_or_embeddings: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        add_response_ending: bool = False
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model, while forcing the outputs.
        This function is useful for extracting activations & logits for a target response, e.g. for soft-suffix attacks.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or of naked embeddings (i.e. no special tokens or padding). Each of shape (seq_len (varying), embedding_size), or a string.
            target_responses_or_embeddings: The target responses to force the model to generate. Each of shape (seq_len (varying), embedding_size), or a string.
            exposed_activations_request: Request specifying which activation layers to extract
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """

        assert len(prompts_or_embeddings) == len(target_responses_or_embeddings)
        if isinstance(prompts_or_embeddings[0], torch.Tensor):
            assert isinstance(target_responses_or_embeddings[0], torch.Tensor)
            assert len(prompts_or_embeddings[0].shape) == 2
            assert len(target_responses_or_embeddings[0].shape) == 2
            # add batch dimension
            prompts_or_embeddings = [prompt.unsqueeze(0) for prompt in prompts_or_embeddings]
            target_responses_or_embeddings = [response.unsqueeze(0) for response in target_responses_or_embeddings]
        else:
            assert isinstance(prompts_or_embeddings[0], str)
            assert isinstance(target_responses_or_embeddings[0], str)

        # Set up hooks for extracting activations
        if exposed_activations_request:
            target_layers = exposed_activations_request.extract_layers_indices
            activations_list = [[] for _ in target_layers]
            fwd_extraction_hooks = [(
                    self._model_block_modules[layer],
                    activation_extraction_hook(destination=activations_list, index=i, debug_mode=self.debug_mode)
                ) for i, layer in enumerate(target_layers)
            ]
        else:
            fwd_extraction_hooks = []

        if isinstance(prompts_or_embeddings[0], str):
            # Add special token chat template & padding!
            messages = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_response}
                ]
                for prompt, target_response in zip(prompts_or_embeddings, target_responses_or_embeddings)
            ]
            tokenized_chat = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                return_tensors="pt",
                return_dict=True
            ).to(self._model.device)
            if self.debug_mode:
                print(f"Tokenized chat: {tokenized_chat}")
                print(f"About to forward with tokenized_chat, with input_ids.shape: {tokenized_chat['input_ids'].shape}")
            with add_fwd_hooks(fwd_extraction_hooks):
                outputs = self._model.forward(**tokenized_chat)
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
            with add_fwd_hooks(fwd_extraction_hooks):
                outputs = self._model.forward(
                    inputs_embeds=chat_embeddings_tensor,
                    attention_mask=attention_masks,
                    #use_cache=False # test to try preventing graph issues...
                )
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
        
        decoded_responses = self._logits_to_strings(torch.stack(trimmed_logits))

        if self.debug_mode:
            print(f"Decoded responses: {decoded_responses}")

        return LLMResponses(
            responses_strings=decoded_responses,
            responses_logits=trimmed_logits, # list length (batch_size), each element shape (response_length, vocab_size)
            activation_layers=activations_list if exposed_activations_request else None # list length (num_req_layers), then list (num_req_tokens), then tensor (batch_size, hidden_size)
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
    # I expect we can just add a sort of custom hooks feature into ActivationRequest ?!!????!!!!

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

    # TODO: fix activation shape...
    print(output.activation_layers)
    print(len(output.activation_layers))
    print(len(output.activation_layers[0]))
    print(output.activation_layers[0][0].shape)

    print("\nEMBEDDING SPACE GENERATION:\n")

    output = llm.generate_responses([llm.string_to_embedding("Hello, how are you?")], ExposedActivationsRequest(extract_layers_indices=[0], token_selection_method=TokenSelectionMethod.ALL_RESPONSE))
    print(output.responses_strings)
    print(output.responses_logits)
    print(len(output.responses_logits))

    # TODO: fix activation shape...
    print(output.activation_layers)
    print(len(output.activation_layers))
    print(len(output.activation_layers[0]))
    print(output.activation_layers[0][0].shape)