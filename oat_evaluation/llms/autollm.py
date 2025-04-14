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

def activation_extraction_hook(destination: List[torch.Tensor], index: int):
    def hook_fn(module, input):
        print(f"Hook number {index} triggered with input: {input}")
        destination[index] = input[0].detach().clone()
    return hook_fn

class AutoLLM(LLM):

    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")

        self._model = AutoModelForCausalLM.from_pretrained(
            # float16 was the default for obfuscated-activations too
            model_path, device_map="cuda", torch_dtype=torch.float16
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
        self.pad_embedding = self._token_ids_to_embeddings(self.pad_token_id)

        self._figure_out_chat_function()

        print(f"Loaded model with left-padding token: {self._tokenizer.pad_token}")

    def generate_responses(
        self,
        prompts: List[str],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None
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
            target_layers = exposed_activations_request.extract_layers_indices
            activations_list = [[torch.zeros(1) for _ in target_layers] for _ in prompts]
            fwd_extraction_hooks = [(
                    self._model_block_modules[layer],
                    activation_extraction_hook(destination=activations_list, index=i)
                ) for i, layer in enumerate(target_layers)
            ]
        else:
            fwd_extraction_hooks = []

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

        with add_fwd_hooks(fwd_extraction_hooks):
            outputs = self._model.generate(tokenized_chat['input_ids'], output_scores=True, return_dict_in_generate=True)

        start_length = tokenized_chat["input_ids"].shape[1]
        decoded_responses = [
            self._tokenizer.decode(seq[start_length:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]

        return LLMResponses(
            responses_strings=decoded_responses,
            responses_logits=outputs.scores,
            activation_layers=activations_list if exposed_activations_request else None
        )

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
        for emb in embeddings:
            seq_len = emb.size(1)
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                # Repeat self.pad_embedding to match pad_len
                padding = self.pad_embedding.expand(1, pad_len, embedding_size)
                padded = torch.cat([padding, emb], dim=1)
            else:
                padded = emb
            padded_embeddings.append(padded)

        return torch.cat(padded_embeddings, dim=0)  # (batch_size, max_seq_len, embedding_size)


    def _figure_out_chat_function(self):
        # Let's try to figure out how to turn embeddings into chat-template embeddings!
        # First, let's establish what "chat-template" embed we actually want.
        # i.e. let's embed pre and post chat and compare them...

        prompt = "How to bake?"

        # Returns (batch_size, seq_len, vocab_size)
        token_ids1 = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        embeddings1 = self._model_embedding_layer(token_ids1["input_ids"])

        messages = [
            [{"role": "user", "content": prompt}]
        ]
        token_ids2 = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
        embeddings2 = self._model_embedding_layer(token_ids2["input_ids"])

        insertion_index = -1
        for i in range(embeddings2.shape[1]):
            if embeddings2[0, i].equal(embeddings1[0, 0]):
                # Insertion here!
                print(f"Found chat template intro length {i}, outro length {embeddings2.shape[1] - i - embeddings1.shape[1]}")
                insertion_index = i
                break

        embeddings2_intro = embeddings2[0, :insertion_index].unsqueeze(0)
        embeddings2_outro = embeddings2[0, insertion_index+embeddings1.shape[1]:].unsqueeze(0)

        self._embeddings_to_chat_embeddings = lambda raw: torch.cat((embeddings2_intro, raw, embeddings2_outro), dim=1)
        
    def string_to_token_ids(self, input_string):
        """Output shape (batch_size, seq_len)"""
        return self._tokenizer(input_string, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self._model.device)

    def _token_ids_to_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Expects input shape (batch_size, seq_len). Outputs shape (batch_size, seq_len, embedding_size)."""
        return self._model_embedding_layer(token_ids)

    def generate_responses_forced(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor]],
        target_responses_or_embeddings: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model, while forcing the outputs.
        This function is useful for extracting activations & logits for a target response, e.g. for soft-suffix attacks.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or of naked embeddings (i.e. no special tokens or padding)
            target_responses_or_embeddings: The target responses to force the model to generate
            exposed_activations_request: Request specifying which activation layers to extract
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """

        raise NotImplementedError("This function is not fully implemented yet...")

        # Set up hooks for extracting activations
        if exposed_activations_request:
            target_layers = exposed_activations_request.extract_layers_indices
            activations_list = [[torch.zeros(1) for _ in target_layers] for _ in prompts]
            fwd_extraction_hooks = [(
                    self._model_block_modules[layer],
                    activation_extraction_hook(destination=activations_list, index=i)
                ) for i, layer in enumerate(target_layers)
            ]
        else:
            fwd_extraction_hooks = []

        if isinstance(prompts_or_embeddings[0], str):
            # Add special token chat template & padding!
            messages = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts_or_embeddings
            ]
            tokenized_chat = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True
            ).to(self._model.device)

            with add_fwd_hooks(fwd_extraction_hooks):
                outputs = self._model.forward(tokenized_chat['input_ids'], output_scores=True, return_dict_in_generate=True)
        else:
            chat_embeddings = [self._embeddings_to_chat_embeddings(embed) for embed in prompts_or_embeddings]
            print(f"Chat embeddings: {chat_embeddings}")
            chat_embeddings_tensor = self._left_pad_embeddings(chat_embeddings)
            print(f"Chat embeddings tensor: {chat_embeddings_tensor}")

            with add_fwd_hooks(fwd_extraction_hooks):
                outputs = self._model.forward(chat_embeddings_tensor, output_scores=True, return_dict_in_generate=True)

        print(f"Outputs: {outputs}")
        if isinstance(outputs, torch.Tensor):
            print(f"Outputs is a tensor of shape {outputs.shape}")

        # Slice out just the newly generated tokens, then decode to text
        try:
            start_length = tokenized_chat["input_ids"].shape[1]
            decoded_responses = [
                self._tokenizer.decode(seq[start_length:], skip_special_tokens=True)
                for seq in outputs.sequences
            ]
        except Exception as e:
            print(f"Error trying to extract start length: {e}")
            decoded_responses = [
                self._tokenizer.decode(outputs.sequences, skip_special_tokens=True)
            ]

        print(f"Decoded responses: {decoded_responses}")

        return LLMResponses(
            responses_strings=decoded_responses,
            responses_logits=outputs.scores,
            activation_layers=activations_list if exposed_activations_request else None
        )


    def string_to_embedding(self, string: str) -> torch.Tensor:
        """Converts a prompt/response string to a "naked" embedding tensor. i.e. Does not add any special tokens or padding."""
        # Returns shape (1, seq_len, embedding_size)

        return self._token_ids_to_embeddings(self._string_to_token_ids(string))

    def _get_block_modules(self) -> List[torch.nn.Module]:
        """Get the transformer block modules for hooking."""
        blocks = []
        for name, module in self._model.named_modules():
            # For Gemma models, the transformer blocks are in model.layers
            if isinstance(module, torch.nn.Module) and hasattr(module, 'self_attn'):
                blocks.append(module)
        return blocks

    @property
    def num_layers(self) -> int:
        """The number of transformer blocks in the model (i.e. the max number of activation layers to extract)"""
        return len(self._model_block_modules)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary of the model (number of possible tokens, i.e. number of columns in the logits tensor)"""
        return self._model.lm_head.out_features

    @property
    def embedding_size(self) -> int:
        """The number of columns in embedding tensors"""
        return self._model.embeddings.word_embeddings.embedding_dim

    @property
    def name(self) -> str:
        """The name of the model"""
        return self.__class__.__name__

    # TODO: consider how to support attacks injected into activation space...
    # We'll need to be able to extract activations, give them to an arbitrary attack function, then continue generation....
    # I expect we can just add a sort of custom hooks feature into ActivationRequest ?!!????!!!!

if __name__ == "__main__":
    gemma = AutoLLM("/workspace/gemma_2_9b_instruct")
    output = gemma.generate_responses(["Hello, how are you?"], ExposedActivationsRequest(extract_layers_indices=[0], token_selection_method=TokenSelectionMethod.ALL_RESPONSE))
    print(output.responses_strings)
    print(output.responses_logits)
    print(output.activation_layers)