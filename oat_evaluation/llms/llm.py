from abc import ABC, abstractmethod
from enum import Enum
from http.client import responses
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch


class TokenSelectionMethod(Enum):
    ALL_RESPONSE = "all_response"
    LAST_RESPONSE_TOKEN = "last_response_token"
    LAST_USER_TOKEN = "last_user_token"


class ExposedActivationsRequest():
    # TODO: add argument for funcions to run over extracted layers during forward pass (via hooks)
    def __init__(self, extract_layers_indices: List[int], token_selection_method: TokenSelectionMethod):
        self.extract_layers_indices = extract_layers_indices
        self.token_selection_method = token_selection_method


class LLMResponses():
    def __init__(self, responses_strings: List[str], responses_logits: Optional[list[torch.Tensor]], activation_layers: Optional[list[list[torch.Tensor]]] = None):
        """
        Initialize LLMResponses with generated responses, logits, and activation layers.
        
        Args:
            responses_strings: List of generated response strings, length = batch_size
            responses_logits: List of tensors of shape (sequence_length, vocab_size) containing logits, length = batch_size
            activation_layers: List of activation tensors organized as:
                - Inner list: length = batch_size 
                - Inner inner list: length = num_req_layers
                - Each tensor: shape = (num_req_tokens, hidden_size)
        """
        # FIXME: These checks don't work, since logits and activation layers are sometimes not set
        # bs_strings = len(responses_strings)
        # bs_logits = len(responses_logits)
        # bs_activations = len(activation_layers[0]) if bs_strings else 0

        # if not (bs_strings == bs_logits == bs_activations):
        #      raise ValueError(f"Batch size mismatch: "
        #                       f"strings ({bs_strings}), "
        #                       f"logits ({bs_logits}), "
        #                       f"activations ({bs_activations})")

        self.responses_strings = responses_strings
        self.responses_logits = responses_logits
        self.activation_layers = activation_layers 

    @property
    def batch_size(self) -> int:
        return max(len(self.responses_strings), len(self.responses_logits))
    
    def __add__(self, other: 'LLMResponses') -> 'LLMResponses':
        """
        Concatenates this LLMResponses object with another.

        Args:
            other: Another LLMResponses object.

        Returns:
            A new LLMResponses object containing the combined data.

        Raises:
            TypeError: If 'other' is not an LLMResponses instance.
        """
        if not isinstance(other, LLMResponses):
            return NotImplementedError(f"LLMResponse merging not implemented for type {type(other)}")
        responses_logits = self.responses_logits + other.responses_logits if self.responses_logits and other.responses_logits else None
        activation_layers = self.activation_layers + other.activation_layers if self.activation_layers and other.activation_layers else None
        return LLMResponses(responses_strings=self.responses_strings + other.responses_strings,
                            responses_logits=responses_logits,
                            activation_layers=activation_layers)


class LLM(ABC):
    @abstractmethod
    def generate_responses(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor], list[list[dict[str, str]]]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        max_new_tokens: int = 64,
        *args,
        **kwargs,
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or a tensor of embeddings
            exposed_activations_request: Request specifying which activation layers to extract
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """
        pass

    @abstractmethod
    def generate_responses_forced(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor]],
        target_responses_or_embeddings: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None,
        add_response_ending: bool = False,
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model, while forcing the outputs.
        This function is useful for extracting activations & logits for a target response, e.g. for soft-suffix attacks.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or of naked embeddings (i.e. no special tokens or padding). Each of shape (seq_len (varying), embedding_size), or a string.
            target_responses_or_embeddings: The target responses to force the model to generate. Each of shape (seq_len (varying), embedding_size), or a string.
            exposed_activations_request: Request specifying which activation layers to extract
            add_response_ending: Whether to add the response ending token
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """
        pass

    @abstractmethod
    def string_to_embedding(self, string: str) -> torch.Tensor:
        """Converts a prompt/response string to a "naked" embedding tensor. i.e. Does not add any special tokens or padding. Returns shape (seq_len, embedding_size)."""
        pass

    @abstractmethod
    def string_to_token_ids(self, input_string: str, add_response_ending: bool = False) -> torch.Tensor:
        """
        Convert a string to token IDs.
        
        Args:
            input_string: The string to tokenize
            add_response_ending: Whether to add response ending tokens
            
        Returns:
            Tensor of shape (seq_len)
        """
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """The number of layers in the model (specifically, the max number of activation layers to extract)"""
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        """The number of parameters in the model"""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """The size of the vocabulary of the model (number of possible tokens, i.e. number of columns in the logits tensor)"""
        pass

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """The number of columns in embedding tensors"""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device of the model"""
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the model"""
        pass

    @property
    def name(self) -> str:
        """The name of the model"""
        return self.__class__.__name__
    
    # TODO: consider how to support attacks injected into activation space...
    # We'll need to be able to extract activations, give them to an arbitrary attack function, then continue generation....
    # I expect we can just add a sort of "hooks" feature into ActivationRequest ?!!????!!!!
