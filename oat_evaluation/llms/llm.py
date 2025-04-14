from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

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
    def __init__(self, responses_strings: List[str], responses_logits: torch.Tensor, activation_layers: List[torch.Tensor]):
        """
        Initialize LLMResponses with generated responses, logits, and activation layers.
        
        Args:
            responses_strings: List of generated response strings, length = batch_size
            responses_logits: Tensor of shape (batch_size, sequence_length, vocab_size) containing logits
            activation_layers: List of activation tensors organized as:
                - Inner list: length = num_req_layers (one entry per requested layer)
                - Each tensor: shape = (batch_size, num_req_tokens, hidden_size), where:
                  * num_req_tokens = 1 for LAST_RESPONSE_TOKEN/LAST_USER_TOKEN
                  * num_req_tokens = sequence_length for ALL_RESPONSE
        """
        self.responses_strings = responses_strings
        self.responses_logits = responses_logits
        self.activation_layers = activation_layers 

    @property
    def batch_size(self) -> int:
        return max(len(self.responses_strings), self.responses_logits.shape[0])

    @property
    def activation_layers_indices(self) -> List[int]:
        return self.activation_layers_indices

class LLM(ABC):
    @abstractmethod
    def generate_responses(
        self,
        prompts_or_embeddings: Union[List[str], List[torch.Tensor]],
        exposed_activations_request: Optional[ExposedActivationsRequest] = None
    ) -> LLMResponses:
        """
        Generate responses for the given prompts using the model.
        
        Args:
            prompts_or_embeddings: The prompts to generate responses for, as a list of strings or a tensor of embeddings
            exposed_activations_request: Request specifying which activation layers to extract
            
        Returns:
            LLMResponses containing the generated responses, their logits, and the extracted activation layers.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def string_to_embedding(self, string: str) -> torch.Tensor:
        """Converts a prompt/response string to a "naked" embedding tensor. i.e. Does not add any special tokens or padding. Returns shape (1, seq_len, embedding_size)."""
        pass

    @abstractmethod
    def string_to_token_ids(self, input_string):
        """Output shape (batch_size, seq_len)"""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """The number of layers in the model (specifically, the max number of activation layers to extract)"""
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
    def name(self) -> str:
        """The name of the model"""
        return self.__class__.__name__

    # TODO: consider how to support attacks injected into activation space...
    # We'll need to be able to extract activations, give them to an arbitrary attack function, then continue generation....
    # I expect we can just add a sort of "hooks" feature into ActivationRequest ?!!????!!!!
