from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union

import torch

from oat_evaluation.llms.llm import ExposedActivationsRequest, LLMResponses

class Probe(ABC):
    @abstractmethod
    def compute_scores(
        self,
        responses: LLMResponses,
    ) -> List[torch.Tensor]:  # List of length batch_size
        """
        Compute probe scores for the given responses (one per response).
        
        Args:
            responses: The responses to compute probe scores for
            
        Returns:
            List of probe scores (one per response).
        """
        pass

    @property
    @abstractmethod
    def activation_extraction_request(self) -> ExposedActivationsRequest:
        """The request for the activation extraction"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the probe"""
        return self.__class__.__name__
