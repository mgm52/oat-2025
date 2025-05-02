from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

class HarmfulnessGrades():
    def __init__(self, refusals: List[bool], other_metrics: List[Dict[str, float]]):
        self.refusals = refusals # whether theses responses count as "refusals"
        self.other_metrics = other_metrics # dictionary of other metrics like strongreject scores, specificity, etc
    
    def __str__(self) -> str:
        output = f"Harmfulness grades:\n  Refusals (mean {np.mean(self.refusals)}): {self.refusals}"
        if len(self.other_metrics) > 0:
            for metric in self.other_metrics[0].keys():
                metric_values = [metrics.get(metric, None) for metrics in self.other_metrics]
                output += f'\n  "{metric}" (mean {np.mean(metric_values)}): {metric_values}'
        return output

class HarmGrader(ABC):
    @abstractmethod
    def grade_harm(self, prompts: List[str], responses: List[str]) -> HarmfulnessGrades:
        pass

    @property
    def name(self) -> str:
        """The name of the grader"""
        return self.__class__.__name__