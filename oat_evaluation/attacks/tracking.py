from contextlib import contextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from oat_evaluation.attacks.attack import AttackDetails
from oat_evaluation.llms.metrics import LLMMetricsTracker
import warnings

@dataclass
class AttackTracker:
    """Tracks API usage and token counts during an attack."""
    num_api_calls: int = 0
    num_total_tokens: int = 0
    api_usd_cost: float = 0.0
    local_flop_cost: int = 0
    used_api_llm: bool = False
    
    def to_attack_details(self) -> AttackDetails:
        """Convert tracker state to AttackDetails."""
        # Get aggregated metrics from all LLMs
        metrics = LLMMetricsTracker.get_aggregated_metrics()
        
        return AttackDetails(
            num_api_calls=metrics.num_api_calls,
            num_total_tokens=metrics.num_total_tokens,
            api_usd_cost=metrics.api_usd_cost,
            local_flop_cost=metrics.local_flop_cost,
            used_api_llm=metrics.num_api_calls > 0
        )

@contextmanager
def track_attack():
    """Context manager for tracking API usage and token counts during an attack."""
    # Clear any existing metrics
    LLMMetricsTracker.clear_metrics()
    
    tracker = AttackTracker()
    try:
        yield tracker
    finally:
        # Warn if tracker fields were modified directly, as these modifications
        # will be overwritten by the final aggregated metrics.
        if tracker.num_api_calls != 0:
            warnings.warn(
                "AttackTracker.num_api_calls was modified directly within the 'track_attack' context. "
                "This value will be overwritten by aggregated metrics.", UserWarning
            )
        if tracker.num_total_tokens != 0:
            warnings.warn(
                "AttackTracker.num_total_tokens was modified directly within the 'track_attack' context. "
                "This value will be overwritten by aggregated metrics.", UserWarning
            )
        if tracker.api_usd_cost != 0.0:
            warnings.warn(
                "AttackTracker.api_usd_cost was modified directly within the 'track_attack' context. "
                "This value will be overwritten by aggregated metrics.", UserWarning
            )
        if tracker.local_flop_cost != 0:
            warnings.warn(
                "AttackTracker.local_flop_cost was modified directly within the 'track_attack' context. "
                "This value will be overwritten by aggregated metrics.", UserWarning
            )
        if tracker.used_api_llm is True: # Default is False
            warnings.warn(
                "AttackTracker.used_api_llm was modified directly within the 'track_attack' context. "
                "This value will be overwritten by aggregated metrics.", UserWarning
            )

        # Update tracker with final metrics
        metrics = LLMMetricsTracker.get_aggregated_metrics()
        tracker.num_api_calls = metrics.num_api_calls
        tracker.num_total_tokens = metrics.num_total_tokens
        tracker.api_usd_cost = metrics.api_usd_cost
        tracker.local_flop_cost = metrics.local_flop_cost
        tracker.used_api_llm = metrics.num_api_calls > 0
