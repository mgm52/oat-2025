import threading
from dataclasses import dataclass, field
from typing import Dict, Optional
from oat_evaluation.llms.llm import LLM

@dataclass
class LLMMetrics:
    """Thread-local metrics for a single LLM instance."""
    num_total_tokens: int = 0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    local_flop_cost: int = 0
    api_usd_cost: float = 0.0
    num_api_calls: int = 0

class LLMMetricsTracker:
    """Thread-local storage for tracking metrics across multiple LLMs."""
    _thread_local = threading.local()
    
    @classmethod
    def get_metrics(cls, llm: LLM) -> LLMMetrics:
        """Get metrics for a specific LLM instance in the current thread."""
        if not hasattr(cls._thread_local, 'metrics'):
            cls._thread_local.metrics = {}
        
        # Use id() of LLM as key to ensure uniqueness
        llm_id = id(llm)
        if llm_id not in cls._thread_local.metrics:
            cls._thread_local.metrics[llm_id] = LLMMetrics()
        
        return cls._thread_local.metrics[llm_id]
    
    @classmethod
    def update_metrics(cls, llm: LLM, num_input_tokens: int, num_output_tokens: int) -> None:
        """Update metrics for a specific LLM instance."""
        metrics = cls.get_metrics(llm)
        metrics.num_input_tokens += num_input_tokens
        metrics.num_output_tokens += num_output_tokens
        metrics.num_total_tokens += num_input_tokens + num_output_tokens
        
        # Update API-specific metrics
        if hasattr(llm, 'usd_per_input_token') and hasattr(llm, 'usd_per_output_token'):
            metrics.num_api_calls += 1
            assert llm.usd_per_input_token > 0 and llm.usd_per_output_token > 0
            metrics.api_usd_cost += (
                num_input_tokens * llm.usd_per_input_token +
                num_output_tokens * llm.usd_per_output_token
            )
    
    @classmethod
    def add_local_flops(cls, llm: LLM, flops: int) -> None:
        """Update local FLOP count for a specific LLM instance."""
        metrics = cls.get_metrics(llm)
        metrics.local_flop_cost += flops
    
    @classmethod
    def get_aggregated_metrics(cls) -> LLMMetrics:
        """Get aggregated metrics across all LLMs in the current thread."""
        if not hasattr(cls._thread_local, 'metrics'):
            return LLMMetrics()
        
        aggregated = LLMMetrics()
        for metrics in cls._thread_local.metrics.values():
            aggregated.num_total_tokens += metrics.num_total_tokens
            aggregated.num_input_tokens += metrics.num_input_tokens
            aggregated.num_output_tokens += metrics.num_output_tokens
            aggregated.local_flop_cost += metrics.local_flop_cost
            aggregated.api_usd_cost += metrics.api_usd_cost
            aggregated.num_api_calls += metrics.num_api_calls
        
        return aggregated
    
    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all metrics in the current thread."""
        if hasattr(cls._thread_local, 'metrics'):
            cls._thread_local.metrics.clear() 