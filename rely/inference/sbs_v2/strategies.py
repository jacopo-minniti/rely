# rely/inference/sbs/strategies.py

import logging
import random
import math
import uuid
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SamplingStrategy:
    """Base class for sampling strategies."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        """Distributes samples uniformly across active beams."""
        num_active_beams = len(sbs_instance.active_beams)
        if num_active_beams == 0:
            return []
        
        total_samples_budget = sbs_instance.config.n_total_samples
        
        base_samples = total_samples_budget // num_active_beams
        remainder = total_samples_budget % num_active_beams
        
        samples_per_beam = [base_samples] * num_active_beams
        if remainder > 0:
            indices_for_extra = random.sample(range(num_active_beams), remainder)
            for idx in indices_for_extra:
                samples_per_beam[idx] += 1
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution (uniform): {samples_per_beam}")
            
        return samples_per_beam

    def update_candidate_uncertainty(self, candidate_node: 'SBSNode', generation_result: Dict[str, Any]):
        """Update the uncertainty of a newly generated candidate node."""
        pass # Default is no-op

    def calculate_candidate_uncertainties(self, sbs_instance: 'StepBeamSearch', question: str, candidate_data: List[Dict[str, Any]]) -> Optional[List[float]]:
        """Calculate uncertainties for a batch of new candidates."""
        return None

    def requires_logprobs(self) -> bool:
        return False

class UniformStrategy(SamplingStrategy):
    """Strategy that uses uniform sampling and no uncertainty."""
    pass

class PumStrategy(SamplingStrategy):
    """Uses PUM for uncertainty score."""
    def __init__(self, uncertainty_task_queue, uncertainty_result_queue):
        self.uncertainty_task_queue = uncertainty_task_queue
        self.uncertainty_result_queue = uncertainty_result_queue

    def _get_uncertainties_from_server(self, sbs_instance, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        request_id = str(uuid.uuid4())
        payload = {"request_id": request_id, "worker_rank": sbs_instance.worker_rank, "prompts": prompts}
        self.uncertainty_task_queue.put(payload)
        
        while True:
            response = self.uncertainty_result_queue.get()
            if response.get("request_id") == request_id:
                return response["uncertainties"]
            self.uncertainty_result_queue.put(response)
            time.sleep(0.01)

    def calculate_candidate_uncertainties(self, sbs_instance: 'StepBeamSearch', question: str, candidate_data: List[Dict[str, Any]]) -> Optional[List[float]]:
        if not candidate_data:
            return []
        
        prompts_for_uncertainty = []
        for data in candidate_data:
            parent_node = data['parent_node']
            gen_text = data['generation_result']['text']
            snippet = gen_text.rstrip() + '

'
            new_full_text = parent_node.full_text + snippet
            prompts_for_uncertainty.append(sbs_instance.create_prompt(question, new_full_text))
        
        return self._get_uncertainties_from_server(sbs_instance, prompts_for_uncertainty)

class TokenEntropyStrategy(SamplingStrategy):
    """Uses token entropy for uncertainty score."""
    
    def requires_logprobs(self) -> bool:
        return True

    def _calculate_token_entropy(self, log_probs_data, k: int) -> float:
        if not log_probs_data:
            return 0.5
        
        entropies = []
        if hasattr(log_probs_data, 'top_logprobs') and log_probs_data.top_logprobs:
            for token_log_probs_dict in log_probs_data.top_logprobs:
                if token_log_probs_dict:
                    top_k_log_probs = list(token_log_probs_dict.values())[:k]
                    if top_k_log_probs:
                        probs = [math.exp(log_p) for log_p in top_k_log_probs]
                        total_prob = sum(probs)
                        if total_prob > 0:
                            probs = [p / total_prob for p in probs]
                            entropy = -sum(p * math.log(p) for p in probs if p > 0)
                            entropies.append(entropy)
        return sum(entropies) / len(entropies) if entropies else 0.5

    def update_candidate_uncertainty(self, candidate_node: 'SBSNode', generation_result: Dict[str, Any]):
        logprobs = generation_result.get('logprobs')
        k = generation_result.get('entropy_k', 20) # A bit of a hack to pass k
        candidate_node.uncertainty = self._calculate_token_entropy(logprobs, k)
