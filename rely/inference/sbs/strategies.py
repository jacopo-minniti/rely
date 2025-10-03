# rely/inference/sbs/strategies.py

import logging
import random
import math
import uuid
import time
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rely.inference.sbs.main import StepBeamSearch
    from rely.inference.sbs.utils import SBSNode

logger = logging.getLogger(__name__)

class SamplingStrategy:
    """Base class for sampling strategies."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        raise NotImplementedError

    def update_candidate_uncertainty(self, candidate_node: 'SBSNode', generation_result: Dict[str, Any]):
        """Update the uncertainty of a newly generated candidate node."""
        pass # Default is no-op

    def requires_logprobs(self) -> bool:
        return False

class UniformStrategy(SamplingStrategy):
    """Distributes samples uniformly."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
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

class PumStrategy(SamplingStrategy):
    """Distributes samples based on PUM uncertainty."""
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

    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        if not sbs_instance.active_beams:
            return []

        # 1. Get uncertainties
        uncertainty_prompts = [sbs_instance.create_prompt(question, beam.full_text) for beam in sbs_instance.active_beams]
        uncertainty_scores = self._get_uncertainties_from_server(sbs_instance, uncertainty_prompts)
        
        for i, beam in enumerate(sbs_instance.active_beams):
            if i < len(uncertainty_scores):
                beam.uncertainty = uncertainty_scores[i]

        # 2. Calculate distribution (logic from sbs_pum.py)
        num_beams = len(uncertainty_scores)
        total_samples = sbs_instance.config.n_total_samples

        total_uncertainty = sum(uncertainty_scores)
        if total_uncertainty == 0:
            # Fallback to even distribution
            samples_per_beam = total_samples // num_beams
            remainder = total_samples % num_beams
            distribution = [samples_per_beam] * num_beams
            for i in range(remainder):
                distribution[i] += 1
            return distribution

        normalized_scores = [score / total_uncertainty for score in uncertainty_scores]
        sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

        total_assigned = sum(sample_distribution)
        remainder = total_samples - total_assigned
        if remainder > 0:
            sorted_indices = sorted(range(num_beams), key=lambda k: uncertainty_scores[k], reverse=True)
            for i in range(remainder):
                sample_distribution[sorted_indices[i % num_beams]] += 1
        
        zero_indices = [i for i, s in enumerate(sample_distribution) if s == 0]
        if zero_indices:
            for i in zero_indices:
                donatable_beams = sorted([(s, j) for j, s in enumerate(sample_distribution) if s > 1], reverse=True)
                if donatable_beams:
                    max_index = donatable_beams[0][1]
                    sample_distribution[max_index] -= 1
                    sample_distribution[i] += 1
                else:
                    break
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Uncertainty scores: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

        return sample_distribution


class PumPerValueStrategy(PumStrategy):
    """Distributes samples based on PUM uncertainty multiplied by beam value."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        if not sbs_instance.active_beams:
            return []

        # 1. Get uncertainties
        uncertainty_prompts = [sbs_instance.create_prompt(question, beam.full_text) for beam in sbs_instance.active_beams]
        uncertainty_scores = self._get_uncertainties_from_server(sbs_instance, uncertainty_prompts)
        
        for i, beam in enumerate(sbs_instance.active_beams):
            if i < len(uncertainty_scores):
                beam.uncertainty = uncertainty_scores[i]

        # New part: multiply by value
        value_scores = [beam.value for beam in sbs_instance.active_beams]
        distribution_scores = [pum * val for pum, val in zip(uncertainty_scores, value_scores)]

        # 2. Calculate distribution (logic from sbs_pum.py)
        num_beams = len(distribution_scores)
        total_samples = sbs_instance.config.n_total_samples

        total_score = sum(distribution_scores)
        if total_score == 0:
            # Fallback to even distribution
            samples_per_beam = total_samples // num_beams
            remainder = total_samples % num_beams
            distribution = [samples_per_beam] * num_beams
            for i in range(remainder):
                distribution[i] += 1
            return distribution

        normalized_scores = [score / total_score for score in distribution_scores]
        sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

        total_assigned = sum(sample_distribution)
        remainder = total_samples - total_assigned
        if remainder > 0:
            sorted_indices = sorted(range(num_beams), key=lambda k: distribution_scores[k], reverse=True)
            for i in range(remainder):
                sample_distribution[sorted_indices[i % num_beams]] += 1
        
        zero_indices = [i for i, s in enumerate(sample_distribution) if s == 0]
        if zero_indices:
            for i in zero_indices:
                donatable_beams = sorted([(s, j) for j, s in enumerate(sample_distribution) if s > 1], reverse=True)
                if donatable_beams:
                    max_index = donatable_beams[0][1]
                    sample_distribution[max_index] -= 1
                    sample_distribution[i] += 1
                else:
                    break
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Uncertainty scores: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Value scores: {[f'{s:.3f}' for s in value_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Distribution scores: {[f'{s:.3f}' for s in distribution_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

        return sample_distribution

class TokenEntropyStrategy(SamplingStrategy):
    """Distributes samples based on token entropy from the previous step."""
    
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

    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        if not sbs_instance.active_beams:
            return []

        is_first_step = all(beam.depth == 0 for beam in sbs_instance.active_beams)
        if is_first_step or len(sbs_instance.active_beams) == 1:
            # On first step, no uncertainty is available, so distribute uniformly.
            num_beams = len(sbs_instance.active_beams)
            samples_per_beam = sbs_instance.config.n_total_samples // num_beams
            remainder = sbs_instance.config.n_total_samples % num_beams
            distribution = [samples_per_beam + (1 if i < remainder else 0) for i in range(num_beams)]
            if sbs_instance.config.verbose:
                logger.info(f"[Rank {sbs_instance.worker_rank}] First step/single beam: Using equal distribution: {distribution}")
            return distribution

        # Use uncertainty from previous step
        uncertainty_scores = [beam.uncertainty for beam in sbs_instance.active_beams]
        
        # Same distribution calculation as PUM
        num_beams = len(uncertainty_scores)
        total_samples = sbs_instance.config.n_total_samples
        total_uncertainty = sum(uncertainty_scores)
        if total_uncertainty == 0:
            samples_per_beam = total_samples // num_beams
            remainder = total_samples % num_beams
            distribution = [samples_per_beam] * num_beams
            for i in range(remainder):
                distribution[i] += 1
            return distribution

        normalized_scores = [score / total_uncertainty for score in uncertainty_scores]
        sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

        total_assigned = sum(sample_distribution)
        remainder = total_samples - total_assigned
        if remainder > 0:
            sorted_indices = sorted(range(num_beams), key=lambda k: uncertainty_scores[k], reverse=True)
            for i in range(remainder):
                sample_distribution[sorted_indices[i % num_beams]] += 1
        
        while any(s == 0 for s in sample_distribution):
            donor_idx = -1
            max_samples = 1
            for i, s in enumerate(sample_distribution):
                if s > max_samples:
                    max_samples = s
                    donor_idx = i
            if donor_idx == -1: break
            try:
                receiver_idx = sample_distribution.index(0)
            except ValueError:
                break
            sample_distribution[donor_idx] -= 1
            sample_distribution[receiver_idx] += 1

        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Previous step uncertainties: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

        return sample_distribution


class UCBStrategy(SamplingStrategy):
    """UCB-like strategy: Uses uniform sampling but scores beams with value + c * uncertainty."""
    def __init__(self, uncertainty_task_queue, uncertainty_result_queue, c=1.0):
        self.uncertainty_task_queue = uncertainty_task_queue
        self.uncertainty_result_queue = uncertainty_result_queue
        self.c = c  # UCB exploration parameter

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

    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        """Uniform distribution - same as UniformStrategy."""
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
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution (UCB uniform): {samples_per_beam}")
            
        return samples_per_beam

    def calculate_ucb_scores(self, sbs_instance: 'StepBeamSearch', question: str) -> None:
        """Calculate UCB scores for all active beams: value + c * uncertainty."""
        if not sbs_instance.active_beams:
            return

        # Get uncertainties from PUM model
        uncertainty_prompts = [sbs_instance.create_prompt(question, beam.full_text) for beam in sbs_instance.active_beams]
        uncertainty_scores = self._get_uncertainties_from_server(sbs_instance, uncertainty_prompts)
        
        # Update beams with UCB scores
        for i, beam in enumerate(sbs_instance.active_beams):
            if i < len(uncertainty_scores):
                beam.uncertainty = uncertainty_scores[i]
                # UCB score: value + c * uncertainty
                ucb_score = beam.value + self.c * uncertainty_scores[i]
                beam.ucb_score = ucb_score
            else:
                beam.uncertainty = 0.5
                beam.ucb_score = beam.value + self.c * 0.5

        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] UCB scores calculated:")
            for i, beam in enumerate(sbs_instance.active_beams):
                logger.info(f"  Beam {i}: value={beam.value:.3f}, uncertainty={beam.uncertainty:.3f}, ucb_score={beam.ucb_score:.3f}")
    
    def get_ucb_parameter(self) -> float:
        """Get the UCB exploration parameter."""
        return self.c