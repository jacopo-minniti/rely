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

def _distribute_samples_proportionally(scores: List[float], total_samples: int, num_beams: int) -> List[int]:
    """Helper to distribute samples proportionally to a list of scores."""
    total_score = sum(scores)
    if total_score == 0:
        # Fallback to even distribution
        base_samples = total_samples // num_beams
        remainder = total_samples % num_beams
        distribution = [base_samples + (1 if i < remainder else 0) for i in range(num_beams)]
        return distribution

    normalized_scores = [score / total_score for score in scores]
    sample_distribution = [int(norm_score * total_samples) for norm_score in normalized_scores]

    total_assigned = sum(sample_distribution)
    remainder = total_samples - total_assigned
    if remainder > 0:
        sorted_indices = sorted(range(num_beams), key=lambda k: scores[k], reverse=True)
        for i in range(remainder):
            sample_distribution[sorted_indices[i % num_beams]] += 1
    
    # Ensure no beam gets 0 samples if possible
    zero_indices = [i for i, s in enumerate(sample_distribution) if s == 0]
    if zero_indices:
        for i in zero_indices:
            # Find a beam with > 1 sample to donate
            donatable_beams = sorted([(s, j) for j, s in enumerate(sample_distribution) if s > 1], reverse=True)
            if donatable_beams:
                donor_index = donatable_beams[0][1]
                sample_distribution[donor_index] -= 1
                sample_distribution[i] += 1
            else:
                # Cannot re-allocate, break to avoid infinite loops
                break
    return sample_distribution


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

class PumBasedStrategy(SamplingStrategy):
    """Base class for strategies that use the PUM uncertainty model."""
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

class PumStrategy(PumBasedStrategy):
    """Distributes samples based on PUM uncertainty."""
    def distribute_samples(self, sbs_instance: 'StepBeamSearch', question: str) -> List[int]:
        if not sbs_instance.active_beams:
            return []

        # 1. Get uncertainties
        uncertainty_prompts = [sbs_instance.create_prompt(question, beam.full_text) for beam in sbs_instance.active_beams]
        uncertainty_scores = self._get_uncertainties_from_server(sbs_instance, uncertainty_prompts)
        
        for i, beam in enumerate(sbs_instance.active_beams):
            if i < len(uncertainty_scores):
                beam.uncertainty = uncertainty_scores[i]

        # 2. Calculate distribution
        sample_distribution = _distribute_samples_proportionally(
            uncertainty_scores,
            sbs_instance.config.n_total_samples,
            len(sbs_instance.active_beams)
        )
        
        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Uncertainty scores: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

        return sample_distribution


class PumPerValueStrategy(PumBasedStrategy):
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

        # 2. Multiply by value to get distribution scores
        value_scores = [beam.value for beam in sbs_instance.active_beams]
        distribution_scores = [pum * val for pum, val in zip(uncertainty_scores, value_scores)]

        # 3. Calculate distribution
        sample_distribution = _distribute_samples_proportionally(
            distribution_scores,
            sbs_instance.config.n_total_samples,
            len(sbs_instance.active_beams)
        )
        
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
        
        # Calculate distribution
        sample_distribution = _distribute_samples_proportionally(
            uncertainty_scores,
            sbs_instance.config.n_total_samples,
            len(sbs_instance.active_beams)
        )

        if sbs_instance.config.verbose:
            logger.info(f"[Rank {sbs_instance.worker_rank}] Previous step uncertainties: {[f'{s:.3f}' for s in uncertainty_scores]}")
            logger.info(f"[Rank {sbs_instance.worker_rank}] Sample distribution: {sample_distribution}")

        return sample_distribution


class UCBStrategy(PumBasedStrategy):
    """UCB-like strategy: Uses uniform sampling but scores beams with value + c * uncertainty."""
    def __init__(self, uncertainty_task_queue, uncertainty_result_queue, c=1.0):
        super().__init__(uncertainty_task_queue, uncertainty_result_queue)
        self.c = c  # UCB exploration parameter

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

    def calculate_ucb_scores(self, sbs_instance: 'StepBeamSearch', question: str, nodes: List['SBSNode']) -> None:
        """Calculate UCB scores for the given nodes: value + c * uncertainty."""
        if not nodes:
            return

        # Get uncertainties from PUM model
        uncertainty_prompts = [sbs_instance.create_prompt(question, beam.full_text) for beam in nodes]
        uncertainty_scores = self._get_uncertainties_from_server(sbs_instance, uncertainty_prompts)
        
        # Update nodes with UCB scores
        for i, beam in enumerate(nodes):
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
            for i, beam in enumerate(nodes):
                logger.info(f"  Beam {i}: value={beam.value:.3f}, uncertainty={beam.uncertainty:.3f}, ucb_score={beam.ucb_score:.3f}")
    
    def get_ucb_parameter(self) -> float:
        """Get the UCB exploration parameter."""
        return self.c
