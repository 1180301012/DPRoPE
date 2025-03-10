import torch
import torch.nn.functional as F
from transformers.utils import logging

logger = logging.get_logger(__name__)




class DPRopeClass:
    def __init__(self, rope_theta: int=10000, origin_max_position: int=4096, head_dim: int=128, scaling_factor: int=4, target_max_position: int=None, intervals: int=360):
        self.rope_theta = rope_theta
        self.origin_max_position = origin_max_position
        self.head_dim = head_dim
        self.scaling_factor = scaling_factor
        self.target_max_position = target_max_position
        if not target_max_position:
            self.target_max_position = self.origin_max_position * self.scaling_factor
        self.intervals = intervals

        self.all_scaling_methods = [self._extrapolation, self._interpolation] # Add custom scaling methods here: NTK, YaRN...

    def _get_inv_freq(self):
        return 1 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))


    def _extrapolation(self, seq_len: int):
        all_inv_freqs = self._get_inv_freq().repeat(seq_len, 1).transpose(0, 1)
        all_scaling_factors = torch.ones((self.head_dim // 2,))
        return all_inv_freqs, all_scaling_factors
    
    def _interpolation(self, seq_len: int):
        all_inv_freqs = self._get_inv_freq().repeat(seq_len, 1).transpose(0, 1) / self.scaling_factor
        all_scaling_factors = torch.full((self.head_dim // 2,), self.scaling_factor)
        return all_inv_freqs, all_scaling_factors

    def _custom_scaling_methods(self, seq_len: int):
        # Implementation for other scaling methods: NTK,YaRN...
        # return all_inv_freqs, all_scaling_factors
        pass

    def _get_rotation_angles(self, seq_len: int, scaling_methods: callable):
        all_inv_freqs, all_scaling_factors = scaling_methods(seq_len)
        all_positions = torch.arange(0, seq_len).repeat(self.head_dim // 2, 1) # [head_dim // 2, max_position]
        all_angles = all_inv_freqs * all_positions
        
        all_angles = all_angles % (torch.pi * 2)
        
        all_angles_count = torch.full((self.head_dim // 2, self.intervals), torch.finfo(torch.float16).tiny)
        for idx in range(all_angles.shape[0]):
            for jdx in range(all_angles.shape[1]):
                position_map = torch.floor((self.intervals / (torch.pi * 2)) * all_angles[idx][jdx])
                all_angles_count[idx][int(position_map)] += 1
        
        all_angles_distribution = all_angles_count / seq_len

        return all_angles_distribution, all_scaling_factors


    def _compute_distribution_similarity(self, origin_distribution, scaling_distribution):
        similarity = torch.zeros((self.head_dim // 2,))
        for idx in range(self.head_dim // 2):
            similarity[idx] = F.kl_div(scaling_distribution[idx].log(), origin_distribution[idx], reduction='mean')
        return similarity

    def get_dprope_inv_freq(self):
        logger.warning(f"{'+'*10}Compute original distribution.{'+'*10}")
        origin_angles_distribution, _ = self._get_rotation_angles(self.origin_max_position, self._extrapolation)
        target_scaling_factors = []
        similarities = []
        logger.warning(f"{'+'*10}Compute scaling distribution.{'+'*10}")
        for func in self.all_scaling_methods:
            logger.warning(f"{'+'*10}Compute {func.__name__} distribution.{'+'*10}")
            scaling_angles_distribution, scaling_factors = self._get_rotation_angles(self.target_max_position, func)
            similarity = self._compute_distribution_similarity(origin_angles_distribution, scaling_angles_distribution)
            similarities.append(similarity.tolist())
            target_scaling_factors.append(scaling_factors.tolist())
        
        target_scaling_factors = torch.tensor(target_scaling_factors)
        similarities = torch.tensor(similarities)
        min_indices = torch.argmin(similarities, dim=0)

        result_scaling_factors = target_scaling_factors[min_indices, torch.arange(target_scaling_factors.size(1))]

        inv_freq = self._get_inv_freq()
        inv_freq = inv_freq / result_scaling_factors
        return inv_freq, result_scaling_factors # shape=(1, self.head_dim // 2) represent inv_freq and scaling factors for every dim.

        

        
        




if __name__ == "__main__":
    dp = DPRopeClass()
    inv_freq, result_scaling_factors = dp.get_dprope_inv_freq()
    # replace your model's inv_freq or use result_scaling_factors for scaling RoPE.