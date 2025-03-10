# DPRoPE

This repository constitutes the implementation of the paper presented at EMNLP 2024: [Extending Context Window of Large Language Models from a Distributional Perspective](https://aclanthology.org/2024.emnlp-main.414.pdf)

### Quick Start
```python
from DPRoPE import DPRoPEClass
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--origin_max_position", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--scaling_factor", type=int, default=4)
    parser.add_argument("--target_max_position", type=int, default=None)
    parser.add_argument("--intervals", type=int, default=360)

    args = parser.parse_args()

    dprope = DPRoPEClass(
        rope_theta=rope_theta, 
        origin_max_position=origin_max_position,
        head_dim=head_dim,
        scaling_factor=scaling_factor,
        target_max_position=target_max_position,
        intervals=intervals
        )

    inv_freq, scaling_factors = dprope.get_dprope_inv_freq()
```

By default, **DPRoPE** selects between **extrapolation** and **interpolation**. If you wish to incorporate additional methods, please implement a ```_custom_scaling_method``` and integrate it into ```all_scaling_methods```.Please refer to ```DRRoPE.py``` for further details and implementation.

Below is an implementation example:
```python
from DPRoPE import DPRoPEClass


class CustomDPRoPE(DPRoPEClass):
    def __init__(self):
        super().__init__()
        self.all_scaling_methods.append(self._custom_scaling_method)
    
    def _custom_scaling_method(self):
        # Your methods here such as YaRN, NTK etc.
        # DPRoPE will choice by angle distribution similarity.
        pass
```

### Citation
```
@inproceedings{DBLP:conf/acl/FeiNZH0D024,
  author       = {Weizhi Fei and
                  Xueyan Niu and
                  Pingyi Zhou and
                  Lu Hou and
                  Bo Bai and
                  Lei Deng and
                  Wei Han},
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
  title        = {Extending Context Window of Large Language Models via Semantic Compression},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
  pages        = {5169--5181},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.findings-acl.306},
  doi          = {10.18653/V1/2024.FINDINGS-ACL.306},
  timestamp    = {Tue, 08 Oct 2024 07:48:00 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/FeiNZH0D024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```