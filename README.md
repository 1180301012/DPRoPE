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
@inproceedings{wu-etal-2024-extending,
    title = "Extending Context Window of Large Language Models from a Distributional Perspective",
    author = "Wu, Yingsheng  and
      Gu, Yuxuan  and
      Feng, Xiaocheng  and
      Zhong, Weihong  and
      Xu, Dongliang  and
      Yang, Qing  and
      Liu, Hongtao  and
      Qin, Bing",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.414/",
    doi = "10.18653/v1/2024.emnlp-main.414",
    pages = "7288--7301",
    abstract = "Scaling the rotary position embedding (RoPE) has become a common method for extending the context window of RoPE-based large language models (LLMs). However, existing scaling methods often rely on empirical approaches and lack a profound understanding of the internal distribution within RoPE, resulting in suboptimal performance in extending the context window length. In this paper, we propose to optimize the context window extending task from the view of rotary angle distribution. Specifically, we first estimate the distribution of the rotary angles within the model and analyze the extent to which length extension perturbs this distribution. Then, we present a novel extension strategy that minimizes the disturbance between rotary angle distributions to maintain consistency with the pre-training phase, enhancing the model`s capability to generalize to longer sequences. Experimental results compared to the strong baseline methods demonstrate that our approach reduces by up to 72{\%} of the distributional disturbance when extending LLaMA2`s context window to 8k, and reduces by up to 32{\%} when extending to 16k. On the LongBench-E benchmark, our method achieves an average improvement of up to 4.33{\%} over existing state-of-the-art methods. Furthermore, Our method maintains the model`s performance on the Hugging Face Open LLM benchmark after context window extension, with only an average performance fluctuation ranging from -0.12 to +0.22."
}
```
