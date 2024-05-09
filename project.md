# Research project

We propose a model that leverages the state of the art in foundation models trained on massive datasets, along with another simpler model trained on a smaller dataset specific to the domain of the time series to be forecasted. This combination aims to bring out the best of both worlds: increasing prediction accuracy compared to individual model results, as well as reducing computational costs by avoiding expensive training and fine-tuning routines.

## Table of Contents

- [Research project](#research-project)
  - [Research problem](#research-problem)
  - [Related work](#related-work)
  - [Ongoing Tasks](#ongoing-tasks)
  - [Proposal](#proposal)
  - [Initial Results](#initial-results)
  - [Initial paper (Bracis)](#initial-paper-bracis)
  - [Future works](#future-works)

## Research problem

The research problem revolves around the **high computational costs associated with extensive training routines and fine-tuning of foundation models**. These models, trained on massive datasets, offer state-of-the-art performance but **require significant computational resources and time for training and optimization**. The challenge aims to finding efficient methods to **leverage the benefits of these foundation models** while **mitigating the computational complexity**, especially in domains such as time series forecasting where computational efficiency is crucial for real-world applications.

## Related work

* Zhou, K., He, Y., Cai, J., & Han, J. (2023). [Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models](https://proceedings.neurips.cc/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html). In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Main Conference Track*.

* Sun, H., Liu, Y., Wang, Z., Jiang, C., & Han, J. (2023). [DIME-FM: Distilling Multimodal and Efficient Foundation Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.pdf). In *International Conference on Computer Vision 2023*.

* Hinton, G., Vinyals, O., & Dean, J. (2014). [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531). In *Advances in Neural Information Processing Systems (NIPS)*.

* Liu, J., Yang, B., Wang, C., & Yang, Y. (Preprint). [Large Language Model Guided Knowledge Distillation for Time Series Anomaly Detection](https://arxiv.org/abs/2401.15123).

## Ongoing Tasks
[Experiments](https://github.com/fialhocoelho/nexus/experiments.md)

## Proposal
[Proposal](https://github.com/fialhocoelho/nexus/proposal.md)

## Initial Results
[Proof of Concept](https://github.com/fialhocoelho/nexus/poc.md)

## Initial paper (Bracis)

* Distillations:
    * TimeGPT without tuning + residualMLP
    * Chronos without tuning + residualMLP
    * (TimeGPT + Chronos) without tuning + residualMLP

## Future works
* How to fine-tune the Chronos model?
* Canonical correlation analysis (Similarity from model's internal representation)
* Feature Distillation: Create a distillation related to internal know-how ([Initializing Models with Larger Ones](https://arxiv.org/pdf/2311.18823)).
