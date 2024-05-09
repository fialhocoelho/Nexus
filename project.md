# Research project
We propose a model that leverages the state of the art in foundation models trained on massive datasets, along with another simpler model trained on a smaller dataset specific to the domain of the time series to be forecasted. This combination aims to bring out the best of both worlds: increasing prediction accuracy compared to individual model results, as well as reducing computational costs by avoiding expensive training and fine-tuning routines.

Here, we describe an experimental design and organization for the development of [Nexus](https://github.com/fialhocoelho/Nexus/) model.

## Research problem

The research problem revolves around the **high computational costs associated with extensive training routines and fine-tuning of foundation models**. These models, trained on massive datasets, offer state-of-the-art performance but **require significant computational resources and time for training and optimization**. The challenge aims to finding efficient methods to **leverage the benefits of these foundation models** while **mitigating the computational complexity**, especially in domains such as time series forecasting where computational efficiency is crucial for real-world applications.

## Related work

* [Zhou et al.,Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models - Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Main Conference Track](https://proceedings.neurips.cc/paper_files/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html)
* [Sun et al., Dime-fm: Distilling multimodal and efficient foundation models - International Conference on Computer Vision 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_DIME-FM__DIstilling_Multimodal_and_Efficient_Foundation_Models_ICCV_2023_paper.pdf)
* [Hinton et al., Distilling the Knowledge in a Neural Network -In Advances in Neural Information Processing Systems (NIPS), 2014](https://arxiv.org/pdf/1503.02531)
* [Liu et al., Large Language Model Guided Knowledge Distillation for Time Series Anomaly Detection - PrePrint](https://arxiv.org/abs/2401.15123)

## Ongoing Tasks
[Experiments](https://github.com/fialhocoelho/nexus/experiments.md)

## Proposal
[Experiments](https://github.com/fialhocoelho/nexus/experiments.md)

## Initial Results
[PoC](https://github.com/fialhocoelho/nexus/poc.md)