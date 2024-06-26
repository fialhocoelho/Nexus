# Experiments setup

This step outlines the experimental design and organization for the development of [Nexus](https://github.com/fialhocoelho/Nexus/) model: a forecasting model using foundation models,
proposing a knowledge distillation approach to achieve robust forecasts. The experiments aim to explore various aspects of model training, tuning, and inference to optimize the forecasting performance.
This section explains the selection of teacher models, student models, calibration strategies, and which baseline models will be used to compare a generated forecast data.
It also addresses key issues to be resolved before the experimentation process starts, including data selection, handling missing values, and the incorporation of exogenous variables related to Santos oceanic variables.

## Table of Contents

- [Models](#models)
  - [Teacher models](#teacher-models)
  - [Student model](#student-model)
  - [Calibration model](#calibration-model)
- [Baseline models](#baseline-models)
- [Issues to be Addressed](#issues-to-be-addressed)
- [Variables](#variables)
  - [Endogenous](#endogenous)
  - [Exogenous](#exogenous)
- [Granularity](#granularity)
- [Inference Window](#inference-window)
- [Experiment Organization](#experiment-organization)
- [Proposed Initial Model Diagram](#proposed-initial-model-diagram)
- [Tables](#tables)
  - [1st Stage of Experiments (Teacher models)](#1st-stage-of-experiments-teacher-models)
  - [2nd Stage of Experiments (Student Model)](#2nd-stage-of-experiments-student-model)
  - [3rd Stage of Experiments (Baseline)](#3rd-stage-of-experiments-baseline)

### Models
#### Teacher models 
* [TimeGPT](https://arxiv.org/abs/2310.03589) (Foundation Model used for PoC)
* [Chronos](https://arxiv.org/html/2403.07815v1) (Foundation Model)
* [lag-llama](https://arxiv.org/abs/2310.08278) (Foundation Model)

#### Student model
* Residual MLP

#### Calibration model
* Optuna

### Storytelling Baseline models
* [NHITS](https://arxiv.org/abs/2201.12886) (State-Of-Art model for timeseries prediction using ML)
* GNN OMAE (Graph Model)
* [TFT](https://arxiv.org/abs/1912.09363) (SoA for Multi-horizon Time Series Forecasting)
* SARIMAX (ARIMA-like model for Seasonal univariate + exog)

### Issues to be Addressed:
* Should we use data only from Praticagem or also from other regions? 
* Some models do not accept missing data, and some series have missing data. Should we use interpolation or zero filling techniques?
* Should we use SOFS or other data from numerical simulations as endo/exogenous variables for input to the models?
    * **Problem:** Granularity of 3h. To deal with this in models that do not support series with different granularities, we would have to:
        * Interpolate to have measurements with a granularity of 1h or
        * Decide to run the models with a granularity of 3h to be able to use the SOFS as a baseline.
* Actually, the most used foundation models are not the best choice for a good performance for long-term prediction, as they are specialized in short-term prediction, given the nature of the diversity of the data used to train the models. In the TimeGPT paper, they mention empirically considering that long-term predictions are those with more than 24 points.
    * **Problem:** If we deal with granularity less than 1h, to predict a day in the future, we would need predictions greater than 24 points. Of the mentioned foundation model implementations, only TimeGPT presents an approach that uses pre-trained weights for long-term prediction, but they mention that this model is still incipient and was trained with a lower variety of data compared to the model for short-term prediction (<=24 prediction points).
* Should we use estimates of carbon production for training/inference of the models?
    * **Problem:** Learning curve x time for project delivery.
* Should we use other variables from satellites/images?
    * **Problem:** The foundation models chosen, by definition, do not work with encoded data. It would be necessary to alter the model to handle this data. In this case, the challenge would be to retrain them with this data domain. These models use weights already calculated from other GenAI-based SoA models (e.g., T5, Llama, etc.).

### Variables
#### Endogenous
* Current: Praticagem and other regions
* Sea Surface Height (SSH): Praticagem and other regions
* Astronomical tide: Praticagem and other regions
#### Exogenous
* Astronomical tide: Praticagem and other regions

### Granularity
I believe the ideal granularity for our testing scenario would be a periodicity of 1 hour, due to the limitation of foundation models in handling long-term predictions.

### Inference Window
As it involves zero-shot learning strategies, I believe we can investigate an optimal window that correlates the size of the window with the computational cost/tokens used for training/inference. For proof of concept inference using TimeGPT, an incremental windowing strategy was used with padding = 1, where it consists of N predictions, where N is the number of measurement points in the test series. For elucidation:
* 1st prediction step:
    * **Inferred data:** N measurement points of the training series minus the last 23 points of the same series.
    * **Predicted data:** 24 points to be validated with the last 23 points of training plus the 1st measured point of the test series.  
* 2nd prediction step: 
    * **Inferred data:** N measurement points of the training series minus the last 22 points of the same series.
    * **Predicted data:** 24 points to be validated with the last 22 points of training plus the first 2 measured points of the test series.  
* 24th prediction step: 
    * **Inferred data:** Total of N measurement points of the training series.
    * **Predicted data:** 24 points to be validated with the first 24 points measured of the test series.
* N-th prediction step: 
    * **Inferred data:** Total of N measurement points of the training series plus N - 24 points of the test series.
    * **Predicted data:** 24 points to be validated with the last 24 points of the test series.

### Experiment Organization
The experiments will be divided into 5 stages with objectives:
1. Teacher models (pre-trained models)
1. Student model (To be trained model)
1. Calibration model
1. Ablation Study
1. Validation/Optimization (Running complete pipeline)

### Proposed Initial Model Diagram
The diagram below exemplifies an initial sketch of the proposed model with the aim of understanding the purpose of each element. 

![nexus-diagram](images/nexus_diagram.png)

### Tables
#### 1st Stage of Experiments (Teacher models)

| Models    | Mode               | Exec Env      | Regions    | Endog Vars    | Exog Vars | #Fine-Tuning steps | #Epochs |
|-----------|--------------------|---------------|------------|---------------|-----------|--------------------|---------|
| TimeGPT   | Multivariate       | Nixtla API    | Praticagem | curr, ssh, at | at        | N/A                | N/A     |
| Chronos   | Univariate         | Xeon + TitanV | Praticagem | curr          | at        | N/A                | N/A     |
| N-HITS    | Multivariate       | Xeon + TitanV | Praticagem | curr, ssh, at | at        | N/A                | 200     |


#### 2nd Stage of Experiments (Student Model) TBD

| Models       | Mode         | Exec Env      | Endog Vars    | Distillation loss                 | Exog Vars | #FT steps | #Epochs |
|--------------|--------------|---------------|---------------|-----------------------------------|-----------|-----------|---------|
| Residual MLP | Multivariate | Xeon + TitanV | curr, ssh, at | TimeGPT(ŷ), Chronos(ŷ), N-HITS(ŷ) | at        | N/A       | 200     |


#### 3rd Stage of tunning Student model Calibration (Baseline)

| Models        | Plataform  | Exec Env      | Main Hyperparameters    | Others HP    | #Fine-Tuning steps | Parallel? |
|---------------|------------|---------------|-------------------------|--------------|--------------------|-----------|
| Residual MLP  | Optuna     | Xeon + TitanV | Beta (Distil. Balancer) | TBD          | TBD                | Yes       |


#### 4th Stage of Experiments (Baseline)

| Models    | Mode               | Exec Env      | Regions    | Endog Vars    | Exog Vars | #Fine-Tuning steps | #Epochs |
|-----------|--------------------|---------------|------------|---------------|-----------|--------------------|---------|
| GNN OMAE  | Multivariate       | Xeon + TitanV | Praticagem | curr, ssh, at | at        | 100 (optuna)       | 200     |
| SARIMAX   | Univariate         | Xeon + TitanV | Praticagem | curr, ssh, at | at        | 100 (optuna)       | N/A     |
