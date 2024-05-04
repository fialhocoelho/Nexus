# PoC: Foundation Models for Time Series Forecasting 

This Proof of Concept aims to validate if a foundation model can achieve similar results when compared with state-of-the-art time series forecasting techniques.

## Table of Contents

- [PoC: Foundation Models for Time Series Forecasting](#poc-foundation-models-for-time-series-forecasting)
  - [Table of Contents](#table-of-contents)
  - [Data](#data)
    - [Chosen Data for experiments](#chosen-data-for-experiments)
    - [The dataset](#the-dataset)
    - [Splitting the Dataset](#splitting-the-dataset)
    - [Granularity](#granularity)
  - [Chosen Model: Nixtla TimeGPT-1](#chosen-model-nixtla-timegpt-1)
  - [Experiments](#experiments)
    - [Inference Window](#inference-window)
    - [Baseline](#baseline)
    - [Metrics](#metrics)
  - [Results](#results)
    - [Predicted Range](#predicted-range)
  - [Conclusion](#conclusion)

## Data

### Chosen Data for experiments

The Santos-São Vicente-Bertioga Estuarine System (SSVBES), situated on the southeastern Brazilian coast, is influenced by three main forcing processes: astronomical tide, river discharge, and meteorological tide. The latter is driven by the synoptic winds that blow over the nearby continental shelf, following Ekman dynamics, and enter Santos Bay as gravity waves. When winds blow from the North-Northeast, meteorological tide results in a reduction of sea surface height (SSH); when winds blow from the South-Southwest, it leads to an increase in SSH [1]. 

### The dataset

Our dataset spans **two years of data** from January 1, 2019, to September 1, 2021 (a total of 974 days) from the **Praticagem** measuring station from SSVBES. This location is equipped with sensors that collect measurements of oceanic variables: **SSH (Sea Surface Height)** and **water current speed**. For endogenous and exogenous data, we utilize numerical simulated data, specifically, **astronomical tide**. The input features contain *less than 4% missing data*. For this experiment, we *interpolated the missing data using a simple linear method* since the TimeGPT-1 model does not accept gaps in inference data. For a mature pipeline, a robust set of methods for filling data can and should be tested to verify the impact on prediction results.

### Splitting the Dataset

Furthermore, the dataset is split into two sequential sets: the train set and the test set. The train set comprises the first 20 months of the time series data, while the test set comprises the last 4 months.

### Granularity

The monitoring/simulated data was aggregated using a 60-minute step between windows, resulting in 3 measured points in the flow data for each hour. We chose this granularity due to the limitation of foundation models in handling long-term predictions (more than 24 forecast points).

## Chosen Model: Nixtla TimeGPT-1

Foundation models rely on their ability to generalize across different areas, especially with new data not present during training. In our quest for a foundation model capable of achieving results close to state-of-the-art forecasting methods using a zero-shot approach, we also considered the trade-off between accuracy and forecast generation time, as well as the challenge posed by limited availability of large data windows for training. It was during this search that we encountered a highly referenced model known as [TimeGPT-1](https://arxiv.org/abs/2310.03589). It leverages large public time series datasets to train TimeGPT-1, a Transformer-based model with self-attention mechanisms[2]. It captures diverse temporal patterns across various domains by leveraging a diverse dataset. TimeGPT-1 employs an encoder-decoder structure with residual connections and layer normalization, generating forecasts based on historical values and local positional encoding. The model's attention-based mechanisms aim to accurately predict future distributions by capturing the diversity of past events[2].

## Experiments

### Inference Window

In the paper [3], a large inference windows (larger than 5k measured points) cannot be set due to computer resources limitations. The paper above uses 20 months of data to train the models and an inference window with 168 measured points (7 days) to forecast the next 24 hours using sliding windowing as we can see in [figure 1](#fig1). 
For the experiment, we used Nixtla TimeGPT-1 API to send data using incremental windowing to forecast the next 24 measured points ([figure 1](#fig1)). The incremental window was chosen to check the performance of the related model inferring a large data to be inferred. 

Given it involves zero-shot learning strategies, we propose investigating an optimal window that correlates window size with computational cost/tokens used for training/inference. **For this proof of concept inference using TimeGPT-1, we employed an incremental windowing strategy with padding = 1.** Here's how it works:

* **1st prediction step:**
    * **Inferred data:** N measurement points from the training series minus the last 23 points of the same series.
    * **Predicted data:** 24 points to be validated with the last 23 points of training plus the 1st measured point of the test series.  

* **2nd prediction step:** 
    * **Inferred data:** N measurement points from the training series minus the last 22 points of the same series.
    * **Predicted data:** 24 points to be validated with the last 22 points of training plus the first 2 measured points of the test series.  

* **24th prediction step:** 
    * **Inferred data:** Total of N measurement points from the training series.
    * **Predicted data:** 24 points to be validated with the first 24 points measured of the test series.

* **Last prediction step (2880th step):** 
    * **Inferred data:** Total of N measurement points from the training series plus N - 24 points of the test series.
    * **Predicted data:** 24 points to be validated with the last 24 points of the test series.

<br/><br/>
<p id='fig1'></p>

![nexus-diagram](images/nexus_windowing_h.png)
<center><h4>Figure 1: Types of Windowing</h4></center>

### Baseline

To measure the experiment results, we will use the paper [3] as a baseline.

### Metrics

For all experiments, presented results using Index of Agreement (IoA)[4] comparison metrics.

The Index of Agreement ($IoA$) is calculated using the formula:

$$
IoA = 1 - \frac{\sum_{i=1}^{n} (O_i - P_i)^2}{\sum_{i=1}^{n} (|P_i - \overline{O}| + |O_i - \overline{O}|)^2}
$$

where:
- $O_i$ represents the observed values,
- $P_i$ represents the predicted values,
- $\overline{O}$ is the mean of the observed values, and
- $n$ is the number of observations.

This formula assesses the agreement between observed and predicted values. A value of 1 indicates perfect agreement, while lower values indicate less agreement.

## Results

### Predicted Range

Due to computational resource limitations, only the first 20 days of the test dataset were generated. As a result, the metrics for prediction evaluation may exhibit disturbances compared to the benchmark data, which were compared with 4 months of data.


![ioa results animation](images/timegpt_poc.gif)

## Conclusion

TBD.
