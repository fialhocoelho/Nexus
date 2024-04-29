# Nexus
Where Converging Foundation Models and Knowledge Distillation Promote Robust Time Series Forecasting

### About Nexus model:
TBD

### Experiment setup:

#### Models: 
- TimeGPT: A foundation model utilized for Proof of Concept (PoC) purposes, known for its capabilities in short-term prediction.
- Chronos: Another foundation model with versatile applications in time series forecasting.
- lag-llama: A foundation model known for its robust performance in various time series prediction tasks.
- GNN OMAE: A graph neural network model designed for optimizing marine operations and efficiency.
- MLP: A standard machine learning baseline model, commonly used as a reference for comparison in time series prediction tasks.
- NHITS: An advanced model recognized as the state-of-the-art in time series prediction within certain domains.
- SARIMAX: A classic statistical model often used as a baseline for seasonal time series prediction with exogenous variables.
- Knowledge Distillation model: TBD - Further details pending.

#### Issues to be addressed:
- Utilizing SOFS as an endo/exogenous variable: 
  - Problem: Granularity of 3h poses challenges for models not supporting different granularities.
  - Solution: Consider interpolation for 1h granularity or adapt models for 3h granularity.
  - Note: Foundation models may not excel in long-term predictions (> 24 points).
- Incorporating carbon production estimation measures:
  - Problem: Balancing learning curve vs. project delivery time.
- Leveraging other variables from satellites/images:
  - Problem: Foundation models don't handle encoded data, requiring model adaptation and retraining.

#### Variables:
  - Endogenous
    - Current: Praticagem and other regions
    - Sea Surface Height (SSH): Praticagem and other regions
    - Astronomical tide: Praticagem and other regions
  - Exogenous
    - Astronomical tide: Praticagem and other regions

#### Granularity:
  - Ideal granularity: 1-hour periodicity for test scenarios.

#### Inference Window:
  - An optimal window size correlates with computational cost/tokens used for training/inference.
  - For TimeGPT inference, an incremental windowing strategy was employed, where N predictions were made, each step incorporating additional points from the test series.

Example:
1st prediction step: 
- Inferred data: N measurement points from the training series minus the last 23 points.
- Predicted data: 24 points validated with the last 23 training points plus the 1st test point.
2nd prediction step: 
- Inferred data: N measurement points from training minus the last 22 points.
- Predicted data: 24 points validated with the last 22 training points plus the 2nd and 1st test points.
24th prediction step: 
- Inferred data: Total of N measurement points from training.
- Predicted data: 24 points validated with the 24th first test points.
N-th prediction step: 
- Inferred data: Total of N measurement points from training plus the N - 24 points of the test series.
- Predicted data: 24 points validated with the last 24 test points.

#### Table of experiments
| Models | Time Series Type | Execution Environment | Endogenous Variables | Exogenous Variables | Number of Model Runs | Number of Fine-Tuning Rounds | Epochs |
|--------|------------------|------------------------|----------------------|---------------------|----------------------|------------------------------|--------|
| TimeGPT | Multivariate with varying granularities | Nixtla Cluster (API as a service) | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | Astronomical tide: Praticagem and other regions | 1 | 500 (Fine-tuning on API call) | N/A |
| Chronos | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| lag-llama | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| GNN OMAE | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| MLP | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| NHITS | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| SARIMAX | Univariate with seasonal variations | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |
| Knowledge Distillation model | Multivariate with varying granularities | On-premises with 2x Xeon Platinum, 16GB DDR4 * 8, and a Titan V GPU | Praticagem and other regions: Praticagem and other regions; Sea Surface Height (SSH): Praticagem and other regions; Astronomical tide: Praticagem and other regions | None | TBD | 100 (Tuned with Optuna) | 200 |

