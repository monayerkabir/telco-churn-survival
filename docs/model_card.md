# Model card: Telco churn models

Short model card in the spirit of [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993).
All numbers come from [`reports/metrics.json`](../reports/metrics.json) and
[notebook 02](../notebooks/02_modeling.ipynb); they are reproduced exactly by `make all`.

| | |
|---|---|
| **Version** | `20261028T111359Z-e4e7657` (the run exported to [`model_bundle/`](../model_bundle/)) |
| **Type** | Binary classifier + survival model + decision rule |
| **Framework** | scikit-learn 1.9 pipelines, LightGBM 4.7, lifelines 0.30 |
| **License** | MIT |
| **Contact** | Open an issue on this repository |

## Models

| Component | Details |
|---|---|
| **Churn classifier** | LightGBM tuned with Optuna (40 TPE trials, 5-fold stratified CV, ROC-AUC objective) inside a scikit-learn `Pipeline` (feature engineering + one-hot encoding refit per fold). Calibration was selected by out-of-fold Brier score from none / Platt / isotonic; **none** won (already calibrated). Output: P(churn). |
| **Time-to-churn model** | Cox proportional hazards **stratified by contract type** (lifelines, L2 penalty 0.01). Covariates exclude tenure and everything derived from it. Output: survival curve S(t \| x), conditioned on the customer's current tenure. |
| **Decision rule** | Offer the retention incentive when P(churn) ≥ **0.18**, the profit-maximising threshold on out-of-fold training predictions under the assumptions below. |

## Intended use

- **Primary use:** rank *current* customers for retention outreach, size revenue at risk
  and explore what-if scenarios. It is decision support for a retention or CRM team, with a
  human deciding on the campaign.
- **Users:** analysts and campaign managers, through the API, the Streamlit app or batch
  scoring.

## Out of scope

- Decisions about individuals' **pricing, credit, eligibility or service level**.
- **Causal** questions such as "what happens if we move this customer to a two-year
  contract?". The models learn associations, and contract type partly reflects a
  customer's existing intent to stay.
- Any real operator's customers without re-training and re-validation. The data is a
  public, semi-synthetic sample.

## Training and evaluation data

- **Source:** IBM Telco Customer Churn sample (7,043 customers, 21 columns, one snapshot),
  pinned by SHA-256.
- **Label:** `Churn = Yes` means the customer left within the last month (26.5%). For
  survival, `tenure` is the time axis and active customers are right-censored.
- **Split:** stratified 80/20 (seed 42): 5,634 training and 1,409 held-out test customers.
  The test set was used once, for final evaluation.
- **Cleaning:** the 11 blank `TotalCharges` values belong to tenure-0 customers and are set
  to 0. A pandera schema validates types, levels, ranges and cross-column consistency.
  Tenure-0 customers are excluded from survival fitting.

## Performance (held-out test set)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | **0.848** | 0.826 – 0.868 |
| PR-AUC (no-skill: 0.265) | **0.666** | 0.616 – 0.713 |
| Brier score | **0.135** | 0.124 – 0.144 |
| Log loss | 0.414 | 0.387 – 0.440 |
| Survival C-index (Harrell) | **0.855** | – |
| Integrated Brier score, months 1–71 (Kaplan-Meier baseline: 0.183) | **0.083** | – |

The cross-validated ROC-AUC on the training set was 0.850, against 0.846 for a
logistic-regression baseline.

### Across customer groups (test set)

| Group | Customers | Churn rate | Mean predicted | ROC-AUC | Brier | Targeted at 0.18 |
|---|---|---|---|---|---|---|
| Female | 687 | 28.1% | 26.6% | 0.846 | 0.139 | 50% |
| Male | 722 | 25.1% | 26.5% | 0.850 | 0.130 | 50% |
| Not senior | 1,187 | 23.3% | 24.0% | 0.849 | 0.125 | 45% |
| Senior citizen | 222 | 44.1% | 40.5% | 0.787 | 0.187 | 77% |
| Month-to-month | 773 | 42.6% | 43.0% | 0.758 | 0.196 | 84% |
| One-year contract | 300 | 12.0% | 10.1% | 0.749 | 0.098 | 18% |
| Two-year contract | 336 | 2.7% | 3.5% | 0.731 | 0.026 | 0% |

- **Gender:** no performance or targeting gap. Gender's SHAP importance is near zero, and
  gender is kept as an input only because the source data includes it; dropping it is a
  reasonable choice for production.
- **Senior citizens:** ranked less well (on a small group) and slightly under-predicted.
  This is the group to monitor first.
- **Contract:** within-segment ranking (0.73–0.76) is weaker than the overall figure, which
  partly reflects separating month-to-month customers from contract holders. Calibration
  holds within each segment.

## Business assumptions

The threshold depends on assumptions, not data: a **\$50** offer cost, a **30%** success
rate (share of would-be churners retained) and a **12-month** value horizon (see
`churn.config.BusinessConfig`). On the test set this targets 50% of customers, reaches 87% of
churners and earns about **\$51k** of expected profit, against \$28k for offering everyone.
Notebook 02 shows how the threshold moves with the assumptions. The success rate should be
measured with a randomised hold-out before scaling up.

## Limitations and risks

- **Semi-synthetic, single snapshot:** covariates are measured at the snapshot, not at
  sign-up, and the 26.5% "last month" churn rate is unrealistically high for a monthly
  label. Taken literally, earlier churners are missing (left truncation), which biases
  absolute survival levels; rankings are less affected.
- **Two different questions:** the classifier predicts the snapshot label, while the
  survival model gives P(churn within 12 months | active today). Their numbers differ for
  the same customer (they rank customers similarly: Spearman 0.77).
- **Proportional hazards:** five small violations remain after stratifying on contract.
- **Horizon:** nothing is observed beyond 72 months. "Expected remaining tenure" is
  restricted to that window.
- **Feedback loop:** once offers are sent, outcomes of targeted customers change. Keep a
  randomised control group to evaluate and retrain on unbiased data.

## Monitoring (recommended)

- Calibration (predicted vs observed churn) overall and per segment, monthly.
- Input drift (population stability index) on contract mix, tenure, charges and payment
  method.
- Realised offer success rate from the control group, fed back into `BusinessConfig`.

## Reproducibility

Seeds are fixed everywhere and LightGBM runs deterministically. `make all` rebuilds the
environment, data, models, metrics and notebooks from a fresh clone, with identical metrics.
Every run writes a versioned directory with `metrics.json` (config snapshot, git SHA, data
checksum and library versions). The deployment bundle carries a SHA-256 manifest that is
verified at load time.
