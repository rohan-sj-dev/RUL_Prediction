**Paper 1 was read from source** (copy in `data/raw/`). The notes for papers 2–7 are written
from standard knowledge of these works

| # | Paper | Role in our project | Reader |
|---|---|---|---|
| 1 | Saxena, Goebel, Simon, Eklund. *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM 2008. | Dataset and scoring function
| 2 | Heimes. *Recurrent Neural Networks for Remaining Useful Life Estimation.* PHM 2008. | Capped RUL target
| 3 | Babu, Zhao, Li. *Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life.* DASFAA 2016. | First CNN baseline
| 4 | Zheng, Ristovski, Farahat, Gupta. *Long Short-Term Memory Network for Remaining Useful Life Estimation.* IEEE ICPHM 2017. | LSTM baseline and multi-condition normalization
| 5 | Li, Ding, Sun. *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks.* Reliability Eng. & System Safety, 2018. | Strongest benchmark and windowing setup
| 6 | Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv 2021. | Prediction interval
| 7 | Lundberg, Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. | Explainability

---

## 1. Saxena et al. (2008): the dataset paper

**What it does.** Describes how the C-MAPSS data was generated with NASA's C-MAPSS simulator,
a MATLAB/Simulink model of a 90,000 lb-thrust commercial turbofan, for the PHM'08 data challenge.

**How the data was generated**
- Each engine starts with random **initial wear**: efficiency and flow start between 0.99 and 1.0
  of healthy (Eq. 9). This is why every engine looks slightly different at cycle 1.
- Degradation follows an **exponential** health curve, `h(t) = 1 − d − exp(a·t^b)`, with
  `a ∈ [0.001, 0.003]` and `b ∈ [1.4, 1.6]` (Eq. 5, 10). Damage accelerates near the end, which
  supports the capped/piecewise RUL target in paper 2.
- The health index is the **minimum of several operating margins** (fan, HPC and LPC stall
  margins, and EGT margin). The engine "fails" when it reaches 0. These margins are **not** in
  the data; models must infer health from sensors.
- **Noise** comes from several sources: a mixture of two noise distributions on the trajectory,
  filtered through the engine dynamics, plus sensor measurement noise. Smoothing or windowing
  is therefore necessary.
- Between-flight maintenance is modelled as process noise, so the degradation is **not locally
  monotonic** (Fig. 5).
- The **6 operating conditions** are combinations of altitude (0–42k ft), Mach (0–0.84) and
  throttle angle TRA (20–100), which is why FD002/FD004 need per-condition normalization.
- Test-set RULs range from 10 to 150 cycles.

**Sensor names (Table 2): our `s1`–`s21` follow this order**

| col | name | meaning | col | name | meaning |
|---|---|---|---|---|---|
| s1 | T2 | Fan inlet temp | s12 | phi | Fuel flow / Ps30 |
| s2 | T24 | LPC outlet temp | s13 | NRf | Corrected fan speed |
| s3 | T30 | HPC outlet temp | s14 | NRc | Corrected core speed |
| s4 | T50 | LPT outlet temp | s15 | BPR | Bypass ratio |
| s5 | P2 | Fan inlet pressure | s16 | farB | Burner fuel-air ratio |
| s6 | P15 | Bypass-duct pressure | s17 | htBleed | Bleed enthalpy |
| s7 | P30 | HPC outlet pressure | s18 | Nf_dmd | Demanded fan speed |
| s8 | Nf | Physical fan speed | s19 | PCNfR_dmd | Demanded corrected fan speed |
| s9 | Nc | Physical core speed | s20 | W31 | HPT coolant bleed |
| s10 | epr | Engine pressure ratio | s21 | W32 | LPT coolant bleed |
| s11 | Ps30 | HPC outlet static pressure | | | |

**Why some sensors are constant.** The sensors that are constant in FD001 (s1, s5, s6, s10, s16,
s18, s19) are mostly **inlet conditions and demanded set-points**. At a single sea-level
condition these don't change with wear. We will show this in the EDA and report the physical
reason.

**Scoring function (Section VII, Eq. 11).** With `d = predicted RUL − true RUL`:

- `s = exp(−d/13) − 1` when `d < 0` (early prediction, milder penalty)
- `s = exp(d/10) − 1` when `d ≥ 0` (late prediction, harsher penalty)

The total score is the sum over all test engines; lower is better.

The paper's equation has an apparent typo: it states `a1 = 10` for the `d < 0` branch and
`a2 = 13` for the late branch, which would penalize *early* predictions more. That contradicts
its own text ("late predictions were more heavily penalized") and Figure 9. Use the convention
above, which every later paper follows, and mention the discrepancy in the report.

**Further ideas from the paper**
- Near-failure predictions should count more than far-off ones.
- Look at **consistency across engines**, not only the aggregate score. We can report the
  per-engine error spread.

**Dataset readme discrepancy:** `readme.txt` lists FD004 as 248 train / 249 test, but the
actual files contain **249 train / 248 test** (verified by `src/check_data.py`).

**What we take from it:** sensor names for plots and SHAP labels, the scoring function, the
reason for windowing and smoothing, and the reason for condition-wise normalization.

---

## 2. Heimes (2008): capped RUL target

**What it does.** A recurrent neural network trained with an extended Kalman filter. It was among
the top entries of the PHM'08 challenge **[verify]**.

**Key idea.** Early in an engine's life, wear has no visible effect on the sensors, so a target
that falls linearly from the very first cycle asks the model to predict something it cannot see.
Heimes instead used a **piecewise-linear target**: RUL is held constant at a maximum
(≈ 130 cycles **[verify]**) and only decreases linearly near the end.

**What we take from it:** RUL capping. We use a cap of 125, the value adopted by paper 5, and
will include an ablation with and without the cap to show its effect.

---

## 3. Babu, Zhao, Li (2016): first CNN for RUL

**What it does.** The first deep CNN applied to C-MAPSS RUL regression. Sensor windows are
treated as a 2D input (time × sensors) and passed through convolution and pooling layers, then
a regression head. It is compared with MLP, SVR and RVR **[verify]**.

**Reported RMSE [verify]:** FD001 ≈ 18.4 · FD002 ≈ 30.3 · FD003 ≈ 19.8 · FD004 ≈ 29.2

**What we take from it:** shows that deep models learn features automatically and beat
shallow models given only raw sensors. It is a historical baseline for our results table.

---

## 4. Zheng et al. (2017): LSTM for RUL

**What it does.** Stacked LSTM layers followed by fully connected layers, using a
piecewise-linear target. For the multi-condition subsets, operating conditions are identified
and sensors are **normalized per condition** **[verify]**.

**Reported RMSE [verify]:** FD001 ≈ 16.1 · FD002 ≈ 24.5 · FD003 ≈ 16.2 · FD004 ≈ 28.2

**What we take from it:** the LSTM baseline, and support for our per-condition (KMeans)
normalization on FD002 and FD004.

---

## 5. Li, Ding, Sun (2018): deep 1D-CNN benchmark

**What it does.** Several 1D convolution layers that convolve **along time only** (kernel about
10 × 1), with dropout and a fully connected head. Inputs are min–max normalized sensor windows,
and the RUL cap is 125. The window length is 30 for FD001 **[verify per subset]**.

**Reported RMSE [verify]:** FD001 ≈ 12.6 · FD002 ≈ 22.4 · FD003 ≈ 12.6 · FD004 ≈ 23.3

**What we take from it:** our windowing setup (length 30, cap 125) and our **target to match:
FD001 RMSE ≈ 12–13**. It shows that a simple well-tuned CNN can beat LSTMs on this data.

---

## 6. Angelopoulos & Bates (2021): conformal prediction

**What it does.** A tutorial on **conformal prediction**, which wraps any trained model to
produce prediction intervals with a guaranteed coverage level, with no distributional
assumptions.

**Split conformal regression, in short:**
1. Train the model on the training set.
2. On a separate **calibration set**, compute residuals `|y − ŷ|`.
3. Let `q` be the `⌈(n+1)(1−α)⌉ / n` quantile of these residuals.
4. For a new input, the interval is `[ŷ − q, ŷ + q]`. It covers the true value with probability
   ≥ 1 − α, averaged over test points.
- **Conformalized quantile regression (CQR)** gives intervals that adapt per input, for example
  narrower near failure, rather than a fixed width.

**Caveat for our data.** The guarantee assumes calibration and test points are
**exchangeable**. Consecutive cycles of one engine are strongly correlated, so we must
calibrate on **held-out engines**, ideally one point per engine, and not on random rows.

**What we take from it:** the uncertainty layer (MAPIE 1.x `SplitConformalRegressor` /
`ConformalizedQuantileRegressor`) and the coverage and interval-width metrics.

---

## 7. Lundberg & Lee (2017): SHAP

**What it does.** Shows that several explanation methods (LIME, DeepLIFT, layer-wise relevance
and others) belong to one family of **additive feature attributions**. **Shapley values** are
the unique attribution with local accuracy, missingness and consistency. The paper proposes
KernelSHAP (model-agnostic) and model-specific approximations, including Deep SHAP.

**How we use it**
- **Tree models (XGBoost / LightGBM):** `shap.TreeExplainer`, which is fast and exact. It comes
  from the follow-up work by Lundberg et al. (2020).
- **Deep models:** `shap.DeepExplainer` or `GradientExplainer` on the input windows,
  aggregated over time to give per-sensor importance.
- Global view: which sensors matter across the fleet. Local view: why this engine got this
  RUL prediction (dashboard).

**What we take from it:** the explainability layer. A useful check is whether SHAP's top
sensors match the physically meaningful ones from paper 1 (e.g. T50, Ps30, phi).

---

## What the project needs from the literature

| Need | From | Project phase |
|---|---|---|
| Sensor names and the physical reason some are constant | 1 | Phase 2 (EDA) |
| NASA scoring function, using the corrected convention | 1 | Phase 4 onward |
| RUL cap (125) plus an ablation without the cap | 2, 5 | Phase 3 |
| Per-operating-condition normalization | 1, 4 | Phase 3 |
| Window length 30, 1D-CNN architecture | 5 | Phase 5 |
| Baseline numbers for the results table (CNN 2016, LSTM 2017, DCNN 2018) | 3, 4, 5 | Phase 5 / report |
| Calibrate conformal intervals on held-out engines | 6 | Phase 6 |
| TreeSHAP for tree models, Deep/Gradient SHAP for neural nets | 7 | Phase 6 |
| Report per-engine error spread, not just averages | 1 | Phase 4 onward |

## Gap our project addresses
The works above report mostly **point-estimate accuracy** (RMSE and score). Our contribution
is a single system that combines:

1. calibrated **uncertainty** (conformal intervals),
2. sensor-level **explanations** (SHAP), and
3. a **cost-based maintenance decision** built on those intervals and compared against
   reactive and fixed-interval policies.

## Optional further reading
- Saxena et al. *Metrics for Evaluating Performance of Prognostics Techniques.* PHM 2008. More
  prognostics metrics; useful for the evaluation section.
- Lundberg et al. *From Local Explanations to Global Understanding with Explainable AI for
  Trees.* Nature Machine Intelligence, 2020. The TreeSHAP paper.
