# v2 benchmark analysis

## Row AUPRC (seed mean ± std) and row prior

| network | prior | rf_baseline | rf_noise | rf_gmm | rf_smote | rf_raw_baseline | lr_baseline | gcn_baseline | resid |
|---|---|---|---|---|---|---|---|---|---|
| net3 | 0.091 | 0.094±0.010 | 0.108±0.011 | 0.100±0.012 | 0.090±0.009 | 0.105±0.010 | 0.095±0.006 | 0.113±0.014 | 0.101±0.010 |
| d_town | 0.099 | 0.255±0.044 | 0.245±0.043 | 0.210±0.058 | 0.250±0.039 | 0.206±0.054 | 0.307±0.006 | 0.307±0.015 | 0.101±0.004 |
| l_town | 0.085 | 0.434±0.060 | 0.397±0.018 | 0.408±0.033 | 0.446±0.027 | 0.078±0.016 | 0.418±0.009 | 0.451±0.017 | 0.136±0.032 |
| ky2 | 0.084 | 0.191±0.135 | 0.323±0.075 | 0.160±0.104 | 0.169±0.114 | 0.093±0.010 | 0.144±0.042 | 0.086±0.016 | 0.077±0.006 |
| ky3 | 0.099 | 0.281±0.050 | 0.255±0.027 | 0.384±0.051 | 0.278±0.064 | 0.129±0.022 | 0.196±0.045 | 0.188±0.094 | 0.071±0.003 |
| ky5 | 0.097 | 0.242±0.046 | 0.296±0.069 | 0.232±0.055 | 0.233±0.078 | 0.114±0.035 | 0.245±0.033 | 0.231±0.059 | 0.084±0.009 |
| ky6 | 0.092 | 0.148±0.019 | 0.152±0.011 | 0.155±0.006 | 0.147±0.017 | 0.109±0.022 | 0.127±0.015 | 0.124±0.012 | 0.083±0.009 |
| ky7 | 0.084 | 0.237±0.015 | 0.273±0.033 | 0.219±0.048 | 0.245±0.029 | 0.218±0.047 | 0.276±0.067 | 0.300±0.037 | 0.081±0.021 |
| **macro** | 0.091 | **0.235** | **0.256** | **0.234** | **0.233** | **0.131** | **0.226** | **0.225** | **0.092** |

## Scenario AUPRC, max-pool (seed mean) and scenario prior

| network | prior | rf_baseline | rf_noise | rf_gmm | rf_smote | rf_raw_baseline | lr_baseline | gcn_baseline | resid |
|---|---|---|---|---|---|---|---|---|---|
| net3 | 0.259 | 0.294 | 0.280 | 0.278 | 0.282 | 0.288 | 0.354 | 0.323 | 0.324 |
| d_town | 0.265 | 0.338 | 0.317 | 0.300 | 0.324 | 0.313 | 0.373 | 0.375 | 0.251 |
| l_town | 0.254 | 0.765 | 0.700 | 0.706 | 0.820 | 0.254 | 0.939 | 0.706 | 0.724 |
| ky2 | 0.254 | 0.490 | 0.652 | 0.479 | 0.499 | 0.289 | 0.570 | 0.269 | 0.302 |
| ky3 | 0.265 | 0.409 | 0.402 | 0.397 | 0.400 | 0.297 | 0.358 | 0.330 | 0.247 |
| ky5 | 0.265 | 0.384 | 0.378 | 0.330 | 0.374 | 0.286 | 0.398 | 0.318 | 0.361 |
| ky6 | 0.259 | 0.327 | 0.322 | 0.297 | 0.324 | 0.286 | 0.401 | 0.325 | 0.343 |
| ky7 | 0.254 | 0.414 | 0.523 | 0.374 | 0.437 | 0.361 | 0.677 | 0.556 | 0.269 |

## Deltas vs rf_baseline (row AUPRC, paired over networks)

| arm | Δ mean | wins | Wilcoxon p | network bootstrap 95% | MDE (between-network) |
|---|---|---|---|---|---|
| rf_noise | +0.021 | 5/8 | 0.461 | [-0.010, +0.057] | 0.048 |
| rf_gmm | -0.002 | 3/8 | 0.383 | [-0.026, +0.031] | 0.040 |
| rf_smote | -0.003 | 2/8 | 0.461 | [-0.009, +0.003] | 0.009 |
| rf_raw_baseline | -0.104 | 1/8 | 0.016 | [-0.185, -0.040] | 0.103 |
| lr_baseline | -0.009 | 4/8 | 0.742 | [-0.039, +0.018] | 0.039 |
| gcn_baseline | -0.010 | 4/8 | 0.844 | [-0.054, +0.028] | 0.055 |
| resid | -0.144 | 1/8 | 0.016 | [-0.205, -0.084] | 0.081 |

Baseline lift over prior: mean +0.144, Wilcoxon p = 0.008; per network: net3 +0.004, d_town +0.155, l_town +0.350, ky2 +0.107, ky3 +0.182, ky5 +0.145, ky6 +0.055, ky7 +0.153

## Early warning (rf_baseline vs resid): WSTL on leak scenarios, first-3h detection

| network | WSTL rf | WSTL resid | chance | first-3h rf | first-3h resid | FA/normal rf | FA/normal resid |
|---|---|---|---|---|---|---|---|
| net3 | 0.303 | 0.376 | 0.242 | 0.103 | 0.230 | 1.05 | 1.00 |
| d_town | 0.253 | 0.256 | 0.251 | 0.020 | 0.003 | 1.12 | 1.00 |
| l_town | 0.677 | 0.403 | 0.245 | 0.520 | 0.273 | 1.26 | 1.00 |
| ky2 | 0.408 | 0.269 | 0.240 | 0.207 | 0.110 | 1.16 | 1.76 |
| ky3 | 0.391 | 0.232 | 0.250 | 0.150 | 0.010 | 1.06 | 1.00 |
| ky5 | 0.373 | 0.259 | 0.240 | 0.097 | 0.087 | 1.09 | 1.00 |
| ky6 | 0.341 | 0.265 | 0.248 | 0.087 | 0.053 | 1.10 | 1.00 |
| ky7 | 0.323 | 0.286 | 0.243 | 0.153 | 0.053 | 1.01 | 1.00 |

## RSI (seed mean ± seed std; pooled scenario bootstrap 95%)

- net3 RSI(resid,rf_baseline): -0.024 ± 0.040 [-0.065, +0.011]
- net3 RSI(rf_noise,rf_baseline): +0.028 ± 0.017 [+0.008, +0.053]
- net3 RSI(rf_gmm,rf_baseline): +0.023 ± 0.019 [-0.001, +0.048]
- net3 RSI(rf_smote,rf_baseline): +0.008 ± 0.023 [-0.009, +0.025]
- net3 RSI(gcn_baseline,rf_baseline): -0.010 ± 0.023 [-0.057, +0.035]
- d_town RSI(resid,rf_baseline): -0.067 ± 0.046 [-0.198, +0.043]
- d_town RSI(rf_noise,rf_baseline): +0.011 ± 0.034 [-0.008, +0.043]
- d_town RSI(rf_gmm,rf_baseline): -0.007 ± 0.055 [-0.040, +0.035]
- d_town RSI(rf_smote,rf_baseline): +0.009 ± 0.031 [-0.013, +0.028]
- d_town RSI(gcn_baseline,rf_baseline): +0.015 ± 0.032 [-0.017, +0.049]
- l_town RSI(resid,rf_baseline): -0.257 ± 0.154 [-0.381, -0.154]
- l_town RSI(rf_noise,rf_baseline): +0.027 ± 0.095 [-0.006, +0.057]
- l_town RSI(rf_gmm,rf_baseline): +0.033 ± 0.096 [+0.003, +0.066]
- l_town RSI(rf_smote,rf_baseline): -0.044 ± 0.070 [-0.064, -0.026]
- l_town RSI(gcn_baseline,rf_baseline): +0.076 ± 0.130 [+0.036, +0.106]
- ky2 RSI(resid,rf_baseline): +0.073 ± 0.159 [+0.025, +0.124]
- ky2 RSI(rf_noise,rf_baseline): -0.030 ± 0.123 [-0.097, +0.028]
- ky2 RSI(rf_gmm,rf_baseline): -0.020 ± 0.049 [-0.039, +0.010]
- ky2 RSI(rf_smote,rf_baseline): -0.031 ± 0.085 [-0.056, -0.009]
- ky2 RSI(gcn_baseline,rf_baseline): +0.116 ± 0.089 [+0.063, +0.161]
- ky3 RSI(resid,rf_baseline): -0.048 ± 0.038 [-0.154, +0.037]
- ky3 RSI(rf_noise,rf_baseline): -0.019 ± 0.036 [-0.078, +0.053]
- ky3 RSI(rf_gmm,rf_baseline): +0.115 ± 0.019 [+0.039, +0.166]
- ky3 RSI(rf_smote,rf_baseline): +0.007 ± 0.027 [-0.013, +0.027]
- ky3 RSI(gcn_baseline,rf_baseline): -0.013 ± 0.016 [-0.080, +0.036]
- ky5 RSI(resid,rf_baseline): -0.135 ± 0.063 [-0.253, -0.045]
- ky5 RSI(rf_noise,rf_baseline): +0.060 ± 0.045 [+0.015, +0.100]
- ky5 RSI(rf_gmm,rf_baseline): +0.045 ± 0.046 [+0.007, +0.078]
- ky5 RSI(rf_smote,rf_baseline): +0.002 ± 0.024 [-0.017, +0.020]
- ky5 RSI(gcn_baseline,rf_baseline): +0.055 ± 0.066 [-0.022, +0.121]
- ky6 RSI(resid,rf_baseline): -0.081 ± 0.101 [-0.160, -0.015]
- ky6 RSI(rf_noise,rf_baseline): +0.009 ± 0.040 [-0.023, +0.038]
- ky6 RSI(rf_gmm,rf_baseline): +0.037 ± 0.027 [+0.017, +0.051]
- ky6 RSI(rf_smote,rf_baseline): +0.003 ± 0.009 [-0.008, +0.015]
- ky6 RSI(gcn_baseline,rf_baseline): -0.022 ± 0.038 [-0.082, +0.025]
- ky7 RSI(resid,rf_baseline): -0.012 ± 0.112 [-0.136, +0.100]
- ky7 RSI(rf_noise,rf_baseline): -0.073 ± 0.050 [-0.123, -0.024]
- ky7 RSI(rf_gmm,rf_baseline): +0.022 ± 0.057 [-0.013, +0.057]
- ky7 RSI(rf_smote,rf_baseline): -0.016 ± 0.029 [-0.033, -0.002]
- ky7 RSI(gcn_baseline,rf_baseline): -0.080 ± 0.064 [-0.155, +0.027]

## Scenario-prior sweep (subsampled normal scenarios): rf_baseline scen AUPRC max-pool and RSI(resid, rf_baseline)

- N normals = 20: net3: prior 0.64, scen 0.631, RSI -0.050; d_town: prior 0.64, scen 0.717, RSI -0.087; l_town: prior 0.63, scen 0.896, RSI -0.259; ky2: prior 0.63, scen 0.760, RSI +0.014; ky3: prior 0.64, scen 0.725, RSI -0.170; ky5: prior 0.64, scen 0.712, RSI -0.207; ky6: prior 0.64, scen 0.695, RSI -0.072; ky7: prior 0.63, scen 0.712, RSI -0.046
- N normals = 40: net3: prior 0.47, scen 0.497, RSI -0.013; d_town: prior 0.47, scen 0.548, RSI -0.077; l_town: prior 0.46, scen 0.836, RSI -0.274; ky2: prior 0.46, scen 0.646, RSI +0.051; ky3: prior 0.47, scen 0.596, RSI -0.108; ky5: prior 0.47, scen 0.573, RSI -0.180; ky6: prior 0.47, scen 0.532, RSI -0.082; ky7: prior 0.46, scen 0.580, RSI -0.024
- N normals = 100: net3: prior 0.26, scen 0.294, RSI -0.024; d_town: prior 0.26, scen 0.338, RSI -0.067; l_town: prior 0.25, scen 0.765, RSI -0.257; ky2: prior 0.25, scen 0.490, RSI +0.073; ky3: prior 0.26, scen 0.409, RSI -0.048; ky5: prior 0.26, scen 0.384, RSI -0.135; ky6: prior 0.26, scen 0.327, RSI -0.081; ky7: prior 0.25, scen 0.414, RSI -0.012
