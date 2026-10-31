# v2 benchmark analysis

## Row AUPRC (seed mean ± std) and row prior

| network | prior | rf_baseline | rf_noise | rf_gmm | rf_smote | rf_raw_baseline | lr_baseline | gcn_baseline | resid |
|---|---|---|---|---|---|---|---|---|---|
| net3 | 0.091 | 0.181±0.019 | 0.168±0.052 | 0.183±0.046 | 0.174±0.036 | 0.153±0.019 | 0.143±0.040 | 0.144±0.037 | 0.112±0.010 |
| d_town | 0.099 | 0.452±0.060 | 0.318±0.101 | 0.285±0.156 | 0.434±0.062 | 0.324±0.088 | 0.452±0.170 | 0.414±0.147 | 0.119±0.007 |
| l_town | 0.078 | 0.490±0.016 | 0.315±0.180 | 0.447±0.035 | 0.478±0.019 | 0.156±0.046 | 0.322±0.112 | 0.353±0.113 | 0.125±0.033 |
| ky2 | 0.077 | 0.270±0.157 | 0.189±0.094 | 0.218±0.104 | 0.278±0.154 | 0.085±0.013 | 0.280±0.110 | 0.298±0.094 | 0.071±0.006 |
| ky3 | 0.099 | 0.279±0.031 | 0.218±0.069 | 0.224±0.048 | 0.272±0.057 | 0.116±0.024 | 0.155±0.077 | 0.151±0.072 | 0.093±0.021 |
| ky5 | 0.097 | 0.193±0.034 | 0.157±0.028 | 0.174±0.013 | 0.195±0.029 | 0.131±0.019 | 0.176±0.042 | 0.185±0.033 | 0.087±0.009 |
| ky6 | 0.085 | 0.153±0.018 | 0.131±0.039 | 0.143±0.027 | 0.154±0.020 | 0.131±0.050 | 0.162±0.039 | 0.203±0.044 | 0.085±0.036 |
| ky7 | 0.077 | 0.168±0.050 | 0.120±0.018 | 0.136±0.027 | 0.152±0.025 | 0.119±0.040 | 0.195±0.058 | 0.202±0.043 | 0.071±0.009 |
| **macro** | 0.088 | **0.273** | **0.202** | **0.226** | **0.267** | **0.152** | **0.236** | **0.244** | **0.095** |

## Scenario AUPRC, max-pool (seed mean) and scenario prior

| network | prior | rf_baseline | rf_noise | rf_gmm | rf_smote | rf_raw_baseline | lr_baseline | gcn_baseline | resid |
|---|---|---|---|---|---|---|---|---|---|
| net3 | 0.259 | 0.337 | 0.329 | 0.335 | 0.323 | 0.321 | 0.352 | 0.317 | 0.353 |
| d_town | 0.265 | 0.647 | 0.506 | 0.507 | 0.659 | 0.598 | 0.652 | 0.629 | 0.239 |
| l_town | 0.248 | 0.876 | 0.642 | 0.864 | 0.854 | 0.397 | 0.782 | 0.656 | 0.716 |
| ky2 | 0.248 | 0.593 | 0.576 | 0.499 | 0.620 | 0.276 | 0.549 | 0.469 | 0.291 |
| ky3 | 0.265 | 0.576 | 0.511 | 0.481 | 0.555 | 0.284 | 0.416 | 0.337 | 0.309 |
| ky5 | 0.265 | 0.415 | 0.437 | 0.395 | 0.413 | 0.273 | 0.397 | 0.326 | 0.343 |
| ky6 | 0.254 | 0.347 | 0.324 | 0.346 | 0.353 | 0.320 | 0.391 | 0.391 | 0.387 |
| ky7 | 0.248 | 0.486 | 0.383 | 0.423 | 0.472 | 0.339 | 0.512 | 0.536 | 0.291 |

## Deltas vs rf_baseline (row AUPRC, paired over networks)

| arm | Δ mean | wins | Wilcoxon p | network bootstrap 95% | MDE (between-network) |
|---|---|---|---|---|---|
| rf_noise | -0.071 | 0/8 | 0.008 | [-0.108, -0.039] | 0.050 |
| rf_gmm | -0.047 | 1/8 | 0.016 | [-0.085, -0.020] | 0.046 |
| rf_smote | -0.006 | 3/8 | 0.195 | [-0.012, +0.000] | 0.008 |
| rf_raw_baseline | -0.121 | 0/8 | 0.008 | [-0.191, -0.060] | 0.093 |
| lr_baseline | -0.037 | 4/8 | 0.383 | [-0.085, +0.004] | 0.062 |
| gcn_baseline | -0.030 | 3/8 | 0.383 | [-0.077, +0.013] | 0.063 |
| resid | -0.178 | 0/8 | 0.008 | [-0.255, -0.110] | 0.103 |

Baseline lift over prior: mean +0.185, Wilcoxon p = 0.008; per network: net3 +0.090, d_town +0.353, l_town +0.412, ky2 +0.193, ky3 +0.180, ky5 +0.096, ky6 +0.068, ky7 +0.091

## Early warning (rf_baseline vs resid): WSTL on leak scenarios, first-3h / in-window detection

| network | WSTL rf | WSTL resid | chance | first-3h rf | first-3h resid | in-window rf | in-window resid | FA/normal rf | FA/normal resid |
|---|---|---|---|---|---|---|---|---|---|
| net3 | 0.288 | 0.376 | 0.242 | 0.070 | 0.217 | 0.100 | 0.267 | 1.06 | 1.00 |
| d_town | 0.506 | 0.316 | 0.251 | 0.137 | 0.000 | 0.563 | 0.000 | 1.11 | 1.00 |
| l_town | 0.708 | 0.403 | 0.245 | 0.603 | 0.280 | 0.937 | 0.633 | 1.29 | 1.00 |
| ky2 | 0.468 | 0.270 | 0.240 | 0.357 | 0.097 | 0.417 | 0.100 | 1.12 | 1.37 |
| ky3 | 0.509 | 0.286 | 0.250 | 0.410 | 0.073 | 0.510 | 0.150 | 1.14 | 1.00 |
| ky5 | 0.373 | 0.265 | 0.239 | 0.180 | 0.050 | 0.223 | 0.070 | 1.09 | 1.00 |
| ky6 | 0.413 | 0.298 | 0.248 | 0.197 | 0.110 | 0.263 | 0.113 | 1.18 | 1.00 |
| ky7 | 0.365 | 0.283 | 0.243 | 0.287 | 0.057 | 0.357 | 0.077 | 1.00 | 1.00 |

## RSI (seed mean ± seed std; pooled scenario bootstrap 95%)

- net3 RSI(resid,rf_baseline): -0.085 ± 0.057 [-0.188, +0.006]
- net3 RSI(rf_noise,rf_baseline): -0.005 ± 0.017 [-0.056, +0.040]
- net3 RSI(rf_gmm,rf_baseline): +0.004 ± 0.021 [-0.032, +0.043]
- net3 RSI(rf_smote,rf_baseline): +0.008 ± 0.022 [-0.009, +0.027]
- net3 RSI(gcn_baseline,rf_baseline): -0.017 ± 0.021 [-0.074, +0.034]
- d_town RSI(resid,rf_baseline): +0.074 ± 0.068 [+0.021, +0.135]
- d_town RSI(rf_noise,rf_baseline): +0.006 ± 0.107 [-0.027, +0.044]
- d_town RSI(rf_gmm,rf_baseline): -0.027 ± 0.074 [-0.063, +0.015]
- d_town RSI(rf_smote,rf_baseline): -0.030 ± 0.050 [-0.051, -0.006]
- d_town RSI(gcn_baseline,rf_baseline): -0.020 ± 0.139 [-0.066, +0.029]
- l_town RSI(resid,rf_baseline): -0.205 ± 0.102 [-0.359, -0.077]
- l_town RSI(rf_noise,rf_baseline): +0.059 ± 0.151 [-0.013, +0.117]
- l_town RSI(rf_gmm,rf_baseline): -0.031 ± 0.065 [-0.062, -0.005]
- l_town RSI(rf_smote,rf_baseline): +0.010 ± 0.080 [-0.010, +0.033]
- l_town RSI(gcn_baseline,rf_baseline): +0.082 ± 0.043 [-0.000, +0.167]
- ky2 RSI(resid,rf_baseline): +0.103 ± 0.092 [+0.038, +0.168]
- ky2 RSI(rf_noise,rf_baseline): -0.064 ± 0.116 [-0.104, -0.023]
- ky2 RSI(rf_gmm,rf_baseline): +0.042 ± 0.025 [+0.013, +0.070]
- ky2 RSI(rf_smote,rf_baseline): -0.019 ± 0.043 [-0.042, +0.002]
- ky2 RSI(gcn_baseline,rf_baseline): +0.152 ± 0.049 [+0.124, +0.181]
- ky3 RSI(resid,rf_baseline): +0.081 ± 0.087 [+0.018, +0.141]
- ky3 RSI(rf_noise,rf_baseline): +0.004 ± 0.062 [-0.022, +0.037]
- ky3 RSI(rf_gmm,rf_baseline): +0.040 ± 0.060 [+0.013, +0.070]
- ky3 RSI(rf_smote,rf_baseline): +0.014 ± 0.025 [-0.009, +0.037]
- ky3 RSI(gcn_baseline,rf_baseline): +0.110 ± 0.047 [+0.057, +0.161]
- ky5 RSI(resid,rf_baseline): -0.034 ± 0.026 [-0.095, +0.025]
- ky5 RSI(rf_noise,rf_baseline): -0.058 ± 0.050 [-0.110, -0.006]
- ky5 RSI(rf_gmm,rf_baseline): +0.001 ± 0.048 [-0.041, +0.045]
- ky5 RSI(rf_smote,rf_baseline): +0.004 ± 0.024 [-0.022, +0.028]
- ky5 RSI(gcn_baseline,rf_baseline): +0.081 ± 0.074 [+0.019, +0.133]
- ky6 RSI(resid,rf_baseline): -0.108 ± 0.117 [-0.186, -0.024]
- ky6 RSI(rf_noise,rf_baseline): +0.001 ± 0.081 [-0.027, +0.033]
- ky6 RSI(rf_gmm,rf_baseline): -0.008 ± 0.047 [-0.033, +0.019]
- ky6 RSI(rf_smote,rf_baseline): -0.005 ± 0.015 [-0.022, +0.012]
- ky6 RSI(gcn_baseline,rf_baseline): +0.007 ± 0.070 [-0.043, +0.058]
- ky7 RSI(resid,rf_baseline): +0.097 ± 0.120 [-0.003, +0.169]
- ky7 RSI(rf_noise,rf_baseline): +0.055 ± 0.080 [+0.007, +0.091]
- ky7 RSI(rf_gmm,rf_baseline): +0.031 ± 0.078 [-0.008, +0.069]
- ky7 RSI(rf_smote,rf_baseline): -0.003 ± 0.040 [-0.027, +0.018]
- ky7 RSI(gcn_baseline,rf_baseline): -0.016 ± 0.126 [-0.069, +0.039]

## Scenario-prior sweep (first n normal scenarios kept per network): rf_baseline scen AUPRC max-pool and RSI(resid, rf_baseline)

- N normals = 20: net3: prior 0.64, scen 0.665, RSI -0.115; d_town: prior 0.64, scen 0.868, RSI -0.083; l_town: prior 0.62, scen 0.948, RSI -0.252; ky2: prior 0.62, scen 0.809, RSI +0.001; ky3: prior 0.64, scen 0.811, RSI -0.078; ky5: prior 0.64, scen 0.733, RSI -0.079; ky6: prior 0.63, scen 0.706, RSI -0.067; ky7: prior 0.62, scen 0.754, RSI +0.055
- N normals = 40: net3: prior 0.47, scen 0.535, RSI -0.080; d_town: prior 0.47, scen 0.780, RSI -0.005; l_town: prior 0.45, scen 0.923, RSI -0.238; ky2: prior 0.45, scen 0.716, RSI +0.055; ky3: prior 0.47, scen 0.718, RSI +0.004; ky5: prior 0.47, scen 0.599, RSI -0.052; ky6: prior 0.46, scen 0.552, RSI -0.092; ky7: prior 0.45, scen 0.636, RSI +0.081
- N normals = 100: net3: prior 0.26, scen 0.337, RSI -0.085; d_town: prior 0.26, scen 0.647, RSI +0.074; l_town: prior 0.25, scen 0.876, RSI -0.205; ky2: prior 0.25, scen 0.593, RSI +0.103; ky3: prior 0.26, scen 0.576, RSI +0.081; ky5: prior 0.26, scen 0.415, RSI -0.034; ky6: prior 0.25, scen 0.347, RSI -0.108; ky7: prior 0.25, scen 0.486, RSI +0.097
