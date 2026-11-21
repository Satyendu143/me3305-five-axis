# Numerical output catalogue

The PDF explains all 51 research figures and the inspection replay. This catalogue indexes the machine-readable outputs shipped with the course project. CSV counts exclude the header. Array shapes are in NumPy order. Commands and geometry use mm and rad internally. Export columns ending in _um or _urad use micrometres or microradians.

### results/abbe.npz

| Array | Shape | Type |
| --- | --- |
| `x` | (401,) | float64 |
| `theta` | (401,) | float64 |
| `angle_sensor` | (401,) | float64 |
| `target` | (401,) | float64 |
| `before` | (401,) | float64 |
| `after` | (401,) | float64 |
| `collocated` | (401,) | float64 |
| `rms_before` | () | float64 |
| `rms_after` | () | float64 |

### results/audit_environment.json

Top-level fields: `python`, `platform`, `packages`, `note`

### results/audit_example.npz

| Array | Shape | Type |
| --- | --- |
| `q` | (160, 5) | float64 |
| `residual` | (160, 3) | float64 |
| `shifted_residual` | (160, 3) | float64 |
| `cov` | (160, 3, 3) | float64 |
| `old_threshold` | () | float64 |
| `new_threshold` | () | float64 |

### results/audit_metrics.json

Top-level fields: `synthetic`, `seeds`, `noise_trials`, `mc_draws`, `raw_fits`, `parametric_MC_refits`, `region_records`, `shift_records`, `aggregate`, `shift_aggregate`, `paired_radius`, `mean_MC_covariance_relative_difference`, `mean_split_MC_relative_difference`, `wall_seconds`, `protocol`

### results/audit_nonlinear_MC.csv

72 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `truth_seed` | 831 |
| `family` | matched |
| `noise_law` | iid |
| `mc_draws` | 1024 |
| `relative_MC_vs_first_order` | 0.06190721709999184 |
| `relative_split_MC_difference` | 0.12178490776341348 |
| `relative_empirical_vs_first_order` | 0.17828842249574234 |
| `ratio_naive_to_joint_trace` | 5.7344398317897785 |
| `parametric_refit_mean_shift_um` | 0.00915057975983223 |

### results/audit_region_trials.csv

50,400 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `truth_seed` | 831 |
| `family` | matched |
| `noise_law` | iid |
| `noise_trial` | 0 |
| `method` | rotary_only |
| `observed_coverage` | 0.984375 |
| `mean_radius_um` | 1.412179131653932 |
| `point_prediction_rms_um` | 0.17518718540429307 |

### results/audit_shift_trials.csv

21,600 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `truth_seed` | 831 |
| `family` | matched |
| `noise_law` | iid |
| `noise_trial` | 0 |
| `state` | before_shift |
| `observed_coverage` | 0.984375 |
| `mean_radius_um` | 1.3796398578724915 |

### results/audit_verification.json

Recorded checks or example metrics:

```json
{
  "batch_exact_fk_vs_scalar_max_mm": 7.105427357601002e-14,
  "batch_measurement_vs_independent_frame_max_mm": 5.684341886080802e-14,
  "batch_nuisance_offset_vs_scalar": "pass",
  "correlated_joint_covariance_vs_full_block": "pass",
  "naive_ablation_retains_both_marginal_covariances": "pass",
  "correlated_raw_MC_relative_covariance_error": 0.010943507141405753,
  "conformal_rejects_invalid_inputs": "pass",
  "conformal_finite_sample_unbounded_branch": "pass",
  "correlated_covariance_positive_definite": "pass"
}
```

### results/backlash.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (1001,) | float64 |
| `desired` | (1001,) | float64 |
| `before` | (1001,) | float64 |
| `after` | (1001,) | float64 |
| `sensor_before` | (1001,) | float64 |
| `sensor_after` | (1001,) | float64 |
| `estimated_width_um` | () | float64 |
| `rms_before` | () | float64 |
| `rms_after` | () | float64 |

### results/ballbar_compensation.npz

| Array | Shape | Type |
| --- | --- |
| `A_angle` | (81,) | float64 |
| `A_none` | (81,) | float64 |
| `A_positioning` | (81,) | float64 |
| `A_measured_full` | (81,) | float64 |
| `C_angle` | (81,) | float64 |
| `C_none` | (81,) | float64 |
| `C_positioning` | (81,) | float64 |
| `C_measured_full` | (81,) | float64 |

### results/ballbar_training.csv

675 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `axis` | A |
| `a_rad` | -1.0471975511965976 |
| `c_rad` | 0.0 |
| `L_mm` | 0.0 |
| `H_mm` | 50.0 |
| `nx` | 1.0 |
| `ny` | 0.0 |
| `nz` | 0.0 |
| `direction_index` | 0 |
| `reading_mm` | 0.045091684247892096 |

### results/calibration.npz

| Array | Shape | Type |
| --- | --- |
| `A` | (675, 44) | float64 |
| `y` | (675,) | float64 |
| `truth_scaled` | (44,) | float64 |
| `joint` | (44,) | float64 |
| `separate` | (44,) | float64 |
| `alternating` | (44,) | float64 |
| `nonlinear` | (44,) | float64 |

### results/calibration_to_compensation.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (121,) | float64 |
| `before_um` | (121,) | float64 |
| `after_um` | (121,) | float64 |
| `before_urad` | (121,) | float64 |
| `after_urad` | (121,) | float64 |
| `true_coeff` | (5, 6, 4) | float64 |
| `true_pige` | (11,) | float64 |
| `estimated_coeff` | (5, 6, 4) | float64 |
| `estimated_pige` | (11,) | float64 |

### results/component_measurements.npz

| Array | Shape | Type |
| --- | --- |
| `axis_0_channel_0` | (31, 4) | float64 |
| `axis_0_channel_1` | (31, 4) | float64 |
| `axis_0_channel_2` | (31, 4) | float64 |
| `axis_0_channel_3` | (31, 4) | float64 |
| `axis_0_channel_4` | (31, 4) | float64 |
| `axis_0_channel_5` | (31, 4) | float64 |
| `axis_1_channel_0` | (31, 4) | float64 |
| `axis_1_channel_1` | (31, 4) | float64 |
| `axis_1_channel_2` | (31, 4) | float64 |
| `axis_1_channel_3` | (31, 4) | float64 |
| `axis_1_channel_4` | (31, 4) | float64 |
| `axis_1_channel_5` | (31, 4) | float64 |
| `axis_2_channel_0` | (31, 4) | float64 |
| `axis_2_channel_1` | (31, 4) | float64 |
| `axis_2_channel_2` | (31, 4) | float64 |
| `axis_2_channel_3` | (31, 4) | float64 |
| `axis_2_channel_4` | (31, 4) | float64 |
| `axis_2_channel_5` | (31, 4) | float64 |
| `axis_3_channel_0` | (31, 4) | float64 |
| `axis_3_channel_1` | (31, 4) | float64 |
| `axis_3_channel_2` | (31, 4) | float64 |
| `axis_3_channel_3` | (31, 4) | float64 |
| `axis_3_channel_4` | (31, 4) | float64 |
| `axis_3_channel_5` | (31, 4) | float64 |
| `axis_4_channel_0` | (61, 4) | float64 |
| `axis_4_channel_1` | (61, 4) | float64 |
| `axis_4_channel_2` | (61, 4) | float64 |
| `axis_4_channel_3` | (61, 4) | float64 |
| `axis_4_channel_4` | (61, 4) | float64 |
| `axis_4_channel_5` | (61, 4) | float64 |

### results/dashboard_verification.json

Recorded checks or example metrics:

```json
{
  "tabs": 4,
  "callbacks_checked": 15,
  "rendered_objects": [
    "VBox",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure",
    "Figure"
  ],
  "scope": "Headless Python callback execution; no signed-in Colab frontend tested."
}
```

### results/drive_commands.csv

41 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `x_mm` | 45.032092551125665 |
| `y_mm` | 10.625697827087073 |
| `z_mm` | -133.0638770996509 |
| `a_rad` | 0.5585416814236465 |
| `c_rad` | 4.700149914782398e-05 |
| `predicted_tcp_error_um` | 9.122628878489854e-07 |
| `predicted_axis_error_urad` | 6.851511906056997e-07 |

### results/environment.json

Top-level fields: `python`, `packages`

### results/eq29_audit.npz

| Array | Shape | Type |
| --- | --- |
| `angles` | (181,) | float64 |
| `error_printed` | (181, 6) | float64 |
| `error_corrected` | (181, 6) | float64 |

### results/estimated_map.npz

| Array | Shape | Type |
| --- | --- |
| `pos_um` | (121,) | float64 |
| `ori_urad` | (121,) | float64 |
| `missing_roll_um` | (121,) | float64 |
| `estimated_coeff` | (5, 6, 4) | float64 |
| `estimated_pige` | (11,) | float64 |

### results/example_cl.csv

41 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `X_mm` | 45.0 |
| `Y_mm` | 0.0 |
| `Z_mm` | 20.0 |
| `I` | 0.0 |
| `J` | -0.5299192642332049 |
| `K` | 0.848048096156426 |

### results/helix_animation.html

Interactive 3D helix. Open this self-contained file in a browser to inspect the simulated tool trajectory.

### results/helix_full.npz

| Array | Shape | Type |
| --- | --- |
| `target` | (121, 3) | float64 |
| `target_o` | (121, 3) | float64 |
| `t` | (121,) | float64 |
| `q` | (121, 5) | float64 |
| `p` | (121, 3) | float64 |
| `o` | (121, 3) | float64 |
| `pos_um` | (121,) | float64 |
| `ori_urad` | (121,) | float64 |
| `contour_um` | (121,) | float64 |

### results/helix_none.npz

| Array | Shape | Type |
| --- | --- |
| `target` | (121, 3) | float64 |
| `target_o` | (121, 3) | float64 |
| `t` | (121,) | float64 |
| `q` | (121, 5) | float64 |
| `p` | (121, 3) | float64 |
| `o` | (121, 3) | float64 |
| `pos_um` | (121,) | float64 |
| `ori_urad` | (121,) | float64 |
| `contour_um` | (121,) | float64 |

### results/helix_positioning.npz

| Array | Shape | Type |
| --- | --- |
| `target` | (121, 3) | float64 |
| `target_o` | (121, 3) | float64 |
| `t` | (121,) | float64 |
| `q` | (121, 5) | float64 |
| `p` | (121, 3) | float64 |
| `o` | (121, 3) | float64 |
| `pos_um` | (121,) | float64 |
| `ori_urad` | (121,) | float64 |
| `contour_um` | (121,) | float64 |

### results/independent_truth.json

Top-level fields: `linear`, `a_coeff`, `c_coeff`, `c_scale`, `pige`, `tool_home`, `part_home`

### results/inspection_verification.json

Recorded checks or example metrics:

```json
{
  "rank_and_invalid_calibration": "pass",
  "quadratic_score_and_missing_measurement": "pass",
  "circular_ROI_and_empty_selection": "pass",
  "confusion_and_undefined_metrics": "pass",
  "reference_frozen_injection_and_export": "pass",
  "widget_callbacks": "pass"
}
```

### results/iteration.npz

| Array | Shape | Type |
| --- | --- |
| `maxima` | (6, 2) | float64 |

### results/journal_environment.json

Top-level fields: `python`, `numpy`, `scipy`, `matplotlib`, `platform`, `study_wall_seconds`

### results/journal_metrics.json

Top-level fields: `seeds`, `synthetic`, `representation`, `uncertainty`, `wall_seconds`

### results/journal_model_selection.csv

72 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `config` | compact |
| `truth_family` | harmonic |
| `truth_seed` | 831 |
| `chosen` | fourier4 |
| `wrapped_cubic` | 610.0752027364516 |
| `fourier2` | 10.085650754275227 |
| `spline5` | 24.674449629295168 |
| `fourier4` | 1.4992502819530287 |
| `spline9` | 1.7581401235762737 |

### results/journal_prediction_examples.npz

| Array | Shape | Type |
| --- | --- |
| `matched__pred` | (160, 3) | float64 |
| `matched__measured` | (160, 3) | float64 |
| `matched__threshold` | () | float64 |
| `harmonic_mismatch__pred` | (160, 3) | float64 |
| `harmonic_mismatch__measured` | (160, 3) | float64 |
| `harmonic_mismatch__threshold` | () | float64 |
| `localized_mismatch__pred` | (160, 3) | float64 |
| `localized_mismatch__measured` | (160, 3) | float64 |
| `localized_mismatch__threshold` | () | float64 |

### results/journal_representation_trials.csv

864 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `config` | compact |
| `truth_family` | harmonic |
| `truth_seed` | 831 |
| `turns` | 1 |
| `model` | wrapped_cubic |
| `selected_model` | wrapped_cubic |
| `parameters` | 31 |
| `tcp_rms_um` | 7.629260104638053 |
| `tcp_max_um` | 11.813982864297087 |
| `axis_rms_urad` | 11.702494176756748 |

### results/journal_tcp_covariance.npz

| Array | Shape | Type |
| --- | --- |
| `q` | (160, 5) | float64 |
| `rotary_cov` | (160, 3, 3) | float64 |
| `full_cov` | (160, 3, 3) | float64 |
| `observed_cov` | (160, 3, 3) | float64 |
| `H` | (160, 3, 43) | float64 |
| `G` | (160, 3, 65) | float64 |
| `K` | (160, 3, 65) | float64 |
| `Dw` | (288, 65) | float64 |
| `q_all` | (12, 160, 5) | float64 |
| `full_cov_all` | (12, 160, 3, 3) | float64 |

### results/journal_tcp_coverage_trials.csv

3,600 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `case` | matched |
| `truth_seed` | 831 |
| `noise_trial` | 0 |
| `rotary_latent_coverage` | 0.484375 |
| `full_latent_coverage` | 0.96875 |
| `full_observed_coverage` | 0.953125 |
| `conformal_observed_coverage` | 1.0 |
| `conformal_threshold` | 2.9666151744866838 |
| `mean_region_radius_um` | 1.5378225411800877 |
| `tcp_prediction_rms_um` | 0.24734657483218475 |

### results/journal_tcp_sd_control.csv

36 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `case` | matched |
| `truth_seed` | 831 |
| `relative_sd_error` | 0.07766050341664314 |

### results/journal_verification.json

Recorded checks or example metrics:

```json
{
  "spline5_periodic_C2_and_datum": "pass",
  "spline5_full_rank": 43,
  "spline5_exact_vs_sensitivity": "pass",
  "spline9_periodic_C2_and_datum": "pass",
  "spline9_full_rank": 67,
  "spline9_exact_vs_sensitivity": "pass",
  "localized_numpy_truth_periodicity": "pass",
  "second_truth_no_estimator_imports": "pass",
  "108_column_TCP_derivative_step_stability": "pass",
  "TCP_covariance_positive_and_auxiliary_PSD": "pass",
  "shared_nuisance_covariance_vs_raw_MC_relative_error": 0.012697704523937734,
  "incomplete_auxiliary_covariance_differs_from_joint": "pass",
  "conformal_finite_sample_order_statistic": "pass",
  "conformal_exchangeable_rank_control": 0.958,
  "live_covariance_sensor_SD_scaling": "pass",
  "99_percent_region_requires_more_calibration": "pass",
  "journal_widget_callbacks_and_unbounded_remedy": "pass"
}
```

### results/metrics.json

Top-level fields: `provenance`, `seed`, `geometry`, `large_errors`, `estimated_maps`, `sensitivity_um`, `eq29_corrected_max_error_scaled`, `calibration`, `calibration_to_compensation`, `observability`, `uncertainty`, `geometry_condition`, `thermal`, `backlash`, `servo`, `abbe`, `sensor_faults`, `singularity`

### results/model.json

Top-level fields: `coeff`, `pige`, `convention`, `q_scale`, `tool_home`, `part_home`

### results/paper_setup_reference.json

Top-level fields: `source`, `A`, `C`

### results/publication_design_compact.npz

| Array | Shape | Type |
| --- | --- |
| `pool` | (320, 4) | float64 |
| `task` | (192, 4) | float64 |
| `heldout` | (128, 4) | float64 |
| `space_filling` | (48,) | int64 |
| `d_optimal` | (48,) | int64 |
| `task_i_optimal` | (48,) | int64 |

### results/publication_design_extended.npz

| Array | Shape | Type |
| --- | --- |
| `pool` | (320, 4) | float64 |
| `task` | (192, 4) | float64 |
| `heldout` | (128, 4) | float64 |
| `space_filling` | (48,) | int64 |
| `d_optimal` | (48,) | int64 |
| `task_i_optimal` | (48,) | int64 |

### results/publication_design_reference.npz

| Array | Shape | Type |
| --- | --- |
| `pool` | (320, 4) | float64 |
| `task` | (192, 4) | float64 |
| `heldout` | (128, 4) | float64 |
| `space_filling` | (48,) | int64 |
| `d_optimal` | (48,) | int64 |
| `task_i_optimal` | (48,) | int64 |

### results/publication_design_summary.csv

36 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `mismatch` | 0.0 |
| `poses` | 16 |
| `method` | space_filling |
| `mean_rms_um` | 0.35510624597427437 |
| `seed_cluster_ci95` | [0.3479292653292253, 0.3609903688236591] |
| `noise_only_rms_um` | 0.3285995900197282 |
| `bias_um` | 0.00048024093455456725 |
| `min_rank` | 43 |

### results/publication_design_trials.csv

10,368 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `config` | compact |
| `mismatch` | 0.0 |
| `truth_seed` | 831 |
| `noise_trial` | 0 |
| `poses` | 16 |
| `scalar_readings` | 96 |
| `method` | space_filling |
| `rms_um` | 0.3081860163487889 |
| `noise_only_rms_um` | 0.2710267199358489 |
| `noiseless_rotary_bias_um` | 0.000451690497403468 |
| `rank` | 43 |
| `condition` | 403.07336495688577 |

### results/publication_dynamic_ablations.csv

36 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `thermal_mode` | physical |
| `duration_s` | 30.0 |
| `turns` | 2.5 |
| `scheme` | uncompensated |
| `tcp_rms_um` | 458.44868733606893 |
| `contour_rms_um` | 79.38126017629025 |
| `axis_rms_urad` | 7691.471368074168 |
| `max_command_jump_linear_mm` | 0.09123827711389865 |

### results/publication_dynamic_accelerated_30.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (3001,) | float64 |
| `minutes` | (3001,) | float64 |
| `RTD` | (3001, 3) | float64 |
| `thermal_true` | (3001, 3) | float64 |
| `thermal_estimate` | (3001, 3) | float64 |
| `probe_error_um` | (3001, 3) | float64 |
| `uncompensated__pos_um` | (3001,) | float64 |
| `uncompensated__contour_um` | (3001,) | float64 |
| `uncompensated__q_command` | (3001, 5) | float64 |
| `uncompensated__q_motor` | (3001, 5) | float64 |
| `uncompensated__output_encoder` | (3001, 5) | float64 |
| `geometry__pos_um` | (3001,) | float64 |
| `geometry__contour_um` | (3001,) | float64 |
| `geometry__q_command` | (3001, 5) | float64 |
| `geometry__q_motor` | (3001, 5) | float64 |
| `geometry__output_encoder` | (3001, 5) | float64 |
| `geometry_ff__pos_um` | (3001,) | float64 |
| `geometry_ff__contour_um` | (3001,) | float64 |
| `geometry_ff__q_command` | (3001, 5) | float64 |
| `geometry_ff__q_motor` | (3001, 5) | float64 |
| `geometry_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_backlash__pos_um` | (3001,) | float64 |
| `geometry_backlash__contour_um` | (3001,) | float64 |
| `geometry_backlash__q_command` | (3001, 5) | float64 |
| `geometry_backlash__q_motor` | (3001, 5) | float64 |
| `geometry_backlash__output_encoder` | (3001, 5) | float64 |
| `geometry_backlash_ff__pos_um` | (3001,) | float64 |
| `geometry_backlash_ff__contour_um` | (3001,) | float64 |
| `geometry_backlash_ff__q_command` | (3001, 5) | float64 |
| `geometry_backlash_ff__q_motor` | (3001, 5) | float64 |
| `geometry_backlash_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal__pos_um` | (3001,) | float64 |
| `geometry_thermal__contour_um` | (3001,) | float64 |
| `geometry_thermal__q_command` | (3001, 5) | float64 |
| `geometry_thermal__q_motor` | (3001, 5) | float64 |
| `geometry_thermal__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal_ff__pos_um` | (3001,) | float64 |
| `geometry_thermal_ff__contour_um` | (3001,) | float64 |
| `geometry_thermal_ff__q_command` | (3001, 5) | float64 |
| `geometry_thermal_ff__q_motor` | (3001, 5) | float64 |
| `geometry_thermal_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal_backlash__pos_um` | (3001,) | float64 |
| `geometry_thermal_backlash__contour_um` | (3001,) | float64 |
| `geometry_thermal_backlash__q_command` | (3001, 5) | float64 |
| `geometry_thermal_backlash__q_motor` | (3001, 5) | float64 |
| `geometry_thermal_backlash__output_encoder` | (3001, 5) | float64 |
| `all_remedies__pos_um` | (3001,) | float64 |
| `all_remedies__contour_um` | (3001,) | float64 |
| `all_remedies__q_command` | (3001, 5) | float64 |
| `all_remedies__q_motor` | (3001, 5) | float64 |
| `all_remedies__output_encoder` | (3001, 5) | float64 |

### results/publication_dynamic_accelerated_60.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (6001,) | float64 |
| `minutes` | (6001,) | float64 |
| `RTD` | (6001, 3) | float64 |
| `thermal_true` | (6001, 3) | float64 |
| `thermal_estimate` | (6001, 3) | float64 |
| `probe_error_um` | (6001, 3) | float64 |
| `uncompensated__pos_um` | (6001,) | float64 |
| `uncompensated__contour_um` | (6001,) | float64 |
| `uncompensated__q_command` | (6001, 5) | float64 |
| `uncompensated__q_motor` | (6001, 5) | float64 |
| `uncompensated__output_encoder` | (6001, 5) | float64 |
| `geometry__pos_um` | (6001,) | float64 |
| `geometry__contour_um` | (6001,) | float64 |
| `geometry__q_command` | (6001, 5) | float64 |
| `geometry__q_motor` | (6001, 5) | float64 |
| `geometry__output_encoder` | (6001, 5) | float64 |
| `geometry_ff__pos_um` | (6001,) | float64 |
| `geometry_ff__contour_um` | (6001,) | float64 |
| `geometry_ff__q_command` | (6001, 5) | float64 |
| `geometry_ff__q_motor` | (6001, 5) | float64 |
| `geometry_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_backlash__pos_um` | (6001,) | float64 |
| `geometry_backlash__contour_um` | (6001,) | float64 |
| `geometry_backlash__q_command` | (6001, 5) | float64 |
| `geometry_backlash__q_motor` | (6001, 5) | float64 |
| `geometry_backlash__output_encoder` | (6001, 5) | float64 |
| `geometry_backlash_ff__pos_um` | (6001,) | float64 |
| `geometry_backlash_ff__contour_um` | (6001,) | float64 |
| `geometry_backlash_ff__q_command` | (6001, 5) | float64 |
| `geometry_backlash_ff__q_motor` | (6001, 5) | float64 |
| `geometry_backlash_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal__pos_um` | (6001,) | float64 |
| `geometry_thermal__contour_um` | (6001,) | float64 |
| `geometry_thermal__q_command` | (6001, 5) | float64 |
| `geometry_thermal__q_motor` | (6001, 5) | float64 |
| `geometry_thermal__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal_ff__pos_um` | (6001,) | float64 |
| `geometry_thermal_ff__contour_um` | (6001,) | float64 |
| `geometry_thermal_ff__q_command` | (6001, 5) | float64 |
| `geometry_thermal_ff__q_motor` | (6001, 5) | float64 |
| `geometry_thermal_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal_backlash__pos_um` | (6001,) | float64 |
| `geometry_thermal_backlash__contour_um` | (6001,) | float64 |
| `geometry_thermal_backlash__q_command` | (6001, 5) | float64 |
| `geometry_thermal_backlash__q_motor` | (6001, 5) | float64 |
| `geometry_thermal_backlash__output_encoder` | (6001, 5) | float64 |
| `all_remedies__pos_um` | (6001,) | float64 |
| `all_remedies__contour_um` | (6001,) | float64 |
| `all_remedies__q_command` | (6001, 5) | float64 |
| `all_remedies__q_motor` | (6001, 5) | float64 |
| `all_remedies__output_encoder` | (6001, 5) | float64 |

### results/publication_dynamic_physical_30.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (3001,) | float64 |
| `minutes` | (3001,) | float64 |
| `RTD` | (3001, 3) | float64 |
| `thermal_true` | (3001, 3) | float64 |
| `thermal_estimate` | (3001, 3) | float64 |
| `probe_error_um` | (3001, 3) | float64 |
| `uncompensated__pos_um` | (3001,) | float64 |
| `uncompensated__contour_um` | (3001,) | float64 |
| `uncompensated__q_command` | (3001, 5) | float64 |
| `uncompensated__q_motor` | (3001, 5) | float64 |
| `uncompensated__output_encoder` | (3001, 5) | float64 |
| `geometry__pos_um` | (3001,) | float64 |
| `geometry__contour_um` | (3001,) | float64 |
| `geometry__q_command` | (3001, 5) | float64 |
| `geometry__q_motor` | (3001, 5) | float64 |
| `geometry__output_encoder` | (3001, 5) | float64 |
| `geometry_ff__pos_um` | (3001,) | float64 |
| `geometry_ff__contour_um` | (3001,) | float64 |
| `geometry_ff__q_command` | (3001, 5) | float64 |
| `geometry_ff__q_motor` | (3001, 5) | float64 |
| `geometry_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_backlash__pos_um` | (3001,) | float64 |
| `geometry_backlash__contour_um` | (3001,) | float64 |
| `geometry_backlash__q_command` | (3001, 5) | float64 |
| `geometry_backlash__q_motor` | (3001, 5) | float64 |
| `geometry_backlash__output_encoder` | (3001, 5) | float64 |
| `geometry_backlash_ff__pos_um` | (3001,) | float64 |
| `geometry_backlash_ff__contour_um` | (3001,) | float64 |
| `geometry_backlash_ff__q_command` | (3001, 5) | float64 |
| `geometry_backlash_ff__q_motor` | (3001, 5) | float64 |
| `geometry_backlash_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal__pos_um` | (3001,) | float64 |
| `geometry_thermal__contour_um` | (3001,) | float64 |
| `geometry_thermal__q_command` | (3001, 5) | float64 |
| `geometry_thermal__q_motor` | (3001, 5) | float64 |
| `geometry_thermal__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal_ff__pos_um` | (3001,) | float64 |
| `geometry_thermal_ff__contour_um` | (3001,) | float64 |
| `geometry_thermal_ff__q_command` | (3001, 5) | float64 |
| `geometry_thermal_ff__q_motor` | (3001, 5) | float64 |
| `geometry_thermal_ff__output_encoder` | (3001, 5) | float64 |
| `geometry_thermal_backlash__pos_um` | (3001,) | float64 |
| `geometry_thermal_backlash__contour_um` | (3001,) | float64 |
| `geometry_thermal_backlash__q_command` | (3001, 5) | float64 |
| `geometry_thermal_backlash__q_motor` | (3001, 5) | float64 |
| `geometry_thermal_backlash__output_encoder` | (3001, 5) | float64 |
| `all_remedies__pos_um` | (3001,) | float64 |
| `all_remedies__contour_um` | (3001,) | float64 |
| `all_remedies__q_command` | (3001, 5) | float64 |
| `all_remedies__q_motor` | (3001, 5) | float64 |
| `all_remedies__output_encoder` | (3001, 5) | float64 |

### results/publication_dynamic_physical_60.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (6001,) | float64 |
| `minutes` | (6001,) | float64 |
| `RTD` | (6001, 3) | float64 |
| `thermal_true` | (6001, 3) | float64 |
| `thermal_estimate` | (6001, 3) | float64 |
| `probe_error_um` | (6001, 3) | float64 |
| `uncompensated__pos_um` | (6001,) | float64 |
| `uncompensated__contour_um` | (6001,) | float64 |
| `uncompensated__q_command` | (6001, 5) | float64 |
| `uncompensated__q_motor` | (6001, 5) | float64 |
| `uncompensated__output_encoder` | (6001, 5) | float64 |
| `geometry__pos_um` | (6001,) | float64 |
| `geometry__contour_um` | (6001,) | float64 |
| `geometry__q_command` | (6001, 5) | float64 |
| `geometry__q_motor` | (6001, 5) | float64 |
| `geometry__output_encoder` | (6001, 5) | float64 |
| `geometry_ff__pos_um` | (6001,) | float64 |
| `geometry_ff__contour_um` | (6001,) | float64 |
| `geometry_ff__q_command` | (6001, 5) | float64 |
| `geometry_ff__q_motor` | (6001, 5) | float64 |
| `geometry_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_backlash__pos_um` | (6001,) | float64 |
| `geometry_backlash__contour_um` | (6001,) | float64 |
| `geometry_backlash__q_command` | (6001, 5) | float64 |
| `geometry_backlash__q_motor` | (6001, 5) | float64 |
| `geometry_backlash__output_encoder` | (6001, 5) | float64 |
| `geometry_backlash_ff__pos_um` | (6001,) | float64 |
| `geometry_backlash_ff__contour_um` | (6001,) | float64 |
| `geometry_backlash_ff__q_command` | (6001, 5) | float64 |
| `geometry_backlash_ff__q_motor` | (6001, 5) | float64 |
| `geometry_backlash_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal__pos_um` | (6001,) | float64 |
| `geometry_thermal__contour_um` | (6001,) | float64 |
| `geometry_thermal__q_command` | (6001, 5) | float64 |
| `geometry_thermal__q_motor` | (6001, 5) | float64 |
| `geometry_thermal__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal_ff__pos_um` | (6001,) | float64 |
| `geometry_thermal_ff__contour_um` | (6001,) | float64 |
| `geometry_thermal_ff__q_command` | (6001, 5) | float64 |
| `geometry_thermal_ff__q_motor` | (6001, 5) | float64 |
| `geometry_thermal_ff__output_encoder` | (6001, 5) | float64 |
| `geometry_thermal_backlash__pos_um` | (6001,) | float64 |
| `geometry_thermal_backlash__contour_um` | (6001,) | float64 |
| `geometry_thermal_backlash__q_command` | (6001, 5) | float64 |
| `geometry_thermal_backlash__q_motor` | (6001, 5) | float64 |
| `geometry_thermal_backlash__output_encoder` | (6001, 5) | float64 |
| `all_remedies__pos_um` | (6001,) | float64 |
| `all_remedies__contour_um` | (6001,) | float64 |
| `all_remedies__q_command` | (6001, 5) | float64 |
| `all_remedies__q_motor` | (6001, 5) | float64 |
| `all_remedies__output_encoder` | (6001, 5) | float64 |

### results/publication_environment.json

Top-level fields: `python`, `numpy`, `platform`

### results/publication_grid.csv

24 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `thermal_mode` | physical |
| `dt_s` | 0.04 |
| `scheme` | uncompensated |
| `tcp_rms_um` | 248.85473827103087 |
| `contour_rms_um` | 61.92896765301724 |

### results/publication_metrics.json

Top-level fields: `scope`, `configurations`, `seeds`, `noise_repeats`, `design`, `representation`, `uncertainty`, `dynamic`, `first_pass_wall_seconds`, `execution_note`

### results/publication_paired_comparisons.csv

27 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `mismatch` | 0.0 |
| `poses` | 16 |
| `comparison` | task_i_optimal minus space_filling |
| `delta_um` | -0.09196879690728343 |
| `seed_cluster_ci95` | [-0.09997920484262142, -0.08321327008127144] |

### results/publication_representation_summary.csv

10 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `turns` | 1 |
| `model` | uncompensated |
| `mean_rms_um` | 64.90316849829192 |
| `seed_cluster_ci95` | [54.40746423645331, 74.72721517342808] |
| `mean_max_um` | 77.79499855947944 |
| `parameters` | 0 |

### results/publication_representation_trials.csv

360 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `config` | compact |
| `truth_seed` | 831 |
| `turns` | 1 |
| `model` | uncompensated |
| `parameters` | 0 |
| `tcp_rms_um` | 53.50220073221967 |
| `tcp_max_um` | 63.126118315677175 |
| `axis_rms_urad` | 31.44633540331944 |

### results/publication_uncertainty_control.npz

| Array | Shape | Type |
| --- | --- |
| `predicted_sd` | (384,) | float64 |
| `monte_carlo_sd` | (384,) | float64 |
| `parameter_noise` | (43, 400) | float64 |
| `heldout` | (128, 4) | float64 |

### results/publication_verification.json

Recorded checks or example metrics:

```json
{
  "task_design_unique_nested": "pass",
  "task_design_full_rank": 43,
  "exact_two_pole_step_and_semigroup": "pass",
  "task_variance_gain_vs_direct_inverse": 0.0,
  "wrapped_cubic2_exact_vs_sensitivity": 8.966727711594013e-15,
  "periodic2_exact_vs_sensitivity": 1.3316921507336118e-14,
  "periodic4_exact_vs_sensitivity": 1.337435229992365e-14,
  "cubic2_exact_vs_sensitivity": 8.398724655609244e-15,
  "wrapped_baseline_bounded_plus_scale": "pass",
  "wrapped_seam_continuity_and_scale_identifiability": "pass",
  "omitted_harmonics_zero_control": "pass",
  "physical_clock_and_full_factorial": "pass",
  "full_rank_covariance_formula": "pass",
  "rank_deficient_covariance_not_zero": "pass",
  "bootstrap_centred_on_seed_clusters": "pass",
  "final_dashboard_exact_zoh_button_callback": "pass"
}
```

### results/research_calibration.csv

60 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `a_rad` | -1.0471975511965976 |
| `c_rad` | 4.319689898685965 |
| `L_mm` | 140.0 |
| `H_mm` | 110.0 |
| `dx_mm` | 0.02622221923864434 |
| `dy_mm` | 0.046950242116658054 |
| `dz_mm` | -0.012748684361455723 |
| `rx_equiv_mm` | -0.0045458432033094985 |
| `ry_equiv_mm` | -0.000928221454598864 |
| `rz_equiv_mm` | 0.0018985920093756439 |

### results/research_design.npz

| Array | Shape | Type |
| --- | --- |
| `trial_results` | (216, 5) | float64 |
| `pool` | (320, 4) | float64 |
| `heldout` | (96, 4) | float64 |
| `uniform` | (48,) | int64 |
| `d_optimal` | (48,) | int64 |

### results/research_independence.npz

| Array | Shape | Type |
| --- | --- |
| `q` | (80, 5) | float64 |
| `matched` | (80, 2) | float64 |
| `truth_model` | (80, 2) | float64 |

### results/research_integrated.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (1501,) | float64 |
| `p_target` | (1501, 3) | float64 |
| `o_target` | (1501, 3) | float64 |
| `minutes` | (1501,) | float64 |
| `RTD` | (1501, 3) | float64 |
| `thermal_true` | (1501, 3) | float64 |
| `thermal_estimate` | (1501, 3) | float64 |
| `probe_error_um` | (1501, 3) | float64 |
| `uncompensated__q_command` | (1501, 5) | float64 |
| `uncompensated__q_motor` | (1501, 5) | float64 |
| `uncompensated__q_load` | (1501, 5) | float64 |
| `uncompensated__p` | (1501, 3) | float64 |
| `uncompensated__o` | (1501, 3) | float64 |
| `uncompensated__error_vector_um` | (1501, 3) | float64 |
| `uncompensated__pos_um` | (1501,) | float64 |
| `uncompensated__contour_um` | (1501,) | float64 |
| `uncompensated__ori_urad` | (1501,) | float64 |
| `uncompensated__output_encoder` | (1501, 5) | float64 |
| `geometry__q_command` | (1501, 5) | float64 |
| `geometry__q_motor` | (1501, 5) | float64 |
| `geometry__q_load` | (1501, 5) | float64 |
| `geometry__p` | (1501, 3) | float64 |
| `geometry__o` | (1501, 3) | float64 |
| `geometry__error_vector_um` | (1501, 3) | float64 |
| `geometry__pos_um` | (1501,) | float64 |
| `geometry__contour_um` | (1501,) | float64 |
| `geometry__ori_urad` | (1501,) | float64 |
| `geometry__output_encoder` | (1501, 5) | float64 |
| `geometry_thermal__q_command` | (1501, 5) | float64 |
| `geometry_thermal__q_motor` | (1501, 5) | float64 |
| `geometry_thermal__q_load` | (1501, 5) | float64 |
| `geometry_thermal__p` | (1501, 3) | float64 |
| `geometry_thermal__o` | (1501, 3) | float64 |
| `geometry_thermal__error_vector_um` | (1501, 3) | float64 |
| `geometry_thermal__pos_um` | (1501,) | float64 |
| `geometry_thermal__contour_um` | (1501,) | float64 |
| `geometry_thermal__ori_urad` | (1501,) | float64 |
| `geometry_thermal__output_encoder` | (1501, 5) | float64 |
| `geometry_thermal_backlash__q_command` | (1501, 5) | float64 |
| `geometry_thermal_backlash__q_motor` | (1501, 5) | float64 |
| `geometry_thermal_backlash__q_load` | (1501, 5) | float64 |
| `geometry_thermal_backlash__p` | (1501, 3) | float64 |
| `geometry_thermal_backlash__o` | (1501, 3) | float64 |
| `geometry_thermal_backlash__error_vector_um` | (1501, 3) | float64 |
| `geometry_thermal_backlash__pos_um` | (1501,) | float64 |
| `geometry_thermal_backlash__contour_um` | (1501,) | float64 |
| `geometry_thermal_backlash__ori_urad` | (1501,) | float64 |
| `geometry_thermal_backlash__output_encoder` | (1501, 5) | float64 |
| `all_remedies__q_command` | (1501, 5) | float64 |
| `all_remedies__q_motor` | (1501, 5) | float64 |
| `all_remedies__q_load` | (1501, 5) | float64 |
| `all_remedies__p` | (1501, 3) | float64 |
| `all_remedies__o` | (1501, 3) | float64 |
| `all_remedies__error_vector_um` | (1501, 3) | float64 |
| `all_remedies__pos_um` | (1501,) | float64 |
| `all_remedies__contour_um` | (1501,) | float64 |
| `all_remedies__ori_urad` | (1501,) | float64 |
| `all_remedies__output_encoder` | (1501, 5) | float64 |
| `calibration__beta` | (4, 3) | float64 |
| `calibration__alpha` | (5,) | float64 |
| `calibration__width` | (5,) | float64 |
| `calibration__thermal_time` | (601,) | float64 |
| `calibration__thermal_T` | (601, 3) | float64 |
| `calibration__thermal_displacement` | (601, 3) | float64 |
| `calibration__servo_time` | (400,) | float64 |
| `calibration__servo_command` | (400, 5) | float64 |
| `calibration__servo_encoder` | (400, 5) | float64 |
| `calibration__backlash_command` | (1203, 5) | float64 |
| `calibration__backlash_encoder` | (1203, 5) | float64 |

### results/research_metrics.json

Top-level fields: `scope`, `trials`, `fits`, `one_revolution`, `three_revolutions`, `implementation_check`, `design`, `sensor_coverage`, `integrated`, `dynamic_conditions`, `second_path`

### results/research_models.json

Top-level fields: `cubic`, `periodic`

### results/research_second_path.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (3001,) | float64 |
| `uncompensated` | (3001,) | float64 |
| `geometry` | (3001,) | float64 |
| `geometry_thermal` | (3001,) | float64 |
| `geometry_thermal_backlash` | (3001,) | float64 |
| `all_remedies` | (3001,) | float64 |

### results/research_static_one_revolution_cubic.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_static_one_revolution_periodic.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_static_one_revolution_uncompensated.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_static_three_revolutions_cubic.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_static_three_revolutions_periodic.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_static_three_revolutions_uncompensated.npz

| Array | Shape | Type |
| --- | --- |
| `c_deg` | (241,) | float64 |
| `p` | (241, 3) | float64 |
| `q` | (241, 5) | float64 |
| `pos_um` | (241,) | float64 |
| `ori_urad` | (241,) | float64 |

### results/research_verification.json

Recorded checks or example metrics:

```json
{
  "independent_plant_import_audit": "NumPy only",
  "100_pose_direct_vs_homogeneous_max_mm": 2.086444348496176e-13,
  "Fourier_wrap_and_separate_scale": "pass",
  "zero_datum_maps": "pass",
  "43_parameter_full_frame_sensitivity": 2.369585353672113e-10,
  "equal_budget_unique_nested_design": "pass",
  "mixed_tilt_full_rank": 43,
  "orientation_cannot_fix_missing_tilts": 25,
  "orientation_information_positive_semidefinite": "pass",
  "first_order_inverse_matched_control": 8.881784197001252e-16,
  "calibration_reproducibility": "pass",
  "sensor_estimated_play_width_relative_error": 0.002498797933020089,
  "coupled_remedy_ratio": 0.2088442295770802,
  "mismatched_plant_not_numerical_zero": "pass",
  "servo_state_causality": "pass",
  "play_state_causality": "pass",
  "dashboard_pure_callbacks": "pass"
}
```

### results/sensor_faults.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (601,) | float64 |
| `truth` | (601,) | float64 |
| `sensors` | (601, 3) | float64 |
| `before` | (601,) | float64 |
| `after` | (601,) | float64 |
| `alarm` | (601,) | bool |
| `rms_before` | () | float64 |
| `rms_after` | () | float64 |

### results/servo.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (5000,) | float64 |
| `desired` | (5000,) | float64 |
| `before` | (5000,) | float64 |
| `after` | (5000,) | float64 |
| `sensor_before` | (5000,) | float64 |
| `sensor_after` | (5000,) | float64 |
| `estimated_tau_ms` | () | float64 |
| `rms_before` | () | float64 |
| `rms_after` | () | float64 |

### results/thermal.npz

| Array | Shape | Type |
| --- | --- |
| `t` | (361,) | float64 |
| `T` | (361, 3) | float64 |
| `sensors` | (361, 3) | float64 |
| `before` | (361, 3) | float64 |
| `after` | (361, 3) | float64 |
| `prediction` | (361, 3) | float64 |
| `beta` | (4, 3) | float64 |
| `train_condition` | () | float64 |
| `rms_before` | () | float64 |
| `rms_after` | () | float64 |

### results/uncertainty.npz

| Array | Shape | Type |
| --- | --- |
| `samples` | (44, 100) | float64 |
| `predicted_std` | (44,) | float64 |

### results/verification.json

Recorded checks or example metrics:

```json
{
  "SE3_vs_scipy_expm": 2.2315482794965646e-14,
  "small_angle_and_translation_limit": "pass",
  "IK_FK_100_poses_with_home_offsets_mm": 1.9898368414232615e-13,
  "general_Paden_Kahan_two_branches": "pass",
  "singularity_branch_keeps_previous_C": "pass",
  "exact_error_transform_stays_in_SE3": "pass",
  "symbolic_corrected_Eq29": "pass",
  "corrected_inverse_at_trigonometric_edge_cases": 3.469446951953614e-18,
  "printed_Eq29_counterexample": {
    "eps_xa_true_urad": 30,
    "printed_urad": -30.00000000000003,
    "delta_za_true_um": 5,
    "printed_um": 3.5000000000000004
  },
  "44_parameter_sensitivity_vs_exact_finite_difference": 7.958078650310297e-10,
  "noiseless_linear_calibration_recovery_scaled": 6.990685808005992e-12,
  "noiseless_exact_ballbar_recovery_scaled": 8.686212569275487e-05,
  "radial_only_unidentifiability_rank": 40,
  "equal_height_degeneracy_rank": 5,
  "ballbar_three_directions_Rtest_information_equivalence": "pass",
  "full_compensation_max_um": 1.0756090398242875e-05,
  "damped_compensation_mm": 7.418283949015283e-14,
  "PCS_datum_zero": "pass",
  "CSV_sensor_roundtrip": "pass",
  "thermal_remedy_ratio": 0.039409216867141904,
  "backlash_remedy_ratio": 0.03341922061701887,
  "servo_remedy_ratio": 3.828162892792911e-05,
  "abbe_remedy_ratio": 0.07059276725661458,
  "sensor_faults_remedy_ratio": 0.06551248131916577
}
```

### results/volume_map.npz

| Array | Shape | Type |
| --- | --- |
| `drive_commands` | (343, 5) | float64 |
| `nominal` | (343, 3) | float64 |
| `error_mm` | (343, 3) | float64 |
| `norm_um` | (343,) | float64 |

### inspection/inspection_example.csv

64 data rows. Fields and first saved record:

| Field | Value |
| --- | --- |
| `pose_index` | 0 |
| `a_deg` | -33.56575602922871 |
| `c_deg` | 143.45157386168478 |
| `rx_um` | 3.6947620935306986 |
| `ry_um` | 0.5972224183850017 |
| `rz_um` | 1.1021849438357378 |
| `score` | 7.638901151266317 |
| `threshold` | 2.9666151744866838 |
| `selected` | True |
| `valid` | True |
| `statistical_alarm` | True |
| `quality_error` | False |

### inspection/inspection_metrics.json

Recorded checks or example metrics:

```json
{
  "scope": "One saved synthetic example; no repeated-trial or physical detection claim",
  "alpha": 0.05,
  "normal_calibration_records": 96,
  "assessment_records": 64,
  "cases": [
    {
      "case": "normal",
      "records": 64,
      "alarms": 0,
      "invalid": 0,
      "labels_available": true,
      "tp": 0,
      "fp": 0,
      "fn": 0,
      "tn": 64,
      "precision": null,
      "recall": null,
      "false_positive_rate": 0.0,
      "f1": null
    },
    {
      "case": "offset",
      "records": 64,
      "alarms": 64,
      "invalid": 0,
      "labels_available": true,
      "tp": 64,
      "fp": 0,
      "fn": 0,
      "tn": 0,
      "precision": 1.0,
      "recall": 1.0,
      "false_positive_rate": null,
      "f1": 1.0
    },
    {
      "case": "sparse_spikes",
      "records": 64,
      "alarms": 8,
      "invalid": 0,
      "labels_available": true,
      "tp": 8,
      "fp": 0,
      "fn": 0,
      "tn": 56,
      "precision": 1.0,
      "recall": 1.0,
      "false_positive_rate": 0.0,
      "f1": 1.0
    },
    {
      "case": "dropout",
      "records": 64,
      "alarms": 8,
      "invalid": 8,
      "labels_available": true,
      "tp": 8,
      "fp": 0,
      "fn": 0,
      "tn": 56,
      "precision": 1.0,
      "recall": 1.0,
      "false_positive_rate": 0.0,
      "f1": 1.0
    },
    {
      "case": "harmonic_mismatch",
      "records": 64,
      "alarms": 29,
      "invalid": 0,
      "labels_available": false
    },
    {
      "case": "localized_mismatch",
      "records": 64,
      "alarms": 27,
      "invalid": 0,
      "labels_available": false
    }
  ]
}
```

### inspection/inspection_spike_example.png

Rendered sparse-spike inspection example. The report shows the same image alongside an interpretation of the alarm counts.
