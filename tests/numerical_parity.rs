use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    SparseTokenMask, VJepa2_1Model, VJepaConfig, VJepaEncoderConfig, VJepaLoadOptions,
    VJepaModelVariant, VJepaPredictorConfig, VJepaPreprocessConfig, apply_token_mask,
};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use serde::Deserialize;
use std::{collections::BTreeSet, env, process::Command};

type B = burn::backend::NdArray<f32>;

#[derive(Debug, Deserialize)]
struct TorchParityOutput {
    predictions: Vec<f32>,
    targets: Vec<f32>,
}

#[test]
fn tiny_sparse_forward_matches_independent_torch_fixture() {
    if Command::new("python3")
        .arg("-c")
        .arg("import torch, safetensors")
        .status()
        .map(|status| !status.success())
        .unwrap_or(true)
    {
        eprintln!(
            "skipping torch parity fixture because python3 torch/safetensors are unavailable"
        );
        return;
    }

    let device = Default::default();
    let config = parity_config();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let context = SparseTokenMask::new(vec![0, 2, 5, 7], config.num_patches()).expect("context");
    let target = SparseTokenMask::new(vec![1, 3, 4, 6], config.num_patches()).expect("target");
    let video_values = deterministic_video_values();
    let video = Tensor::<B, 5>::from_data(
        TensorData::new(
            video_values,
            [
                1,
                3,
                config.num_frames,
                config.image_size,
                config.image_size,
            ],
        ),
        &device,
    );

    let tempdir = tempfile::tempdir().expect("tempdir");
    let weights_path = tempdir.path().join("tiny.safetensors");
    let torch_output_path = tempdir.path().join("torch-output.json");
    let mut store = SafetensorsStore::from_file(&weights_path).overwrite(true);
    model.save_into(&mut store).expect("save tiny weights");

    let burn_output = model
        .predict_dense_targets(video, &context, &target)
        .expect("burn forward");
    let burn_predictions = burn_output
        .predictions
        .to_data()
        .to_vec::<f32>()
        .expect("burn prediction values");
    let burn_targets = burn_output
        .targets
        .to_data()
        .to_vec::<f32>()
        .expect("burn target values");

    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vjepa_tiny_parity.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(&weights_path)
        .arg(&torch_output_path)
        .output()
        .expect("run torch fixture");
    assert!(
        output.status.success(),
        "torch fixture failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let torch_output: TorchParityOutput = serde_json::from_slice(
        &std::fs::read(&torch_output_path).expect("read torch fixture output"),
    )
    .expect("parse torch fixture output");

    let prediction_diff = max_abs_diff(&burn_predictions, &torch_output.predictions);
    let target_diff = max_abs_diff(&burn_targets, &torch_output.targets);
    eprintln!("tiny parity prediction_diff={prediction_diff:e} target_diff={target_diff:e}");
    assert!(
        prediction_diff <= 5.0e-4 && target_diff <= 5.0e-4,
        "tiny parity exceeded tolerance: prediction_diff={prediction_diff:e}, target_diff={target_diff:e}"
    );
}

#[test]
fn sparse_forward_hot_path_has_no_backend_readbacks() {
    let model_source = include_str!("../src/model.rs");
    let temporal_source = include_str!("../src/temporal.rs");
    let model_source = model_source
        .split("#[cfg(test)]")
        .next()
        .unwrap_or(model_source);
    let temporal_source = temporal_source
        .split("#[cfg(test)]")
        .next()
        .unwrap_or(temporal_source);
    let production_source = format!("{model_source}\n{temporal_source}");
    for marker in ["into_data(", ".to_data("] {
        assert!(
            !production_source.contains(marker),
            "sparse model hot path should not force backend readback marker {marker}"
        );
    }
}

#[test]
fn cached_sparse_forward_paths_do_not_rebuild_position_tensors() {
    let model_source = include_str!("../src/model.rs");
    let temporal_source = include_str!("../src/temporal.rs");
    let encoder_hot_path = source_between(
        model_source,
        "pub fn forward_sparse_tokens_with_plan",
        "#[cfg(feature = \"sparse-patchify-wgpu\")]",
    );
    let predictor_hot_path = source_between(
        model_source,
        "pub fn forward_sparse_with_plan",
        "#[derive(Debug)]\npub struct DensePredictionOutput",
    );
    let temporal_wgpu_hot_path = source_between(
        temporal_source,
        "fn forward_sparse_patchified_masks",
        "}\n}\n\n#[derive(Clone, Copy, Debug, PartialEq)]\npub struct TemporalSparseMaskConfig",
    );

    for (label, source) in [
        ("encoder sparse plan path", encoder_hot_path),
        ("predictor sparse plan path", predictor_hot_path),
        ("temporal WGPU sparse patchify path", temporal_wgpu_hot_path),
    ] {
        assert!(
            !source.contains("TensorData::new"),
            "{label} should use cached backend tensors instead of rebuilding TensorData"
        );
        assert!(
            !source.contains("position_tensor::<"),
            "{label} should use cached sparse positional tensors"
        );
    }
}

#[test]
fn tiny_safetensors_loader_round_trips_burn_weights() {
    let device = Default::default();
    let config = parity_config();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let context = SparseTokenMask::new(vec![0, 2, 5, 7], config.num_patches()).expect("context");
    let target = SparseTokenMask::new(vec![1, 3, 4, 6], config.num_patches()).expect("target");
    let video = parity_video(&config);
    let expected = model
        .predict_dense_targets(video.clone(), &context, &target)
        .expect("expected forward")
        .predictions
        .to_data()
        .to_vec::<f32>()
        .expect("expected values");

    let tempdir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        tempdir.path().join("config.json"),
        serde_json::to_vec_pretty(&config).expect("serialize config"),
    )
    .expect("write config");
    let weights_path = tempdir.path().join("model.safetensors");
    let mut store = SafetensorsStore::from_file(&weights_path).overwrite(true);
    model.save_into(&mut store).expect("save tiny weights");

    let (loaded, _config, report) = VJepaLoadOptions {
        allow_partial: false,
        pytorch_adapter: false,
        upstream_vjepa21_names: false,
        ..VJepaLoadOptions::default()
    }
    .load_model::<B>(tempdir.path(), &device)
    .expect("load tiny model");
    assert!(
        report.missing.is_empty(),
        "missing tensors: {:?}",
        report.missing
    );
    assert!(report.errors.is_empty(), "load errors: {:?}", report.errors);
    let actual = loaded
        .predict_dense_targets(video, &context, &target)
        .expect("loaded forward")
        .predictions
        .to_data()
        .to_vec::<f32>()
        .expect("loaded values");

    let diff = max_abs_diff(&expected, &actual);
    assert!(diff <= 1.0e-6, "loader round-trip diff={diff:e}");
}

#[test]
fn tiny_hf_vjepa2_fixture_matches_burn_loader_and_forward() {
    if Command::new("python3")
        .arg("-c")
        .arg("import torch, transformers, safetensors")
        .status()
        .map(|status| !status.success())
        .unwrap_or(true)
    {
        eprintln!(
            "skipping HF V-JEPA2 parity fixture because python3 torch/transformers/safetensors are unavailable"
        );
        return;
    }

    let tempdir = tempfile::tempdir().expect("tempdir");
    let model_dir = tempdir.path().join("hf-vjepa2-tiny");
    let torch_output_path = tempdir.path().join("hf-output.json");
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vjepa_hf_tiny_parity.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(&model_dir)
        .arg(&torch_output_path)
        .output()
        .expect("run HF V-JEPA2 fixture");
    assert!(
        output.status.success(),
        "HF V-JEPA2 fixture failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let torch_output: TorchParityOutput =
        serde_json::from_slice(&std::fs::read(&torch_output_path).expect("read HF fixture output"))
            .expect("parse HF fixture output");

    let device = Default::default();
    let (model, config, report) = VJepaLoadOptions {
        allow_partial: false,
        ..VJepaLoadOptions::default()
    }
    .load_model::<B>(&model_dir, &device)
    .expect("load HF tiny V-JEPA2 fixture");
    assert!(
        report.missing.is_empty(),
        "missing tensors: {:?}",
        report.missing
    );
    assert!(report.errors.is_empty(), "load errors: {:?}", report.errors);
    assert_eq!(config.model_type, "vjepa2");
    assert_eq!(config.num_patches(), 8);

    let context = SparseTokenMask::new((0..config.num_patches()).collect(), config.num_patches())
        .expect("dense context mask");
    let target = SparseTokenMask::new(vec![1, 3, 4, 6], config.num_patches()).expect("target");
    let video = hf_parity_video(&config);
    let dense = model.encode_video(video, None);
    let predictor = model
        .predictor
        .forward_sparse(dense.tokens.clone(), &context, &target, dense.grid, 1)
        .expect("Burn predictor on HF fixture");
    let burn_predictions = predictor
        .target_predictions
        .to_data()
        .to_vec::<f32>()
        .expect("burn HF prediction values");
    let burn_targets = apply_token_mask(dense.tokens, target.to_tensor(1, &device))
        .to_data()
        .to_vec::<f32>()
        .expect("burn HF target values");

    let prediction_diff = max_abs_diff(&burn_predictions, &torch_output.predictions);
    let target_diff = max_abs_diff(&burn_targets, &torch_output.targets);
    eprintln!("HF tiny parity prediction_diff={prediction_diff:e} target_diff={target_diff:e}");
    assert!(
        prediction_diff <= 5.0e-4 && target_diff <= 5.0e-4,
        "HF tiny parity exceeded tolerance: prediction_diff={prediction_diff:e}, target_diff={target_diff:e}"
    );
}

#[test]
#[ignore = "requires a local Meta V-JEPA 2.1 checkpoint fixture"]
fn real_vjepa_checkpoint_loads_when_fixture_is_set() {
    let Some(checkpoint_dir) = env::var_os("BURN_JEPA_VJEPA21_CHECKPOINT_DIR") else {
        eprintln!("skipping real V-JEPA checkpoint smoke; set BURN_JEPA_VJEPA21_CHECKPOINT_DIR");
        return;
    };
    let checkpoint_dir = std::path::PathBuf::from(checkpoint_dir);
    let device = Default::default();
    let allow_partial = env_bool("BURN_JEPA_VJEPA21_ALLOW_PARTIAL");
    let (model, config, report) = VJepaLoadOptions {
        config_name: env::var("BURN_JEPA_VJEPA21_CONFIG")
            .unwrap_or_else(|_| "config.json".to_string()),
        weights_name: env::var("BURN_JEPA_VJEPA21_WEIGHTS")
            .unwrap_or_else(|_| "model.safetensors".to_string()),
        allow_partial,
        ..VJepaLoadOptions::default()
    }
    .load_model::<B>(&checkpoint_dir, &device)
    .expect("load real V-JEPA checkpoint fixture");

    eprintln!(
        "real V-JEPA load report: applied={} missing={} skipped={} errors={}",
        report.applied.len(),
        report.missing.len(),
        report.skipped.len(),
        report.errors.len()
    );
    assert!(report.errors.is_empty(), "load errors: {:?}", report.errors);
    assert!(
        !report.applied.is_empty(),
        "real checkpoint smoke did not apply any tensors; missing={:?} skipped={:?}",
        report.missing,
        report.skipped
    );
    assert!(
        config.encoder.embed_dim >= 768 && config.predictor.embed_dim >= 384,
        "checkpoint config was not mapped to a production V-JEPA shape: encoder={} predictor={}",
        config.encoder.embed_dim,
        config.predictor.embed_dim
    );
    if !allow_partial {
        assert!(
            report.missing.is_empty(),
            "missing tensors: {:?}",
            report.missing
        );
    }

    if env_bool("BURN_JEPA_VJEPA21_FORWARD_PARITY") {
        let torch_output = real_hf_micro_forward(&checkpoint_dir);
        let context = SparseTokenMask::new(vec![0], 1).expect("micro context");
        let target = SparseTokenMask::new(vec![0], 1).expect("micro target");
        let dense = model.encode_video(real_micro_parity_video(&config), None);
        let predictor = model
            .predictor
            .forward_sparse(
                dense.tokens.clone(),
                &context,
                &target,
                dense.grid,
                env_usize("BURN_JEPA_VJEPA21_MASK_INDEX", 1),
            )
            .expect("real checkpoint micro predictor parity");
        let burn_predictions = predictor
            .target_predictions
            .to_data()
            .to_vec::<f32>()
            .expect("real burn prediction values");
        let burn_targets = apply_token_mask(dense.tokens, target.to_tensor(1, &device))
            .to_data()
            .to_vec::<f32>()
            .expect("real burn target values");
        let prediction_diff = max_abs_diff(&burn_predictions, &torch_output.predictions);
        let target_diff = max_abs_diff(&burn_targets, &torch_output.targets);
        eprintln!(
            "real V-JEPA micro parity prediction_diff={prediction_diff:e} target_diff={target_diff:e}"
        );
        assert!(
            prediction_diff <= 5.0e-4 && target_diff <= 5.0e-4,
            "real V-JEPA micro parity exceeded tolerance: prediction_diff={prediction_diff:e}, target_diff={target_diff:e}"
        );
    } else if env_bool("BURN_JEPA_VJEPA21_FORWARD_SMOKE") {
        let context = SparseTokenMask::evenly_spaced(
            config.num_patches(),
            env_usize("BURN_JEPA_VJEPA21_CONTEXT_TOKENS", 64),
        );
        let target = test_target_mask_for_context(
            &context,
            env_usize("BURN_JEPA_VJEPA21_TARGET_TOKENS", 16),
        );
        let video = Tensor::<B, 5>::zeros(
            [
                1,
                config.in_channels,
                config.num_frames,
                config.image_size,
                config.image_size,
            ],
            &device,
        );
        let output = model
            .predict_dense_targets(video, &context, &target)
            .expect("real checkpoint sparse forward smoke");
        assert_eq!(output.predictions.shape().dims::<3>()[1], target.len());
        assert_eq!(output.targets.shape().dims::<3>()[1], target.len());
    }
}

fn parity_config() -> VJepaConfig {
    VJepaConfig {
        model_type: "vjepa2_1_tiny_parity".to_string(),
        variant: VJepaModelVariant::VitBase384,
        image_size: 16,
        patch_size: 8,
        num_frames: 2,
        tubelet_size: 1,
        in_channels: 3,
        encoder: VJepaEncoderConfig {
            embed_dim: 24,
            depth: 1,
            num_heads: 3,
            mlp_ratio: 2.0,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
            interpolate_rope: true,
            modality_embedding: false,
            n_output_distillation: 1,
        },
        predictor: VJepaPredictorConfig {
            embed_dim: 24,
            depth: 1,
            num_heads: 3,
            mlp_ratio: 2.0,
            num_mask_tokens: 1,
            output_dim: Some(24),
            return_all_tokens: false,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
        },
        preprocess: VJepaPreprocessConfig::default(),
    }
}

fn parity_video(config: &VJepaConfig) -> Tensor<B, 5> {
    let device = Default::default();
    Tensor::<B, 5>::from_data(
        TensorData::new(
            deterministic_video_values(),
            [
                1,
                3,
                config.num_frames,
                config.image_size,
                config.image_size,
            ],
        ),
        &device,
    )
}

fn deterministic_video_values() -> Vec<f32> {
    let len = 3 * 2 * 16 * 16;
    (0..len).map(|i| ((i % 29) as f32 - 14.0) / 31.0).collect()
}

fn hf_parity_video(config: &VJepaConfig) -> Tensor<B, 5> {
    let device = Default::default();
    let len = config.num_frames * config.in_channels * config.image_size * config.image_size;
    let values = (0..len)
        .map(|i| ((i % 31) as f32 - 15.0) / 23.0)
        .collect::<Vec<_>>();
    let mut burn_values = Vec::with_capacity(len);
    for channel in 0..config.in_channels {
        for frame in 0..config.num_frames {
            for row in 0..config.image_size {
                for col in 0..config.image_size {
                    let hf_index = (((frame * config.in_channels + channel) * config.image_size
                        + row)
                        * config.image_size)
                        + col;
                    burn_values.push(values[hf_index]);
                }
            }
        }
    }
    Tensor::<B, 5>::from_data(
        TensorData::new(
            burn_values,
            [
                1,
                config.in_channels,
                config.num_frames,
                config.image_size,
                config.image_size,
            ],
        ),
        &device,
    )
}

fn real_micro_parity_video(config: &VJepaConfig) -> Tensor<B, 5> {
    let device = Default::default();
    let frames = config.tubelet_size.max(1);
    let height = config.patch_size.max(1);
    let width = config.patch_size.max(1);
    let len = frames * config.in_channels * height * width;
    let values = (0..len)
        .map(|i| ((i % 31) as f32 - 15.0) / 23.0)
        .collect::<Vec<_>>();
    let mut burn_values = Vec::with_capacity(len);
    for channel in 0..config.in_channels {
        for frame in 0..frames {
            for row in 0..height {
                for col in 0..width {
                    let hf_index =
                        (((frame * config.in_channels + channel) * height + row) * width) + col;
                    burn_values.push(values[hf_index]);
                }
            }
        }
    }
    Tensor::<B, 5>::from_data(
        TensorData::new(burn_values, [1, config.in_channels, frames, height, width]),
        &device,
    )
}

fn real_hf_micro_forward(checkpoint_dir: &std::path::Path) -> TorchParityOutput {
    if Command::new("python3")
        .arg("-c")
        .arg("import torch, transformers")
        .status()
        .map(|status| !status.success())
        .unwrap_or(true)
    {
        panic!("BURN_JEPA_VJEPA21_FORWARD_PARITY requires python3 torch and transformers");
    }
    let tempdir = tempfile::tempdir().expect("real parity tempdir");
    let output_path = tempdir.path().join("real-hf-output.json");
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vjepa_hf_real_micro_forward.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(checkpoint_dir)
        .arg(&output_path)
        .output()
        .expect("run real HF V-JEPA micro fixture");
    assert!(
        output.status.success(),
        "real HF V-JEPA micro fixture failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&std::fs::read(&output_path).expect("read real HF fixture output"))
        .expect("parse real HF fixture output")
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "vectors differ in length");
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn test_target_mask_for_context(
    context: &SparseTokenMask,
    target_tokens: usize,
) -> SparseTokenMask {
    let context_set = context.indices().iter().copied().collect::<BTreeSet<_>>();
    let target = (0..context.dense_len())
        .filter(|index| !context_set.contains(index))
        .take(target_tokens.max(1))
        .collect::<Vec<_>>();
    SparseTokenMask::new(target, context.dense_len()).expect("target mask")
}

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = source
        .split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("missing source marker {start}"));
    source
        .split(end)
        .next()
        .unwrap_or_else(|| panic!("missing source marker {end}"))
}

fn env_bool(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}
