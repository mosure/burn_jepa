use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    SparseTokenMask, VJepa2_1Model, VJepaConfig, VJepaEncoderConfig, VJepaLoadOptions,
    VJepaModelVariant, VJepaPredictorConfig, VJepaPreprocessConfig,
};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use serde::Deserialize;
use std::process::Command;

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

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "vectors differ in length");
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}
