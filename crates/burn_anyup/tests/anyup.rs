use burn::tensor::{Int, Tensor, TensorData};
use burn_anyup::{
    AnyUp, AnyUpAttentionMode, AnyUpConfig, AnyUpHighResFeatureMemory,
    AnyUpHighResFeatureMemoryConfig, AnyUpLoadOptions, AnyUpSparseFeatureMemoryWriteMode,
    AnyUpSparseFeatureUpdateMode, AnyUpSparseOutputPlan, sparse_low_features_to_nchw,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

type B = burn::backend::NdArray<f32>;

#[test]
fn anyup_forward_returns_requested_output_resolution_and_feature_dim() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = Tensor::<B, 4>::ones([1, 3, 8, 8], &device);
    let features = Tensor::<B, 4>::ones([1, 5, 2, 2], &device);

    let output = model.forward(image, features, Some([6, 4]), Some(1));

    assert_eq!(output.shape().dims::<4>(), [1, 5, 6, 4]);
}

#[test]
fn chunked_attention_matches_unchunked_attention() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);

    let full = model.forward(image.clone(), features.clone(), None, None);
    let chunked = model.forward(image, features, None, Some(1));

    assert_close(&values(full), &values(chunked), 1.0e-4);
}

#[test]
fn upstream_masked_attention_chunks_match() {
    let device = Default::default();
    let config =
        AnyUpConfig::tiny_for_tests().with_attention_mode(AnyUpAttentionMode::UpstreamMasked);
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);

    let full = model.forward(image.clone(), features.clone(), None, None);
    let chunked = model.forward(image, features, None, Some(1));

    assert_close(&values(full), &values(chunked), 1.0e-4);
}

#[test]
fn image_context_decode_matches_full_upsample() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);
    let encoded = model.encode_image(image);

    let direct = model.upsample(encoded.clone(), features.clone(), [8, 8], Some(1));
    let context = model.prepare_encoded_context(encoded, [8, 8], [2, 2]);
    let cached = model.upsample_with_context(&context, features, Some(1));

    assert_close(&values(direct), &values(cached), 1.0e-4);
}

#[test]
fn prepared_image_grid_matches_uncached_rope_path() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);
    let grid = model.prepare_image_grid([8, 8], &device);

    let direct = model.forward(image.clone(), features.clone(), None, Some(1));
    let context = model.prepare_image_context_with_grid(image, &grid, None, [2, 2]);
    let cached = model.upsample_with_context(&context, features, Some(1));

    assert_close(&values(direct), &values(cached), 1.0e-4);
}

#[test]
fn value_projection_commutes_with_anyup_decode() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config, &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);
    let projection = Tensor::<B, 2>::from_data(
        TensorData::new(
            vec![
                0.2, -0.4, 0.7, //
                -0.1, 0.3, 0.2, //
                0.6, 0.1, -0.5, //
                -0.3, 0.8, 0.4, //
                0.5, -0.2, 0.1,
            ],
            [5, 3],
        ),
        &device,
    );
    let mean = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.01, -0.02, 0.03, -0.04, 0.05], [1, 1, 5]),
        &device,
    );
    let context = model.prepare_image_context(image, None, [2, 2]);
    let full = model.upsample_with_context(&context, features.clone(), Some(1));
    let full_projected = project_nchw(full, projection.clone(), mean.clone());
    let low_projected = project_nchw(features.clone(), projection, mean);
    let fast_projected =
        model.upsample_values_with_context(&context, features, low_projected, Some(1));

    assert_close(&values(full_projected), &values(fast_projected), 1.0e-4);
}

#[test]
fn sparse_anyup_matches_dense_forward_on_selected_high_res_tokens() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config.clone(), &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let features = ramp([1, 5, 2, 2], &device);
    let context = model.prepare_image_context(image.clone(), None, [2, 2]);
    let plan = AnyUpSparseOutputPlan::<B>::new(
        vec![0, 5, 17, 33, 63],
        [8, 8],
        [2, 2],
        1,
        config.window_ratio,
        &device,
    )
    .expect("sparse AnyUp plan");

    let dense = model.forward(image, features.clone(), None, Some(1));
    let sparse = model
        .upsample_sparse_with_context(&context, features, &plan)
        .expect("sparse upsample");
    let dense_tokens = gather_dense_output(dense, plan.indices.clone());

    assert_eq!(sparse.features.shape().dims::<3>(), [1, 5, 5]);
    assert_close(&values3(sparse.features), &values3(dense_tokens), 1.0e-4);
}

#[test]
fn sparse_low_res_encoder_features_can_drive_sparse_high_res_anyup() {
    let device = Default::default();
    let config = AnyUpConfig::tiny_for_tests();
    let model = AnyUp::<B>::new(config.clone(), &device).expect("AnyUp model");
    let image = ramp([1, 3, 8, 8], &device);
    let dense_features = ramp([1, 5, 2, 2], &device);
    let low_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0, 1, 2, 3], [1, 4]), &device);
    let sparse_features = gather_dense_output(dense_features.clone(), low_indices.clone());
    let restored = sparse_low_features_to_nchw(
        sparse_features.clone(),
        low_indices.clone(),
        [2, 2],
        &device,
    )
    .expect("restore sparse low features");
    let context = model.prepare_image_context(image, None, [2, 2]);
    let plan = AnyUpSparseOutputPlan::<B>::new(
        vec![1, 18, 47],
        [8, 8],
        [2, 2],
        1,
        config.window_ratio,
        &device,
    )
    .expect("sparse AnyUp plan");

    let direct = model
        .upsample_sparse_with_context(&context, restored, &plan)
        .expect("direct sparse upsample");
    let from_sparse = model
        .upsample_sparse_low_features_with_context(&context, sparse_features, low_indices, &plan)
        .expect("sparse low feature upsample");

    assert_close(
        &values3(from_sparse.features),
        &values3(direct.features),
        1.0e-4,
    );
}

#[test]
fn high_res_sparse_feature_memory_updates_only_observed_pixels() {
    let device = Default::default();
    let mut memory = AnyUpHighResFeatureMemory::<B>::new(
        AnyUpHighResFeatureMemoryConfig::default(),
        1,
        [2, 3],
        2,
        &device,
    )
    .expect("AnyUp high-res memory");

    let first = memory
        .update_tokens(
            Tensor::<B, 3>::from_data(
                TensorData::new(vec![10.0, 11.0, 20.0, 21.0], [1, 2, 2]),
                &device,
            ),
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0, 4], [1, 2]), &device),
        )
        .expect("first sparse update");
    assert_eq!(first.updated_tokens, 2);
    assert_close(
        &values3(first.features),
        &[
            10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0,
        ],
        1.0e-5,
    );
    assert_close(
        &values2(first.observed),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1.0e-5,
    );

    let second = memory
        .update_tokens(
            Tensor::<B, 3>::from_data(TensorData::new(vec![30.0, 31.0], [1, 1, 2]), &device),
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![2], [1, 1]), &device),
        )
        .expect("second sparse update");
    assert_close(
        &values3(second.features),
        &[
            10.0, 11.0, 0.0, 0.0, 30.0, 31.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0,
        ],
        1.0e-5,
    );
    assert_close(
        &values2(second.age_frames),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1.0e-5,
    );
}

#[test]
fn high_res_sparse_feature_memory_write_modes_match() {
    let device = Default::default();
    let assign_config = AnyUpHighResFeatureMemoryConfig {
        update_mode: AnyUpSparseFeatureUpdateMode::Ema { alpha: 0.25 },
        write_mode: AnyUpSparseFeatureMemoryWriteMode::ScatterNdAssign,
        ..Default::default()
    };
    let delta_config = AnyUpHighResFeatureMemoryConfig {
        write_mode: AnyUpSparseFeatureMemoryWriteMode::ScatterAddDelta,
        ..assign_config
    };
    let mut assign = AnyUpHighResFeatureMemory::<B>::new(assign_config, 1, [2, 3], 2, &device)
        .expect("assign memory");
    let mut delta = AnyUpHighResFeatureMemory::<B>::new(delta_config, 1, [2, 3], 2, &device)
        .expect("delta memory");
    let first_tokens = Tensor::<B, 3>::from_data(
        TensorData::new(vec![10.0, 11.0, 20.0, 21.0], [1, 2, 2]),
        &device,
    );
    let first_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0, 4], [1, 2]), &device);
    let second_tokens = Tensor::<B, 3>::from_data(
        TensorData::new(vec![14.0, 19.0, 30.0, 31.0], [1, 2, 2]),
        &device,
    );
    let second_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0, 2], [1, 2]), &device);

    assign
        .update_tokens(first_tokens.clone(), first_indices.clone())
        .expect("assign first update");
    delta
        .update_tokens(first_tokens, first_indices)
        .expect("delta first update");
    let assign = assign
        .update_tokens(second_tokens.clone(), second_indices.clone())
        .expect("assign second update");
    let delta = delta
        .update_tokens(second_tokens, second_indices)
        .expect("delta second update");

    assert_close(&values3(delta.features), &values3(assign.features), 1.0e-5);
    assert_close(&values2(delta.observed), &values2(assign.observed), 1.0e-5);
    assert_close(
        &values2(delta.age_frames),
        &values2(assign.age_frames),
        1.0e-5,
    );
}

#[test]
fn tiny_anyup_matches_torch_fixture_and_loads_upstream_names() {
    let Some((dir, fixture)) = torch_fixture() else {
        eprintln!(
            "skipping AnyUp torch parity fixture because python3 torch/safetensors are unavailable"
        );
        return;
    };
    let device = Default::default();
    let image = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.image.clone(), fixture.image_shape),
        &device,
    );
    let features = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.features.clone(), fixture.features_shape),
        &device,
    );

    let mut efficient = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("model");
    let report = AnyUpLoadOptions::default()
        .load_into(
            &mut efficient,
            dir.path().join("efficient.safetensors"),
            &device,
        )
        .expect("load efficient safetensors");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path.contains("cross_decode.cross_attn.q_proj.weight")),
        "q projection should load: {report:?}"
    );
    let output = efficient.forward(image.clone(), features.clone(), None, None);
    assert_close(&values(output), &fixture.output, 1.0e-4);

    let mut fused = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("model");
    let report = AnyUpLoadOptions::default()
        .load_into(
            &mut fused,
            dir.path().join("paper_fused.safetensors"),
            &device,
        )
        .expect("load fused safetensors");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path == "cross_decode.cross_attn.q_proj.weight"),
        "fused q projection should be split: {report:?}"
    );
    let output = fused.forward(image.clone(), features.clone(), None, Some(1));
    assert_close(&values(output), &fixture.chunked_output, 1.0e-4);

    let mut upstream_masked = AnyUp::<B>::new(
        AnyUpConfig::tiny_for_tests().with_attention_mode(AnyUpAttentionMode::UpstreamMasked),
        &device,
    )
    .expect("model");
    let report = AnyUpLoadOptions::default()
        .load_into(
            &mut upstream_masked,
            dir.path().join("paper_fused.safetensors"),
            &device,
        )
        .expect("load upstream masked safetensors");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path == "cross_decode.cross_attn.q_proj.weight"),
        "fused q projection should be split for upstream-masked mode: {report:?}"
    );
    let output = upstream_masked.forward(image.clone(), features.clone(), None, None);
    assert_close(&values(output), &fixture.paper_output, 1.0e-4);
    let output = upstream_masked.forward(image.clone(), features.clone(), None, Some(1));
    assert_close(&values(output), &fixture.paper_chunked_output, 1.0e-4);

    let mut pth = AnyUp::<B>::new(
        AnyUpConfig::tiny_for_tests().with_attention_mode(AnyUpAttentionMode::UpstreamMasked),
        &device,
    )
    .expect("model");
    let report = AnyUpLoadOptions::default()
        .load_into(&mut pth, dir.path().join("paper_fused.pth"), &device)
        .expect("load fused pth");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path == "cross_decode.cross_attn.k_proj.bias"),
        "fused k bias should be split from pth: {report:?}"
    );
    let output = pth.forward(image, features, None, None);
    assert_close(&values(output), &fixture.paper_output, 1.0e-4);
}

#[test]
#[ignore = "downloads or reads the published upstream AnyUp checkpoint"]
fn real_multi_backbone_checkpoint_matches_torch_efficient_reference() {
    let Some(checkpoint) = real_checkpoint() else {
        eprintln!(
            "skipping real AnyUp checkpoint parity; set BURN_ANYUP_REAL_CHECKPOINT or BURN_ANYUP_DOWNLOAD_REAL=1"
        );
        return;
    };
    let (dir, fixture) = real_torch_fixture(&checkpoint).expect("real torch fixture");
    let _dir = dir;
    let device = Default::default();
    let image = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.image.clone(), fixture.image_shape),
        &device,
    );
    let features = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.features.clone(), fixture.features_shape),
        &device,
    );
    let mut model = AnyUp::<B>::new(AnyUpConfig::default(), &device).expect("AnyUp model");
    let report = AnyUpLoadOptions::default()
        .load_into(&mut model, checkpoint, &device)
        .expect("load real AnyUp checkpoint");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path == "cross_decode.cross_attn.q_proj.weight"),
        "fused q projection should be split from the real checkpoint: {report:?}"
    );
    let output = model.forward(image.clone(), features.clone(), None, Some(2));
    assert_close(&values(output), &fixture.output, 1.0e-3);
    let chunked = model.forward(image, features, None, Some(1));
    assert_close(&values(chunked), &fixture.chunked_output, 1.0e-3);
}

#[test]
#[ignore = "downloads or reads the published upstream AnyUp checkpoint"]
fn real_multi_backbone_checkpoint_matches_torch_upstream_masked_reference() {
    let Some(checkpoint) = real_checkpoint() else {
        eprintln!(
            "skipping real AnyUp checkpoint parity; set BURN_ANYUP_REAL_CHECKPOINT or BURN_ANYUP_DOWNLOAD_REAL=1"
        );
        return;
    };
    let (dir, fixture) = real_torch_fixture(&checkpoint).expect("real torch fixture");
    let _dir = dir;
    let device = Default::default();
    let image = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.image.clone(), fixture.image_shape),
        &device,
    );
    let features = Tensor::<B, 4>::from_data(
        TensorData::new(fixture.features.clone(), fixture.features_shape),
        &device,
    );
    let mut model = AnyUp::<B>::new(AnyUpConfig::upstream_masked(), &device).expect("AnyUp model");
    let report = AnyUpLoadOptions::default()
        .load_into(&mut model, checkpoint, &device)
        .expect("load real AnyUp checkpoint");
    assert!(
        report
            .applied
            .iter()
            .any(|path| path == "cross_decode.cross_attn.q_proj.weight"),
        "fused q projection should be split from the real checkpoint: {report:?}"
    );
    let output = model.forward(image.clone(), features.clone(), None, None);
    assert_close(&values(output), &fixture.paper_output, 1.0e-3);
    let chunked = model.forward(image, features, None, Some(64));
    assert_close(&values(chunked), &fixture.paper_chunked_output, 1.0e-3);
}

#[derive(Debug, Deserialize)]
struct TorchFixture {
    image: Vec<f32>,
    image_shape: [usize; 4],
    features: Vec<f32>,
    features_shape: [usize; 4],
    output: Vec<f32>,
    chunked_output: Vec<f32>,
    paper_output: Vec<f32>,
    paper_chunked_output: Vec<f32>,
    output_shape: [usize; 4],
}

fn torch_fixture() -> Option<(tempfile::TempDir, TorchFixture)> {
    let status = Command::new("python3")
        .arg("-c")
        .arg("import torch, safetensors")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let dir = tempfile::tempdir().ok()?;
    let status = Command::new("python3")
        .arg("tests/fixtures/anyup_tiny_parity.py")
        .arg(dir.path())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let text = std::fs::read_to_string(dir.path().join("fixture.json")).ok()?;
    let fixture = serde_json::from_str::<TorchFixture>(&text).ok()?;
    assert_eq!(fixture.output_shape, [1, 5, 8, 8]);
    Some((dir, fixture))
}

fn real_torch_fixture(checkpoint: &Path) -> Result<(tempfile::TempDir, TorchFixture), String> {
    let status = Command::new("python3")
        .arg("-c")
        .arg("import torch")
        .status()
        .map_err(|err| err.to_string())?;
    if !status.success() {
        return Err("python3 torch import failed".to_string());
    }
    let dir = tempfile::tempdir().map_err(|err| err.to_string())?;
    let status = Command::new("python3")
        .arg("tests/fixtures/anyup_real_parity.py")
        .arg(checkpoint)
        .arg(dir.path())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .map_err(|err| err.to_string())?;
    if !status.success() {
        return Err("real AnyUp parity fixture generation failed".to_string());
    }
    let text =
        std::fs::read_to_string(dir.path().join("fixture.json")).map_err(|err| err.to_string())?;
    let fixture = serde_json::from_str::<TorchFixture>(&text).map_err(|err| err.to_string())?;
    assert_eq!(fixture.output_shape, [1, 768, 32, 32]);
    Ok((dir, fixture))
}

fn real_checkpoint() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("BURN_ANYUP_REAL_CHECKPOINT") {
        return PathBuf::from(path).canonicalize().ok();
    }
    if std::env::var("BURN_ANYUP_DOWNLOAD_REAL").ok().as_deref() != Some("1") {
        return None;
    }
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/burn-anyup-checkpoints/anyup_multi_backbone.pth");
    if out.exists() {
        return Some(out);
    }
    std::fs::create_dir_all(out.parent()?).ok()?;
    let status = Command::new("curl")
        .arg("-L")
        .arg("--fail")
        .arg("--show-error")
        .arg("https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth")
        .arg("-o")
        .arg(&out)
        .status()
        .ok()?;
    status.success().then_some(out)
}

fn ramp(
    shape: [usize; 4],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 4> {
    let values = (0..shape.iter().product::<usize>())
        .map(|index| (index as f32).sin() * 0.01)
        .collect::<Vec<_>>();
    Tensor::<B, 4>::from_data(TensorData::new(values, shape), device)
}

fn values(tensor: Tensor<B, 4>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values2(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn project_nchw(
    tensor: Tensor<B, 4>,
    projection: Tensor<B, 2>,
    mean: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    let centered = tensor
        .permute([0, 2, 3, 1])
        .reshape([batch, height * width, channels])
        - mean.repeat_dim(0, batch).repeat_dim(1, height * width);
    centered
        .reshape([batch * height * width, channels])
        .matmul(projection)
        .reshape([batch, height, width, 3])
        .permute([0, 3, 1, 2])
}

fn gather_dense_output(tensor: Tensor<B, 4>, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    let tokens = tensor
        .permute([0, 2, 3, 1])
        .reshape([batch, height * width, channels]);
    let gather_indices = indices.unsqueeze_dim::<3>(2).repeat_dim(2, channels);
    tokens.gather(1, gather_indices)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}
