use crate::{TransformerBlock, VJepa2_1Model, VJepaConfig};
use anyhow::{Context, Result, bail, ensure};
use burn::module::Param;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{
    ApplyResult, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct VJepaLoadOptions {
    pub config_name: String,
    pub weights_name: String,
    pub allow_partial: bool,
    pub pytorch_adapter: bool,
    pub upstream_vjepa21_names: bool,
    pub pytorch_top_level_key: Option<String>,
}

impl Default for VJepaLoadOptions {
    fn default() -> Self {
        Self {
            config_name: "config.json".to_string(),
            weights_name: "model.safetensors".to_string(),
            allow_partial: true,
            pytorch_adapter: true,
            upstream_vjepa21_names: true,
            pytorch_top_level_key: None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoadReport {
    pub applied: Vec<String>,
    pub missing: Vec<String>,
    pub skipped: Vec<String>,
    pub errors: Vec<String>,
}

impl VJepaLoadOptions {
    pub fn load_model<B: Backend>(
        &self,
        dir: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<(VJepa2_1Model<B>, VJepaConfig, LoadReport)> {
        let dir = dir.as_ref();
        let config = load_config_from_hf_dir(dir, &self.config_name)?;
        let weights_path = dir.join(&self.weights_name);
        if !weights_path.exists() {
            bail!("missing V-JEPA weights {}", weights_path.display());
        }
        let mut model = VJepa2_1Model::new(&config, device);
        let force_partial_for_hf_safetensors = self.upstream_vjepa21_names
            && is_hf_vjepa2_config(&config)
            && !matches!(
                weights_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or_default(),
                "pt" | "pth"
            );
        let result = match weights_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
        {
            "pt" | "pth" => {
                let mut store =
                    PytorchStore::from_file(&weights_path).allow_partial(self.allow_partial);
                if let Some(key) = &self.pytorch_top_level_key {
                    store = store.with_top_level_key(key);
                }
                if self.upstream_vjepa21_names {
                    store = apply_upstream_key_remapping_pytorch(store);
                }
                let result = model
                    .load_from(&mut store)
                    .with_context(|| format!("load weights from {}", weights_path.display()))?;
                if self.upstream_vjepa21_names
                    && self.pytorch_top_level_key.is_none()
                    && result.applied.is_empty()
                {
                    load_nested_upstream_vjepa21_pytorch(
                        &mut model,
                        &weights_path,
                        self.allow_partial,
                    )
                    .with_context(|| {
                        format!(
                            "load nested upstream V-JEPA checkpoint from {}",
                            weights_path.display()
                        )
                    })?
                } else {
                    result
                }
            }
            _ => {
                let mut store = SafetensorsStore::from_file(&weights_path)
                    .allow_partial(self.allow_partial || force_partial_for_hf_safetensors);
                if self.pytorch_adapter {
                    store = store.with_from_adapter(PyTorchToBurnAdapter);
                }
                if self.upstream_vjepa21_names {
                    store = apply_upstream_key_remapping_safetensors(store);
                }
                model
                    .load_from(&mut store)
                    .with_context(|| format!("load weights from {}", weights_path.display()))?
            }
        };
        let mut report = LoadReport {
            applied: result.applied,
            missing: result
                .missing
                .into_iter()
                .map(|(path, reason)| format!("{path}: {reason}"))
                .collect(),
            skipped: result.skipped,
            errors: result
                .errors
                .into_iter()
                .map(|err| format!("{err:?}"))
                .collect(),
        };
        if force_partial_for_hf_safetensors {
            apply_hf_vjepa2_fused_safetensors(
                &mut model,
                &config,
                &weights_path,
                device,
                &mut report,
            )
            .with_context(|| {
                format!(
                    "apply fused HF V-JEPA tensors from {}",
                    weights_path.display()
                )
            })?;
            if !self.allow_partial && !report.missing.is_empty() {
                bail!(
                    "missing V-JEPA tensors after HF adapter: {}",
                    report.missing.join(", ")
                );
            }
        }
        Ok((model, config, report))
    }
}

fn load_nested_upstream_vjepa21_pytorch<B: Backend>(
    model: &mut VJepa2_1Model<B>,
    weights_path: &Path,
    allow_partial: bool,
) -> Result<ApplyResult> {
    let mut encoder_store = PytorchStore::from_file(weights_path)
        .allow_partial(true)
        .with_top_level_key("ema_encoder");
    encoder_store = apply_upstream_nested_encoder_key_remapping_pytorch(encoder_store);
    let mut result = model
        .load_from(&mut encoder_store)
        .context("load upstream ema_encoder")?;
    if result.applied.is_empty() {
        let mut encoder_store = PytorchStore::from_file(weights_path)
            .allow_partial(true)
            .with_top_level_key("encoder");
        encoder_store = apply_upstream_nested_encoder_key_remapping_pytorch(encoder_store);
        result = model
            .load_from(&mut encoder_store)
            .context("load upstream encoder")?;
    }
    if !allow_partial && !result.missing.is_empty() {
        bail!(
            "nested upstream V-JEPA checkpoint did not fully load; missing {} tensors",
            result.missing.len()
        );
    }
    zero_encoder_modality_embeddings(model);
    Ok(result)
}

fn zero_encoder_modality_embeddings<B: Backend>(model: &mut VJepa2_1Model<B>) {
    // Official nested Meta checkpoints store these as [1, 1, D], while this
    // port stores [1, D]. Keep skipped modality embeddings deterministic.
    let video = model.encoder.video_mod_embed.val();
    let [video_rows, video_dim] = video.shape().dims::<2>();
    let video_device = video.device();
    model.encoder.video_mod_embed = Param::from_tensor(Tensor::<B, 2>::zeros(
        [video_rows, video_dim],
        &video_device,
    ));

    let image = model.encoder.image_mod_embed.val();
    let [image_rows, image_dim] = image.shape().dims::<2>();
    let image_device = image.device();
    model.encoder.image_mod_embed = Param::from_tensor(Tensor::<B, 2>::zeros(
        [image_rows, image_dim],
        &image_device,
    ));
}

fn is_hf_vjepa2_config(config: &VJepaConfig) -> bool {
    config.model_type == "vjepa2"
}

fn apply_hf_vjepa2_fused_safetensors<B: Backend>(
    model: &mut VJepa2_1Model<B>,
    config: &VJepaConfig,
    weights_path: &Path,
    device: &B::Device,
    report: &mut LoadReport,
) -> Result<()> {
    let bytes =
        std::fs::read(weights_path).with_context(|| format!("read {}", weights_path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", weights_path.display()))?;
    let mut satisfied = BTreeSet::new();

    set_conv3d(
        &mut model.encoder.image_patch_embed.proj,
        &tensors,
        "encoder.embeddings.patch_embeddings.proj",
        device,
        &mut report.applied,
    )?;
    satisfied.insert("encoder.image_patch_embed.proj.weight".to_string());
    satisfied.insert("encoder.image_patch_embed.proj.bias".to_string());

    set_param2(
        &mut model.encoder.video_mod_embed,
        vec![0.0; config.encoder.embed_dim],
        [1, config.encoder.embed_dim],
        device,
    );
    set_param2(
        &mut model.encoder.image_mod_embed,
        vec![0.0; config.encoder.embed_dim],
        [1, config.encoder.embed_dim],
        device,
    );
    report
        .applied
        .push("encoder.video_mod_embed (zero HF compatibility)".to_string());
    report
        .applied
        .push("encoder.image_mod_embed (zero HF compatibility)".to_string());
    satisfied.insert("encoder.video_mod_embed".to_string());
    satisfied.insert("encoder.image_mod_embed".to_string());

    for (index, block) in model.encoder.blocks.iter_mut().enumerate() {
        set_qkv(
            block,
            &tensors,
            &format!("encoder.layer.{index}.attention"),
            &format!("encoder.blocks.{index}.attn.qkv"),
            config.encoder.embed_dim,
            device,
            &mut report.applied,
        )?;
        satisfied.insert(format!("encoder.blocks.{index}.attn.qkv.weight"));
        satisfied.insert(format!("encoder.blocks.{index}.attn.qkv.bias"));
    }
    for (index, block) in model.predictor.blocks.iter_mut().enumerate() {
        set_qkv(
            block,
            &tensors,
            &format!("predictor.layer.{index}.attention"),
            &format!("predictor.blocks.{index}.attn.qkv"),
            config.predictor.embed_dim,
            device,
            &mut report.applied,
        )?;
        satisfied.insert(format!("predictor.blocks.{index}.attn.qkv.weight"));
        satisfied.insert(format!("predictor.blocks.{index}.attn.qkv.bias"));
    }
    set_mask_tokens(
        &mut model.predictor.mask_tokens,
        &tensors,
        config.predictor.embed_dim,
        device,
        &mut report.applied,
    )?;
    for index in 0..model.predictor.mask_tokens.len() {
        satisfied.insert(format!("predictor.mask_tokens.{index}"));
    }

    report.missing.retain(|missing| {
        let path = missing
            .split_once(':')
            .map(|(path, _)| path)
            .unwrap_or(missing.as_str());
        !satisfied.contains(path)
    });
    Ok(())
}

fn set_qkv<B: Backend>(
    block: &mut TransformerBlock<B>,
    tensors: &SafeTensors<'_>,
    hf_prefix: &str,
    burn_prefix: &str,
    dim: usize,
    device: &B::Device,
    applied: &mut Vec<String>,
) -> Result<()> {
    let q_weight = tensor_values(tensors, &format!("{hf_prefix}.query.weight"), &[dim, dim])?;
    let k_weight = tensor_values(tensors, &format!("{hf_prefix}.key.weight"), &[dim, dim])?;
    let v_weight = tensor_values(tensors, &format!("{hf_prefix}.value.weight"), &[dim, dim])?;
    let mut fused_weight = Vec::with_capacity(dim * dim * 3);
    fused_weight.extend(q_weight);
    fused_weight.extend(k_weight);
    fused_weight.extend(v_weight);
    set_linear_weight(
        &mut block.attn.qkv,
        transpose_2d(&fused_weight, dim * 3, dim),
        [dim, dim * 3],
        device,
    );

    let q_bias = tensor_values(tensors, &format!("{hf_prefix}.query.bias"), &[dim])?;
    let k_bias = tensor_values(tensors, &format!("{hf_prefix}.key.bias"), &[dim])?;
    let v_bias = tensor_values(tensors, &format!("{hf_prefix}.value.bias"), &[dim])?;
    let mut fused_bias = Vec::with_capacity(dim * 3);
    fused_bias.extend(q_bias);
    fused_bias.extend(k_bias);
    fused_bias.extend(v_bias);
    set_linear_bias(&mut block.attn.qkv, fused_bias, [dim * 3], device);

    applied.push(format!("{burn_prefix}.weight (fused HF q/k/v)"));
    applied.push(format!("{burn_prefix}.bias (fused HF q/k/v)"));
    Ok(())
}

fn set_conv3d<B: Backend>(
    conv: &mut burn::nn::conv::Conv3d<B>,
    tensors: &SafeTensors<'_>,
    prefix: &str,
    device: &B::Device,
    applied: &mut Vec<String>,
) -> Result<()> {
    let weight_view = tensors
        .tensor(&format!("{prefix}.weight"))
        .with_context(|| format!("read {prefix}.weight"))?;
    let shape = tensor_shape(&weight_view);
    ensure!(shape.len() == 5, "{prefix}.weight must be rank 5");
    set_param5(
        &mut conv.weight,
        f32_values(&weight_view)?,
        [shape[0], shape[1], shape[2], shape[3], shape[4]],
        device,
    );
    let bias = tensor_values(tensors, &format!("{prefix}.bias"), &[shape[0]])?;
    if let Some(conv_bias) = &mut conv.bias {
        set_param1(conv_bias, bias, [shape[0]], device);
    }
    applied.push("encoder.image_patch_embed.proj.weight (copied HF patch embed)".to_string());
    applied.push("encoder.image_patch_embed.proj.bias (copied HF patch embed)".to_string());
    Ok(())
}

fn set_mask_tokens<B: Backend>(
    mask_tokens: &mut [Param<Tensor<B, 2>>],
    tensors: &SafeTensors<'_>,
    pred_dim: usize,
    device: &B::Device,
    applied: &mut Vec<String>,
) -> Result<()> {
    let view = tensors
        .tensor("predictor.embeddings.mask_tokens")
        .context("read predictor.embeddings.mask_tokens")?;
    let shape = tensor_shape(&view);
    ensure!(
        shape.as_slice() == [mask_tokens.len(), 1, 1, pred_dim],
        "unexpected predictor mask token shape: {:?}",
        shape
    );
    let values = f32_values(&view)?;
    for (index, token) in mask_tokens.iter_mut().enumerate() {
        let start = index * pred_dim;
        let end = start + pred_dim;
        set_param2(token, values[start..end].to_vec(), [1, pred_dim], device);
        applied.push(format!(
            "predictor.mask_tokens.{index} (sliced HF mask_tokens)"
        ));
    }
    Ok(())
}

fn set_linear_weight<B: Backend>(
    linear: &mut Linear<B>,
    values: Vec<f32>,
    shape: [usize; 2],
    device: &B::Device,
) {
    set_param2(&mut linear.weight, values, shape, device);
}

fn set_linear_bias<B: Backend>(
    linear: &mut Linear<B>,
    values: Vec<f32>,
    shape: [usize; 1],
    device: &B::Device,
) {
    if let Some(bias) = &mut linear.bias {
        set_param1(bias, values, shape, device);
    }
}

fn set_param1<B: Backend>(
    param: &mut Param<Tensor<B, 1>>,
    values: Vec<f32>,
    shape: [usize; 1],
    device: &B::Device,
) {
    *param = Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), device));
}

fn set_param2<B: Backend>(
    param: &mut Param<Tensor<B, 2>>,
    values: Vec<f32>,
    shape: [usize; 2],
    device: &B::Device,
) {
    *param = Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), device));
}

fn set_param5<B: Backend>(
    param: &mut Param<Tensor<B, 5>>,
    values: Vec<f32>,
    shape: [usize; 5],
    device: &B::Device,
) {
    *param = Param::from_tensor(Tensor::from_data(TensorData::new(values, shape), device));
}

fn tensor_values(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("read tensor {name}"))?;
    let shape = tensor_shape(&view);
    ensure!(
        shape.as_slice() == expected_shape,
        "tensor {name} shape mismatch: got {:?}, expected {:?}",
        shape,
        expected_shape
    );
    f32_values(&view)
}

fn tensor_shape(view: &safetensors::tensor::TensorView<'_>) -> Vec<usize> {
    view.shape().to_vec()
}

fn f32_values(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    ensure!(
        view.dtype() == Dtype::F32,
        "only f32 V-JEPA safetensors are supported, got {:?}",
        view.dtype()
    );
    Ok(view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn transpose_2d(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

fn apply_upstream_key_remapping_safetensors(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"^module\.", "")
        .with_key_remapping(r"^ema_encoder\.", "encoder.")
        .with_key_remapping(r"^target_encoder\.", "encoder.")
        .with_key_remapping(r"^backbone\.", "encoder.")
        .with_key_remapping(r"^predictor\.", "predictor.")
        .with_key_remapping(
            r"^encoder\.embeddings\.patch_embeddings\.",
            "encoder.patch_embed.",
        )
        .with_key_remapping(r"^encoder\.layer\.(\d+)\.", "encoder.blocks.$1.")
        .with_key_remapping(r"^encoder\.layernorm\.", "encoder.norms_block.0.")
        .with_key_remapping(
            r"^predictor\.embeddings\.predictor_embeddings\.",
            "predictor.predictor_embed.",
        )
        .with_key_remapping(r"^predictor\.layer\.(\d+)\.", "predictor.blocks.$1.")
        .with_key_remapping(r"^predictor\.layernorm\.", "predictor.norm.")
        .with_key_remapping(r"^predictor\.proj\.", "predictor.target_proj.")
        .with_key_remapping(r"\.attention\.proj\.", ".attn.proj.")
}

fn apply_upstream_key_remapping_pytorch(store: PytorchStore) -> PytorchStore {
    store
        .with_key_remapping(r"^module\.", "")
        .with_key_remapping(r"^ema_encoder\.", "encoder.")
        .with_key_remapping(r"^target_encoder\.", "encoder.")
        .with_key_remapping(r"^backbone\.", "encoder.")
        .with_key_remapping(r"^predictor\.", "predictor.")
        .with_key_remapping(
            r"^encoder\.embeddings\.patch_embeddings\.",
            "encoder.patch_embed.",
        )
        .with_key_remapping(r"^encoder\.layer\.(\d+)\.", "encoder.blocks.$1.")
        .with_key_remapping(r"^encoder\.layernorm\.", "encoder.norms_block.0.")
        .with_key_remapping(
            r"^predictor\.embeddings\.predictor_embeddings\.",
            "predictor.predictor_embed.",
        )
        .with_key_remapping(r"^predictor\.layer\.(\d+)\.", "predictor.blocks.$1.")
        .with_key_remapping(r"^predictor\.layernorm\.", "predictor.norm.")
        .with_key_remapping(r"^predictor\.proj\.", "predictor.target_proj.")
        .with_key_remapping(r"\.attention\.proj\.", ".attn.proj.")
}

fn apply_upstream_nested_encoder_key_remapping_pytorch(store: PytorchStore) -> PytorchStore {
    store
        .with_key_remapping(
            r"^module\.backbone\.patch_embed_img\.",
            "encoder.image_patch_embed.",
        )
        .with_key_remapping(r"^module\.backbone\.patch_embed\.", "encoder.patch_embed.")
        .with_key_remapping(r"^module\.backbone\.blocks\.(\d+)\.", "encoder.blocks.$1.")
        .with_key_remapping(
            r"^module\.backbone\.norms_block\.(\d+)\.",
            "encoder.norms_block.$1.",
        )
        .with_key_remapping(r"\.norm([12])\.weight$", ".norm$1.gamma")
        .with_key_remapping(r"\.norm([12])\.bias$", ".norm$1.beta")
        .with_key_remapping(r"\.norms_block\.(\d+)\.weight$", ".norms_block.$1.gamma")
        .with_key_remapping(r"\.norms_block\.(\d+)\.bias$", ".norms_block.$1.beta")
}

pub fn load_config_from_hf_dir(dir: impl AsRef<Path>, name: &str) -> Result<VJepaConfig> {
    let path = dir.as_ref().join(name);
    VJepaConfig::from_json_file(&path).with_context(|| format!("load {}", path.display()))
}

pub fn checkpoint_tensor_prefixes(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let bytes = std::fs::read(path.as_ref())
        .with_context(|| format!("read {}", path.as_ref().display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", path.as_ref().display()))?;
    let mut prefixes = tensors
        .names()
        .iter()
        .map(|name| name.split('.').next().unwrap_or(name).to_string())
        .collect::<Vec<_>>();
    prefixes.sort();
    prefixes.dedup();
    Ok(prefixes)
}

pub fn default_hf_snapshot_dir() -> PathBuf {
    PathBuf::from("/home/mosure/.cache/huggingface/hub/models--facebook--vjepa2/snapshots")
}
