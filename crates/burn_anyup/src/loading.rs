use crate::model::AnyUp;
use anyhow::{Context, Result, bail, ensure};
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_store::{
    ApplyResult, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
    TensorSnapshot,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AnyUpLoadOptions {
    pub weights_name: String,
    pub allow_partial: bool,
    pub pytorch_adapter: bool,
    pub pytorch_top_level_key: Option<String>,
    pub upstream_anyup_names: bool,
}

impl Default for AnyUpLoadOptions {
    fn default() -> Self {
        Self {
            weights_name: "model.safetensors".to_string(),
            allow_partial: true,
            pytorch_adapter: true,
            pytorch_top_level_key: None,
            upstream_anyup_names: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnyUpLoadReport {
    pub applied: Vec<String>,
    pub missing: Vec<String>,
    pub skipped: Vec<String>,
    pub errors: Vec<String>,
}

impl AnyUpLoadOptions {
    pub fn load_into<B: Backend>(
        &self,
        model: &mut AnyUp<B>,
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<AnyUpLoadReport> {
        let path = resolve_weights_path(path.as_ref(), &self.weights_name);
        ensure!(path.exists(), "missing AnyUp weights {}", path.display());
        let result = match path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
        {
            "pt" | "pth" => {
                let mut raw_store = PytorchStore::from_file(&path).allow_partial(true);
                if let Some(key) = &self.pytorch_top_level_key {
                    raw_store = raw_store.with_top_level_key(key);
                }
                let qk_result = load_fused_qk_if_present(model, &mut raw_store, device)
                    .with_context(|| {
                        format!("split AnyUp fused q/k tensors in {}", path.display())
                    })?;

                let mut store = PytorchStore::from_file(&path).allow_partial(self.allow_partial);
                if let Some(key) = &self.pytorch_top_level_key {
                    store = store.with_top_level_key(key);
                }
                if self.upstream_anyup_names {
                    store = remap_anyup_pytorch(store);
                }
                let mut result = model
                    .load_from(&mut store)
                    .with_context(|| format!("load AnyUp weights from {}", path.display()))?;
                result.applied.extend(qk_result);
                result
            }
            _ => {
                let mut raw_store = SafetensorsStore::from_file(&path).allow_partial(true);
                let qk_result = load_fused_qk_if_present(model, &mut raw_store, device)
                    .with_context(|| {
                        format!("split AnyUp fused q/k tensors in {}", path.display())
                    })?;

                let mut store =
                    SafetensorsStore::from_file(&path).allow_partial(self.allow_partial);
                if self.pytorch_adapter {
                    store = store.with_from_adapter(PyTorchToBurnAdapter);
                }
                if self.upstream_anyup_names {
                    store = remap_anyup_safetensors(store);
                }
                let mut result = model
                    .load_from(&mut store)
                    .with_context(|| format!("load AnyUp weights from {}", path.display()))?;
                result.applied.extend(qk_result);
                result
            }
        };
        Ok(report_from_apply(result))
    }
}

fn resolve_weights_path(path: &Path, weights_name: &str) -> PathBuf {
    if path.is_dir() {
        path.join(weights_name)
    } else {
        path.to_path_buf()
    }
}

fn report_from_apply(result: ApplyResult) -> AnyUpLoadReport {
    AnyUpLoadReport {
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
    }
}

fn remap_anyup_pytorch(store: PytorchStore) -> PytorchStore {
    remap_anyup_store(store)
}

fn remap_anyup_safetensors(store: SafetensorsStore) -> SafetensorsStore {
    remap_anyup_store(store)
}

trait AnyUpRemap: Sized {
    fn remap_key(self, from: &str, to: &str) -> Self;
}

impl AnyUpRemap for PytorchStore {
    fn remap_key(self, from: &str, to: &str) -> Self {
        self.with_key_remapping(from, to)
    }
}

impl AnyUpRemap for SafetensorsStore {
    fn remap_key(self, from: &str, to: &str) -> Self {
        self.with_key_remapping(from, to)
    }
}

fn remap_anyup_store<S: AnyUpRemap>(mut store: S) -> S {
    for stem in [
        "image_encoder",
        "key_encoder",
        "query_encoder",
        "aggregation",
    ] {
        store = store
            .remap_key(&format!(r"^{stem}\.0\."), &format!("{stem}.pre."))
            .remap_key(&format!(r"^{stem}\.1\."), &format!("{stem}.blocks.0."))
            .remap_key(&format!(r"^{stem}\.2\."), &format!("{stem}.blocks.1."));
    }
    store = store
        .remap_key(r"^key_features_encoder\.0\.", "key_features_encoder.pre.")
        .remap_key(
            r"^key_features_encoder\.1\.",
            "key_features_encoder.blocks.0.",
        )
        .remap_key(
            r"^key_features_encoder\.2\.",
            "key_features_encoder.blocks.1.",
        )
        .remap_key(r"\.block\.0\.weight$", ".norm1.gamma")
        .remap_key(r"\.block\.0\.bias$", ".norm1.beta")
        .remap_key(r"\.block\.2\.weight$", ".conv1.weight")
        .remap_key(r"\.block\.3\.weight$", ".norm2.gamma")
        .remap_key(r"\.block\.3\.bias$", ".norm2.beta")
        .remap_key(r"\.block\.5\.weight$", ".conv2.weight")
        .remap_key(r"^cross_decode\.conv2d\.", "cross_decode.conv.")
        .remap_key(
            r"^cross_decode\.norm_q\.weight$",
            "cross_decode.norm_q.gamma",
        )
        .remap_key(
            r"^cross_decode\.norm_k\.weight$",
            "cross_decode.norm_k.gamma",
        )
        .remap_key(
            r"^cross_decode\.cross_attn\.norm_q\.weight$",
            "cross_decode.norm_q.gamma",
        )
        .remap_key(
            r"^cross_decode\.cross_attn\.norm_k\.weight$",
            "cross_decode.norm_k.gamma",
        );
    store
}

fn load_fused_qk_if_present<B, S>(
    model: &mut AnyUp<B>,
    store: &mut S,
    device: &B::Device,
) -> Result<Vec<String>>
where
    B: Backend,
    S: ModuleStore,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let Some(weight_snapshot) = store
        .get_snapshot("cross_decode.cross_attn.attention.in_proj_weight")?
        .cloned()
    else {
        return Ok(Vec::new());
    };
    let bias_snapshot = store
        .get_snapshot("cross_decode.cross_attn.attention.in_proj_bias")?
        .cloned();
    let qk_dim = model.config.qk_dim;
    let shape = weight_snapshot.shape.clone();
    ensure!(
        shape.len() == 2 && shape[0] >= 2 * qk_dim && shape[1] == qk_dim,
        "AnyUp fused q/k weight has shape {:?}, expected [{}, {}]",
        shape,
        3 * qk_dim,
        qk_dim
    );
    let values = tensor_values(&weight_snapshot)?;
    let q_weight = values[0..qk_dim * qk_dim].to_vec();
    let k_weight = values[qk_dim * qk_dim..2 * qk_dim * qk_dim].to_vec();
    model.cross_decode.cross_attn.q_proj.weight = Param::from_tensor(Tensor::<B, 4>::from_data(
        TensorData::new(q_weight, [qk_dim, qk_dim, 1, 1]),
        device,
    ));
    model.cross_decode.cross_attn.k_proj.weight = Param::from_tensor(Tensor::<B, 4>::from_data(
        TensorData::new(k_weight, [qk_dim, qk_dim, 1, 1]),
        device,
    ));

    let mut applied = vec![
        "cross_decode.cross_attn.q_proj.weight".to_string(),
        "cross_decode.cross_attn.k_proj.weight".to_string(),
    ];
    if let Some(snapshot) = bias_snapshot {
        let shape = snapshot.shape.clone();
        if shape.len() != 1 || shape[0] < 2 * qk_dim {
            bail!(
                "AnyUp fused q/k bias has shape {:?}, expected [{}]",
                shape,
                3 * qk_dim
            );
        }
        let values = tensor_values(&snapshot)?;
        model.cross_decode.cross_attn.q_proj.bias =
            Some(Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(values[0..qk_dim].to_vec(), [qk_dim]),
                device,
            )));
        model.cross_decode.cross_attn.k_proj.bias =
            Some(Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(values[qk_dim..2 * qk_dim].to_vec(), [qk_dim]),
                device,
            )));
        applied.push("cross_decode.cross_attn.q_proj.bias".to_string());
        applied.push("cross_decode.cross_attn.k_proj.bias".to_string());
    }
    Ok(applied)
}

fn tensor_values(snapshot: &TensorSnapshot) -> Result<Vec<f32>> {
    snapshot
        .to_data()
        .context("materialize tensor snapshot")?
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("{err:?}"))
}
