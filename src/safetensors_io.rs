use crate::{VJepa2_1Model, VJepaConfig};
use anyhow::{Context, Result, bail};
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore};
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
                model
                    .load_from(&mut store)
                    .with_context(|| format!("load weights from {}", weights_path.display()))?
            }
            _ => {
                let mut store =
                    SafetensorsStore::from_file(&weights_path).allow_partial(self.allow_partial);
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
        let report = LoadReport {
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
        Ok((model, config, report))
    }
}

fn apply_upstream_key_remapping_safetensors(store: SafetensorsStore) -> SafetensorsStore {
    store
        .with_key_remapping(r"^module\.", "")
        .with_key_remapping(r"^ema_encoder\.", "encoder.")
        .with_key_remapping(r"^target_encoder\.", "encoder.")
        .with_key_remapping(r"^backbone\.", "encoder.")
        .with_key_remapping(r"^predictor\.", "predictor.")
}

fn apply_upstream_key_remapping_pytorch(store: PytorchStore) -> PytorchStore {
    store
        .with_key_remapping(r"^module\.", "")
        .with_key_remapping(r"^ema_encoder\.", "encoder.")
        .with_key_remapping(r"^target_encoder\.", "encoder.")
        .with_key_remapping(r"^backbone\.", "encoder.")
        .with_key_remapping(r"^predictor\.", "predictor.")
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
