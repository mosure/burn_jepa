use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::time::UNIX_EPOCH;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Bytes, DType, FloatDType, Tensor};
use burn_store::{
    ApplyResult, BurnpackStore, ModuleAdapter, ModuleSnapshot, ModuleStore, TensorSnapshot,
};
use ciborium::Value;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{AnyUp, AnyUpConfig, TttEncoderConfig, VJepa2_1Model, VJepaConfig, VJepaTttModel};

pub const DEFAULT_BURN_JEPA_MODEL_ROOT_URL: &str = "https://aberration.technology/model/burn_jepa";
pub const DEFAULT_BURN_JEPA_MODEL_BASE_URL: &str =
    "https://aberration.technology/model/burn_jepa/vjepa2_1_ttt";
pub const DEFAULT_BURN_ANYUP_MODEL_ROOT_URL: &str =
    "https://aberration.technology/model/burn_anyup";
pub const DEFAULT_BURN_ANYUP_MODEL_BASE_URL: &str =
    "https://aberration.technology/model/burn_anyup/anyup_multi_backbone";
pub const DEFAULT_BURN_ANYUP_CHECKPOINT_PATH: &str =
    "target/burn-anyup-checkpoints/anyup_multi_backbone.pth";
pub const DEFAULT_BURNPACK_SHARD_MAX_BYTES: u64 = 20 * 1024 * 1024;
pub const DEFAULT_BURN_JEPA_MODEL_CACHE_ROOT_DIR: &str = ".burn_jepa";
pub const DEFAULT_BURN_JEPA_MODEL_CACHE_SUBDIR: &str = "models/burn_jepa";
pub const DEFAULT_BURN_ANYUP_MODEL_CACHE_SUBDIR: &str = "models/burn_anyup";
const BURNPACK_HEADER_SIZE: usize = 10;
const BURNPACK_MAGIC_NUMBER: u32 = 0x4255_524E;
const BURNPACK_TENSOR_ALIGNMENT: u64 = 256;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_ATTEMPTS: u32 = 4;
#[cfg(not(target_arch = "wasm32"))]
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
#[cfg(not(target_arch = "wasm32"))]
const READ_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(not(target_arch = "wasm32"))]
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BurnJepaPackageModelKind {
    #[default]
    Base,
    Ttt,
}

impl BurnJepaPackageModelKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Ttt => "ttt",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum BurnJepaModelProfile {
    #[serde(rename = "vjepa2_1_base", alias = "vjepa21_base", alias = "base")]
    Vjepa21Base,
    #[default]
    #[serde(rename = "vjepa2_1_ttt", alias = "vjepa21_ttt", alias = "ttt")]
    Vjepa21Ttt,
}

impl BurnJepaModelProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vjepa21Base => "vjepa2_1_base",
            Self::Vjepa21Ttt => "vjepa2_1_ttt",
        }
    }

    pub const fn model_kind(self) -> BurnJepaPackageModelKind {
        match self {
            Self::Vjepa21Base => BurnJepaPackageModelKind::Base,
            Self::Vjepa21Ttt => BurnJepaPackageModelKind::Ttt,
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "vjepa2_1_ttt",
            "vjepa2.1-ttt",
            "vjepa21-ttt",
            "ttt",
            "vjepa2_1_base",
            "vjepa2.1-base",
            "vjepa21-base",
            "base",
        ]
    }
}

impl fmt::Display for BurnJepaModelProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BurnJepaModelProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "vjepa2_1_base"
            | "vjepa2.1-base"
            | "vjepa21-base"
            | "vjepa2-base"
            | "base"
            | "vjepa-base"
            | "vjepa2_1_vit_base_384"
            | "vjepa2_1_vitb" => Ok(Self::Vjepa21Base),
            "vjepa2_1_ttt" | "vjepa2.1-ttt" | "vjepa21-ttt" | "vjepa2-ttt" | "ttt"
            | "trained-ttt" | "production-ttt" | "vjepa-ttt" => Ok(Self::Vjepa21Ttt),
            other => Err(format!(
                "unsupported burn_jepa model profile `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

pub fn burn_jepa_model_profile_base_url(profile: BurnJepaModelProfile) -> String {
    format!(
        "{}/{}",
        DEFAULT_BURN_JEPA_MODEL_ROOT_URL.trim_end_matches('/'),
        profile.as_str()
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum BurnAnyUpModelProfile {
    #[default]
    #[serde(
        rename = "anyup_multi_backbone",
        alias = "multi_backbone",
        alias = "multi-backbone",
        alias = "default",
        alias = "paper"
    )]
    AnyupMultiBackbone,
}

impl BurnAnyUpModelProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnyupMultiBackbone => "anyup_multi_backbone",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "anyup_multi_backbone",
            "anyup-multi-backbone",
            "multi_backbone",
            "multi-backbone",
            "default",
            "paper",
        ]
    }
}

impl fmt::Display for BurnAnyUpModelProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BurnAnyUpModelProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anyup_multi_backbone"
            | "anyup-multi-backbone"
            | "multi_backbone"
            | "multi-backbone"
            | "default"
            | "paper" => Ok(Self::AnyupMultiBackbone),
            other => Err(format!(
                "unsupported burn_anyup model profile `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

pub fn burn_anyup_model_profile_base_url(profile: BurnAnyUpModelProfile) -> String {
    format!(
        "{}/{}",
        DEFAULT_BURN_ANYUP_MODEL_ROOT_URL.trim_end_matches('/'),
        profile.as_str()
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BurnJepaPipelinePackageManifest {
    pub version: u32,
    pub model_kind: BurnJepaPackageModelKind,
    pub record_dtype: Option<String>,
    pub burnpack: String,
    pub parts_manifest: String,
    pub model_base_url: String,
    pub jepa_config: VJepaConfig,
    pub ttt_config: Option<TttEncoderConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BurnAnyUpPackageManifest {
    pub version: u32,
    pub record_dtype: Option<String>,
    pub burnpack: String,
    pub parts_manifest: String,
    pub model_base_url: String,
    pub anyup_config: AnyUpConfig,
}

impl Default for BurnAnyUpPackageManifest {
    fn default() -> Self {
        Self {
            version: 1,
            record_dtype: None,
            burnpack: "anyup.bpk".to_string(),
            parts_manifest: "anyup.bpk.parts.json".to_string(),
            model_base_url: DEFAULT_BURN_ANYUP_MODEL_BASE_URL.to_string(),
            anyup_config: AnyUpConfig::default(),
        }
    }
}

impl BurnAnyUpPackageManifest {
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse burn_anyup package manifest")
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serialize burn_anyup package manifest")
    }

    pub fn with_burnpack_paths(mut self, burnpack_path: &Path) -> Self {
        self.burnpack = burnpack_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("anyup.bpk")
            .to_string();
        self.parts_manifest = burnpack_parts_manifest_path(burnpack_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("anyup.bpk.parts.json")
            .to_string();
        self
    }
}

impl Default for BurnJepaPipelinePackageManifest {
    fn default() -> Self {
        Self {
            version: 1,
            model_kind: BurnJepaPackageModelKind::Base,
            record_dtype: None,
            burnpack: "jepa.bpk".to_string(),
            parts_manifest: "jepa.bpk.parts.json".to_string(),
            model_base_url: DEFAULT_BURN_JEPA_MODEL_BASE_URL.to_string(),
            jepa_config: VJepaConfig::default(),
            ttt_config: None,
        }
    }
}

impl BurnJepaPipelinePackageManifest {
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse burn_jepa package manifest")
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serialize burn_jepa package manifest")
    }

    pub fn with_burnpack_paths(mut self, burnpack_path: &Path) -> Self {
        self.burnpack = burnpack_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("jepa.bpk")
            .to_string();
        self.parts_manifest = burnpack_parts_manifest_path(burnpack_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("jepa.bpk.parts.json")
            .to_string();
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BurnpackPartsManifest {
    pub version: u32,
    pub source_file: String,
    pub source_modified_unix_ms: u64,
    pub total_bytes: u64,
    pub max_part_bytes: u64,
    pub parts: Vec<BurnpackPartEntry>,
}

impl Default for BurnpackPartsManifest {
    fn default() -> Self {
        Self {
            version: 1,
            source_file: String::new(),
            source_modified_unix_ms: 0,
            total_bytes: 0,
            max_part_bytes: DEFAULT_BURNPACK_SHARD_MAX_BYTES,
            parts: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BurnpackPartEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub tensors: usize,
}

#[derive(Clone, Debug)]
pub struct BurnpackPartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnJepaModelBootstrapConfig {
    pub cache_root: Option<PathBuf>,
    pub model_profile: BurnJepaModelProfile,
    pub model_base_url: String,
    pub manifest_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnAnyUpModelBootstrapConfig {
    pub cache_root: Option<PathBuf>,
    pub model_profile: BurnAnyUpModelProfile,
    pub model_base_url: String,
    pub manifest_url: Option<String>,
}

impl Default for BurnAnyUpModelBootstrapConfig {
    fn default() -> Self {
        let model_profile = BurnAnyUpModelProfile::default();
        Self {
            cache_root: None,
            model_profile,
            model_base_url: burn_anyup_model_profile_base_url(model_profile),
            manifest_url: None,
        }
    }
}

impl BurnAnyUpModelBootstrapConfig {
    pub fn for_profile(model_profile: BurnAnyUpModelProfile) -> Self {
        Self {
            model_profile,
            model_base_url: burn_anyup_model_profile_base_url(model_profile),
            ..Self::default()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_env_overrides(self) -> Self {
        apply_anyup_bootstrap_env_overrides(self)
    }
}

impl Default for BurnJepaModelBootstrapConfig {
    fn default() -> Self {
        let model_profile = BurnJepaModelProfile::default();
        Self {
            cache_root: None,
            model_profile,
            model_base_url: burn_jepa_model_profile_base_url(model_profile),
            manifest_url: None,
        }
    }
}

impl BurnJepaModelBootstrapConfig {
    pub fn for_profile(model_profile: BurnJepaModelProfile) -> Self {
        Self {
            model_profile,
            model_base_url: burn_jepa_model_profile_base_url(model_profile),
            ..Self::default()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_env_overrides(self) -> Self {
        apply_model_bootstrap_env_overrides(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnJepaModelPackageFiles {
    pub cache_root: PathBuf,
    pub manifest_path: PathBuf,
    pub parts_manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
    pub model_base_url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnAnyUpModelPackageFiles {
    pub cache_root: PathBuf,
    pub manifest_path: PathBuf,
    pub parts_manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
    pub model_base_url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnJepaModelDeployBundleReport {
    pub output_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub parts_manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BurnAnyUpModelDeployBundleReport {
    pub output_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub parts_manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawBurnpackMetadata {
    tensors: BTreeMap<String, RawTensorDescriptor>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawTensorDescriptor {
    dtype: Value,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    param_id: Option<u64>,
}

#[derive(Clone, Debug)]
struct TensorRecord {
    name: String,
    descriptor: RawTensorDescriptor,
}

#[derive(Clone, Debug, Default)]
struct BurnJepaF16SaveAdapter;

impl ModuleAdapter for BurnJepaF16SaveAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::F32 {
            return snapshot.clone();
        }
        let original = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(original()?.convert_dtype(DType::F16))),
            DType::F16,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Default)]
struct BurnJepaF16LoadAdapter;

impl ModuleAdapter for BurnJepaF16LoadAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::F16 {
            return snapshot.clone();
        }
        let original = snapshot.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(original()?.convert_dtype(DType::F32))),
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, Default)]
struct ForceFloat32Mapper;

impl<B: Backend> ModuleMapper<B> for ForceFloat32Mapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = if tensor.dtype() == DType::F32 {
            tensor
        } else {
            tensor.cast(FloatDType::F32)
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn force_module_float32<B, M>(model: M) -> M
where
    B: Backend,
    M: Module<B>,
{
    let mut mapper = ForceFloat32Mapper;
    model.map(&mut mapper)
}

pub fn save_module_burnpack<B, M>(model: &M, output: impl AsRef<Path>) -> Result<PathBuf>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let output = normalize_burnpack_path(output.as_ref());
    ensure_parent_dir(&output)?;
    let mut store = BurnpackStore::from_file(&output)
        .auto_extension(false)
        .overwrite(true)
        .metadata("record_dtype", "f16")
        .with_to_adapter(BurnJepaF16SaveAdapter);
    model
        .save_into(&mut store)
        .map_err(|err| anyhow!("save burnpack {}: {err}", output.display()))?;
    Ok(output)
}

pub fn save_vjepa_burnpack<B: Backend>(
    model: &VJepa2_1Model<B>,
    output: impl AsRef<Path>,
) -> Result<PathBuf> {
    save_module_burnpack::<B, _>(model, output)
}

pub fn save_ttt_burnpack<B: Backend>(
    model: &VJepaTttModel<B>,
    output: impl AsRef<Path>,
) -> Result<PathBuf> {
    save_module_burnpack::<B, _>(model, output)
}

pub fn save_anyup_burnpack<B: Backend>(
    model: &AnyUp<B>,
    output: impl AsRef<Path>,
) -> Result<PathBuf> {
    save_module_burnpack::<B, _>(model, output)
}

pub fn load_vjepa_burnpack<B: Backend>(
    config: &VJepaConfig,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(VJepa2_1Model<B>, ApplyResult)> {
    let mut model = VJepa2_1Model::new(config, device);
    let mut store = BurnpackStore::from_file(path.as_ref())
        .auto_extension(false)
        .validate(true)
        .with_from_adapter(BurnJepaF16LoadAdapter);
    let result = model
        .load_from(&mut store)
        .map_err(|err| anyhow!("load V-JEPA burnpack {}: {err}", path.as_ref().display()))?;
    let model = force_module_float32::<B, _>(model);
    Ok((model, result))
}

pub fn load_ttt_burnpack<B: Backend>(
    config: &VJepaConfig,
    ttt_config: TttEncoderConfig,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(VJepaTttModel<B>, ApplyResult)> {
    let base = VJepa2_1Model::new(config, device);
    let mut model = VJepaTttModel::from_model(base, ttt_config, device)?;
    let mut store = BurnpackStore::from_file(path.as_ref())
        .auto_extension(false)
        .validate(true)
        .with_from_adapter(BurnJepaF16LoadAdapter);
    let result = model.load_from(&mut store).map_err(|err| {
        anyhow!(
            "load TTT V-JEPA burnpack {}: {err}",
            path.as_ref().display()
        )
    })?;
    let model = force_module_float32::<B, _>(model);
    Ok((model, result))
}

pub fn load_vjepa_burnpack_parts<B: Backend>(
    config: &VJepaConfig,
    parts: &[Vec<u8>],
    device: &B::Device,
) -> Result<(VJepa2_1Model<B>, ApplyResult)> {
    let mut model = VJepa2_1Model::new(config, device);
    let result = apply_burnpack_parts::<B, _>(&mut model, parts)?;
    let model = force_module_float32::<B, _>(model);
    Ok((model, result))
}

pub fn load_ttt_burnpack_parts<B: Backend>(
    config: &VJepaConfig,
    ttt_config: TttEncoderConfig,
    parts: &[Vec<u8>],
    device: &B::Device,
) -> Result<(VJepaTttModel<B>, ApplyResult)> {
    let base = VJepa2_1Model::new(config, device);
    let mut model = VJepaTttModel::from_model(base, ttt_config, device)?;
    let result = apply_burnpack_parts::<B, _>(&mut model, parts)?;
    let model = force_module_float32::<B, _>(model);
    Ok((model, result))
}

pub fn load_anyup_burnpack_parts<B: Backend>(
    config: &AnyUpConfig,
    parts: &[Vec<u8>],
    device: &B::Device,
) -> Result<(AnyUp<B>, ApplyResult)> {
    let mut model = AnyUp::new(config.clone(), device)?;
    let result = apply_burnpack_parts::<B, _>(&mut model, parts)?;
    let model = force_module_float32::<B, _>(model);
    Ok((model, result))
}

pub fn apply_burnpack_parts<B, M>(model: &mut M, parts: &[Vec<u8>]) -> Result<ApplyResult>
where
    B: Backend,
    M: Module<B> + ModuleSnapshot<B>,
{
    if parts.is_empty() {
        bail!("burnpack parts bundle is empty");
    }
    let mut applied = BTreeSet::new();
    let mut skipped = BTreeSet::new();
    let mut missing = BTreeSet::new();
    let mut unused = BTreeSet::new();
    let mut errors = Vec::new();
    for part in parts {
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(part.clone())))
            .allow_partial(true)
            .validate(true)
            .with_from_adapter(BurnJepaF16LoadAdapter);
        let result = model
            .load_from(&mut store)
            .map_err(|err| anyhow!("apply burnpack part: {err}"))?;
        applied.extend(result.applied);
        skipped.extend(result.skipped);
        missing.extend(result.missing);
        unused.extend(result.unused);
        errors.extend(result.errors);
    }
    missing.retain(|(path, _container_stack)| !applied.contains(path));
    Ok(ApplyResult {
        applied: applied.into_iter().collect(),
        skipped: skipped.into_iter().collect(),
        missing: missing.into_iter().collect(),
        unused: unused.into_iter().collect(),
        errors,
    })
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn write_pipeline_package_manifest(
    manifest_path: impl AsRef<Path>,
    manifest: &BurnJepaPipelinePackageManifest,
) -> Result<PathBuf> {
    let manifest_path = manifest_path.as_ref();
    ensure_parent_dir(manifest_path)?;
    fs::write(manifest_path, manifest.to_json_string()?)
        .with_context(|| format!("write package manifest {}", manifest_path.display()))?;
    Ok(manifest_path.to_path_buf())
}

pub fn write_anyup_package_manifest(
    manifest_path: impl AsRef<Path>,
    manifest: &BurnAnyUpPackageManifest,
) -> Result<PathBuf> {
    let manifest_path = manifest_path.as_ref();
    ensure_parent_dir(manifest_path)?;
    fs::write(manifest_path, manifest.to_json_string()?)
        .with_context(|| format!("write AnyUp package manifest {}", manifest_path.display()))?;
    Ok(manifest_path.to_path_buf())
}

pub fn resolve_package_manifest_entry_path(
    manifest_path: &Path,
    entry_path: &str,
) -> Result<PathBuf> {
    let entry = Path::new(entry_path);
    if entry.is_absolute() {
        return Ok(entry.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry))
        .ok_or_else(|| anyhow!("invalid package manifest path {}", manifest_path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_burn_jepa_model_cache_root() -> Result<PathBuf> {
    default_burn_jepa_model_cache_root_with_config(&apply_model_bootstrap_env_overrides(
        BurnJepaModelBootstrapConfig::default(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_burn_jepa_model_cache_root_with_config(
    config: &BurnJepaModelBootstrapConfig,
) -> Result<PathBuf> {
    if let Some(cache_root) = &config.cache_root {
        return Ok(expand_home_path(cache_root.clone()));
    }
    let home = user_home_dir().context("failed to resolve user home directory for model cache")?;
    Ok(home
        .join(DEFAULT_BURN_JEPA_MODEL_CACHE_ROOT_DIR)
        .join(DEFAULT_BURN_JEPA_MODEL_CACHE_SUBDIR)
        .join(config.model_profile.as_str()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_burn_anyup_model_cache_root() -> Result<PathBuf> {
    default_burn_anyup_model_cache_root_with_config(&apply_anyup_bootstrap_env_overrides(
        BurnAnyUpModelBootstrapConfig::default(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_burn_anyup_model_cache_root_with_config(
    config: &BurnAnyUpModelBootstrapConfig,
) -> Result<PathBuf> {
    if let Some(cache_root) = &config.cache_root {
        return Ok(expand_home_path(cache_root.clone()));
    }
    let home = user_home_dir().context("failed to resolve user home directory for model cache")?;
    Ok(home
        .join(DEFAULT_BURN_JEPA_MODEL_CACHE_ROOT_DIR)
        .join(DEFAULT_BURN_ANYUP_MODEL_CACHE_SUBDIR)
        .join(config.model_profile.as_str()))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_jepa_model_package() -> Result<BurnJepaModelPackageFiles> {
    resolve_or_bootstrap_burn_jepa_model_package_with_config(&apply_model_bootstrap_env_overrides(
        BurnJepaModelBootstrapConfig::default(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_jepa_model_package_with_config(
    config: &BurnJepaModelBootstrapConfig,
) -> Result<BurnJepaModelPackageFiles> {
    resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress(config, |_| {})
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress<F>(
    config: &BurnJepaModelBootstrapConfig,
    progress: F,
) -> Result<BurnJepaModelPackageFiles>
where
    F: Fn(String),
{
    let config = normalized_model_bootstrap_config(config);
    let cache_root = default_burn_jepa_model_cache_root_with_config(&config)?;
    progress(format!(
        "resolving burn_jepa model cache under {}",
        cache_root.display()
    ));
    fs::create_dir_all(&cache_root)
        .with_context(|| format!("create model cache directory {}", cache_root.display()))?;
    let manifest_path = cache_root.join("manifest.json");
    if let Some(files) = cached_model_package_files(&cache_root, &manifest_path)? {
        progress("using cached burn_jepa package manifest".to_string());
        return Ok(files);
    }

    let manifest_url = config
        .manifest_url
        .clone()
        .unwrap_or_else(|| join_url(&config.model_base_url, "manifest.json"));
    progress(format!(
        "downloading burn_jepa package manifest {manifest_url}"
    ));
    ensure_file_cached(&manifest_path, &manifest_url, true)?;

    let manifest_json = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let manifest = BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)
        .with_context(|| format!("parse package manifest {}", manifest_path.display()))?;
    let parts_manifest_url = resolve_manifest_entry_url(&manifest_url, &manifest.parts_manifest);
    let parts_manifest_path = safe_cache_entry_path(&cache_root, &manifest.parts_manifest)?;

    progress(format!(
        "downloading burn_jepa parts manifest {parts_manifest_url}"
    ));
    ensure_file_cached(&parts_manifest_path, &parts_manifest_url, true)?;
    let parts_manifest = read_parts_manifest(&parts_manifest_path)?;
    if parts_manifest.parts.is_empty() {
        bail!(
            "burn_jepa parts manifest {} contains no parts",
            parts_manifest_path.display()
        );
    }

    let total_parts = parts_manifest.parts.len();
    for (index, part) in parts_manifest.parts.iter().enumerate() {
        let part_path = safe_cache_entry_path(&cache_root, &part.path)?;
        if part_matches_cache(&part_path, part)? {
            progress(format!(
                "cached burn_jepa shard {}/{}",
                index + 1,
                total_parts
            ));
            continue;
        }
        let part_url = resolve_manifest_entry_url(&parts_manifest_url, &part.path);
        progress(format!(
            "downloading burn_jepa shard {}/{}",
            index + 1,
            total_parts
        ));
        ensure_file_cached(&part_path, &part_url, false)?;
        if !part_matches_cache(&part_path, part)? {
            bail!(
                "downloaded burn_jepa shard `{}` does not match manifest entry",
                part_path.display()
            );
        }
    }

    cached_model_package_files(&cache_root, &manifest_path)?.ok_or_else(|| {
        anyhow!(
            "burn_jepa package cache remained incomplete after download: {}",
            cache_root.display()
        )
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_anyup_model_package() -> Result<BurnAnyUpModelPackageFiles> {
    resolve_or_bootstrap_burn_anyup_model_package_with_config(&apply_anyup_bootstrap_env_overrides(
        BurnAnyUpModelBootstrapConfig::default(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_anyup_model_package_with_config(
    config: &BurnAnyUpModelBootstrapConfig,
) -> Result<BurnAnyUpModelPackageFiles> {
    resolve_or_bootstrap_burn_anyup_model_package_with_config_and_progress(config, |_| {})
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_or_bootstrap_burn_anyup_model_package_with_config_and_progress<F>(
    config: &BurnAnyUpModelBootstrapConfig,
    progress: F,
) -> Result<BurnAnyUpModelPackageFiles>
where
    F: Fn(String),
{
    let config = normalized_anyup_bootstrap_config(config);
    let cache_root = default_burn_anyup_model_cache_root_with_config(&config)?;
    progress(format!(
        "resolving burn_anyup model cache under {}",
        cache_root.display()
    ));
    fs::create_dir_all(&cache_root).with_context(|| {
        format!(
            "create AnyUp model cache directory {}",
            cache_root.display()
        )
    })?;
    let manifest_path = cache_root.join("manifest.json");
    if let Some(files) = cached_anyup_package_files(&cache_root, &manifest_path)? {
        progress("using cached burn_anyup package manifest".to_string());
        return Ok(files);
    }

    let manifest_url = config
        .manifest_url
        .clone()
        .unwrap_or_else(|| join_url(&config.model_base_url, "manifest.json"));
    progress(format!(
        "downloading burn_anyup package manifest {manifest_url}"
    ));
    ensure_file_cached(&manifest_path, &manifest_url, true)?;

    let manifest_json = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read AnyUp package manifest {}", manifest_path.display()))?;
    let manifest = BurnAnyUpPackageManifest::from_json_str(&manifest_json)
        .with_context(|| format!("parse AnyUp package manifest {}", manifest_path.display()))?;
    let parts_manifest_url = resolve_manifest_entry_url(&manifest_url, &manifest.parts_manifest);
    let parts_manifest_path = safe_cache_entry_path(&cache_root, &manifest.parts_manifest)?;

    progress(format!(
        "downloading burn_anyup parts manifest {parts_manifest_url}"
    ));
    ensure_file_cached(&parts_manifest_path, &parts_manifest_url, true)?;
    let parts_manifest = read_parts_manifest(&parts_manifest_path)?;
    if parts_manifest.parts.is_empty() {
        bail!(
            "burn_anyup parts manifest {} contains no parts",
            parts_manifest_path.display()
        );
    }

    let total_parts = parts_manifest.parts.len();
    for (index, part) in parts_manifest.parts.iter().enumerate() {
        let part_path = safe_cache_entry_path(&cache_root, &part.path)?;
        if part_matches_cache(&part_path, part)? {
            progress(format!(
                "cached burn_anyup shard {}/{}",
                index + 1,
                total_parts
            ));
            continue;
        }
        let part_url = resolve_manifest_entry_url(&parts_manifest_url, &part.path);
        progress(format!(
            "downloading burn_anyup shard {}/{}",
            index + 1,
            total_parts
        ));
        ensure_file_cached(&part_path, &part_url, false)?;
        if !part_matches_cache(&part_path, part)? {
            bail!(
                "downloaded burn_anyup shard `{}` does not match manifest entry",
                part_path.display()
            );
        }
    }

    cached_anyup_package_files(&cache_root, &manifest_path)?.ok_or_else(|| {
        anyhow!(
            "burn_anyup package cache remained incomplete after download: {}",
            cache_root.display()
        )
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn normalized_model_bootstrap_config(
    config: &BurnJepaModelBootstrapConfig,
) -> BurnJepaModelBootstrapConfig {
    let mut config = config.clone();
    if config.model_base_url == DEFAULT_BURN_JEPA_MODEL_BASE_URL
        && config.model_profile != BurnJepaModelProfile::default()
    {
        config.model_base_url = burn_jepa_model_profile_base_url(config.model_profile);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn normalized_anyup_bootstrap_config(
    config: &BurnAnyUpModelBootstrapConfig,
) -> BurnAnyUpModelBootstrapConfig {
    let mut config = config.clone();
    if config.model_base_url == DEFAULT_BURN_ANYUP_MODEL_BASE_URL
        && config.model_profile != BurnAnyUpModelProfile::default()
    {
        config.model_base_url = burn_anyup_model_profile_base_url(config.model_profile);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
pub fn burn_jepa_model_package_cache_complete(manifest_path: &Path) -> Result<bool> {
    let cache_root = manifest_path
        .parent()
        .ok_or_else(|| anyhow!("invalid package manifest path {}", manifest_path.display()))?;
    Ok(cached_model_package_files(cache_root, manifest_path)?.is_some())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn burn_anyup_model_package_cache_complete(manifest_path: &Path) -> Result<bool> {
    let cache_root = manifest_path.parent().ok_or_else(|| {
        anyhow!(
            "invalid AnyUp package manifest path {}",
            manifest_path.display()
        )
    })?;
    Ok(cached_anyup_package_files(cache_root, manifest_path)?.is_some())
}

pub fn write_burn_jepa_model_deploy_bundle(
    manifest_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    overwrite: bool,
) -> Result<BurnJepaModelDeployBundleReport> {
    let manifest_path = manifest_path.as_ref();
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        if !overwrite {
            let mut entries = fs::read_dir(output_dir)
                .with_context(|| format!("read output directory {}", output_dir.display()))?;
            if entries.next().transpose()?.is_some() {
                bail!(
                    "deploy bundle output directory `{}` is not empty; pass --overwrite",
                    output_dir.display()
                );
            }
        } else {
            fs::remove_dir_all(output_dir)
                .with_context(|| format!("remove old deploy bundle {}", output_dir.display()))?;
        }
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("create deploy bundle directory {}", output_dir.display()))?;

    let manifest_json = fs::read_to_string(manifest_path)
        .with_context(|| format!("read package manifest {}", manifest_path.display()))?;
    let mut manifest = BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)
        .with_context(|| format!("parse package manifest {}", manifest_path.display()))?;
    let source_parts_manifest_path =
        resolve_package_manifest_entry_path(manifest_path, &manifest.parts_manifest)?;
    let mut parts_manifest = read_parts_manifest(&source_parts_manifest_path)?;

    let parts_manifest_name = file_name_string(&source_parts_manifest_path)?;
    let deploy_parts_manifest_path = output_dir.join(&parts_manifest_name);
    let mut deploy_part_paths = Vec::with_capacity(parts_manifest.parts.len());
    for part in &mut parts_manifest.parts {
        let source_part_path = resolve_part_entry_path(&source_parts_manifest_path, &part.path)?;
        let part_name = file_name_string(&source_part_path)?;
        let deploy_part_path = output_dir.join(&part_name);
        fs::copy(&source_part_path, &deploy_part_path).with_context(|| {
            format!(
                "copy burnpack shard {} -> {}",
                source_part_path.display(),
                deploy_part_path.display()
            )
        })?;
        part.path = part_name;
        deploy_part_paths.push(deploy_part_path);
    }
    manifest.parts_manifest = parts_manifest_name;
    manifest.burnpack = Path::new(&manifest.burnpack)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("jepa.bpk")
        .to_string();
    fs::write(
        &deploy_parts_manifest_path,
        serde_json::to_string_pretty(&parts_manifest)?,
    )
    .with_context(|| {
        format!(
            "write deploy parts manifest {}",
            deploy_parts_manifest_path.display()
        )
    })?;
    let deploy_manifest_path = output_dir.join("manifest.json");
    write_pipeline_package_manifest(&deploy_manifest_path, &manifest)?;
    Ok(BurnJepaModelDeployBundleReport {
        output_dir: output_dir.to_path_buf(),
        manifest_path: deploy_manifest_path,
        parts_manifest_path: deploy_parts_manifest_path,
        part_paths: deploy_part_paths,
        total_bytes: parts_manifest.total_bytes,
    })
}

pub fn write_burn_anyup_model_deploy_bundle(
    manifest_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    overwrite: bool,
) -> Result<BurnAnyUpModelDeployBundleReport> {
    let manifest_path = manifest_path.as_ref();
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        if !overwrite {
            let mut entries = fs::read_dir(output_dir)
                .with_context(|| format!("read output directory {}", output_dir.display()))?;
            if entries.next().transpose()?.is_some() {
                bail!(
                    "AnyUp deploy bundle output directory `{}` is not empty; pass --overwrite",
                    output_dir.display()
                );
            }
        } else {
            fs::remove_dir_all(output_dir).with_context(|| {
                format!("remove old AnyUp deploy bundle {}", output_dir.display())
            })?;
        }
    }
    fs::create_dir_all(output_dir)
        .with_context(|| format!("create deploy bundle directory {}", output_dir.display()))?;

    let manifest_json = fs::read_to_string(manifest_path)
        .with_context(|| format!("read AnyUp package manifest {}", manifest_path.display()))?;
    let mut manifest = BurnAnyUpPackageManifest::from_json_str(&manifest_json)
        .with_context(|| format!("parse AnyUp package manifest {}", manifest_path.display()))?;
    let source_parts_manifest_path =
        resolve_package_manifest_entry_path(manifest_path, &manifest.parts_manifest)?;
    let mut parts_manifest = read_parts_manifest(&source_parts_manifest_path)?;

    let parts_manifest_name = file_name_string(&source_parts_manifest_path)?;
    let deploy_parts_manifest_path = output_dir.join(&parts_manifest_name);
    let mut deploy_part_paths = Vec::with_capacity(parts_manifest.parts.len());
    for part in &mut parts_manifest.parts {
        let source_part_path = resolve_part_entry_path(&source_parts_manifest_path, &part.path)?;
        let part_name = file_name_string(&source_part_path)?;
        let deploy_part_path = output_dir.join(&part_name);
        fs::copy(&source_part_path, &deploy_part_path).with_context(|| {
            format!(
                "copy AnyUp burnpack shard {} -> {}",
                source_part_path.display(),
                deploy_part_path.display()
            )
        })?;
        part.path = part_name;
        deploy_part_paths.push(deploy_part_path);
    }
    manifest.parts_manifest = parts_manifest_name;
    manifest.burnpack = Path::new(&manifest.burnpack)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("anyup.bpk")
        .to_string();
    fs::write(
        &deploy_parts_manifest_path,
        serde_json::to_string_pretty(&parts_manifest)?,
    )
    .with_context(|| {
        format!(
            "write AnyUp deploy parts manifest {}",
            deploy_parts_manifest_path.display()
        )
    })?;
    let deploy_manifest_path = output_dir.join("manifest.json");
    write_anyup_package_manifest(&deploy_manifest_path, &manifest)?;
    Ok(BurnAnyUpModelDeployBundleReport {
        output_dir: output_dir.to_path_buf(),
        manifest_path: deploy_manifest_path,
        parts_manifest_path: deploy_parts_manifest_path,
        part_paths: deploy_part_paths,
        total_bytes: parts_manifest.total_bytes,
    })
}

pub fn write_burnpack_parts_for_browser(
    burnpack_path: impl AsRef<Path>,
    max_part_bytes: u64,
    overwrite: bool,
) -> Result<BurnpackPartsReport> {
    let burnpack_path = burnpack_path.as_ref();
    if !burnpack_path.exists() {
        bail!(
            "burnpack does not exist for sharding: {}",
            burnpack_path.display()
        );
    }
    let max_part_bytes = max_part_bytes.max(1);
    let total_bytes = fs::metadata(burnpack_path)
        .with_context(|| format!("read burnpack metadata {}", burnpack_path.display()))?
        .len();
    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    if manifest_path.exists()
        && !overwrite
        && manifest_has_all_parts(&manifest_path, Some(burnpack_path))?
    {
        let manifest = read_parts_manifest(&manifest_path)?;
        let part_paths = manifest
            .parts
            .iter()
            .map(|entry| resolve_part_entry_path(&manifest_path, &entry.path))
            .collect::<Result<Vec<_>>>()?;
        return Ok(BurnpackPartsReport {
            manifest_path,
            part_paths,
            total_bytes: manifest.total_bytes,
        });
    }

    if overwrite {
        cleanup_existing_parts(&manifest_path)?;
    }
    ensure_parent_dir(&manifest_path)?;

    let mut source = fs::File::open(burnpack_path)
        .with_context(|| format!("open burnpack {}", burnpack_path.display()))?;
    let (version, metadata_size, metadata) = read_burnpack_metadata(&mut source, burnpack_path)?;
    let data_start = aligned_data_section_start(metadata_size as usize) as u64;
    let mut tensor_records = metadata
        .tensors
        .iter()
        .map(|(name, descriptor)| TensorRecord {
            name: name.clone(),
            descriptor: descriptor.clone(),
        })
        .collect::<Vec<_>>();
    if tensor_records.is_empty() {
        bail!("burnpack {} contains no tensors", burnpack_path.display());
    }
    tensor_records.sort_by_key(|record| record.descriptor.data_offsets.0);

    let source_file = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("invalid burnpack path {}", burnpack_path.display()))?;
    let groups = split_tensor_records(tensor_records, max_part_bytes, &metadata.metadata);
    let mut parts = Vec::with_capacity(groups.len());
    let mut part_paths = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let part_name = format!("{source_file}.part-{index:05}.bpk");
        let part_path = burnpack_path.with_file_name(&part_name);
        if part_path.exists() && overwrite {
            fs::remove_file(&part_path)
                .with_context(|| format!("remove stale part {}", part_path.display()))?;
        }
        write_burnpack_part(
            &mut source,
            &part_path,
            version,
            data_start,
            &metadata.metadata,
            group,
        )?;
        let bytes = fs::metadata(&part_path)
            .with_context(|| format!("stat part {}", part_path.display()))?
            .len();
        parts.push(BurnpackPartEntry {
            path: part_name,
            bytes,
            sha256: sha256_file(&part_path)?,
            tensors: group.len(),
        });
        part_paths.push(part_path);
    }

    let manifest = BurnpackPartsManifest {
        version: 1,
        source_file: source_file.to_string(),
        source_modified_unix_ms: file_modified_unix_ms(burnpack_path).unwrap_or(0),
        total_bytes,
        max_part_bytes,
        parts,
    };
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write parts manifest {}", manifest_path.display()))?;
    Ok(BurnpackPartsReport {
        manifest_path,
        part_paths,
        total_bytes,
    })
}

pub fn read_parts_manifest(path: impl AsRef<Path>) -> Result<BurnpackPartsManifest> {
    let path = path.as_ref();
    let bytes = fs::read(path)
        .with_context(|| format!("read burnpack parts manifest {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse burnpack parts manifest {}", path.display()))
}

pub fn burnpack_dtype_counts(path: impl AsRef<Path>) -> Result<BTreeMap<String, usize>> {
    let path = path.as_ref();
    let mut store = BurnpackStore::from_file(path)
        .auto_extension(false)
        .validate(true);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|err| anyhow!("inspect burnpack {}: {err}", path.display()))?;
    let mut counts = BTreeMap::new();
    for snapshot in snapshots.values() {
        *counts.entry(format!("{:?}", snapshot.dtype)).or_insert(0) += 1;
    }
    Ok(counts)
}

pub fn burnpack_parts_dtype_counts(
    parts_manifest_path: impl AsRef<Path>,
) -> Result<BTreeMap<String, usize>> {
    let parts_manifest_path = parts_manifest_path.as_ref();
    let manifest = read_parts_manifest(parts_manifest_path)?;
    let mut counts = BTreeMap::new();
    for part in &manifest.parts {
        let part_path = resolve_part_entry_path(parts_manifest_path, &part.path)?;
        for (dtype, count) in burnpack_dtype_counts(&part_path)? {
            *counts.entry(dtype).or_insert(0) += count;
        }
    }
    Ok(counts)
}

pub fn module_dtype_counts<B, M>(model: &M) -> BTreeMap<String, usize>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut counts = BTreeMap::new();
    for snapshot in model.collect(None, None, false) {
        *counts.entry(format!("{:?}", snapshot.dtype)).or_insert(0) += 1;
    }
    counts
}

pub fn resolve_part_entry_path(manifest_path: &Path, entry_path: &str) -> Result<PathBuf> {
    let entry = Path::new(entry_path);
    if entry.is_absolute() {
        return Ok(entry.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry))
        .ok_or_else(|| anyhow!("invalid manifest path {}", manifest_path.display()))
}

pub fn manifest_has_all_parts(
    manifest_path: &Path,
    source_burnpack_path: Option<&Path>,
) -> Result<bool> {
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(false),
    };
    if manifest.parts.is_empty() {
        return Ok(false);
    }
    if let Some(source_path) = source_burnpack_path
        && source_path.exists()
        && !manifest_matches_source_file(&manifest, source_path)
    {
        return Ok(false);
    }
    for part in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &part.path)?;
        if !part_matches_cache(&path, part)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn normalize_burnpack_path(path: &Path) -> PathBuf {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bpk"))
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    }
}

fn read_burnpack_metadata(
    source: &mut fs::File,
    burnpack_path: &Path,
) -> Result<(u16, u32, RawBurnpackMetadata)> {
    source
        .seek(SeekFrom::Start(0))
        .with_context(|| format!("seek burnpack {}", burnpack_path.display()))?;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    source
        .read_exact(&mut header)
        .with_context(|| format!("read burnpack header {}", burnpack_path.display()))?;
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != BURNPACK_MAGIC_NUMBER {
        bail!(
            "invalid burnpack magic in {}: expected {BURNPACK_MAGIC_NUMBER:#x}, found {magic:#x}",
            burnpack_path.display()
        );
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);
    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    source
        .read_exact(&mut metadata_bytes)
        .with_context(|| format!("read burnpack metadata {}", burnpack_path.display()))?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice())
        .with_context(|| format!("parse burnpack metadata {}", burnpack_path.display()))?;
    Ok((version, metadata_size, metadata))
}

fn split_tensor_records(
    records: Vec<TensorRecord>,
    max_part_bytes: u64,
    source_metadata: &BTreeMap<String, String>,
) -> Vec<Vec<TensorRecord>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    for record in records {
        let mut candidate = current.clone();
        candidate.push(record.clone());
        let candidate_bytes =
            estimate_part_total_bytes(&candidate, source_metadata).unwrap_or(u64::MAX);
        if !current.is_empty() && candidate_bytes > max_part_bytes {
            groups.push(current);
            current = vec![record];
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

fn estimate_part_total_bytes(
    records: &[TensorRecord],
    source_metadata: &BTreeMap<String, String>,
) -> Result<u64> {
    let mut tensors = BTreeMap::new();
    let mut payload_bytes = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        payload_bytes = align_offset(payload_bytes, BURNPACK_TENSOR_ALIGNMENT);
        let mut descriptor = record.descriptor.clone();
        descriptor.data_offsets = (payload_bytes, payload_bytes.saturating_add(tensor_bytes));
        payload_bytes = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }
    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .context("estimate burnpack part metadata size")?;
    Ok(aligned_data_section_start(metadata_bytes.len()) as u64 + payload_bytes)
}

fn write_burnpack_part(
    source: &mut fs::File,
    destination: &Path,
    version: u16,
    data_start: u64,
    source_metadata: &BTreeMap<String, String>,
    records: &[TensorRecord],
) -> Result<()> {
    let mut tensors = BTreeMap::new();
    let mut next_offset = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        next_offset = align_offset(next_offset, BURNPACK_TENSOR_ALIGNMENT);
        let mut descriptor = record.descriptor.clone();
        descriptor.data_offsets = (next_offset, next_offset.saturating_add(tensor_bytes));
        next_offset = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }
    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .context("serialize burnpack part metadata")?;
    let metadata_size =
        u32::try_from(metadata_bytes.len()).context("burnpack part metadata exceeds u32")?;
    ensure_parent_dir(destination)?;
    let mut out = fs::File::create(destination)
        .with_context(|| format!("create burnpack part {}", destination.display()))?;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    header[0..4].copy_from_slice(&BURNPACK_MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header)
        .with_context(|| format!("write part header {}", destination.display()))?;
    out.write_all(&metadata_bytes)
        .with_context(|| format!("write part metadata {}", destination.display()))?;
    let data_section_start = aligned_data_section_start(metadata_bytes.len());
    let current = BURNPACK_HEADER_SIZE + metadata_bytes.len();
    if data_section_start > current {
        write_zeros(&mut out, data_section_start - current)?;
    }

    let mut part_data_offset = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    for record in records {
        let start = record.descriptor.data_offsets.0;
        let end = record.descriptor.data_offsets.1;
        let mut remaining = end.saturating_sub(start);
        let aligned_part_offset = align_offset(part_data_offset, BURNPACK_TENSOR_ALIGNMENT);
        if aligned_part_offset > part_data_offset {
            write_zeros(&mut out, (aligned_part_offset - part_data_offset) as usize)?;
            part_data_offset = aligned_part_offset;
        }
        source
            .seek(SeekFrom::Start(data_start.saturating_add(start)))
            .context("seek source burnpack tensor data")?;
        while remaining > 0 {
            let chunk = remaining.min(buffer.len() as u64) as usize;
            source
                .read_exact(&mut buffer[..chunk])
                .context("read source burnpack tensor data")?;
            out.write_all(&buffer[..chunk])
                .context("write burnpack part tensor data")?;
            remaining -= chunk as u64;
            part_data_offset += chunk as u64;
        }
    }
    out.flush()
        .with_context(|| format!("flush burnpack part {}", destination.display()))?;
    Ok(())
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

fn aligned_data_section_start(metadata_size: usize) -> usize {
    let unaligned_start = (BURNPACK_HEADER_SIZE + metadata_size) as u64;
    align_offset(unaligned_start, BURNPACK_TENSOR_ALIGNMENT) as usize
}

fn write_zeros(writer: &mut fs::File, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    const ZEROS: [u8; 256] = [0; 256];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len());
        writer
            .write_all(&ZEROS[..chunk])
            .context("write burnpack alignment padding")?;
        remaining -= chunk;
    }
    Ok(())
}

fn cleanup_existing_parts(manifest_path: &Path) -> Result<()> {
    let Ok(manifest) = read_parts_manifest(manifest_path) else {
        return Ok(());
    };
    for entry in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &entry.path)?;
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("remove old part {}", path.display()))?;
        }
    }
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent directory {}", parent.display()))?;
    }
    Ok(())
}

fn manifest_matches_source_file(manifest: &BurnpackPartsManifest, source_path: &Path) -> bool {
    let Some(file_name) = source_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if !manifest.source_file.is_empty() && manifest.source_file != file_name {
        return false;
    }
    let Ok(metadata) = fs::metadata(source_path) else {
        return false;
    };
    if manifest.total_bytes > 0 && manifest.total_bytes != metadata.len() {
        return false;
    }
    if manifest.source_modified_unix_ms == 0 {
        return true;
    }
    manifest.source_modified_unix_ms == file_modified_unix_ms(source_path).unwrap_or(0)
}

fn part_matches_cache(path: &Path, part: &BurnpackPartEntry) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let bytes = fs::metadata(path)
        .with_context(|| format!("stat burnpack part {}", path.display()))?
        .len();
    if part.bytes > 0 && bytes != part.bytes {
        return Ok(false);
    }
    if !part.sha256.trim().is_empty() {
        return Ok(sha256_file(path)?.eq_ignore_ascii_case(part.sha256.trim()));
    }
    Ok(true)
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_model_bootstrap_env_overrides(
    mut config: BurnJepaModelBootstrapConfig,
) -> BurnJepaModelBootstrapConfig {
    let had_profile_default_url =
        config.model_base_url == burn_jepa_model_profile_base_url(config.model_profile);
    if let Ok(value) =
        std::env::var("BURN_JEPA_MODEL_PROFILE").or_else(|_| std::env::var("BURN_JEPA_MODEL_NAME"))
        && let Ok(profile) = BurnJepaModelProfile::from_str(&value)
    {
        config.model_profile = profile;
        if had_profile_default_url {
            config.model_base_url = burn_jepa_model_profile_base_url(profile);
        }
    }
    if let Some(root) = std::env::var_os("BURN_JEPA_CACHE_DIR") {
        config.cache_root = Some(
            PathBuf::from(root)
                .join(DEFAULT_BURN_JEPA_MODEL_CACHE_SUBDIR)
                .join(config.model_profile.as_str())
                .to_path_buf(),
        );
    }
    if let Some(root) = std::env::var_os("BURN_JEPA_MODEL_CACHE_DIR") {
        config.cache_root = Some(PathBuf::from(root));
    }
    if let Ok(value) = std::env::var("BURN_JEPA_MODEL_BASE_URL") {
        config.model_base_url = value;
    }
    if let Ok(value) = std::env::var("BURN_JEPA_MODEL_MANIFEST_URL") {
        config.manifest_url = Some(value);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_anyup_bootstrap_env_overrides(
    mut config: BurnAnyUpModelBootstrapConfig,
) -> BurnAnyUpModelBootstrapConfig {
    let had_profile_default_url =
        config.model_base_url == burn_anyup_model_profile_base_url(config.model_profile);
    if let Ok(value) = std::env::var("BURN_ANYUP_MODEL_PROFILE")
        .or_else(|_| std::env::var("BURN_ANYUP_MODEL_NAME"))
        && let Ok(profile) = BurnAnyUpModelProfile::from_str(&value)
    {
        config.model_profile = profile;
        if had_profile_default_url {
            config.model_base_url = burn_anyup_model_profile_base_url(profile);
        }
    }
    if let Some(root) = std::env::var_os("BURN_ANYUP_CACHE_DIR") {
        config.cache_root = Some(
            PathBuf::from(root)
                .join(DEFAULT_BURN_ANYUP_MODEL_CACHE_SUBDIR)
                .join(config.model_profile.as_str())
                .to_path_buf(),
        );
    }
    if let Some(root) = std::env::var_os("BURN_ANYUP_MODEL_CACHE_DIR") {
        config.cache_root = Some(PathBuf::from(root));
    }
    if let Ok(value) = std::env::var("BURN_ANYUP_MODEL_BASE_URL") {
        config.model_base_url = value;
    }
    if let Ok(value) = std::env::var("BURN_ANYUP_MODEL_MANIFEST_URL") {
        config.manifest_url = Some(value);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn cached_model_package_files(
    cache_root: &Path,
    manifest_path: &Path,
) -> Result<Option<BurnJepaModelPackageFiles>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_json = match fs::read_to_string(manifest_path) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let manifest = match BurnJepaPipelinePackageManifest::from_json_str(&manifest_json) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let parts_manifest_path = match safe_cache_entry_path(cache_root, &manifest.parts_manifest) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let parts_manifest = match read_parts_manifest(&parts_manifest_path) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    if parts_manifest.parts.is_empty() {
        return Ok(None);
    }
    let mut part_paths = Vec::with_capacity(parts_manifest.parts.len());
    for part in &parts_manifest.parts {
        let path = match safe_cache_entry_path(cache_root, &part.path) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        if !part_matches_cache(&path, part)? {
            return Ok(None);
        }
        part_paths.push(path);
    }
    Ok(Some(BurnJepaModelPackageFiles {
        cache_root: cache_root.to_path_buf(),
        manifest_path: manifest_path.to_path_buf(),
        parts_manifest_path,
        part_paths,
        total_bytes: parts_manifest.total_bytes,
        model_base_url: manifest.model_base_url,
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn cached_anyup_package_files(
    cache_root: &Path,
    manifest_path: &Path,
) -> Result<Option<BurnAnyUpModelPackageFiles>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_json = match fs::read_to_string(manifest_path) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let manifest = match BurnAnyUpPackageManifest::from_json_str(&manifest_json) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let parts_manifest_path = match safe_cache_entry_path(cache_root, &manifest.parts_manifest) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let parts_manifest = match read_parts_manifest(&parts_manifest_path) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    if parts_manifest.parts.is_empty() {
        return Ok(None);
    }
    let mut part_paths = Vec::with_capacity(parts_manifest.parts.len());
    for part in &parts_manifest.parts {
        let path = match safe_cache_entry_path(cache_root, &part.path) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        if !part_matches_cache(&path, part)? {
            return Ok(None);
        }
        part_paths.push(path);
    }
    Ok(Some(BurnAnyUpModelPackageFiles {
        cache_root: cache_root.to_path_buf(),
        manifest_path: manifest_path.to_path_buf(),
        parts_manifest_path,
        part_paths,
        total_bytes: parts_manifest.total_bytes,
        model_base_url: manifest.model_base_url,
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_file_cached(path: &Path, url: &str, overwrite: bool) -> Result<()> {
    if path.exists() && !overwrite {
        return Ok(());
    }
    ensure_parent_dir(path)?;
    let tmp = temp_download_path(path);
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match download_to_file(url, &tmp) {
            Ok(()) => {
                if path.exists() {
                    fs::remove_file(path)
                        .with_context(|| format!("remove stale cached file {}", path.display()))?;
                }
                fs::rename(&tmp, path).with_context(|| {
                    format!(
                        "install downloaded file {} -> {}",
                        tmp.display(),
                        path.display()
                    )
                })?;
                return Ok(());
            }
            Err(err) => {
                let _ = fs::remove_file(&tmp);
                if attempt == DOWNLOAD_ATTEMPTS {
                    bail!("failed to download model `{url}`: {err}");
                }
                last_error = Some(err);
                std::thread::sleep(retry_delay(attempt));
            }
        }
    }
    bail!(
        "failed to download model `{url}`: {}",
        last_error.unwrap_or_else(|| "unknown download error".to_string())
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn download_to_file(url: &str, destination: &Path) -> Result<(), String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
        .timeout_write(WRITE_TIMEOUT)
        .build();
    let response = agent.get(url).call().map_err(|err| match err {
        ureq::Error::Status(code, response) => {
            format!("HTTP {code} ({})", response.status_text())
        }
        ureq::Error::Transport(transport) => format!("transport error: {transport}"),
    })?;
    let expected_len = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut writer = fs::File::create(destination)
        .map_err(|err| format!("failed to create {}: {err}", destination.display()))?;
    let mut total = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| format!("failed reading response body: {err}"))?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&buffer[..read])
            .map_err(|err| format!("failed writing {}: {err}", destination.display()))?;
        total = total.saturating_add(read as u64);
    }
    writer
        .flush()
        .map_err(|err| format!("failed flushing {}: {err}", destination.display()))?;
    if total == 0 {
        return Err("downloaded file is empty".to_string());
    }
    if let Some(expected) = expected_len
        && expected != total
    {
        return Err(format!(
            "content-length mismatch (expected {expected} bytes, wrote {total} bytes)"
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn safe_cache_entry_path(cache_root: &Path, entry: &str) -> Result<PathBuf> {
    if entry.contains("://") || entry.starts_with('/') {
        let name = entry
            .rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| anyhow!("unsafe model package URL entry `{entry}`"))?;
        return Ok(cache_root.join(name));
    }
    let mut out = PathBuf::from(cache_root);
    for component in Path::new(entry).components() {
        match component {
            std::path::Component::Normal(value) => out.push(value),
            std::path::Component::CurDir => {}
            _ => bail!("unsafe model package path entry `{entry}`"),
        }
    }
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_manifest_entry_url(manifest_url: &str, entry_url: &str) -> String {
    if entry_url.contains("://") || entry_url.starts_with('/') {
        return entry_url.to_string();
    }
    let normalized = entry_url.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

#[cfg(not(target_arch = "wasm32"))]
fn join_url(base: &str, child: &str) -> String {
    let mut out = base.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(child.trim_start_matches('/'));
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn retry_delay(attempt: u32) -> Duration {
    let capped = attempt.min(6);
    Duration::from_millis(600_u64.saturating_mul(1_u64 << capped))
}

#[cfg(not(target_arch = "wasm32"))]
fn temp_download_path(path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bin");
    path.with_file_name(format!("{file}.download-{stamp}.tmp"))
}

#[cfg(not(target_arch = "wasm32"))]
fn user_home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        return Some(home);
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
            return Some(profile);
        }
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        if let (Some(drive), Some(path)) = (drive, path) {
            return Some(PathBuf::from(format!(
                "{}{}",
                drive.to_string_lossy(),
                path.to_string_lossy()
            )));
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn expand_home_path(path: PathBuf) -> PathBuf {
    let path_string = path.to_string_lossy().into_owned();
    if path_string == "~" {
        return std::env::var_os("HOME").map(PathBuf::from).unwrap_or(path);
    }
    if let Some(rest) = path_string.strip_prefix("~/") {
        return std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(rest))
            .unwrap_or(path);
    }
    path
}

fn file_name_string(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("invalid file name in path {}", path.display()))
}

fn file_modified_unix_ms(path: &Path) -> Result<u64> {
    let modified = fs::metadata(path)?.modified()?;
    let duration = modified.duration_since(UNIX_EPOCH)?;
    Ok(duration.as_millis().try_into().unwrap_or(u64::MAX))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VJepaConfig;
    use std::net::TcpListener;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    type B = burn::backend::NdArray<f32>;

    fn write_tiny_package(root: &Path, shard_bytes: u64) -> (PathBuf, BurnpackPartsReport) {
        let device = Default::default();
        let config = VJepaConfig::tiny_for_tests();
        let model = VJepa2_1Model::<B>::new(&config, &device);
        let burnpack = root.join("jepa.bpk");
        save_vjepa_burnpack(&model, &burnpack).expect("save bpk");
        let parts =
            write_burnpack_parts_for_browser(&burnpack, shard_bytes, true).expect("write parts");
        let manifest = BurnJepaPipelinePackageManifest {
            model_kind: BurnJepaPackageModelKind::Base,
            record_dtype: Some("f16".to_string()),
            jepa_config: config,
            model_base_url: "http://127.0.0.1/model".to_string(),
            ..BurnJepaPipelinePackageManifest::default()
        }
        .with_burnpack_paths(&burnpack);
        let manifest_path = root.join("manifest.json");
        write_pipeline_package_manifest(&manifest_path, &manifest).expect("write manifest");
        (manifest_path, parts)
    }

    fn write_tiny_anyup_package(root: &Path, shard_bytes: u64) -> (PathBuf, BurnpackPartsReport) {
        let device = Default::default();
        let config = AnyUpConfig::tiny_for_tests();
        let model = AnyUp::<B>::new(config.clone(), &device).expect("tiny AnyUp");
        let burnpack = root.join("anyup.bpk");
        save_anyup_burnpack(&model, &burnpack).expect("save anyup bpk");
        let parts =
            write_burnpack_parts_for_browser(&burnpack, shard_bytes, true).expect("write parts");
        let manifest = BurnAnyUpPackageManifest {
            record_dtype: Some("f16".to_string()),
            anyup_config: config,
            model_base_url: "http://127.0.0.1/anyup".to_string(),
            ..BurnAnyUpPackageManifest::default()
        }
        .with_burnpack_paths(&burnpack);
        let manifest_path = root.join("manifest.json");
        write_anyup_package_manifest(&manifest_path, &manifest).expect("write manifest");
        (manifest_path, parts)
    }

    #[test]
    fn model_profiles_resolve_distinct_cdn_routes() {
        assert_eq!(
            BurnJepaModelProfile::from_str("base").expect("base profile"),
            BurnJepaModelProfile::Vjepa21Base
        );
        assert_eq!(
            BurnJepaModelProfile::from_str("ttt").expect("ttt profile"),
            BurnJepaModelProfile::Vjepa21Ttt
        );
        assert_eq!(
            burn_jepa_model_profile_base_url(BurnJepaModelProfile::Vjepa21Base),
            "https://aberration.technology/model/burn_jepa/vjepa2_1_base"
        );
        assert_eq!(
            burn_jepa_model_profile_base_url(BurnJepaModelProfile::Vjepa21Ttt),
            DEFAULT_BURN_JEPA_MODEL_BASE_URL
        );
        assert_ne!(
            burn_jepa_model_profile_base_url(BurnJepaModelProfile::Vjepa21Base),
            burn_jepa_model_profile_base_url(BurnJepaModelProfile::Vjepa21Ttt)
        );
        assert_eq!(
            serde_json::to_string(&BurnJepaModelProfile::Vjepa21Base).expect("serialize base"),
            "\"vjepa2_1_base\""
        );
        assert_eq!(
            serde_json::from_str::<BurnJepaModelProfile>("\"vjepa21_ttt\"")
                .expect("deserialize ttt alias"),
            BurnJepaModelProfile::Vjepa21Ttt
        );
    }

    #[test]
    fn anyup_model_profile_resolves_cdn_route() {
        assert_eq!(
            BurnAnyUpModelProfile::from_str("paper").expect("anyup profile"),
            BurnAnyUpModelProfile::AnyupMultiBackbone
        );
        assert_eq!(
            burn_anyup_model_profile_base_url(BurnAnyUpModelProfile::AnyupMultiBackbone),
            DEFAULT_BURN_ANYUP_MODEL_BASE_URL
        );
        assert_eq!(
            serde_json::to_string(&BurnAnyUpModelProfile::AnyupMultiBackbone)
                .expect("serialize anyup"),
            "\"anyup_multi_backbone\""
        );
        assert_eq!(
            serde_json::from_str::<BurnAnyUpModelProfile>("\"multi-backbone\"")
                .expect("deserialize anyup alias"),
            BurnAnyUpModelProfile::AnyupMultiBackbone
        );
    }

    #[test]
    fn tiny_anyup_burnpack_parts_roundtrip() {
        let device = Default::default();
        let config = AnyUpConfig::tiny_for_tests();
        let model = AnyUp::<B>::new(config.clone(), &device).expect("tiny AnyUp");
        let temp = tempfile::tempdir().expect("tempdir");
        let burnpack = temp.path().join("tiny-anyup.bpk");
        save_anyup_burnpack(&model, &burnpack).expect("save anyup bpk");
        let dtype_counts = burnpack_dtype_counts(&burnpack).expect("dtype counts");
        assert!(dtype_counts.get("F16").copied().unwrap_or(0) > 0);
        assert_eq!(dtype_counts.get("F32").copied().unwrap_or(0), 0);
        let report = write_burnpack_parts_for_browser(&burnpack, 1024, true).expect("write parts");
        let parts_dtype_counts =
            burnpack_parts_dtype_counts(&report.manifest_path).expect("parts dtype counts");
        assert_eq!(parts_dtype_counts, dtype_counts);
        let parts = report
            .part_paths
            .iter()
            .map(fs::read)
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("read parts");
        let (loaded, result) =
            load_anyup_burnpack_parts::<B>(&config, &parts, &device).expect("load parts");
        assert!(!result.applied.is_empty());
        assert!(result.missing.is_empty());
        assert!(result.errors.is_empty());
        let image = Tensor::<B, 4>::zeros([1, 3, 16, 16], &device);
        let features = Tensor::<B, 4>::zeros([1, config.qk_dim, 4, 4], &device);
        let output = loaded.forward(image, features, Some([16, 16]), Some(4));
        assert_eq!(output.shape().dims::<4>(), [1, config.qk_dim, 16, 16]);
        let values = output
            .slice([0..1, 0..1, 0..2, 0..2])
            .to_data()
            .to_vec::<f32>()
            .expect("output sample");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn anyup_deploy_bundle_contains_clean_cdn_assets() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source");
        fs::create_dir_all(&source).expect("source dir");
        let (manifest_path, parts) = write_tiny_anyup_package(&source, 1024);
        let output = temp.path().join("deploy");
        let report =
            write_burn_anyup_model_deploy_bundle(&manifest_path, &output, false).expect("bundle");

        assert_eq!(report.manifest_path, output.join("manifest.json"));
        assert!(report.manifest_path.exists());
        assert!(report.parts_manifest_path.exists());
        assert_eq!(report.part_paths.len(), parts.part_paths.len());
        assert!(!output.join("anyup.bpk").exists());
        let manifest = BurnAnyUpPackageManifest::from_json_str(
            &fs::read_to_string(&report.manifest_path).unwrap(),
        )
        .expect("manifest");
        assert_eq!(manifest.parts_manifest, "anyup.bpk.parts.json");
        let parts_manifest = read_parts_manifest(&report.parts_manifest_path).expect("parts");
        assert!(
            parts_manifest
                .parts
                .iter()
                .all(|part| !part.path.contains('/'))
        );
    }

    #[test]
    fn tiny_vjepa_burnpack_parts_roundtrip() {
        let device = Default::default();
        let config = VJepaConfig::tiny_for_tests();
        let model = VJepa2_1Model::<B>::new(&config, &device);
        let temp = tempfile::tempdir().expect("tempdir");
        let burnpack = temp.path().join("tiny-jepa.bpk");
        save_vjepa_burnpack(&model, &burnpack).expect("save bpk");
        let dtype_counts = burnpack_dtype_counts(&burnpack).expect("dtype counts");
        assert!(dtype_counts.get("F16").copied().unwrap_or(0) > 0);
        assert_eq!(dtype_counts.get("F32").copied().unwrap_or(0), 0);
        let report = write_burnpack_parts_for_browser(&burnpack, 1024, true).expect("write parts");
        let parts_dtype_counts =
            burnpack_parts_dtype_counts(&report.manifest_path).expect("parts dtype counts");
        assert_eq!(parts_dtype_counts, dtype_counts);
        assert_eq!(
            report.manifest_path,
            burnpack_parts_manifest_path(&burnpack)
        );
        assert!(!report.part_paths.is_empty());
        assert!(manifest_has_all_parts(&report.manifest_path, Some(&burnpack)).expect("complete"));
        let parts = report
            .part_paths
            .iter()
            .map(fs::read)
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("read parts");
        let (_loaded, result) =
            load_vjepa_burnpack_parts::<B>(&config, &parts, &device).expect("load parts");
        assert!(!result.applied.is_empty());
        assert!(
            result.missing.is_empty(),
            "complete part bundles should not report stale per-shard missing tensors"
        );
        assert!(result.errors.is_empty());
    }

    #[test]
    fn tiny_vjepa_burnpack_parts_inference_roundtrip() {
        let device = Default::default();
        let config = VJepaConfig::tiny_for_tests();
        let model = VJepa2_1Model::<B>::new(&config, &device);
        let temp = tempfile::tempdir().expect("tempdir");
        let burnpack = temp.path().join("tiny-jepa.bpk");
        save_vjepa_burnpack(&model, &burnpack).expect("save bpk");
        let report = write_burnpack_parts_for_browser(&burnpack, 1024, true).expect("write parts");
        let parts = report
            .part_paths
            .iter()
            .map(fs::read)
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("read parts");
        let (loaded, result) =
            load_vjepa_burnpack_parts::<B>(&config, &parts, &device).expect("load parts");
        assert!(result.errors.is_empty());
        assert!(result.missing.is_empty());
        let shape = crate::VJepaRgbaVideoShape::new(1, 4, 32, 32);
        let mut rgba = vec![0u8; shape.num_bytes()];
        for index in (0..rgba.len()).step_by(4) {
            let pixel = index / 4;
            rgba[index] = (pixel % 256) as u8;
            rgba[index + 1] = ((pixel >> 2) % 256) as u8;
            rgba[index + 2] = 127;
            rgba[index + 3] = 255;
        }
        let video = crate::rgba_video_to_tensor::<B>(&rgba, shape, &device).expect("rgba tensor");
        let output = loaded.encode_video(video, None);
        assert_eq!(output.tokens.shape().dims::<3>(), [1, 8, 32]);
        let values = output
            .tokens
            .slice([0..1, 0..4, 0..8])
            .to_data()
            .to_vec::<f32>()
            .expect("token sample");
        assert_eq!(values.len(), 32);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn package_manifest_serializes_public_runtime_urls() {
        let config = VJepaConfig::tiny_for_tests();
        let manifest = BurnJepaPipelinePackageManifest {
            model_kind: BurnJepaPackageModelKind::Ttt,
            record_dtype: Some("f16".to_string()),
            burnpack: "vjepa_ttt.bpk".to_string(),
            parts_manifest: "vjepa_ttt.bpk.parts.json".to_string(),
            model_base_url: DEFAULT_BURN_JEPA_MODEL_BASE_URL.to_string(),
            jepa_config: config,
            ttt_config: Some(TttEncoderConfig::default()),
            ..BurnJepaPipelinePackageManifest::default()
        };
        let json = manifest.to_json_string().expect("manifest json");
        assert!(json.contains("https://aberration.technology/model/burn_jepa/vjepa2_1_ttt"));
        assert!(json.contains("\"record_dtype\": \"f16\""));
        assert!(json.contains("vjepa_ttt.bpk.parts.json"));
        let decoded = BurnJepaPipelinePackageManifest::from_json_str(&json).expect("decode");
        assert_eq!(decoded.model_kind, BurnJepaPackageModelKind::Ttt);
        assert_eq!(decoded.record_dtype.as_deref(), Some("f16"));
        assert!(decoded.ttt_config.is_some());
    }

    #[test]
    fn deploy_bundle_contains_clean_cdn_assets() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source");
        fs::create_dir_all(&source).expect("source dir");
        let (manifest_path, parts) = write_tiny_package(&source, 1024);
        let output = temp.path().join("deploy");
        let report =
            write_burn_jepa_model_deploy_bundle(&manifest_path, &output, false).expect("bundle");

        assert_eq!(report.manifest_path, output.join("manifest.json"));
        assert!(report.manifest_path.exists());
        assert!(report.parts_manifest_path.exists());
        assert_eq!(report.part_paths.len(), parts.part_paths.len());
        assert!(!output.join("jepa.bpk").exists());
        let manifest = BurnJepaPipelinePackageManifest::from_json_str(
            &fs::read_to_string(&report.manifest_path).unwrap(),
        )
        .expect("manifest");
        assert_eq!(manifest.parts_manifest, "jepa.bpk.parts.json");
        let parts_manifest = read_parts_manifest(&report.parts_manifest_path).expect("parts");
        assert!(
            parts_manifest
                .parts
                .iter()
                .all(|part| !part.path.contains('/'))
        );
    }

    #[test]
    fn native_bootstrap_downloads_and_reuses_sharded_package_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("remote");
        fs::create_dir_all(&source).expect("remote dir");
        let (_manifest_path, parts) = write_tiny_package(&source, 1024);
        assert!(parts.part_paths.len() > 1);
        let server = TestServer::serve(source);
        let cache_root = temp.path().join("cache");
        let config = BurnJepaModelBootstrapConfig {
            cache_root: Some(cache_root.clone()),
            model_profile: BurnJepaModelProfile::Vjepa21Base,
            model_base_url: server.base_url.clone(),
            manifest_url: None,
        };

        let first = resolve_or_bootstrap_burn_jepa_model_package_with_config(&config)
            .expect("bootstrap package");
        assert_eq!(first.manifest_path, cache_root.join("manifest.json"));
        assert!(first.parts_manifest_path.exists());
        assert_eq!(first.part_paths.len(), parts.part_paths.len());
        assert!(burn_jepa_model_package_cache_complete(&first.manifest_path).unwrap());
        let requests = server.requests.load(Ordering::SeqCst);
        assert!(requests >= 2 + parts.part_paths.len());

        server.requests.store(0, Ordering::SeqCst);
        let second = resolve_or_bootstrap_burn_jepa_model_package_with_config(&config)
            .expect("reuse package cache");
        assert_eq!(second.manifest_path, first.manifest_path);
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn native_anyup_bootstrap_downloads_and_reuses_sharded_package_cache() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("remote-anyup");
        fs::create_dir_all(&source).expect("remote dir");
        let (_manifest_path, parts) = write_tiny_anyup_package(&source, 1024);
        assert!(parts.part_paths.len() > 1);
        let server = TestServer::serve(source);
        let cache_root = temp.path().join("anyup-cache");
        let config = BurnAnyUpModelBootstrapConfig {
            cache_root: Some(cache_root.clone()),
            model_profile: BurnAnyUpModelProfile::AnyupMultiBackbone,
            model_base_url: server.base_url.clone(),
            manifest_url: None,
        };

        let first = resolve_or_bootstrap_burn_anyup_model_package_with_config(&config)
            .expect("bootstrap AnyUp package");
        assert_eq!(first.manifest_path, cache_root.join("manifest.json"));
        assert!(first.parts_manifest_path.exists());
        assert_eq!(first.part_paths.len(), parts.part_paths.len());
        assert!(burn_anyup_model_package_cache_complete(&first.manifest_path).unwrap());
        let requests = server.requests.load(Ordering::SeqCst);
        assert!(requests >= 2 + parts.part_paths.len());

        server.requests.store(0, Ordering::SeqCst);
        let second = resolve_or_bootstrap_burn_anyup_model_package_with_config(&config)
            .expect("reuse AnyUp package cache");
        assert_eq!(second.manifest_path, first.manifest_path);
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
    }

    struct TestServer {
        base_url: String,
        requests: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn serve(root: PathBuf) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind server");
            listener.set_nonblocking(true).expect("nonblocking");
            let addr = listener.local_addr().expect("addr");
            let requests = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let thread_requests = requests.clone();
            let thread_stop = stop.clone();
            let handle = thread::spawn(move || {
                while !thread_stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            thread_requests.fetch_add(1, Ordering::SeqCst);
                            let mut buffer = [0u8; 2048];
                            let read = stream.read(&mut buffer).unwrap_or(0);
                            let request = String::from_utf8_lossy(&buffer[..read]);
                            let path = request
                                .lines()
                                .next()
                                .and_then(|line| line.split_whitespace().nth(1))
                                .unwrap_or("/");
                            let rel = path.trim_start_matches('/').split('?').next().unwrap_or("");
                            let file_path = root.join(rel);
                            let (status, body) = match fs::read(&file_path) {
                                Ok(bytes) => ("200 OK", bytes),
                                Err(_) => ("404 Not Found", Vec::new()),
                            };
                            let header = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(&body);
                            let _ = stream.flush();
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                base_url: format!("http://{addr}"),
                requests,
                stop,
                handle: Some(handle),
            }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let _ = std::net::TcpStream::connect(
                self.base_url
                    .strip_prefix("http://")
                    .expect("server url has host"),
            );
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
