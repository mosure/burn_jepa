use std::{env, path::PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
#[cfg(not(target_arch = "wasm32"))]
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_jepa::{
    AnyUp, AnyUpConfig, AnyUpLoadOptions, FeatureFrameJepaEncoder, VJepa2_1Model, VJepaConfig,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_jepa::{
    BurnJepaModelBootstrapConfig, DEFAULT_BURN_JEPA_MODEL_BASE_URL, TttBackpropMode,
    TttEncoderConfig, TttLayerPlacement, TttMemoryUpdateSource, TttSupervisionMode,
    VJepaLoadOptions, VJepaTttModel, load_config_from_hf_dir,
    resolve_or_bootstrap_burn_jepa_model_package,
    resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress,
};
use burn_jepa::{
    BurnJepaPackageModelKind, BurnJepaPipelinePackageManifest, load_ttt_burnpack_parts,
    load_vjepa_burnpack_parts,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_jepa::{
    read_parts_manifest, resolve_package_manifest_entry_path, resolve_part_entry_path,
};

use crate::{
    BevyJepaConfig, BevyJepaEncoderSource, DEFAULT_ANYUP_CHECKPOINT_PATH, JepaBevyBackend,
    JepaBevyDevice, log,
};

pub(super) fn load_viewer_encoder(
    config: &BevyJepaConfig,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(FeatureFrameJepaEncoder<JepaBevyBackend>, VJepaConfig)> {
    #[cfg(target_arch = "wasm32")]
    if let Some(package) = crate::platform::camera::model_package() {
        return load_wasm_package_encoder(package, image_size, device);
    }

    #[cfg(not(target_arch = "wasm32"))]
    if config.encoder_source != BevyJepaEncoderSource::TinyTest {
        if let Some(package_manifest_path) = effective_model_manifest_path(config)? {
            return load_native_package_encoder(&package_manifest_path, image_size, device);
        }
    }

    match config.encoder_source {
        BevyJepaEncoderSource::TinyTest => {
            let model_config = tiny_viewer_model_config(image_size);
            let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
            Ok((FeatureFrameJepaEncoder::base(jepa), model_config))
        }
        BevyJepaEncoderSource::BaseCheckpoint => {
            #[cfg(target_arch = "wasm32")]
            {
                bail!(
                    "base-checkpoint wasm encoder requires a burn_jepa .bpk package; pass ?model-manifest=... or ?load-model=false for tiny-test"
                );
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (jepa, mut model_config) =
                    load_base_checkpoint_model(config, image_size, device)?;
                model_config.image_size = image_size;
                Ok((FeatureFrameJepaEncoder::base(jepa), model_config))
            }
        }
        BevyJepaEncoderSource::TrainedTtt => {
            #[cfg(target_arch = "wasm32")]
            {
                bail!(
                    "trained-ttt wasm encoder requires a burn_jepa .bpk package; pass ?model-manifest=... or ?load-model=false for tiny-test"
                );
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let ttt_model_path = effective_ttt_model_path(config)?;
                if !ttt_model_path.exists() {
                    bail!(
                        "explicit trained TTT JEPA encoder checkpoint `{}` does not exist; pass a valid --ttt-model / BURN_JEPA_TTT_MODEL or use a sharded .bpk package with --model-manifest / BURN_JEPA_MODEL_MANIFEST",
                        ttt_model_path.display()
                    );
                }
                let model_config = viewer_model_config(config, image_size)?;
                let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
                let ttt = VJepaTttModel::from_model(base, production_ttt_config(), device)?
                    .load_file(
                        ttt_model_path.clone(),
                        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
                        device,
                    )
                    .with_context(|| {
                        format!(
                            "load trained TTT JEPA encoder `{}`",
                            ttt_model_path.display()
                        )
                    })?;
                Ok((FeatureFrameJepaEncoder::ttt(ttt), model_config))
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn load_wasm_package_encoder(
    package: crate::platform::camera::WasmModelPackage,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(FeatureFrameJepaEncoder<JepaBevyBackend>, VJepaConfig)> {
    let mut manifest = BurnJepaPipelinePackageManifest::from_json_str(&package.manifest_json)?;
    load_package_encoder_from_manifest_and_parts(&mut manifest, &package.parts, image_size, device)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native_package_encoder(
    manifest_path: &Path,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(FeatureFrameJepaEncoder<JepaBevyBackend>, VJepaConfig)> {
    let manifest_json = fs::read_to_string(manifest_path).with_context(|| {
        format!(
            "read burn_jepa package manifest `{}`",
            manifest_path.display()
        )
    })?;
    let mut manifest = BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)
        .with_context(|| {
            format!(
                "parse burn_jepa package manifest `{}`",
                manifest_path.display()
            )
        })?;
    let parts_manifest_path =
        resolve_package_manifest_entry_path(manifest_path, &manifest.parts_manifest)?;
    let parts_manifest = read_parts_manifest(&parts_manifest_path)?;
    let parts = parts_manifest
        .parts
        .iter()
        .map(|entry| {
            let path = resolve_part_entry_path(&parts_manifest_path, &entry.path)?;
            fs::read(&path).with_context(|| format!("read burnpack shard `{}`", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    log(&format!(
        "bevy_jepa: loading {} burn_jepa package `{}` from {} shard(s)",
        manifest.model_kind.as_str(),
        manifest_path.display(),
        parts.len()
    ));
    load_package_encoder_from_manifest_and_parts(&mut manifest, &parts, image_size, device)
}

fn load_package_encoder_from_manifest_and_parts(
    manifest: &mut BurnJepaPipelinePackageManifest,
    parts: &[Vec<u8>],
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(FeatureFrameJepaEncoder<JepaBevyBackend>, VJepaConfig)> {
    let mut model_config = manifest.jepa_config.clone();
    model_config.image_size = image_size;
    match manifest.model_kind {
        BurnJepaPackageModelKind::Base => {
            let (model, report) =
                load_vjepa_burnpack_parts::<JepaBevyBackend>(&model_config, parts, device)
                    .context("load V-JEPA burnpack parts")?;
            ensure_apply_report_ok(&report)?;
            Ok((FeatureFrameJepaEncoder::base(model), model_config))
        }
        BurnJepaPackageModelKind::Ttt => {
            let ttt_config = manifest
                .ttt_config
                .take()
                .context("TTT wasm burn_jepa manifest is missing ttt_config")?;
            let (model, report) = load_ttt_burnpack_parts::<JepaBevyBackend>(
                &model_config,
                ttt_config,
                parts,
                device,
            )
            .context("load TTT V-JEPA burnpack parts")?;
            ensure_apply_report_ok(&report)?;
            Ok((FeatureFrameJepaEncoder::ttt(model), model_config))
        }
    }
}

fn ensure_apply_report_ok(report: &burn_jepa::BurnStoreApplyResult) -> Result<()> {
    if !report.errors.is_empty() {
        bail!(
            "burn_jepa package load reported tensor errors: {:?}",
            report.errors
        );
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_base_checkpoint_model(
    config: &BevyJepaConfig,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(VJepa2_1Model<JepaBevyBackend>, VJepaConfig)> {
    if let Some(checkpoint_dir) = &config.jepa_checkpoint_dir {
        let checkpoint_dir = resolve_repo_relative_path(checkpoint_dir);
        let options = VJepaLoadOptions {
            weights_name: config.jepa_weights_name.clone(),
            ..VJepaLoadOptions::default()
        };
        let (model, model_config, _report) = options
            .load_model(&checkpoint_dir, device)
            .with_context(|| {
                format!("load V-JEPA 2.1 checkpoint `{}`", checkpoint_dir.display())
            })?;
        return Ok((model, model_config));
    }

    let model_config = viewer_model_config(config, image_size)?;
    Ok((
        VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device),
        model_config,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn viewer_model_config(config: &BevyJepaConfig, image_size: usize) -> Result<VJepaConfig> {
    let mut model_config = if let Some(config_path) = &config.jepa_config_path {
        let config_path = resolve_repo_relative_path(config_path);
        if config_path.exists() {
            VJepaConfig::from_json_file(&config_path)
                .with_context(|| format!("load V-JEPA 2.1 config `{}`", config_path.display()))?
        } else if config.encoder_source == BevyJepaEncoderSource::TrainedTtt {
            bail!(
                "V-JEPA 2.1 config `{}` does not exist; pass --jepa-config or --encoder-source tiny-test",
                config_path.display()
            );
        } else {
            VJepaConfig::default()
        }
    } else if let Some(checkpoint_dir) = &config.jepa_checkpoint_dir {
        let checkpoint_dir = resolve_repo_relative_path(checkpoint_dir);
        load_config_from_hf_dir(&checkpoint_dir, &VJepaLoadOptions::default().config_name)
            .with_context(|| {
                format!("load V-JEPA 2.1 config from `{}`", checkpoint_dir.display())
            })?
    } else {
        VJepaConfig::default()
    };
    model_config.image_size = image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    Ok(model_config)
}

pub(super) fn tiny_viewer_model_config(image_size: usize) -> VJepaConfig {
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn effective_ttt_model_path(config: &BevyJepaConfig) -> Result<PathBuf> {
    if let Some(path) = env::var_os("BURN_JEPA_TTT_MODEL") {
        return Ok(resolve_repo_relative_path(PathBuf::from(path)));
    }
    config
        .ttt_model_path
        .as_ref()
        .map(resolve_repo_relative_path)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "trained TTT JEPA encoder requires a burn_jepa .bpk package manifest or an explicit .mpk checkpoint; pass --model-manifest, set BURN_JEPA_MODEL_MANIFEST, export with `burn-jepa export-bpk`, or pass --ttt-model / BURN_JEPA_TTT_MODEL"
            )
        })
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn effective_model_manifest_path(config: &BevyJepaConfig) -> Result<Option<PathBuf>> {
    for (label, path) in [
        (
            "BURN_JEPA_MODEL_MANIFEST",
            env::var_os("BURN_JEPA_MODEL_MANIFEST").map(PathBuf::from),
        ),
        (
            "BURN_JEPA_MODEL_PACKAGE_MANIFEST",
            env::var_os("BURN_JEPA_MODEL_PACKAGE_MANIFEST").map(PathBuf::from),
        ),
        ("--model-manifest", config.model_manifest_path.clone()),
    ] {
        if let Some(path) = path {
            let path = resolve_repo_relative_path(path);
            if path.exists() {
                return Ok(Some(path));
            }
            bail!(
                "burn_jepa package manifest `{}` from {label} does not exist",
                path.display()
            );
        }
    }

    for path in [PathBuf::from(crate::DEFAULT_MODEL_MANIFEST_PATH)] {
        let path = resolve_repo_relative_path(path);
        if path.exists() {
            return Ok(Some(path));
        }
    }
    if config.model_auto_download && env_model_download_enabled() {
        let package = if config.model_cache_dir.is_none()
            && config.model_base_url == DEFAULT_BURN_JEPA_MODEL_BASE_URL
        {
            resolve_or_bootstrap_burn_jepa_model_package()
        } else {
            let bootstrap = BurnJepaModelBootstrapConfig {
                cache_root: config
                    .model_cache_dir
                    .clone()
                    .map(resolve_repo_relative_path),
                model_base_url: config.model_base_url.clone(),
                manifest_url: env::var("BURN_JEPA_MODEL_MANIFEST_URL").ok(),
            };
            resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress(
                &bootstrap,
                |message| log(&format!("bevy_jepa: {message}")),
            )
        }?;
        return Ok(Some(package.manifest_path));
    }
    Ok(None)
}

#[cfg(not(target_arch = "wasm32"))]
fn env_model_download_enabled() -> bool {
    env::var("BURN_JEPA_MODEL_DOWNLOAD")
        .ok()
        .and_then(|value| parse_bool(&value))
        .unwrap_or(true)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub(super) fn resolve_repo_relative_path(path: impl Into<PathBuf>) -> PathBuf {
    let path = expand_home_path(path.into());
    if path.is_absolute() || path.exists() {
        return path;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(workspace_root) = manifest_dir.parent().and_then(|parent| parent.parent()) else {
        return path;
    };
    let candidate = workspace_root.join(&path);
    if candidate.exists() { candidate } else { path }
}

fn expand_home_path(path: PathBuf) -> PathBuf {
    let path_string = path.to_string_lossy().into_owned();
    if path_string == "~" {
        return env::var_os("HOME").map(PathBuf::from).unwrap_or(path);
    }
    if let Some(rest) = path_string.strip_prefix("~/") {
        return env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(rest))
            .unwrap_or(path);
    }
    path
}

pub(super) fn effective_anyup_weights(config: &BevyJepaConfig) -> Option<PathBuf> {
    if let Some(path) = env::var_os("BURN_ANYUP_WEIGHTS") {
        return Some(resolve_repo_relative_path(PathBuf::from(path)));
    }
    if let Some(path) = &config.anyup_weights {
        return Some(resolve_repo_relative_path(path));
    }
    if config.encoder_source == BevyJepaEncoderSource::TinyTest {
        return None;
    }
    let default_path = resolve_repo_relative_path(DEFAULT_ANYUP_CHECKPOINT_PATH);
    default_path.exists().then_some(default_path)
}

fn ensure_anyup_load_report_has_critical_weights(
    report: &burn_jepa::AnyUpLoadReport,
    path: &std::path::Path,
) -> Result<()> {
    if !report.errors.is_empty() {
        bail!(
            "AnyUp checkpoint `{}` reported load errors: {:?}",
            path.display(),
            report.errors
        );
    }
    let loaded = |needle: &str| report.applied.iter().any(|path| path == needle);
    for critical in [
        "image_encoder.pre.weight",
        "key_encoder.pre.weight",
        "query_encoder.pre.weight",
        "key_features_encoder.pre.basis",
        "aggregation.pre.weight",
        "cross_decode.conv.weight",
        "cross_decode.cross_attn.q_proj.weight",
        "cross_decode.cross_attn.k_proj.weight",
    ] {
        if !loaded(critical) {
            bail!(
                "AnyUp checkpoint `{}` did not load critical tensor `{}`; refusing to show a misleading high-res AnyUp panel",
                path.display(),
                critical
            );
        }
    }
    Ok(())
}

pub(super) fn load_viewer_anyup(
    config: &BevyJepaConfig,
    device: &JepaBevyDevice,
) -> Result<AnyUp<JepaBevyBackend>> {
    let anyup_weights = effective_anyup_weights(config);
    let mut anyup_config = if anyup_weights.is_some() {
        AnyUpConfig::default()
    } else {
        AnyUpConfig::tiny_for_tests()
    }
    .with_attention_mode(config.anyup_attention_mode);
    anyup_config.input_dim = 3;
    let mut anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, device)
        .context("initialize AnyUp viewer model")?;
    if let Some(path) = anyup_weights.as_ref() {
        let report = AnyUpLoadOptions::default()
            .load_into(&mut anyup, path, device)
            .with_context(|| format!("load AnyUp viewer weights `{}`", path.display()))?;
        ensure_anyup_load_report_has_critical_weights(&report, path)?;
        log(&format!(
            "bevy_jepa: loaded AnyUp checkpoint `{}` ({} tensors applied)",
            path.display(),
            report.applied.len()
        ));
    } else {
        log(
            "bevy_jepa: no AnyUp weights configured or found; high-res AnyUp PCA uses the untrained tiny diagnostic module",
        );
    }
    Ok(anyup)
}

#[cfg(not(target_arch = "wasm32"))]
fn production_ttt_config() -> TttEncoderConfig {
    TttEncoderConfig {
        layer_placement: TttLayerPlacement::Explicit,
        layers: vec![3, 7, 11],
        predictor_layers: Vec::new(),
        chunk_tokens: 196,
        ttt_lr: 0.003,
        use_projection: true,
        conv_kernel: 3,
        memory_update: TttMemoryUpdateSource::SelfHidden,
        supervision: TttSupervisionMode::FinalTeacher,
        hybrid_final_steps: 1,
        rollout_blocks: 2,
        backprop_mode: TttBackpropMode::FinalFeature,
        backprop_truncate_blocks: 1,
        freeze_pretrained: true,
        ..TttEncoderConfig::default()
    }
}
