use bevy::app::AppExit;
use bevy_jepa::{
    BevyJepaAnyUpModelPackageProfile, BevyJepaConfig, BevyJepaEncoderSource, BevyJepaFrameSource,
    BevyJepaModelPackageProfile, run_app,
};
use burn_jepa::AnyUpAttentionMode;

#[cfg(not(target_arch = "wasm32"))]
use bevy_jepa::{
    BevyJepaDisplayTransfer, BevyJepaEncodePath, BevyJepaMaskSource, BevyJepaSparseEncodeMode,
    DEFAULT_ANYUP_CHUNK_SIZE, DEFAULT_BOOTSTRAP_CONTEXT_DENSITY, DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, DEFAULT_CONTEXT_DENSITY,
    DEFAULT_HIGH_RES_PCA_EVERY, DEFAULT_IMAGE_SIZE, DEFAULT_MIN_CONTEXT_DENSITY,
    DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES, DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY, DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY,
    DEFAULT_PATCH_DIFF_REFRESH_ENABLED, DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER, DEFAULT_PATCH_DIFF_THRESHOLD,
    DEFAULT_PCA_MIN_SAMPLE_FRAMES, DEFAULT_PCA_SAMPLE_WINDOW_FRAMES, DEFAULT_PCA_UPDATE_EVERY,
    DEFAULT_PCA_UPDATE_ITERATIONS, DEFAULT_PREWARM_SHAPE_BUCKETS,
    DEFAULT_SPARSE_MASK_BUCKET_TOKENS, DEFAULT_VJEPA21_CHECKPOINT_DIR, DEFAULT_VJEPA21_CONFIG_PATH,
    DEFAULT_VJEPA21_WEIGHTS_NAME,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_jepa::DEFAULT_BURN_JEPA_MODEL_BASE_URL;
#[cfg(not(target_arch = "wasm32"))]
use burn_jepa::{
    DEFAULT_BURN_ANYUP_MODEL_BASE_URL, FeatureFrameViewerConfig, PatchDiffRefreshConfig,
    TttRuntimeCollapseGuardAction, TttRuntimeStateConfig, patch_diff_threshold_from_quality,
};

#[cfg(not(target_arch = "wasm32"))]
use clap::{ArgAction, Parser};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[arg(long, default_value_t = BevyJepaEncoderSource::TrainedTtt)]
    encoder_source: BevyJepaEncoderSource,
    #[arg(long, default_value_t = BevyJepaEncodePath::Auto)]
    encode_path: BevyJepaEncodePath,
    #[arg(
        long,
        help = "Local burn_jepa package manifest.json. If omitted, the viewer checks BURN_JEPA_MODEL_MANIFEST, target/burn-jepa-web/model/{model_profile}/manifest.json, then the auto-downloaded cache before falling back to an explicit --ttt-model."
    )]
    model_manifest: Option<PathBuf>,
    #[arg(
        long,
        help = "Exact local cache directory for auto-downloaded burn_jepa model shards."
    )]
    model_cache_dir: Option<PathBuf>,
    #[arg(long, visible_alias = "model-name", default_value_t = BevyJepaModelPackageProfile::default())]
    model_profile: BevyJepaModelPackageProfile,
    #[arg(long, default_value = DEFAULT_BURN_JEPA_MODEL_BASE_URL)]
    model_base_url: String,
    #[arg(long, action = ArgAction::SetTrue, help = "Disable native model package auto-download/cache lookup.")]
    no_model_download: bool,
    #[arg(long, help = "Legacy local .mpk TTT checkpoint override.")]
    ttt_model: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_VJEPA21_CHECKPOINT_DIR)]
    jepa_checkpoint_dir: PathBuf,
    #[arg(long, default_value = DEFAULT_VJEPA21_CONFIG_PATH)]
    jepa_config: PathBuf,
    #[arg(long, default_value = DEFAULT_VJEPA21_WEIGHTS_NAME)]
    jepa_weights_name: String,
    #[arg(long)]
    source: Option<BevyJepaFrameSource>,
    #[arg(long)]
    image_path: Option<PathBuf>,
    #[arg(
        long,
        help = "AnyUp checkpoint path. If omitted, the viewer auto-uses target/burn-anyup-checkpoints/anyup_multi_backbone.pth when present."
    )]
    anyup_weights: Option<PathBuf>,
    #[arg(
        long,
        help = "Local burn_anyup package manifest.json. If omitted, AnyUp checks BURN_ANYUP_MODEL_MANIFEST, target/burn_anyup/{anyup_model_profile}/manifest.json, then the auto-downloaded cache."
    )]
    anyup_model_manifest: Option<PathBuf>,
    #[arg(
        long,
        help = "Exact local cache directory for auto-downloaded burn_anyup model shards."
    )]
    anyup_model_cache_dir: Option<PathBuf>,
    #[arg(long, visible_alias = "anyup-model-name", default_value_t = BevyJepaAnyUpModelPackageProfile::default())]
    anyup_model_profile: BevyJepaAnyUpModelPackageProfile,
    #[arg(long, default_value = DEFAULT_BURN_ANYUP_MODEL_BASE_URL)]
    anyup_model_base_url: String,
    #[arg(long, action = ArgAction::SetTrue, help = "Disable native AnyUp package auto-download/cache lookup.")]
    no_anyup_model_download: bool,
    #[arg(long, default_value_t = AnyUpAttentionMode::EfficientLocal)]
    anyup_attention_mode: AnyUpAttentionMode,
    #[arg(long, default_value_t = BevyJepaMaskSource::PatchDiff)]
    mask_source: BevyJepaMaskSource,
    #[arg(long, default_value_t = BevyJepaDisplayTransfer::Gpu)]
    display_transfer: BevyJepaDisplayTransfer,
    #[arg(long, default_value_t = DEFAULT_IMAGE_SIZE)]
    image_size: usize,
    #[arg(long, default_value_t = DEFAULT_CONTEXT_DENSITY)]
    context_density: f32,
    #[arg(long, default_value_t = DEFAULT_MIN_CONTEXT_DENSITY)]
    min_context_density: f32,
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_CONTEXT_DENSITY)]
    bootstrap_context_density: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_THRESHOLD)]
    patch_diff_threshold: f32,
    #[arg(long)]
    patch_diff_quality: Option<f32>,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY)]
    patch_diff_dense_fallback_density: f32,
    #[arg(
        long,
        action = ArgAction::SetTrue,
        default_value_t = DEFAULT_PATCH_DIFF_REFRESH_ENABLED,
        help = "Enable bounded age/subthreshold/blue-noise patch-diff refresh tokens."
    )]
    patch_diff_refresh: bool,
    #[arg(long = "no-patch-diff-refresh", action = ArgAction::SetTrue, hide = true)]
    no_patch_diff_refresh: bool,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY)]
    patch_diff_subthreshold_decay: f32,
    #[arg(long, default_value_t = 1.0)]
    patch_diff_subthreshold_gain: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER)]
    patch_diff_subthreshold_trigger: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY)]
    patch_diff_subthreshold_max_density: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES)]
    patch_diff_age_refresh_interval_frames: u64,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY)]
    patch_diff_age_refresh_max_density: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY)]
    patch_diff_blue_noise_refresh_density: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY)]
    patch_diff_refresh_max_density: f32,
    #[arg(
        long,
        default_value_t = BevyJepaSparseEncodeMode::BucketedContext,
        help = "How patch-diff masks map to JEPA encode tokens. `bucketed-context` keeps sparse writes exact but widens encoder context to reduce GPU shape churn; `exact` encodes only displayed mask tokens."
    )]
    sparse_encode_mode: BevyJepaSparseEncodeMode,
    #[arg(
        long,
        default_value_t = DEFAULT_SPARSE_MASK_BUCKET_TOKENS,
        help = "Token bucket width used only with --sparse-encode-mode bucketed-context."
    )]
    sparse_mask_bucket_tokens: usize,
    #[arg(
        long,
        action = ArgAction::SetTrue,
        default_value_t = DEFAULT_PREWARM_SHAPE_BUCKETS,
        help = "Prewarm bucketed sparse encode widths during startup. Only applies with --sparse-encode-mode bucketed-context."
    )]
    prewarm_shape_buckets: bool,
    #[arg(long = "no-prewarm-shape-buckets", action = ArgAction::SetTrue, hide = true)]
    no_prewarm_shape_buckets: bool,
    #[arg(long, default_value_t = DEFAULT_ANYUP_CHUNK_SIZE)]
    anyup_q_chunk_size: usize,
    #[arg(long, default_value_t = DEFAULT_PCA_UPDATE_EVERY)]
    pca_update_every: u64,
    #[arg(
        long,
        default_value_t = DEFAULT_PCA_SAMPLE_WINDOW_FRAMES,
        help = "Rolling frame window used to fit the low-res PCA basis."
    )]
    pca_sample_window_frames: usize,
    #[arg(
        long,
        default_value_t = DEFAULT_PCA_MIN_SAMPLE_FRAMES,
        help = "Minimum buffered frames before the first low-res PCA basis update."
    )]
    pca_min_sample_frames: usize,
    #[arg(
        long,
        default_value_t = DEFAULT_PCA_UPDATE_ITERATIONS,
        help = "Oja update iterations per low-res PCA basis update."
    )]
    pca_update_iterations: usize,
    #[arg(long, default_value_t = DEFAULT_HIGH_RES_PCA_EVERY)]
    high_res_pca_every: u64,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_metrics: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    measure_stages: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    sync_measurements: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    ttt_runtime_enabled: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    ttt_update_fast_weight: bool,
    #[arg(long, default_value_t = TttRuntimeStateConfig::default().state_decay_per_frame)]
    ttt_state_decay_per_frame: f64,
    #[arg(long, default_value_t = TttRuntimeStateConfig::default().reset_interval_frames)]
    ttt_reset_interval_frames: u64,
    #[arg(long, default_value_t = TttRuntimeStateConfig::default().metrics_interval_frames)]
    ttt_stability_metrics_every: u64,
    #[arg(long, default_value_t = TttRuntimeStateConfig::default().collapse_guard_enabled, action = ArgAction::Set)]
    ttt_collapse_guard: bool,
    #[arg(long, default_value_t = TttRuntimeCollapseGuardAction::default())]
    ttt_collapse_guard_action: TttRuntimeCollapseGuardAction,
    #[arg(long, default_value_t = TttRuntimeStateConfig::default().collapse_guard_decay)]
    ttt_collapse_guard_decay: f64,
    #[arg(long, default_value_t = DEFAULT_CAMERA_WIDTH)]
    camera_width: u32,
    #[arg(long, default_value_t = DEFAULT_CAMERA_HEIGHT)]
    camera_height: u32,
    #[arg(long, default_value_t = DEFAULT_CAMERA_FPS)]
    camera_fps: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<Cli> for BevyJepaConfig {
    fn from(cli: Cli) -> Self {
        let model_base_url = resolve_model_profile_base_url(cli.model_profile, cli.model_base_url);
        let anyup_model_base_url =
            resolve_anyup_model_profile_base_url(cli.anyup_model_profile, cli.anyup_model_base_url);
        Self {
            encoder_source: cli.encoder_source,
            model_manifest_path: cli.model_manifest,
            model_cache_dir: cli.model_cache_dir,
            model_profile: cli.model_profile,
            model_base_url,
            model_auto_download: !cli.no_model_download,
            ttt_model_path: cli.ttt_model,
            jepa_checkpoint_dir: Some(cli.jepa_checkpoint_dir),
            jepa_config_path: Some(cli.jepa_config),
            jepa_weights_name: cli.jepa_weights_name,
            source: cli.source.unwrap_or(if cli.image_path.is_some() {
                BevyJepaFrameSource::StaticImage
            } else {
                BevyJepaFrameSource::Camera
            }),
            image_path: cli.image_path,
            anyup_weights: cli.anyup_weights,
            anyup_model_manifest_path: cli.anyup_model_manifest,
            anyup_model_cache_dir: cli.anyup_model_cache_dir,
            anyup_model_profile: cli.anyup_model_profile,
            anyup_model_base_url,
            anyup_model_auto_download: !cli.no_anyup_model_download,
            anyup_attention_mode: cli.anyup_attention_mode,
            mask_source: cli.mask_source,
            display_transfer: cli.display_transfer,
            pipeline: FeatureFrameViewerConfig {
                encode_path: cli.encode_path,
                image_size: cli.image_size,
                context_density: cli.context_density,
                min_context_density: cli.min_context_density,
                bootstrap_context_density: cli.bootstrap_context_density,
                patch_diff_threshold: patch_diff_threshold_from_quality(
                    cli.patch_diff_threshold,
                    cli.patch_diff_quality,
                ),
                patch_diff_dense_fallback_density: cli
                    .patch_diff_dense_fallback_density
                    .clamp(0.0, 1.0),
                patch_diff_refresh: PatchDiffRefreshConfig {
                    enabled: cli.patch_diff_refresh && !cli.no_patch_diff_refresh,
                    subthreshold_decay: cli.patch_diff_subthreshold_decay.clamp(0.0, 1.0),
                    subthreshold_gain: cli.patch_diff_subthreshold_gain.max(0.0),
                    subthreshold_trigger: cli.patch_diff_subthreshold_trigger.max(1.0e-6),
                    subthreshold_max_density: cli
                        .patch_diff_subthreshold_max_density
                        .clamp(0.0, 1.0),
                    age_refresh_interval_frames: cli.patch_diff_age_refresh_interval_frames,
                    age_refresh_max_density: cli.patch_diff_age_refresh_max_density.clamp(0.0, 1.0),
                    blue_noise_refresh_density: cli
                        .patch_diff_blue_noise_refresh_density
                        .clamp(0.0, 1.0),
                    max_extra_density: cli.patch_diff_refresh_max_density.clamp(0.0, 1.0),
                    ..PatchDiffRefreshConfig::default()
                },
                sparse_encode_mode: cli.sparse_encode_mode,
                sparse_mask_bucket_tokens: cli.sparse_mask_bucket_tokens,
                prewarm_shape_buckets: cli.prewarm_shape_buckets && !cli.no_prewarm_shape_buckets,
                anyup_q_chunk_size: cli.anyup_q_chunk_size,
                pca_update_every: cli.pca_update_every,
                pca_sample_window_frames: cli.pca_sample_window_frames,
                pca_min_sample_frames: cli.pca_min_sample_frames,
                pca_update_iterations: cli.pca_update_iterations,
                high_res_pca_every: cli.high_res_pca_every,
                ttt_runtime: TttRuntimeStateConfig {
                    enabled: cli.ttt_runtime_enabled,
                    update_fast_weight: cli.ttt_update_fast_weight,
                    state_decay_per_frame: cli.ttt_state_decay_per_frame,
                    reset_interval_frames: cli.ttt_reset_interval_frames,
                    metrics_interval_frames: cli.ttt_stability_metrics_every,
                    collapse_guard_enabled: cli.ttt_collapse_guard,
                    collapse_guard_action: cli.ttt_collapse_guard_action,
                    collapse_guard_decay: cli.ttt_collapse_guard_decay,
                    ..TttRuntimeStateConfig::default()
                },
                measure_stages: cli.measure_stages,
                sync_measurements: cli.sync_measurements,
            },
            show_metrics: cli.show_metrics,
            camera_width: cli.camera_width,
            camera_height: cli.camera_height,
            camera_fps: cli.camera_fps,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_model_profile_base_url(
    model_profile: BevyJepaModelPackageProfile,
    model_base_url: String,
) -> String {
    if model_base_url == DEFAULT_BURN_JEPA_MODEL_BASE_URL {
        burn_jepa::burn_jepa_model_profile_base_url(model_profile)
    } else {
        model_base_url
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_anyup_model_profile_base_url(
    model_profile: BevyJepaAnyUpModelPackageProfile,
    model_base_url: String,
) -> String {
    if model_base_url == DEFAULT_BURN_ANYUP_MODEL_BASE_URL {
        burn_jepa::burn_anyup_model_profile_base_url(model_profile)
    } else {
        model_base_url
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn patch_diff_quality_changes_threshold_not_min_density() {
        let config = BevyJepaConfig::from(Cli::parse_from([
            "bevy_jepa",
            "--patch-diff-quality",
            "0.85",
            "--min-context-density",
            "0.125",
        ]));

        assert_eq!(config.min_context_density, 0.125);
        assert!((config.patch_diff_threshold - 0.15).abs() <= 1.0e-6);
        assert!(
            (config.patch_diff_dense_fallback_density - DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY)
                .abs()
                <= 1.0e-6
        );
    }

    #[test]
    fn patch_diff_dense_fallback_density_is_configurable() {
        let config = BevyJepaConfig::from(Cli::parse_from([
            "bevy_jepa",
            "--patch-diff-dense-fallback-density",
            "0.9",
        ]));

        assert!((config.patch_diff_dense_fallback_density - 0.9).abs() <= 1.0e-6);
    }

    #[test]
    fn sparse_mask_bucket_tokens_is_configurable() {
        let config = BevyJepaConfig::from(Cli::parse_from([
            "bevy_jepa",
            "--sparse-encode-mode",
            "bucketed-context",
            "--sparse-mask-bucket-tokens",
            "64",
        ]));

        assert_eq!(
            config.sparse_encode_mode,
            BevyJepaSparseEncodeMode::BucketedContext
        );
        assert_eq!(config.sparse_mask_bucket_tokens, 64);
    }

    #[test]
    fn model_manifest_is_configurable_and_ttt_model_is_explicit() {
        let default_config = BevyJepaConfig::from(Cli::parse_from(["bevy_jepa"]));
        assert!(default_config.model_manifest_path.is_none());
        assert!(default_config.ttt_model_path.is_none());
        assert!(default_config.model_auto_download);
        assert!(default_config.anyup_model_manifest_path.is_none());
        assert!(default_config.anyup_model_auto_download);
        assert_eq!(
            default_config.anyup_model_profile,
            BevyJepaAnyUpModelPackageProfile::AnyupMultiBackbone
        );
        assert!(
            default_config
                .anyup_model_base_url
                .ends_with("/anyup_multi_backbone")
        );
        assert_eq!(
            default_config.model_profile,
            BevyJepaModelPackageProfile::Vjepa21Ttt
        );

        let config = BevyJepaConfig::from(Cli::parse_from([
            "bevy_jepa",
            "--model-manifest",
            "target/burn-jepa-web/model/manifest.json",
            "--model-cache-dir",
            "target/burn-jepa-cache",
            "--model-base-url",
            "http://127.0.0.1:8091",
            "--no-model-download",
            "--ttt-model",
            "target/local-ttt.mpk",
            "--anyup-model-manifest",
            "target/burn_anyup/anyup_multi_backbone/manifest.json",
            "--anyup-model-cache-dir",
            "target/burn-anyup-cache",
            "--anyup-model-base-url",
            "http://127.0.0.1:8092",
            "--no-anyup-model-download",
            "--anyup-attention-mode",
            "upstream-masked",
        ]));
        assert_eq!(
            config.model_manifest_path.as_deref(),
            Some(std::path::Path::new(
                "target/burn-jepa-web/model/manifest.json"
            ))
        );
        assert_eq!(
            config.ttt_model_path.as_deref(),
            Some(std::path::Path::new("target/local-ttt.mpk"))
        );
        assert_eq!(
            config.model_cache_dir.as_deref(),
            Some(std::path::Path::new("target/burn-jepa-cache"))
        );
        assert_eq!(config.model_base_url, "http://127.0.0.1:8091");
        assert!(!config.model_auto_download);
        assert_eq!(
            config.anyup_model_manifest_path.as_deref(),
            Some(std::path::Path::new(
                "target/burn_anyup/anyup_multi_backbone/manifest.json"
            ))
        );
        assert_eq!(
            config.anyup_model_cache_dir.as_deref(),
            Some(std::path::Path::new("target/burn-anyup-cache"))
        );
        assert_eq!(config.anyup_model_base_url, "http://127.0.0.1:8092");
        assert!(!config.anyup_model_auto_download);
        assert_eq!(
            config.anyup_attention_mode,
            burn_jepa::AnyUpAttentionMode::UpstreamMasked
        );

        let base_config =
            BevyJepaConfig::from(Cli::parse_from(["bevy_jepa", "--model-profile", "base"]));
        assert_eq!(
            base_config.model_profile,
            BevyJepaModelPackageProfile::Vjepa21Base
        );
        assert!(base_config.model_base_url.ends_with("/vjepa2_1_base"));
    }

    #[test]
    fn prewarm_shape_buckets_is_configurable() {
        let config =
            BevyJepaConfig::from(Cli::parse_from(["bevy_jepa", "--prewarm-shape-buckets"]));

        assert!(config.prewarm_shape_buckets);
    }

    #[test]
    fn sparse_encode_mode_defaults_to_bucketed_context() {
        let config = BevyJepaConfig::from(Cli::parse_from(["bevy_jepa"]));

        assert_eq!(
            config.sparse_encode_mode,
            BevyJepaSparseEncodeMode::BucketedContext
        );
        assert!(config.prewarm_shape_buckets);
    }

    #[test]
    fn pca_update_window_is_configurable() {
        let config = BevyJepaConfig::from(Cli::parse_from([
            "bevy_jepa",
            "--pca-update-every",
            "2",
            "--pca-sample-window-frames",
            "12",
            "--pca-min-sample-frames",
            "3",
            "--pca-update-iterations",
            "2",
        ]));

        assert_eq!(config.pca_update_every, 2);
        assert_eq!(config.pca_sample_window_frames, 12);
        assert_eq!(config.pca_min_sample_frames, 3);
        assert_eq!(config.pca_update_iterations, 2);
    }
}

fn main() -> AppExit {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        return run_app(wasm_config_from_url());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let config = BevyJepaConfig::from(Cli::parse());
        if config.source == BevyJepaFrameSource::Camera {
            let request = camera_request_for_config(&config);
            std::thread::spawn(move || {
                bevy_jepa::platform::camera::native_camera_thread_with_request(request);
            });
        }
        run_app(config)
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_config_from_url() -> BevyJepaConfig {
    let mut config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        ..BevyJepaConfig::default()
    };
    if param_bool("load-model").is_some_and(|load| !load) {
        config.encoder_source = BevyJepaEncoderSource::TinyTest;
        config.ttt_model_path = None;
        config.jepa_checkpoint_dir = None;
        config.jepa_config_path = None;
    }
    if let Some(value) = query_param("encoder-source")
        && let Ok(source) = value.parse::<BevyJepaEncoderSource>()
    {
        config.encoder_source = source;
    }
    if let Some(value) = query_param("model-profile")
        .or_else(|| query_param("model-name"))
        .or_else(|| query_param("model"))
        && let Ok(profile) = value.parse::<BevyJepaModelPackageProfile>()
    {
        config.model_profile = profile;
        config.model_base_url = burn_jepa::burn_jepa_model_profile_base_url(profile);
        config.model_manifest_path = None;
        config.ttt_model_path = None;
        config.encoder_source = match profile {
            BevyJepaModelPackageProfile::Vjepa21Base => BevyJepaEncoderSource::BaseCheckpoint,
            BevyJepaModelPackageProfile::Vjepa21Ttt => BevyJepaEncoderSource::TrainedTtt,
        };
    }
    if let Some(value) =
        query_param("anyup-model-profile").or_else(|| query_param("anyup-model-name"))
        && let Ok(profile) = value.parse::<BevyJepaAnyUpModelPackageProfile>()
    {
        config.anyup_model_profile = profile;
        config.anyup_model_base_url = burn_jepa::burn_anyup_model_profile_base_url(profile);
    }
    if let Some(value) =
        query_param("anyup-model-base").or_else(|| query_param("anyup-model-base-url"))
    {
        config.anyup_model_base_url = value;
    }
    if let Some(value) = query_param("anyup-attention-mode").or_else(|| query_param("anyup-mode"))
        && let Ok(mode) = value.parse::<AnyUpAttentionMode>()
    {
        config.anyup_attention_mode = mode;
    }
    if let Some(value) = query_param("source")
        && let Ok(source) = value.parse::<BevyJepaFrameSource>()
    {
        config.source = source;
    }
    if let Some(image_size) = param_usize("image-size").or_else(|| param_usize("size")) {
        config.image_size = image_size;
    }
    if let Some(threshold) = param_f32("patch-diff-threshold") {
        config.patch_diff_threshold = threshold;
    }
    if let Some(quality) = param_f32("patch-diff-quality") {
        config.patch_diff_threshold = burn_jepa::patch_diff_threshold_from_quality(
            config.patch_diff_threshold,
            Some(quality),
        );
    }
    if let Some(enabled) = param_bool("patch-diff-refresh") {
        config.patch_diff_refresh.enabled = enabled;
    }
    if let Some(decay) = param_f32("patch-diff-subthreshold-decay") {
        config.patch_diff_refresh.subthreshold_decay = decay.clamp(0.0, 1.0);
    }
    if let Some(trigger) = param_f32("patch-diff-subthreshold-trigger") {
        config.patch_diff_refresh.subthreshold_trigger = trigger.max(1.0e-6);
    }
    if let Some(density) = param_f32("patch-diff-subthreshold-density") {
        config.patch_diff_refresh.subthreshold_max_density = density.clamp(0.0, 1.0);
    }
    if let Some(frames) = param_u64("patch-diff-age-refresh-frames") {
        config.patch_diff_refresh.age_refresh_interval_frames = frames;
    }
    if let Some(density) = param_f32("patch-diff-age-refresh-density") {
        config.patch_diff_refresh.age_refresh_max_density = density.clamp(0.0, 1.0);
    }
    if let Some(density) = param_f32("patch-diff-blue-noise-density") {
        config.patch_diff_refresh.blue_noise_refresh_density = density.clamp(0.0, 1.0);
    }
    if let Some(density) = param_f32("patch-diff-refresh-density") {
        config.patch_diff_refresh.max_extra_density = density.clamp(0.0, 1.0);
    }
    if let Some(density) = param_f32("context-density") {
        config.context_density = density.clamp(0.0, 1.0);
    }
    if let Some(density) = param_f32("min-context-density") {
        config.min_context_density = density.clamp(0.0, 1.0);
    }
    if let Some(every) = param_u64("high-res-pca-every") {
        config.high_res_pca_every = every;
    }
    config
}

#[cfg(target_arch = "wasm32")]
fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let search = search.strip_prefix('?').unwrap_or(search.as_str());
    for pair in search.split('&').filter(|entry| !entry.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        if key == name {
            return Some(percent_decode_query(value));
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn percent_decode_query(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = &value[index + 1..index + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte as char);
                    index += 3;
                } else {
                    out.push('%');
                    index += 1;
                }
            }
            byte => {
                out.push(byte as char);
                index += 1;
            }
        }
    }
    out
}

#[cfg(target_arch = "wasm32")]
fn param_bool(name: &str) -> Option<bool> {
    match query_param(name)?.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
fn param_usize(name: &str) -> Option<usize> {
    query_param(name)?.parse().ok()
}

#[cfg(target_arch = "wasm32")]
fn param_u64(name: &str) -> Option<u64> {
    query_param(name)?.parse().ok()
}

#[cfg(target_arch = "wasm32")]
fn param_f32(name: &str) -> Option<f32> {
    query_param(name)?.parse().ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn camera_request_for_config(
    config: &BevyJepaConfig,
) -> bevy_jepa::platform::camera::CameraRequest {
    bevy_jepa::platform::camera::CameraRequest::new(
        config.camera_width.max(1),
        config.camera_height.max(1),
        config.camera_fps.max(1),
    )
}
