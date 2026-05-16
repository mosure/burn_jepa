use bevy::app::AppExit;
use bevy_jepa::{BevyJepaConfig, BevyJepaFrameSource, run_app};

#[cfg(not(target_arch = "wasm32"))]
use bevy_jepa::{
    BevyJepaDisplayTransfer, BevyJepaEncodePath, BevyJepaEncoderSource, BevyJepaMaskSource,
    DEFAULT_ANYUP_CHUNK_SIZE, DEFAULT_BOOTSTRAP_CONTEXT_DENSITY, DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, DEFAULT_CONTEXT_DENSITY,
    DEFAULT_HIGH_RES_PCA_EVERY, DEFAULT_IMAGE_SIZE, DEFAULT_MIN_CONTEXT_DENSITY,
    DEFAULT_PATCH_DIFF_THRESHOLD, DEFAULT_PCA_UPDATE_EVERY, DEFAULT_TTT_MODEL_PATH,
    DEFAULT_VJEPA21_CHECKPOINT_DIR, DEFAULT_VJEPA21_CONFIG_PATH, DEFAULT_VJEPA21_WEIGHTS_NAME,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_jepa::AnyUpAttentionMode;

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
    #[arg(long, default_value = DEFAULT_TTT_MODEL_PATH)]
    ttt_model: PathBuf,
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
    #[arg(long)]
    anyup_weights: Option<PathBuf>,
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
    #[arg(long, default_value_t = DEFAULT_ANYUP_CHUNK_SIZE)]
    anyup_q_chunk_size: usize,
    #[arg(long, default_value_t = DEFAULT_PCA_UPDATE_EVERY)]
    pca_update_every: u64,
    #[arg(long, default_value_t = DEFAULT_HIGH_RES_PCA_EVERY)]
    high_res_pca_every: u64,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_metrics: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    measure_stages: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    sync_measurements: bool,
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
        Self {
            encoder_source: cli.encoder_source,
            encode_path: cli.encode_path,
            ttt_model_path: Some(cli.ttt_model),
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
            anyup_attention_mode: cli.anyup_attention_mode,
            mask_source: cli.mask_source,
            display_transfer: cli.display_transfer,
            image_size: cli.image_size,
            context_density: cli.context_density,
            min_context_density: cli.min_context_density,
            bootstrap_context_density: cli.bootstrap_context_density,
            patch_diff_threshold: patch_diff_threshold(
                cli.patch_diff_threshold,
                cli.patch_diff_quality,
            ),
            anyup_q_chunk_size: cli.anyup_q_chunk_size,
            pca_update_every: cli.pca_update_every,
            high_res_pca_every: cli.high_res_pca_every,
            show_metrics: cli.show_metrics,
            measure_stages: cli.measure_stages,
            sync_measurements: cli.sync_measurements,
            camera_width: cli.camera_width,
            camera_height: cli.camera_height,
            camera_fps: cli.camera_fps,
            ..Default::default()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn patch_diff_threshold(threshold: f32, quality: Option<f32>) -> f32 {
    quality
        .map(|quality| 1.0 - quality.clamp(0.0, 1.0))
        .unwrap_or(threshold)
        .clamp(0.0, 1.0)
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
    }
}

fn main() -> AppExit {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        return run_app(BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            ..BevyJepaConfig::default()
        });
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
