use bevy::app::AppExit;
use bevy_jepa::{BevyJepaConfig, run_app};

#[cfg(not(target_arch = "wasm32"))]
use bevy_jepa::{
    BevyJepaDisplayTransfer, BevyJepaMaskSource, DEFAULT_ANYUP_CHUNK_SIZE, DEFAULT_CONTEXT_DENSITY,
    DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_DIFF_THRESHOLD, DEFAULT_PCA_UPDATE_EVERY,
};

#[cfg(not(target_arch = "wasm32"))]
use clap::{ArgAction, Parser};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[arg(long, default_value_t = BevyJepaMaskSource::Autogaze)]
    mask_source: BevyJepaMaskSource,
    #[arg(long, default_value_t = BevyJepaDisplayTransfer::Gpu)]
    display_transfer: BevyJepaDisplayTransfer,
    #[arg(long, default_value_t = DEFAULT_IMAGE_SIZE)]
    image_size: usize,
    #[arg(long, default_value_t = DEFAULT_CONTEXT_DENSITY)]
    context_density: f32,
    #[arg(long, default_value_t = DEFAULT_PATCH_DIFF_THRESHOLD)]
    patch_diff_threshold: f32,
    #[arg(long, default_value_t = DEFAULT_ANYUP_CHUNK_SIZE)]
    anyup_q_chunk_size: usize,
    #[arg(long, default_value_t = DEFAULT_PCA_UPDATE_EVERY)]
    pca_update_every: u64,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    show_metrics: bool,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    measure_stages: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    sync_measurements: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<Cli> for BevyJepaConfig {
    fn from(cli: Cli) -> Self {
        Self {
            mask_source: cli.mask_source,
            display_transfer: cli.display_transfer,
            image_size: cli.image_size,
            context_density: cli.context_density,
            patch_diff_threshold: cli.patch_diff_threshold,
            anyup_q_chunk_size: cli.anyup_q_chunk_size,
            pca_update_every: cli.pca_update_every,
            show_metrics: cli.show_metrics,
            measure_stages: cli.measure_stages,
            sync_measurements: cli.sync_measurements,
            ..Default::default()
        }
    }
}

fn main() -> AppExit {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        return run_app(BevyJepaConfig::default());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        run_app(Cli::parse().into())
    }
}
