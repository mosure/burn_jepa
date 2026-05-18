use bevy::prelude::Resource;
use burn_jepa::{FeatureFrameEncodePath, FeatureFrameMetrics, TokenGridShape};

#[cfg(test)]
use crate::micros_to_ms;
use crate::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncoderSource, BevyJepaFrameSource,
    BevyJepaMaskSource,
};

#[derive(Resource, Clone, Debug, Default)]
pub struct BevyJepaMetrics {
    pub frame_index: u64,
    pub frame_ready: bool,
    pub encoder_source: BevyJepaEncoderSource,
    pub encode_path: FeatureFrameEncodePath,
    pub frame_source: BevyJepaFrameSource,
    pub camera_frame_received: bool,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub context_tokens: usize,
    pub dense_tokens: usize,
    pub grid_height: usize,
    pub grid_width: usize,
    pub patch_size: usize,
    pub stage_metrics: FeatureFrameMetrics,
    pub encode_us: u64,
    pub cache_update_us: u64,
    pub token_view_us: u64,
    pub anyup_context_us: u64,
    pub anyup_decode_us: u64,
    pub low_res_pca_us: u64,
    pub pca_update_us: u64,
    pub pca_sample_window_frames: usize,
    pub pca_sample_frames: usize,
    pub high_res_pca_us: u64,
    pub display_tensor_us: u64,
    pub total_us: u64,
    pub viewer_total_us: u64,
    pub pca_update_applied: bool,
    pub input_frame_index: u64,
    pub input_frames_seen: u64,
    pub completed_frames: u64,
    pub high_res_frames: u64,
    pub input_fps: f64,
    pub low_res_fps: f64,
    pub high_res_fps: f64,
    pub in_flight_frames: usize,
    pub queue_dropped_frames: usize,
    pub queue_overwritten_frames: usize,
    pub stale_completions: usize,
    pub ttt_state_fast_weight_rms: Option<f64>,
    pub ttt_token_spatial_std_rms: Option<f64>,
    pub ttt_mean_pairwise_token_cosine: Option<f64>,
    pub ttt_collapse_score: Option<f64>,
    pub ttt_collapse_guard_triggers: u64,
    pub ttt_collapse_guard_triggered: bool,
    pub last_error: Option<String>,
}

impl BevyJepaMetrics {
    pub fn density(&self) -> f64 {
        if self.dense_tokens == 0 {
            0.0
        } else {
            self.context_tokens as f64 / self.dense_tokens as f64
        }
    }

    pub fn fps(&self) -> f64 {
        self.viewer_fps()
    }

    pub fn core_fps(&self) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.total_us as f64
        }
    }

    pub fn viewer_fps(&self) -> f64 {
        if self.viewer_total_us == 0 {
            self.core_fps()
        } else {
            1_000_000.0 / self.viewer_total_us as f64
        }
    }

    pub fn aligns_with_stage_metrics(&self) -> bool {
        self.encode_us == self.stage_metrics.encode_us
            && self.cache_update_us == self.stage_metrics.cache_update_us
            && self.token_view_us == self.stage_metrics.token_view_us
            && self.anyup_context_us == self.stage_metrics.anyup_context_us
            && self.anyup_decode_us == self.stage_metrics.anyup_decode_us
            && self.low_res_pca_us == self.stage_metrics.low_res_pca_project_us
            && self.pca_update_us == self.stage_metrics.pca_update_us
            && self.pca_sample_window_frames == self.stage_metrics.pca_sample_window_frames
            && self.pca_sample_frames == self.stage_metrics.pca_sample_frames
            && self.high_res_pca_us == self.stage_metrics.pca_project_us
            && self.total_us == self.stage_metrics.total_us
            && self.encode_path == self.stage_metrics.encode_path
            && self.context_tokens == self.stage_metrics.sparse_width
            && self.dense_tokens == self.stage_metrics.dense_tokens_per_frame
            && self.pca_update_applied == self.stage_metrics.pca_update_applied
            && self.ttt_collapse_guard_triggers
                == self.stage_metrics.ttt_runtime.collapse_guard_triggers
    }
}

#[derive(Clone, Debug)]
pub struct BevyJepaStepOutput {
    pub metrics: BevyJepaMetrics,
}

#[derive(Clone, Copy)]
pub(super) struct MetricFrameContext {
    pub frame_index: u64,
    pub encoder_source: BevyJepaEncoderSource,
    pub frame_source: BevyJepaFrameSource,
    pub camera_frame_received: bool,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub context_tokens: usize,
    pub dense_tokens: usize,
    pub grid: TokenGridShape,
    pub patch_size: usize,
}

pub(super) fn bevy_metrics_from_stage(
    frame: MetricFrameContext,
    stage_metrics: FeatureFrameMetrics,
    display_tensor_us: u64,
    viewer_total_us: u64,
) -> BevyJepaMetrics {
    let token_stability = stage_metrics.ttt_runtime.token_stability.clone();
    BevyJepaMetrics {
        frame_index: frame.frame_index,
        frame_ready: true,
        encoder_source: frame.encoder_source,
        encode_path: stage_metrics.encode_path,
        frame_source: frame.frame_source,
        camera_frame_received: frame.camera_frame_received,
        mask_source: frame.mask_source,
        display_transfer: frame.display_transfer,
        context_tokens: frame.context_tokens,
        dense_tokens: frame.dense_tokens,
        grid_height: frame.grid.height,
        grid_width: frame.grid.width,
        patch_size: frame.patch_size,
        encode_us: stage_metrics.encode_us,
        cache_update_us: stage_metrics.cache_update_us,
        token_view_us: stage_metrics.token_view_us,
        anyup_context_us: stage_metrics.anyup_context_us,
        anyup_decode_us: stage_metrics.anyup_decode_us,
        low_res_pca_us: stage_metrics.low_res_pca_project_us,
        pca_update_us: stage_metrics.pca_update_us,
        pca_sample_window_frames: stage_metrics.pca_sample_window_frames,
        pca_sample_frames: stage_metrics.pca_sample_frames,
        high_res_pca_us: stage_metrics.pca_project_us,
        display_tensor_us,
        total_us: stage_metrics.total_us,
        viewer_total_us,
        pca_update_applied: stage_metrics.pca_update_applied,
        input_frame_index: frame.frame_index,
        input_frames_seen: 0,
        completed_frames: 0,
        high_res_frames: u64::from(stage_metrics.anyup_decode_us > 0),
        input_fps: 0.0,
        low_res_fps: 0.0,
        high_res_fps: 0.0,
        in_flight_frames: 0,
        queue_dropped_frames: 0,
        queue_overwritten_frames: 0,
        stale_completions: 0,
        ttt_state_fast_weight_rms: stage_metrics.ttt_runtime.state_fast_weight_rms,
        ttt_token_spatial_std_rms: token_stability
            .as_ref()
            .map(|metrics| metrics.token_spatial_std_rms),
        ttt_mean_pairwise_token_cosine: token_stability
            .as_ref()
            .map(|metrics| metrics.mean_pairwise_token_cosine),
        ttt_collapse_score: token_stability
            .as_ref()
            .map(|metrics| metrics.collapse_score),
        ttt_collapse_guard_triggers: stage_metrics.ttt_runtime.collapse_guard_triggers,
        ttt_collapse_guard_triggered: stage_metrics.ttt_runtime.collapse_guard_triggered,
        last_error: None,
        stage_metrics,
    }
}

#[cfg(test)]
pub(super) fn format_metrics_waiting_line() -> String {
    format_metrics_line(&BevyJepaConfig::default(), &BevyJepaMetrics::default())
}

#[cfg(test)]
pub(super) fn format_metrics_line(config: &BevyJepaConfig, metrics: &BevyJepaMetrics) -> String {
    let source = metrics_source_status(config, metrics);
    format!(
        "src:{:<18} model:{:<13} enc:{:<15} mask:{:<10} seq:{:>5}/{:<5} grid:{:>3}x{:<3} p:{:<2} sparse:{:>5.1}% fps:{:>5.1}/{:>5.1}/{:>5.1} infl:{:>1} drop:{:>4} ovw:{:>4} view:{:>7.2}ms core:{:>7.2}ms disp:{:>6.2}ms enc:{:>6.2}ms cache:{:>6.2}ms any:{:>6.2}/{:>6.2}ms pca:{:>6.2}/{:>6.2}ms ttt:{:>4.2}/{:>4.2} upd:{:<3} {:>3}/{:<3}f",
        source,
        metrics.encoder_source,
        metrics.encode_path,
        metrics.mask_source,
        metrics.input_frame_index,
        metrics.frame_index,
        metrics.grid_height,
        metrics.grid_width,
        metrics.patch_size,
        metrics.density() * 100.0,
        metrics.input_fps,
        metrics.low_res_fps,
        metrics.high_res_fps,
        metrics.in_flight_frames,
        metrics.queue_dropped_frames,
        metrics.queue_overwritten_frames,
        micros_to_ms(metrics.viewer_total_us),
        micros_to_ms(metrics.total_us),
        micros_to_ms(metrics.display_tensor_us),
        micros_to_ms(metrics.encode_us),
        micros_to_ms(metrics.cache_update_us),
        micros_to_ms(metrics.anyup_context_us),
        micros_to_ms(metrics.anyup_decode_us),
        micros_to_ms(metrics.low_res_pca_us),
        micros_to_ms(metrics.high_res_pca_us),
        metrics.ttt_collapse_score.unwrap_or(0.0),
        metrics.ttt_mean_pairwise_token_cosine.unwrap_or(0.0),
        if metrics.pca_update_applied {
            "yes"
        } else {
            "no"
        },
        metrics.pca_sample_frames,
        metrics.pca_sample_window_frames,
    )
}

pub(super) fn metrics_source_status(
    config: &BevyJepaConfig,
    metrics: &BevyJepaMetrics,
) -> &'static str {
    if !metrics.frame_ready {
        return match config.source {
            BevyJepaFrameSource::Camera => "camera:wait",
            BevyJepaFrameSource::StaticImage => "static:wait",
            BevyJepaFrameSource::SyntheticLocalMotion => "synthetic:wait",
        };
    }

    match metrics.frame_source {
        BevyJepaFrameSource::Camera => "camera:live",
        BevyJepaFrameSource::StaticImage => "static:ready",
        BevyJepaFrameSource::SyntheticLocalMotion => "synthetic:ready",
    }
}
