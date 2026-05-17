#![cfg(not(target_arch = "wasm32"))]

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};

use anyhow::{Context, Result};
use bevy_jepa::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncodePath, BevyJepaEncoderSource,
    BevyJepaFrameSource, BevyJepaHeadlessPipeline, BevyJepaMaskSource, BevyJepaSparseEncodeMode,
    FeatureFrameViewerConfig, JepaBevyBackend, JepaBevyDevice, platform,
};
use burn::tensor::backend::Backend;
use burn_jepa::FeatureFrameRequest;
use image::{Rgba, RgbaImage};

const PATCH_SIZE: usize = 16;
const DEFAULT_FRAMES: usize = 18;
const DEFAULT_WARMUP: usize = 4;
const THRESHOLDS: [f32; 4] = [0.0, 0.01, 0.03, 0.06];
const RESOLUTIONS: [usize; 2] = [256, 512];

#[derive(Clone, Copy)]
struct MotionCase {
    label: &'static str,
    base_density: f32,
    jitter_density: f32,
    reshuffle: bool,
    low_contrast: bool,
}

const MOTION_CASES: [MotionCase; 6] = [
    MotionCase {
        label: "static",
        base_density: 0.0,
        jitter_density: 0.0,
        reshuffle: false,
        low_contrast: false,
    },
    MotionCase {
        label: "stable_10",
        base_density: 0.10,
        jitter_density: 0.0,
        reshuffle: false,
        low_contrast: false,
    },
    MotionCase {
        label: "stable_30",
        base_density: 0.30,
        jitter_density: 0.0,
        reshuffle: false,
        low_contrast: false,
    },
    MotionCase {
        label: "stable_60",
        base_density: 0.60,
        jitter_density: 0.0,
        reshuffle: false,
        low_contrast: false,
    },
    MotionCase {
        label: "jitter_60",
        base_density: 0.60,
        jitter_density: 0.18,
        reshuffle: true,
        low_contrast: false,
    },
    MotionCase {
        label: "low_contrast_60",
        base_density: 0.60,
        jitter_density: 0.0,
        reshuffle: false,
        low_contrast: true,
    },
];

#[derive(Clone, Copy)]
struct SweepMode {
    label: &'static str,
    sparse_encode_mode: BevyJepaSparseEncodeMode,
}

const SWEEP_MODES: [SweepMode; 2] = [
    SweepMode {
        label: "bucketed256",
        sparse_encode_mode: BevyJepaSparseEncodeMode::BucketedContext,
    },
    SweepMode {
        label: "exact",
        sparse_encode_mode: BevyJepaSparseEncodeMode::Exact,
    },
];

#[derive(Clone, Copy)]
struct FrameSample {
    write_density: f64,
    encode_density: f64,
    write_tokens: usize,
    encode_tokens: usize,
    write_dense_ordered: bool,
    encode_dense_ordered: bool,
    outer_us: u64,
    viewer_us: u64,
    encode_us: u64,
    cache_us: u64,
    low_res_pca_us: u64,
    pca_update_us: u64,
    display_us: u64,
}

#[derive(Clone)]
struct SummaryRow {
    mode: &'static str,
    resolution: usize,
    threshold: f32,
    motion: &'static str,
    frames: usize,
    mean_write_density: f64,
    mean_encode_density: f64,
    p50_write_density: f64,
    p95_write_density: f64,
    unique_write_widths: usize,
    unique_encode_widths: usize,
    dense_write_frames: usize,
    dense_encode_frames: usize,
    p50_outer_ms: f64,
    p95_outer_ms: f64,
    max_outer_ms: f64,
    p50_viewer_ms: f64,
    p95_viewer_ms: f64,
    p95_encode_ms: f64,
    p95_cache_ms: f64,
    p95_low_res_pca_ms: f64,
    p95_pca_update_ms: f64,
    p95_display_ms: f64,
}

fn main() -> Result<()> {
    let frames = env_usize("BURN_JEPA_FPS_STABILITY_FRAMES", DEFAULT_FRAMES);
    let warmup = env_usize("BURN_JEPA_FPS_STABILITY_WARMUP", DEFAULT_WARMUP);
    let output_dir = env::var_os("BURN_JEPA_FPS_STABILITY_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bevy-jepa-fps-stability"));
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let sender = ensure_camera_sender();
    let mut rows = Vec::new();
    for mode in SWEEP_MODES {
        for resolution in RESOLUTIONS {
            for threshold in THRESHOLDS {
                for motion in MOTION_CASES {
                    rows.push(run_case(
                        &sender, mode, resolution, threshold, motion, frames, warmup,
                    )?);
                }
            }
        }
    }

    let csv = output_dir.join("fps-stability-summary.csv");
    let markdown = output_dir.join("fps-stability-summary.md");
    fs::write(&csv, rows_to_csv(&rows)).with_context(|| format!("write {}", csv.display()))?;
    fs::write(&markdown, rows_to_markdown(&rows))
        .with_context(|| format!("write {}", markdown.display()))?;
    println!("wrote {}", csv.display());
    println!("wrote {}", markdown.display());
    print_headline(&rows);
    Ok(())
}

fn ensure_camera_sender() -> mpsc::SyncSender<RgbaImage> {
    if let Some(sender) = platform::camera::SAMPLE_SENDER.get() {
        return sender.clone();
    }
    let (sender, receiver) = mpsc::sync_channel(1);
    let _ = platform::camera::SAMPLE_RECEIVER.set(Arc::new(Mutex::new(receiver)));
    let _ = platform::camera::SAMPLE_SENDER.set(sender.clone());
    sender
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    sender: &mpsc::SyncSender<RgbaImage>,
    mode: SweepMode,
    resolution: usize,
    threshold: f32,
    motion: MotionCase,
    frames: usize,
    warmup: usize,
) -> Result<SummaryRow> {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TinyTest,
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        display_transfer: BevyJepaDisplayTransfer::Gpu,
        pipeline: FeatureFrameViewerConfig {
            encode_path: BevyJepaEncodePath::Auto,
            image_size: resolution,
            context_density: 1.0,
            min_context_density: 0.0,
            bootstrap_context_density: 1.0,
            patch_diff_threshold: threshold,
            sparse_encode_mode: mode.sparse_encode_mode,
            high_res_pca_every: 0,
            measure_stages: true,
            sync_measurements: true,
            ..FeatureFrameViewerConfig::default()
        },
        show_metrics: false,
        ..BevyJepaConfig::default()
    };
    let mut pipeline = BevyJepaHeadlessPipeline::new(config, device.clone());
    let total_frames = frames + warmup;
    let mut samples = Vec::with_capacity(frames);
    for frame in 0..total_frames {
        send_frame(
            sender,
            motion_frame(resolution, threshold, motion, frame as u64),
        )?;
        let started = Instant::now();
        let output = pipeline.step_with_display_request(FeatureFrameRequest::low_res())?;
        JepaBevyBackend::sync(&device)?;
        let outer_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        if frame >= warmup {
            let metrics = output.metrics;
            let dense_tokens = metrics.dense_tokens.max(1);
            let encode_tokens = encode_width_for_mode(
                metrics.context_tokens,
                dense_tokens,
                mode.sparse_encode_mode,
            );
            samples.push(FrameSample {
                write_density: metrics.density(),
                encode_density: encode_tokens as f64 / dense_tokens as f64,
                write_tokens: metrics.context_tokens,
                encode_tokens,
                write_dense_ordered: metrics.context_tokens == metrics.dense_tokens,
                encode_dense_ordered: encode_tokens == metrics.dense_tokens,
                outer_us,
                viewer_us: metrics.viewer_total_us,
                encode_us: metrics.encode_us,
                cache_us: metrics.cache_update_us,
                low_res_pca_us: metrics.low_res_pca_us,
                pca_update_us: metrics.pca_update_us,
                display_us: metrics.display_tensor_us,
            });
        }
    }
    Ok(summarize(
        mode.label,
        resolution,
        threshold,
        motion.label,
        samples,
    ))
}

fn send_frame(sender: &mpsc::SyncSender<RgbaImage>, frame: RgbaImage) -> Result<()> {
    match sender.try_send(frame) {
        Ok(()) => Ok(()),
        Err(mpsc::TrySendError::Full(frame)) => {
            while platform::camera::receive_image().is_some() {}
            sender
                .try_send(frame)
                .map_err(|err| anyhow::anyhow!("send synthetic camera frame: {err}"))
        }
        Err(mpsc::TrySendError::Disconnected(_)) => {
            anyhow::bail!("synthetic camera receiver disconnected")
        }
    }
}

fn motion_frame(
    image_size: usize,
    threshold: f32,
    motion: MotionCase,
    frame_index: u64,
) -> RgbaImage {
    let grid = image_size / PATCH_SIZE;
    let dense_tokens = grid * grid;
    let density = if threshold <= 0.0 {
        1.0
    } else {
        motion_density(motion, frame_index)
    };
    let active_tokens = ((dense_tokens as f32) * density).round() as usize;
    let seed = if motion.reshuffle { frame_index } else { 0 };
    let active = active_patch_set(dense_tokens, active_tokens.min(dense_tokens), seed);
    let mut image = RgbaImage::from_pixel(
        image_size as u32,
        image_size as u32,
        Rgba([96, 96, 96, 255]),
    );
    let amplitude = if motion.low_contrast { 10i16 } else { 72i16 };
    let sign = if frame_index.is_multiple_of(2) {
        1i16
    } else {
        -1i16
    };
    for token in active {
        let row = token / grid;
        let col = token % grid;
        let hash = mix64(token as u64 ^ frame_index.rotate_left(13));
        let channel = (hash % 3) as usize;
        let mut rgb = [96i16, 96i16, 96i16];
        rgb[channel] += sign * amplitude;
        rgb[(channel + 1) % 3] -= sign * (amplitude / 2);
        rgb[(channel + 2) % 3] += sign * (amplitude / 3);
        let rgba = Rgba([
            rgb[0].clamp(0, 255) as u8,
            rgb[1].clamp(0, 255) as u8,
            rgb[2].clamp(0, 255) as u8,
            255,
        ]);
        for y in row * PATCH_SIZE..(row + 1) * PATCH_SIZE {
            for x in col * PATCH_SIZE..(col + 1) * PATCH_SIZE {
                image.put_pixel(x as u32, y as u32, rgba);
            }
        }
    }
    image
}

fn motion_density(motion: MotionCase, frame_index: u64) -> f32 {
    let phase = ((frame_index % 11) as f32 - 5.0) / 5.0;
    (motion.base_density + motion.jitter_density * phase).clamp(0.0, 1.0)
}

fn encode_width_for_mode(
    write_tokens: usize,
    dense_tokens: usize,
    mode: BevyJepaSparseEncodeMode,
) -> usize {
    match mode {
        BevyJepaSparseEncodeMode::Exact => write_tokens,
        BevyJepaSparseEncodeMode::BucketedContext => {
            if write_tokens >= dense_tokens || dense_tokens < 256 {
                return write_tokens;
            }
            let bucket = 256.min((dense_tokens / 4).max(1));
            write_tokens
                .div_ceil(bucket)
                .saturating_mul(bucket)
                .min(dense_tokens)
        }
    }
}

fn active_patch_set(dense_tokens: usize, active_tokens: usize, seed: u64) -> Vec<usize> {
    let mut ranked = (0..dense_tokens)
        .map(|index| {
            (
                mix64(index as u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                index,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|(score, index)| (*score, *index));
    ranked
        .into_iter()
        .take(active_tokens)
        .map(|(_, index)| index)
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn summarize(
    mode: &'static str,
    resolution: usize,
    threshold: f32,
    motion: &'static str,
    samples: Vec<FrameSample>,
) -> SummaryRow {
    let frames = samples.len();
    let write_widths = samples
        .iter()
        .map(|sample| sample.write_tokens)
        .collect::<BTreeSet<_>>();
    let encode_widths = samples
        .iter()
        .map(|sample| sample.encode_tokens)
        .collect::<BTreeSet<_>>();
    let dense_write_frames = samples
        .iter()
        .filter(|sample| sample.write_dense_ordered)
        .count();
    let dense_encode_frames = samples
        .iter()
        .filter(|sample| sample.encode_dense_ordered)
        .count();
    let mean_write_density = mean_f64(samples.iter().map(|sample| sample.write_density));
    let mean_encode_density = mean_f64(samples.iter().map(|sample| sample.encode_density));
    let mut write_densities = samples
        .iter()
        .map(|sample| (sample.write_density * 1_000_000.0) as u64)
        .collect::<Vec<_>>();
    SummaryRow {
        mode,
        resolution,
        threshold,
        motion,
        frames,
        mean_write_density,
        mean_encode_density,
        p50_write_density: percentile(&mut write_densities, 0.50) as f64 / 1_000_000.0,
        p95_write_density: percentile(&mut write_densities, 0.95) as f64 / 1_000_000.0,
        unique_write_widths: write_widths.len(),
        unique_encode_widths: encode_widths.len(),
        dense_write_frames,
        dense_encode_frames,
        p50_outer_ms: percentile_us(&samples, |sample| sample.outer_us, 0.50),
        p95_outer_ms: percentile_us(&samples, |sample| sample.outer_us, 0.95),
        max_outer_ms: percentile_us(&samples, |sample| sample.outer_us, 1.0),
        p50_viewer_ms: percentile_us(&samples, |sample| sample.viewer_us, 0.50),
        p95_viewer_ms: percentile_us(&samples, |sample| sample.viewer_us, 0.95),
        p95_encode_ms: percentile_us(&samples, |sample| sample.encode_us, 0.95),
        p95_cache_ms: percentile_us(&samples, |sample| sample.cache_us, 0.95),
        p95_low_res_pca_ms: percentile_us(&samples, |sample| sample.low_res_pca_us, 0.95),
        p95_pca_update_ms: percentile_us(&samples, |sample| sample.pca_update_us, 0.95),
        p95_display_ms: percentile_us(&samples, |sample| sample.display_us, 0.95),
    }
}

fn percentile_us(samples: &[FrameSample], value: impl Fn(&FrameSample) -> u64, p: f64) -> f64 {
    let mut values = samples.iter().map(value).collect::<Vec<_>>();
    percentile(&mut values, p) as f64 / 1000.0
}

fn percentile(values: &mut [u64], p: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = (((values.len() - 1) as f64) * p.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn mean_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn rows_to_csv(rows: &[SummaryRow]) -> String {
    let mut out = String::from(
        "mode,resolution,threshold,motion,frames,mean_write_density,mean_encode_density,p50_write_density,p95_write_density,unique_write_widths,unique_encode_widths,dense_write_frames,dense_encode_frames,p50_outer_ms,p95_outer_ms,max_outer_ms,p50_viewer_ms,p95_viewer_ms,p95_encode_ms,p95_cache_ms,p95_low_res_pca_ms,p95_pca_update_ms,p95_display_ms\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
            row.mode,
            row.resolution,
            row.threshold,
            row.motion,
            row.frames,
            row.mean_write_density,
            row.mean_encode_density,
            row.p50_write_density,
            row.p95_write_density,
            row.unique_write_widths,
            row.unique_encode_widths,
            row.dense_write_frames,
            row.dense_encode_frames,
            row.p50_outer_ms,
            row.p95_outer_ms,
            row.max_outer_ms,
            row.p50_viewer_ms,
            row.p95_viewer_ms,
            row.p95_encode_ms,
            row.p95_cache_ms,
            row.p95_low_res_pca_ms,
            row.p95_pca_update_ms,
            row.p95_display_ms,
        ));
    }
    out
}

fn rows_to_markdown(rows: &[SummaryRow]) -> String {
    let mut out = String::from(
        "| Mode | Resolution | Threshold | Motion | Mean write density | Mean encode density | Unique write widths | Unique encode widths | Dense write frames | Dense encode frames | p50 outer ms | p95 outer ms | max outer ms | p95 encode ms | p95 cache ms | p95 PCA upd ms | p95 display ms |\n",
    );
    out.push_str(
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {:.2} | {} | {:.1}% | {:.1}% | {} | {} | {}/{} | {}/{} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            row.mode,
            row.resolution,
            row.threshold,
            row.motion,
            row.mean_write_density * 100.0,
            row.mean_encode_density * 100.0,
            row.unique_write_widths,
            row.unique_encode_widths,
            row.dense_write_frames,
            row.frames,
            row.dense_encode_frames,
            row.frames,
            row.p50_outer_ms,
            row.p95_outer_ms,
            row.max_outer_ms,
            row.p95_encode_ms,
            row.p95_cache_ms,
            row.p95_pca_update_ms,
            row.p95_display_ms,
        ));
    }
    out
}

fn print_headline(rows: &[SummaryRow]) {
    let mut by_resolution = BTreeMap::<usize, Vec<&SummaryRow>>::new();
    for row in rows
        .iter()
        .filter(|row| row.mode == "bucketed256" && (row.threshold - 0.03).abs() < f32::EPSILON)
    {
        by_resolution.entry(row.resolution).or_default().push(row);
    }
    for (resolution, rows) in by_resolution {
        println!("\ndefault bucketed threshold 0.03, {resolution}px:");
        for row in rows {
            println!(
                "  {:<16} density {:>5.1}% widths {:>2} p50 {:>6.2} ms p95 {:>6.2} ms max {:>6.2} ms",
                row.motion,
                row.mean_write_density * 100.0,
                row.unique_encode_widths,
                row.p50_outer_ms,
                row.p95_outer_ms,
                row.max_outer_ms
            );
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}
