use crate::VJepaConfig;
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct JepaDatasetConfig {
    pub kind: JepaDatasetKind,
    pub sample_kind: JepaSampleKind,
    pub train_manifest: Option<PathBuf>,
    pub eval_manifest: Option<PathBuf>,
    pub frames: usize,
    pub stride: usize,
    pub image_size: usize,
    pub synthetic_len: usize,
    pub sample_limit: usize,
    pub repeat_count: usize,
    pub repeat_mode: JepaDatasetRepeatMode,
}

impl Default for JepaDatasetConfig {
    fn default() -> Self {
        Self {
            kind: JepaDatasetKind::Synthetic,
            sample_kind: JepaSampleKind::Video,
            train_manifest: None,
            eval_manifest: None,
            frames: 4,
            stride: 1,
            image_size: 32,
            synthetic_len: 16,
            sample_limit: 0,
            repeat_count: 1,
            repeat_mode: JepaDatasetRepeatMode::Preserve,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaDatasetKind {
    Synthetic,
    Manifest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaSampleKind {
    Image,
    Video,
    PairedVideo,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaDatasetRepeatMode {
    #[default]
    Preserve,
    ContinuousStreams,
    StitchedStream,
    AdversarialStitchedStream,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JepaManifestRow {
    pub clip_id: Option<String>,
    pub domain: Option<String>,
    pub start_frame: Option<usize>,
    pub fps: Option<f32>,
    pub duration: Option<f32>,
    pub caption: Option<String>,
    pub source: Option<String>,
    pub image: Option<PathBuf>,
    pub frames: Option<Vec<PathBuf>>,
    pub frame_dir: Option<PathBuf>,
    pub teacher_frames: Option<Vec<PathBuf>>,
    pub teacher_frame_dir: Option<PathBuf>,
    pub precomputed_context_indices: Option<Vec<usize>>,
    pub precomputed_target_indices: Option<Vec<usize>>,
    pub original_stream: Option<String>,
    pub cache_id: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JepaSampleMetadata {
    pub clip_id: Option<String>,
    pub domain: Option<String>,
    pub start_frame: Option<usize>,
    pub fps: Option<f32>,
    pub duration: Option<f32>,
    pub caption: Option<String>,
    pub source: Option<String>,
    pub precomputed_context_indices: Option<Vec<usize>>,
    pub precomputed_target_indices: Option<Vec<usize>>,
    pub original_stream: Option<String>,
    pub cache_id: Option<String>,
}

#[derive(Clone, Debug)]
pub enum JepaSample {
    Image {
        path: PathBuf,
        metadata: JepaSampleMetadata,
    },
    Video {
        frames: Vec<PathBuf>,
        metadata: JepaSampleMetadata,
    },
    PairedVideo {
        student_frames: Vec<PathBuf>,
        teacher_frames: Vec<PathBuf>,
        metadata: JepaSampleMetadata,
    },
    SyntheticVideo {
        index: usize,
    },
}

#[derive(Debug)]
pub struct JepaTensorBatch<B: Backend> {
    pub student: Tensor<B, 5>,
    pub teacher: Tensor<B, 5>,
    pub metadata: Vec<JepaSampleMetadata>,
}

#[derive(Debug)]
pub struct JepaCpuTensorBatch {
    pub student_values: Vec<f32>,
    pub teacher_values: Option<Vec<f32>>,
    pub shape: [usize; 5],
    pub metadata: Vec<JepaSampleMetadata>,
}

pub trait JepaDataset {
    fn len(&self) -> usize;
    fn sample(&self, index: usize) -> Result<JepaSample>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
pub struct ManifestJepaDataset {
    rows: Vec<JepaSample>,
}

impl ManifestJepaDataset {
    pub fn from_manifest(path: impl AsRef<Path>, sample_kind: JepaSampleKind) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("read manifest {}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut rows = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: JepaManifestRow = serde_json::from_str(line)
                .with_context(|| format!("parse manifest line {}", line_index + 1))?;
            rows.push(row_to_sample(base, row, sample_kind)?);
        }
        ensure!(
            !rows.is_empty(),
            "manifest dataset must contain at least one row"
        );
        Ok(Self { rows })
    }
}

impl JepaDataset for ManifestJepaDataset {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn sample(&self, index: usize) -> Result<JepaSample> {
        self.rows
            .get(index % self.rows.len().max(1))
            .cloned()
            .context("manifest dataset is empty")
    }
}

#[derive(Clone, Debug)]
pub struct SyntheticJepaDataset {
    len: usize,
}

impl SyntheticJepaDataset {
    pub fn new(len: usize) -> Self {
        Self { len: len.max(1) }
    }
}

impl JepaDataset for SyntheticJepaDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn sample(&self, index: usize) -> Result<JepaSample> {
        Ok(JepaSample::SyntheticVideo {
            index: index % self.len,
        })
    }
}

pub fn dataset_from_config(
    config: &JepaDatasetConfig,
    train: bool,
) -> Result<Box<dyn JepaDataset>> {
    let dataset: Box<dyn JepaDataset> = match config.kind {
        JepaDatasetKind::Synthetic => Box::new(SyntheticJepaDataset::new(config.synthetic_len)),
        JepaDatasetKind::Manifest => {
            let path = if train {
                config.train_manifest.as_ref()
            } else {
                config
                    .eval_manifest
                    .as_ref()
                    .or(config.train_manifest.as_ref())
            }
            .context("manifest dataset requires train_manifest or eval_manifest")?;
            Box::new(ManifestJepaDataset::from_manifest(
                path,
                config.sample_kind,
            )?)
        }
    };
    let dataset = limit_dataset_if_needed(config, dataset);
    Ok(repeat_dataset_if_needed(config, dataset))
}

pub fn load_jepa_tensor_batch<B: Backend>(
    sample: &JepaSample,
    dataset: &JepaDatasetConfig,
    model: &VJepaConfig,
    device: &B::Device,
) -> Result<JepaTensorBatch<B>> {
    let batch = load_jepa_cpu_tensor_batch(sample, dataset, model)?;
    Ok(jepa_tensor_batch_from_cpu(batch, device))
}

pub fn jepa_tensor_batch_from_cpu<B: Backend>(
    batch: JepaCpuTensorBatch,
    device: &B::Device,
) -> JepaTensorBatch<B> {
    let student =
        Tensor::<B, 5>::from_data(TensorData::new(batch.student_values, batch.shape), device);
    let teacher = match batch.teacher_values {
        Some(values) => Tensor::<B, 5>::from_data(TensorData::new(values, batch.shape), device),
        None => student.clone(),
    };
    JepaTensorBatch {
        student,
        teacher,
        metadata: batch.metadata,
    }
}

pub fn load_jepa_cpu_tensor_batch(
    sample: &JepaSample,
    dataset: &JepaDatasetConfig,
    model: &VJepaConfig,
) -> Result<JepaCpuTensorBatch> {
    let frames = round_up_to_multiple(
        dataset.frames.max(model.tubelet_size.max(1)),
        model.tubelet_size.max(1),
    );
    let image_size = round_up_to_multiple(
        dataset.image_size.max(model.patch_size.max(1)),
        model.patch_size.max(1),
    );
    let (student, channels) = match sample {
        JepaSample::Image { path, .. } => {
            let paths = select_frame_paths(std::slice::from_ref(path), frames, dataset.stride);
            (load_video_values_from_paths(&paths, image_size)?, 3)
        }
        JepaSample::Video { frames: paths, .. } => {
            let paths = select_frame_paths(paths, frames, dataset.stride);
            (load_video_values_from_paths(&paths, image_size)?, 3)
        }
        JepaSample::PairedVideo { student_frames, .. } => {
            let paths = select_frame_paths(student_frames, frames, dataset.stride);
            (load_video_values_from_paths(&paths, image_size)?, 3)
        }
        JepaSample::SyntheticVideo { index } => {
            let channels = model.in_channels.max(1);
            (
                synthetic_video_values(*index, channels, frames, image_size, image_size),
                channels,
            )
        }
    };
    let teacher = match sample {
        JepaSample::PairedVideo { teacher_frames, .. } => {
            let paths = select_frame_paths(teacher_frames, frames, dataset.stride);
            Some(load_video_values_from_paths(&paths, image_size)?)
        }
        _ => None,
    };
    Ok(JepaCpuTensorBatch {
        student_values: student,
        teacher_values: teacher,
        shape: [1, channels, frames, image_size, image_size],
        metadata: vec![sample.metadata().cloned().unwrap_or_default()],
    })
}

pub fn synthetic_video<B: Backend>(
    index: usize,
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    let values = synthetic_video_values(index, channels, frames, height, width);
    Tensor::<B, 5>::from_data(
        TensorData::new(
            values,
            [
                1,
                channels.max(1),
                frames.max(1),
                height.max(1),
                width.max(1),
            ],
        ),
        device,
    )
}

fn synthetic_video_values(
    index: usize,
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let channels = channels.max(1);
    let frames = frames.max(1);
    let height = height.max(1);
    let width = width.max(1);
    let mut values = Vec::with_capacity(channels * frames * height * width);
    for channel in 0..channels {
        for frame in 0..frames {
            for row in 0..height {
                for col in 0..width {
                    let value = ((index + channel * 13 + frame * 7 + row * 3 + col) % 257) as f32;
                    values.push((value / 128.0) - 1.0);
                }
            }
        }
    }
    values
}

fn row_to_sample(
    base: &Path,
    row: JepaManifestRow,
    sample_kind: JepaSampleKind,
) -> Result<JepaSample> {
    let metadata = row.metadata();
    Ok(match sample_kind {
        JepaSampleKind::Image => {
            let image = row.image.context("image manifest row requires image")?;
            JepaSample::Image {
                path: resolve_path(base, image),
                metadata,
            }
        }
        JepaSampleKind::Video => {
            let frames = row_frames(base, row.frames, row.frame_dir)?;
            JepaSample::Video { frames, metadata }
        }
        JepaSampleKind::PairedVideo => {
            let student_frames = row_frames(base, row.frames, row.frame_dir)?;
            let teacher_frames = row_frames(base, row.teacher_frames, row.teacher_frame_dir)?;
            JepaSample::PairedVideo {
                student_frames,
                teacher_frames,
                metadata,
            }
        }
    })
}

impl JepaManifestRow {
    fn metadata(&self) -> JepaSampleMetadata {
        JepaSampleMetadata {
            clip_id: self.clip_id.clone(),
            domain: self.domain.clone(),
            start_frame: self.start_frame,
            fps: self.fps,
            duration: self.duration,
            caption: self.caption.clone(),
            source: self.source.clone(),
            precomputed_context_indices: self.precomputed_context_indices.clone(),
            precomputed_target_indices: self.precomputed_target_indices.clone(),
            original_stream: self.original_stream.clone(),
            cache_id: self.cache_id.clone(),
        }
    }

    pub fn to_sample(&self, base: &Path, sample_kind: JepaSampleKind) -> Result<JepaSample> {
        row_to_sample(base, self.clone(), sample_kind)
    }
}

impl JepaSample {
    pub fn metadata(&self) -> Option<&JepaSampleMetadata> {
        match self {
            JepaSample::Image { metadata, .. }
            | JepaSample::Video { metadata, .. }
            | JepaSample::PairedVideo { metadata, .. } => Some(metadata),
            JepaSample::SyntheticVideo { .. } => None,
        }
    }
}

fn row_frames(
    base: &Path,
    frames: Option<Vec<PathBuf>>,
    frame_dir: Option<PathBuf>,
) -> Result<Vec<PathBuf>> {
    if let Some(frames) = frames {
        ensure!(
            !frames.is_empty(),
            "video manifest row frames must be non-empty"
        );
        return Ok(frames
            .into_iter()
            .map(|path| resolve_path(base, path))
            .collect());
    }
    let Some(frame_dir) = frame_dir else {
        bail!("video manifest row requires frames or frame_dir");
    };
    let frame_dir = resolve_path(base, frame_dir);
    let mut paths = fs::read_dir(&frame_dir)
        .with_context(|| format!("read frame_dir {}", frame_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    paths.retain(|path| path.is_file());
    ensure!(
        !paths.is_empty(),
        "frame_dir {} did not contain files",
        frame_dir.display()
    );
    Ok(paths)
}

fn resolve_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn select_frame_paths(paths: &[PathBuf], frames: usize, stride: usize) -> Vec<PathBuf> {
    let stride = stride.max(1);
    let frames = frames.max(1);
    let mut selected = paths
        .iter()
        .step_by(stride)
        .take(frames)
        .cloned()
        .collect::<Vec<_>>();
    if let Some(last) = selected.last().cloned() {
        while selected.len() < frames {
            selected.push(last.clone());
        }
    }
    selected
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    let multiple = multiple.max(1);
    value.div_ceil(multiple) * multiple
}

fn load_video_values_from_paths(paths: &[PathBuf], image_size: usize) -> Result<Vec<f32>> {
    ensure!(
        !paths.is_empty(),
        "video sample must contain at least one frame"
    );
    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        let image =
            image::open(path).with_context(|| format!("decode image {}", path.display()))?;
        let image = image.resize_exact(image_size as u32, image_size as u32, FilterType::Triangle);
        frames.push(image.to_rgb8());
    }

    let mut values = Vec::with_capacity(paths.len() * 3 * image_size * image_size);
    for channel in 0..3 {
        for frame in &frames {
            for y in 0..image_size {
                for x in 0..image_size {
                    let pixel = frame.get_pixel(x as u32, y as u32);
                    values.push(pixel[channel] as f32 / 255.0);
                }
            }
        }
    }
    Ok(values)
}

struct RepeatedJepaDataset {
    inner: Box<dyn JepaDataset>,
    repeat_count: usize,
    mode: JepaDatasetRepeatMode,
    window_stride: usize,
}

struct LimitedJepaDataset {
    inner: Box<dyn JepaDataset>,
    len: usize,
}

impl JepaDataset for LimitedJepaDataset {
    fn len(&self) -> usize {
        self.len.max(1)
    }

    fn sample(&self, index: usize) -> Result<JepaSample> {
        self.inner.sample(index % self.len())
    }
}

fn limit_dataset_if_needed(
    config: &JepaDatasetConfig,
    dataset: Box<dyn JepaDataset>,
) -> Box<dyn JepaDataset> {
    if config.sample_limit == 0 {
        return dataset;
    }
    Box::new(LimitedJepaDataset {
        len: config.sample_limit.min(dataset.len()).max(1),
        inner: dataset,
    })
}

impl JepaDataset for RepeatedJepaDataset {
    fn len(&self) -> usize {
        self.inner
            .len()
            .saturating_mul(self.repeat_count.max(1))
            .max(1)
    }

    fn sample(&self, index: usize) -> Result<JepaSample> {
        let base_len = self.inner.len().max(1);
        let virtual_index = index % self.len();
        let repeat = virtual_index / base_len;
        let base_slot = virtual_index % base_len;
        let base_index = match self.mode {
            JepaDatasetRepeatMode::Preserve
            | JepaDatasetRepeatMode::ContinuousStreams
            | JepaDatasetRepeatMode::StitchedStream => base_slot,
            JepaDatasetRepeatMode::AdversarialStitchedStream => {
                if base_slot.is_multiple_of(2) {
                    base_slot / 2
                } else {
                    base_len.saturating_sub(1 + base_slot / 2)
                }
            }
        };
        let mut sample = self.inner.sample(base_index)?;
        if let Some(metadata) = sample_metadata_mut(&mut sample) {
            rewrite_repeated_metadata(
                metadata,
                self.mode,
                repeat,
                base_index,
                virtual_index,
                base_len,
                self.window_stride,
            );
        }
        Ok(sample)
    }
}

fn repeat_dataset_if_needed(
    config: &JepaDatasetConfig,
    dataset: Box<dyn JepaDataset>,
) -> Box<dyn JepaDataset> {
    let repeat_count = config.repeat_count.max(1);
    if repeat_count == 1 && config.repeat_mode == JepaDatasetRepeatMode::Preserve {
        return dataset;
    }
    Box::new(RepeatedJepaDataset {
        inner: dataset,
        repeat_count,
        mode: config.repeat_mode,
        window_stride: config.frames.max(1) * config.stride.max(1),
    })
}

fn sample_metadata_mut(sample: &mut JepaSample) -> Option<&mut JepaSampleMetadata> {
    match sample {
        JepaSample::Image { metadata, .. }
        | JepaSample::Video { metadata, .. }
        | JepaSample::PairedVideo { metadata, .. } => Some(metadata),
        JepaSample::SyntheticVideo { .. } => None,
    }
}

fn rewrite_repeated_metadata(
    metadata: &mut JepaSampleMetadata,
    mode: JepaDatasetRepeatMode,
    repeat: usize,
    base_index: usize,
    virtual_index: usize,
    _base_len: usize,
    window_stride: usize,
) {
    let original_stream = original_stream_key(metadata);
    let original_start = metadata.start_frame.unwrap_or(base_index);
    metadata.original_stream = Some(original_stream.clone());
    metadata.cache_id = Some(format!(
        "{original_stream}:start={original_start}:row={base_index}"
    ));

    match mode {
        JepaDatasetRepeatMode::Preserve => {}
        JepaDatasetRepeatMode::ContinuousStreams => {
            let offset = repeat * 1_000_000usize;
            metadata.start_frame = Some(original_start + offset);
        }
        JepaDatasetRepeatMode::StitchedStream
        | JepaDatasetRepeatMode::AdversarialStitchedStream => {
            metadata.clip_id = Some("stitched_stream".to_string());
            metadata.source = Some("stitched_stream".to_string());
            metadata.domain = Some("stitched".to_string());
            metadata.start_frame = Some(virtual_index * window_stride.max(1));
        }
    }
}

fn original_stream_key(metadata: &JepaSampleMetadata) -> String {
    metadata
        .original_stream
        .as_ref()
        .or(metadata.clip_id.as_ref())
        .or(metadata.source.as_ref())
        .or(metadata.domain.as_ref())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        JepaDataset, JepaDatasetConfig, JepaDatasetKind, JepaDatasetRepeatMode, JepaManifestRow,
        JepaSampleKind, dataset_from_config,
    };
    use std::fs;

    fn write_manifest(rows: &[JepaManifestRow]) -> tempfile::TempDir {
        let temp = tempfile::tempdir().expect("tempdir");
        let frame_dir = temp.path().join("frames");
        fs::create_dir_all(&frame_dir).expect("frame dir");
        for index in 0..2 {
            let image = image::RgbImage::from_pixel(2, 2, image::Rgb([index as u8, 0, 0]));
            image
                .save(frame_dir.join(format!("{index:04}.png")))
                .expect("frame");
        }
        let text = rows
            .iter()
            .map(|row| {
                let mut row = row.clone();
                row.frames = Some(vec![frame_dir.join("0000.png"), frame_dir.join("0001.png")]);
                serde_json::to_string(&row).expect("json")
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(temp.path().join("manifest.jsonl"), text).expect("manifest");
        temp
    }

    fn metadata(dataset: &dyn JepaDataset, index: usize) -> super::JepaSampleMetadata {
        dataset
            .sample(index)
            .expect("sample")
            .metadata()
            .cloned()
            .expect("metadata")
    }

    #[test]
    fn repeated_dataset_can_make_streams_continuous() {
        let temp = write_manifest(&[
            JepaManifestRow {
                clip_id: Some("a".into()),
                domain: Some("d".into()),
                start_frame: Some(0),
                fps: None,
                duration: None,
                caption: None,
                source: Some("src-a".into()),
                image: None,
                frames: None,
                frame_dir: None,
                teacher_frames: None,
                teacher_frame_dir: None,
                precomputed_context_indices: None,
                precomputed_target_indices: None,
                original_stream: None,
                cache_id: None,
            },
            JepaManifestRow {
                clip_id: Some("a".into()),
                domain: Some("d".into()),
                start_frame: Some(8),
                fps: None,
                duration: None,
                caption: None,
                source: Some("src-a".into()),
                image: None,
                frames: None,
                frame_dir: None,
                teacher_frames: None,
                teacher_frame_dir: None,
                precomputed_context_indices: None,
                precomputed_target_indices: None,
                original_stream: None,
                cache_id: None,
            },
        ]);
        let config = JepaDatasetConfig {
            kind: JepaDatasetKind::Manifest,
            sample_kind: JepaSampleKind::Video,
            train_manifest: Some(temp.path().join("manifest.jsonl")),
            eval_manifest: None,
            frames: 2,
            repeat_count: 2,
            repeat_mode: JepaDatasetRepeatMode::ContinuousStreams,
            ..JepaDatasetConfig::default()
        };
        let dataset = dataset_from_config(&config, true).expect("dataset");
        assert_eq!(dataset.len(), 4);
        assert_eq!(metadata(dataset.as_ref(), 0).start_frame, Some(0));
        assert_eq!(metadata(dataset.as_ref(), 1).start_frame, Some(8));
        assert_eq!(metadata(dataset.as_ref(), 2).start_frame, Some(1_000_000));
        assert_eq!(
            metadata(dataset.as_ref(), 2).cache_id.as_deref(),
            Some("a:start=0:row=0")
        );
    }

    #[test]
    fn repeated_dataset_can_force_scene_stitching() {
        let temp = write_manifest(&[
            JepaManifestRow {
                clip_id: Some("a".into()),
                domain: Some("a-domain".into()),
                start_frame: Some(0),
                fps: None,
                duration: None,
                caption: None,
                source: Some("src-a".into()),
                image: None,
                frames: None,
                frame_dir: None,
                teacher_frames: None,
                teacher_frame_dir: None,
                precomputed_context_indices: None,
                precomputed_target_indices: None,
                original_stream: None,
                cache_id: None,
            },
            JepaManifestRow {
                clip_id: Some("b".into()),
                domain: Some("b-domain".into()),
                start_frame: Some(0),
                fps: None,
                duration: None,
                caption: None,
                source: Some("src-b".into()),
                image: None,
                frames: None,
                frame_dir: None,
                teacher_frames: None,
                teacher_frame_dir: None,
                precomputed_context_indices: None,
                precomputed_target_indices: None,
                original_stream: None,
                cache_id: None,
            },
        ]);
        let config = JepaDatasetConfig {
            kind: JepaDatasetKind::Manifest,
            sample_kind: JepaSampleKind::Video,
            train_manifest: Some(temp.path().join("manifest.jsonl")),
            eval_manifest: None,
            frames: 2,
            repeat_count: 2,
            repeat_mode: JepaDatasetRepeatMode::AdversarialStitchedStream,
            ..JepaDatasetConfig::default()
        };
        let dataset = dataset_from_config(&config, true).expect("dataset");
        let first = metadata(dataset.as_ref(), 0);
        let second = metadata(dataset.as_ref(), 1);
        assert_eq!(first.clip_id.as_deref(), Some("stitched_stream"));
        assert_eq!(second.clip_id.as_deref(), Some("stitched_stream"));
        assert_eq!(first.domain.as_deref(), Some("stitched"));
        assert_eq!(first.original_stream.as_deref(), Some("a"));
        assert_eq!(second.original_stream.as_deref(), Some("b"));
        assert_eq!(second.start_frame, Some(2));
    }
}
