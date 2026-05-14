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
    match config.kind {
        JepaDatasetKind::Synthetic => Ok(Box::new(SyntheticJepaDataset::new(config.synthetic_len))),
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
            Ok(Box::new(ManifestJepaDataset::from_manifest(
                path,
                config.sample_kind,
            )?))
        }
    }
}

pub fn load_jepa_tensor_batch<B: Backend>(
    sample: &JepaSample,
    dataset: &JepaDatasetConfig,
    model: &VJepaConfig,
    device: &B::Device,
) -> Result<JepaTensorBatch<B>> {
    let frames = round_up_to_multiple(
        dataset.frames.max(model.tubelet_size.max(1)),
        model.tubelet_size.max(1),
    );
    let image_size = round_up_to_multiple(
        dataset.image_size.max(model.patch_size.max(1)),
        model.patch_size.max(1),
    );
    let student = match sample {
        JepaSample::Image { path, .. } => {
            let paths = select_frame_paths(std::slice::from_ref(path), frames, dataset.stride);
            load_video_from_paths(&paths, image_size, device)?
        }
        JepaSample::Video { frames: paths, .. } => {
            let paths = select_frame_paths(paths, frames, dataset.stride);
            load_video_from_paths(&paths, image_size, device)?
        }
        JepaSample::PairedVideo { student_frames, .. } => {
            let paths = select_frame_paths(student_frames, frames, dataset.stride);
            load_video_from_paths(&paths, image_size, device)?
        }
        JepaSample::SyntheticVideo { index } => synthetic_video(
            *index,
            model.in_channels,
            frames,
            image_size,
            image_size,
            device,
        ),
    };
    let teacher = match sample {
        JepaSample::PairedVideo { teacher_frames, .. } => {
            let paths = select_frame_paths(teacher_frames, frames, dataset.stride);
            load_video_from_paths(&paths, image_size, device)?
        }
        _ => student.clone(),
    };
    Ok(JepaTensorBatch {
        student,
        teacher,
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
    Tensor::<B, 5>::from_data(
        TensorData::new(values, [1, channels, frames, height, width]),
        device,
    )
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

fn load_video_from_paths<B: Backend>(
    paths: &[PathBuf],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>> {
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
    Ok(Tensor::<B, 5>::from_data(
        TensorData::new(values, [1, 3, paths.len(), image_size, image_size]),
        device,
    ))
}
