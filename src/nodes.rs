use crate::{
    DensePredictionOutput, SparseTokenMask, VJepaPipeline, VJepaRgbaVideoShape, VJepaVideoShape,
    rgba_video_to_tensor,
};
use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct SparseJepaTensorPipelineConfig {
    pub context_keep_ratio: f32,
}

impl Default for SparseJepaTensorPipelineConfig {
    fn default() -> Self {
        Self {
            context_keep_ratio: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct SparseJepaPacket<B: Backend> {
    pub video: Tensor<B, 5>,
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
    pub output: DensePredictionOutput<B>,
}

pub trait SparseJepaInputNode<B: Backend> {
    fn next_video(&mut self, device: &B::Device) -> Result<Option<Tensor<B, 5>>>;
}

pub trait SparseJepaOutputNode<B: Backend> {
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct TensorVideoInput<B: Backend> {
    video: Option<Tensor<B, 5>>,
}

impl<B: Backend> TensorVideoInput<B> {
    pub fn new(video: Tensor<B, 5>) -> Self {
        Self { video: Some(video) }
    }
}

impl<B: Backend> SparseJepaInputNode<B> for TensorVideoInput<B> {
    fn next_video(&mut self, _device: &B::Device) -> Result<Option<Tensor<B, 5>>> {
        Ok(self.video.take())
    }
}

#[derive(Clone, Debug)]
pub struct RgbaVideoInput {
    rgba: Option<Vec<u8>>,
    shape: VJepaRgbaVideoShape,
}

impl RgbaVideoInput {
    pub fn new(rgba: Vec<u8>, shape: VJepaRgbaVideoShape) -> Self {
        Self {
            rgba: Some(rgba),
            shape,
        }
    }
}

impl<B: Backend> SparseJepaInputNode<B> for RgbaVideoInput {
    fn next_video(&mut self, device: &B::Device) -> Result<Option<Tensor<B, 5>>> {
        self.rgba
            .take()
            .map(|rgba| rgba_video_to_tensor::<B>(&rgba, self.shape, device))
            .transpose()
    }
}

#[derive(Debug, Default)]
pub struct VecOutputNode<B: Backend> {
    pub packets: Vec<SparseJepaPacket<B>>,
}

impl<B: Backend> VecOutputNode<B> {
    pub fn new() -> Self {
        Self {
            packets: Vec::new(),
        }
    }
}

impl<B: Backend> SparseJepaOutputNode<B> for VecOutputNode<B> {
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()> {
        self.packets.push(packet);
        Ok(())
    }
}

pub struct FnOutputNode<F> {
    f: F,
}

impl<F> FnOutputNode<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<B, F> SparseJepaOutputNode<B> for FnOutputNode<F>
where
    B: Backend,
    F: FnMut(SparseJepaPacket<B>) -> Result<()>,
{
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()> {
        (self.f)(packet)
    }
}

pub struct SparseJepaTensorPipeline<B: Backend, I, O> {
    pipeline: VJepaPipeline<B>,
    input: I,
    output: O,
    config: SparseJepaTensorPipelineConfig,
}

impl<B, I, O> SparseJepaTensorPipeline<B, I, O>
where
    B: Backend,
    I: SparseJepaInputNode<B>,
    O: SparseJepaOutputNode<B>,
{
    pub fn new(pipeline: VJepaPipeline<B>, input: I, output: O) -> Self {
        Self {
            pipeline,
            input,
            output,
            config: SparseJepaTensorPipelineConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SparseJepaTensorPipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn run_next(&mut self, device: &B::Device) -> Result<bool> {
        let Some(video) = self.input.next_video(device)? else {
            return Ok(false);
        };
        let grid = {
            let [_batch, _channels, frames, height, width] = video.shape().dims::<5>();
            crate::TokenGridShape::new(
                frames / self.pipeline.config().tubelet_size.max(1),
                height / self.pipeline.config().patch_size.max(1),
                width / self.pipeline.config().patch_size.max(1),
            )
        };
        let (context_mask, target_mask) =
            crate::make_context_target_masks(grid, self.config.context_keep_ratio);
        let output = self.pipeline.model().predict_dense_targets(
            video.clone(),
            &context_mask,
            &target_mask,
        )?;
        self.output.push(SparseJepaPacket {
            video,
            context_mask,
            target_mask,
            output,
        })?;
        Ok(true)
    }

    pub fn into_output(self) -> O {
        self.output
    }
}

pub fn empty_rgb_video_shape(frames: usize, height: usize, width: usize) -> VJepaVideoShape {
    VJepaVideoShape::new(1, 3, frames, height, width)
}
