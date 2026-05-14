use crate::{
    SparseTokenMask, TokenGridShape, VJepa2_1Model, VJepaConfig, VJepaLoadOptions,
    nodes::{SparseJepaSparsityDriverConfig, resolve_sparsity_driver_masks},
};
use anyhow::{Result, anyhow, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Tensor, TensorData};
use std::path::Path;

pub const VJEPA_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const VJEPA_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const VJEPA_RESCALE_FACTOR: f32 = 1.0 / 255.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VJepaVideoShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VJepaVideoShape {
    pub const fn new(
        batch: usize,
        channels: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Self {
        Self {
            batch,
            channels,
            frames,
            height,
            width,
        }
    }

    pub const fn num_values(&self) -> usize {
        self.batch * self.channels * self.frames * self.height * self.width
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VJepaRgbaVideoShape {
    pub batch: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VJepaRgbaVideoShape {
    pub const fn new(batch: usize, frames: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            frames,
            height,
            width,
        }
    }

    pub const fn num_bytes(&self) -> usize {
        self.batch * self.frames * self.height * self.width * 4
    }
}

#[derive(Debug)]
pub struct VJepaEmbedOutput<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub grid: TokenGridShape,
}

#[derive(Debug)]
pub struct VJepaPipeline<B: Backend> {
    model: VJepa2_1Model<B>,
    config: VJepaConfig,
    sparsity_driver: SparseJepaSparsityDriverConfig,
}

impl<B: Backend> VJepaPipeline<B> {
    pub fn new(model: VJepa2_1Model<B>, config: VJepaConfig) -> Self {
        Self {
            model,
            config,
            sparsity_driver: SparseJepaSparsityDriverConfig::keep_ratio(0.5),
        }
    }

    pub fn random(config: VJepaConfig, device: &B::Device) -> Self {
        let model = VJepa2_1Model::new(&config, device);
        Self::new(model, config)
    }

    pub fn load(path: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        Self::load_with_options(path, VJepaLoadOptions::default(), device)
    }

    pub fn load_with_options(
        path: impl AsRef<Path>,
        options: VJepaLoadOptions,
        device: &B::Device,
    ) -> Result<Self> {
        let (model, config, _report) = options.load_model(path, device)?;
        Ok(Self::new(model, config))
    }

    pub fn with_context_keep_ratio(mut self, ratio: f32) -> Self {
        self.sparsity_driver = SparseJepaSparsityDriverConfig::keep_ratio(ratio.clamp(0.0, 1.0));
        self
    }

    pub fn with_sparsity_driver(mut self, driver: SparseJepaSparsityDriverConfig) -> Self {
        self.sparsity_driver = driver;
        self
    }

    pub fn model(&self) -> &VJepa2_1Model<B> {
        &self.model
    }

    pub fn config(&self) -> &VJepaConfig {
        &self.config
    }

    pub fn embed_video(&self, video: Tensor<B, 5>) -> VJepaEmbedOutput<B> {
        let out = self.model.encode_video(video, None);
        VJepaEmbedOutput {
            tokens: out.tokens,
            grid: out.grid,
        }
    }

    pub fn embed_video_sparse(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseTokenMask,
    ) -> VJepaEmbedOutput<B> {
        let out = self.model.encode_video(video, Some(mask));
        VJepaEmbedOutput {
            tokens: out.tokens,
            grid: out.grid,
        }
    }

    pub fn predict_video_dense(
        &self,
        video: Tensor<B, 5>,
    ) -> Result<crate::DensePredictionOutput<B>> {
        let grid = grid_for_video(&video, &self.config);
        let (context, target) =
            resolve_sparsity_driver_masks(&self.sparsity_driver, &video, &self.config, grid)?;
        self.model.predict_dense_targets(video, &context, &target)
    }

    pub fn tensor_from_frames(
        frames: &[f32],
        shape: VJepaVideoShape,
        device: &B::Device,
    ) -> Result<Tensor<B, 5>> {
        ensure!(
            frames.len() == shape.num_values(),
            "expected {} frame values, got {}",
            shape.num_values(),
            frames.len()
        );
        Ok(Tensor::<B, 5>::from_data(
            TensorData::new(
                frames.to_vec(),
                [
                    shape.batch,
                    shape.channels,
                    shape.frames,
                    shape.height,
                    shape.width,
                ],
            ),
            device,
        ))
    }

    pub fn rgba_video_to_tensor(
        rgba: &[u8],
        shape: VJepaRgbaVideoShape,
        device: &B::Device,
    ) -> Result<Tensor<B, 5>> {
        rgba_video_to_tensor(rgba, shape, device)
    }

    pub fn resize_to_model_input(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        resize_video_spatial(video, self.config.image_size, self.config.image_size)
    }
}

pub fn rgba_video_to_tensor<B: Backend>(
    rgba: &[u8],
    shape: VJepaRgbaVideoShape,
    device: &B::Device,
) -> Result<Tensor<B, 5>> {
    ensure!(
        shape.batch > 0 && shape.frames > 0 && shape.height > 0 && shape.width > 0,
        "RGBA video dimensions must be nonzero"
    );
    ensure!(
        rgba.len() == shape.num_bytes(),
        "expected {} RGBA bytes, got {}",
        shape.num_bytes(),
        rgba.len()
    );
    let pixels = shape.height * shape.width;
    let mut values = Vec::with_capacity(shape.batch * 3 * shape.frames * pixels);
    for batch in 0..shape.batch {
        for channel in 0..3 {
            for frame in 0..shape.frames {
                for pixel in 0..pixels {
                    let offset = (((batch * shape.frames + frame) * pixels + pixel) * 4) + channel;
                    let value = rgba[offset] as f32 * VJEPA_RESCALE_FACTOR;
                    values.push((value - VJEPA_IMAGE_MEAN[channel]) / VJEPA_IMAGE_STD[channel]);
                }
            }
        }
    }
    Ok(Tensor::<B, 5>::from_data(
        TensorData::new(
            values,
            [shape.batch, 3, shape.frames, shape.height, shape.width],
        ),
        device,
    ))
}

fn resize_video_spatial<B: Backend>(
    video: Tensor<B, 5>,
    target_height: usize,
    target_width: usize,
) -> Tensor<B, 5> {
    let [batch, channels, frames, height, width] = video.shape().dims::<5>();
    if height == target_height && width == target_width {
        return video;
    }
    let flat = video
        .permute([0, 2, 1, 3, 4])
        .reshape([batch * frames, channels, height, width]);
    let resized = interpolate(
        flat,
        [target_height, target_width],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );
    resized
        .reshape([batch, frames, channels, target_height, target_width])
        .permute([0, 2, 1, 3, 4])
}

fn grid_for_video<B: Backend>(video: &Tensor<B, 5>, config: &VJepaConfig) -> TokenGridShape {
    let [_batch, _channels, frames, height, width] = video.shape().dims::<5>();
    TokenGridShape::new(
        frames / config.tubelet_size.max(1),
        height / config.patch_size.max(1),
        width / config.patch_size.max(1),
    )
}

pub fn ensure_model_sized_video(shape: VJepaVideoShape, config: &VJepaConfig) -> Result<()> {
    let tubelet_size = config.tubelet_size.max(1);
    let patch_size = config.patch_size.max(1);
    ensure!(shape.channels == 3, "V-JEPA expects RGB input");
    ensure!(
        shape.frames >= tubelet_size && shape.frames.is_multiple_of(tubelet_size),
        "frame count must be a positive multiple of tubelet size"
    );
    ensure!(
        shape.height >= patch_size
            && shape.width >= patch_size
            && shape.height.is_multiple_of(patch_size)
            && shape.width.is_multiple_of(patch_size),
        "height and width must be positive multiples of patch size"
    );
    if shape.batch == 0 {
        return Err(anyhow!("batch size must be nonzero"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32>;

    #[test]
    fn rgba_path_emits_bcthw_normalized_tensor() {
        let device = Default::default();
        let rgba = vec![255_u8, 0, 0, 255, 0, 255, 0, 255];
        let tensor =
            rgba_video_to_tensor::<B>(&rgba, VJepaRgbaVideoShape::new(1, 1, 1, 2), &device)
                .expect("rgba tensor");
        assert_eq!(tensor.shape().dims::<5>(), [1, 3, 1, 1, 2]);
        let values = tensor.into_data().to_vec::<f32>().expect("values");
        assert!((values[0] - ((1.0 - 0.485) / 0.229)).abs() < 1.0e-6);
    }
}
