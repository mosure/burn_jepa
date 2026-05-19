use crate::{JepaReconstructionConfig, JepaReconstructionOutputActivation};
use anyhow::Result;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, PaddingConfig2d};
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

#[derive(Module, Debug)]
pub struct JepaReconstructionUpBlock<B: Backend> {
    pub norm: GroupNorm<B>,
    pub conv: Conv2d<B>,
    #[module(skip)]
    upsample: bool,
}

impl<B: Backend> JepaReconstructionUpBlock<B> {
    pub fn new(
        channels: usize,
        norm_groups: usize,
        epsilon: f64,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        let channels = channels.max(1);
        Self {
            norm: GroupNormConfig::new(norm_groups.max(1), channels)
                .with_epsilon(epsilon)
                .init(device),
            conv: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            upsample,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = if self.upsample {
            let [_, _, height, width] = x.shape().dims::<4>();
            interpolate(
                x,
                [height * 2, width * 2],
                InterpolateOptions::new(InterpolateMode::Nearest),
            )
        } else {
            x
        };
        let residual = x.clone();
        let y = self.conv.forward(activation::gelu(self.norm.forward(x)));
        y + residual
    }
}

#[derive(Module, Debug)]
pub struct JepaReconstructionDecoder<B: Backend> {
    pub input_proj: Conv2d<B>,
    pub blocks: Vec<JepaReconstructionUpBlock<B>>,
    pub output_proj: Conv2d<B>,
    #[module(skip)]
    config: JepaReconstructionConfig,
}

impl<B: Backend> JepaReconstructionDecoder<B> {
    pub fn new(config: JepaReconstructionConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let input_proj =
            Conv2dConfig::new([config.input_dim.max(1), config.hidden_dim.max(1)], [1, 1])
                .init(device);
        let blocks = (0..config.upsample_blocks)
            .flat_map(|_| {
                (0..config.residual_blocks_per_scale.max(1)).map(|block_index| {
                    JepaReconstructionUpBlock::new(
                        config.hidden_dim,
                        config.norm_groups,
                        config.epsilon,
                        block_index == 0,
                        device,
                    )
                })
            })
            .collect();
        let output_proj = Conv2dConfig::new(
            [config.hidden_dim.max(1), config.output_channels.max(1)],
            [3, 3],
        )
        .with_padding(PaddingConfig2d::Same)
        .init(device);
        Ok(Self {
            input_proj,
            blocks,
            output_proj,
            config,
        })
    }

    pub fn config(&self) -> &JepaReconstructionConfig {
        &self.config
    }

    pub fn forward(&self, features: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, grid_h, grid_w] = features.shape().dims::<4>();
        self.forward_to_size(
            features,
            [
                grid_h * self.config.patch_size,
                grid_w * self.config.patch_size,
            ],
        )
    }

    pub fn forward_to_size(&self, features: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4> {
        let mut x = activation::gelu(self.input_proj.forward(features));
        for block in &self.blocks {
            x = block.forward(x);
        }
        let [_, _, height, width] = x.shape().dims::<4>();
        if [height, width] != output_size {
            x = interpolate(
                x,
                output_size,
                InterpolateOptions::new(InterpolateMode::Nearest),
            );
        }
        let x = self.output_proj.forward(x);
        match self.config.output_activation {
            JepaReconstructionOutputActivation::Sigmoid => activation::sigmoid(x),
            JepaReconstructionOutputActivation::Tanh01 => (activation::tanh(x) + 1.0) / 2.0,
            JepaReconstructionOutputActivation::None => x,
        }
    }
}
