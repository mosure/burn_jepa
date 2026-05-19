use crate::{
    JepaReconstructionArchitecture, JepaReconstructionConfig, JepaReconstructionOutputActivation,
};
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
pub struct JepaReconstructionTokenBlock<B: Backend> {
    pub norm: GroupNorm<B>,
    pub spatial: Conv2d<B>,
    pub expand: Conv2d<B>,
    pub project: Conv2d<B>,
    #[module(skip)]
    residual_scale: f64,
}

impl<B: Backend> JepaReconstructionTokenBlock<B> {
    pub fn new(
        channels: usize,
        expansion: usize,
        norm_groups: usize,
        epsilon: f64,
        residual_scale: f64,
        device: &B::Device,
    ) -> Self {
        let channels = channels.max(1);
        let hidden = channels * expansion.max(1);
        Self {
            norm: GroupNormConfig::new(valid_norm_groups(norm_groups, channels), channels)
                .with_epsilon(epsilon)
                .init(device),
            spatial: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            expand: Conv2dConfig::new([channels, hidden], [1, 1]).init(device),
            project: Conv2dConfig::new([hidden, channels], [1, 1]).init(device),
            residual_scale,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let y = self.spatial.forward(x);
        let y = self.expand.forward(activation::gelu(self.norm.forward(y)));
        let y = self.project.forward(activation::gelu(y));
        residual + y.mul_scalar(self.residual_scale)
    }
}

#[derive(Module, Debug)]
pub struct JepaReconstructionPyramidStage<B: Backend> {
    pub proj: Conv2d<B>,
    pub blocks: Vec<JepaReconstructionTokenBlock<B>>,
    #[module(skip)]
    upsample: bool,
}

impl<B: Backend> JepaReconstructionPyramidStage<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        block_count: usize,
        config: &JepaReconstructionConfig,
        upsample: bool,
        device: &B::Device,
    ) -> Self {
        let in_channels = in_channels.max(1);
        let out_channels = out_channels.max(1);
        let proj = Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device);
        let blocks = (0..block_count.max(1))
            .map(|_| {
                JepaReconstructionTokenBlock::new(
                    out_channels,
                    config.convnext_expansion,
                    config.norm_groups,
                    config.epsilon,
                    config.residual_scale,
                    device,
                )
            })
            .collect();
        Self {
            proj,
            blocks,
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
        let mut x = self.proj.forward(x);
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct JepaReconstructionDecoder<B: Backend> {
    pub input_proj: Conv2d<B>,
    pub blocks: Vec<JepaReconstructionUpBlock<B>>,
    pub pyramid_stages: Vec<JepaReconstructionPyramidStage<B>>,
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
        let blocks = match config.architecture {
            JepaReconstructionArchitecture::ResidualUniform => (0..config.upsample_blocks)
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
                .collect(),
            JepaReconstructionArchitecture::PatchConv => {
                (0..config.residual_blocks_per_scale.max(1))
                    .map(|_| {
                        JepaReconstructionUpBlock::new(
                            config.hidden_dim,
                            config.norm_groups,
                            config.epsilon,
                            false,
                            device,
                        )
                    })
                    .collect()
            }
            JepaReconstructionArchitecture::PyramidConvnext
            | JepaReconstructionArchitecture::PatchLinear => Vec::new(),
        };
        let pyramid_channels = pyramid_channels(&config);
        let pyramid_stages = match config.architecture {
            JepaReconstructionArchitecture::ResidualUniform
            | JepaReconstructionArchitecture::PatchLinear
            | JepaReconstructionArchitecture::PatchConv => Vec::new(),
            JepaReconstructionArchitecture::PyramidConvnext => pyramid_channels
                .windows(2)
                .map(|channels| {
                    JepaReconstructionPyramidStage::new(
                        channels[0],
                        channels[1],
                        config.residual_blocks_per_scale,
                        &config,
                        true,
                        device,
                    )
                })
                .collect(),
        };
        let output_channels_in = match config.architecture {
            JepaReconstructionArchitecture::ResidualUniform
            | JepaReconstructionArchitecture::PatchLinear
            | JepaReconstructionArchitecture::PatchConv => config.hidden_dim,
            JepaReconstructionArchitecture::PyramidConvnext => {
                *pyramid_channels.last().unwrap_or(&config.hidden_dim)
            }
        };
        let output_proj = match config.architecture {
            JepaReconstructionArchitecture::PatchLinear
            | JepaReconstructionArchitecture::PatchConv => Conv2dConfig::new(
                [
                    output_channels_in.max(1),
                    config.output_channels.max(1)
                        * config.patch_size.max(1)
                        * config.patch_size.max(1),
                ],
                [1, 1],
            )
            .init(device),
            JepaReconstructionArchitecture::ResidualUniform
            | JepaReconstructionArchitecture::PyramidConvnext => Conv2dConfig::new(
                [output_channels_in.max(1), config.output_channels.max(1)],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Same)
            .init(device),
        };
        Ok(Self {
            input_proj,
            blocks,
            pyramid_stages,
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
        if self.config.architecture == JepaReconstructionArchitecture::PatchLinear {
            x = self.output_proj.forward(x);
            x = patch_linear_to_image(
                x,
                self.config.output_channels,
                self.config.patch_size,
                output_size,
            );
            return self.activate_output(x);
        }
        for block in &self.blocks {
            x = block.forward(x);
        }
        if self.config.architecture == JepaReconstructionArchitecture::PatchConv {
            x = self.output_proj.forward(x);
            x = patch_linear_to_image(
                x,
                self.config.output_channels,
                self.config.patch_size,
                output_size,
            );
            return self.activate_output(x);
        }
        for stage in &self.pyramid_stages {
            x = stage.forward(x);
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
        self.activate_output(x)
    }

    fn activate_output(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self.config.output_activation {
            JepaReconstructionOutputActivation::Sigmoid => activation::sigmoid(x),
            JepaReconstructionOutputActivation::Tanh01 => (activation::tanh(x) + 1.0) / 2.0,
            JepaReconstructionOutputActivation::None => x,
        }
    }
}

fn pyramid_channels(config: &JepaReconstructionConfig) -> Vec<usize> {
    let mut channels = Vec::with_capacity(config.upsample_blocks + 1);
    let mut current = config.hidden_dim.max(1);
    channels.push(current);
    for _ in 0..config.upsample_blocks {
        current = (current / 2).max(config.min_channels.max(1));
        channels.push(current);
    }
    channels
}

fn valid_norm_groups(requested: usize, channels: usize) -> usize {
    let requested = requested.max(1).min(channels.max(1));
    (1..=requested)
        .rev()
        .find(|groups| channels.is_multiple_of(*groups))
        .unwrap_or(1)
}

fn patch_linear_to_image<B: Backend>(
    patches: Tensor<B, 4>,
    output_channels: usize,
    patch_size: usize,
    output_size: [usize; 2],
) -> Tensor<B, 4> {
    let [batch, _, grid_h, grid_w] = patches.shape().dims::<4>();
    let output_channels = output_channels.max(1);
    let patch_size = patch_size.max(1);
    let mut image = patches
        .reshape([
            batch,
            output_channels,
            patch_size,
            patch_size,
            grid_h,
            grid_w,
        ])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([
            batch,
            output_channels,
            grid_h * patch_size,
            grid_w * patch_size,
        ]);
    if [grid_h * patch_size, grid_w * patch_size] != output_size {
        image = interpolate(
            image,
            output_size,
            InterpolateOptions::new(InterpolateMode::Nearest),
        );
    }
    image
}
