use crate::tensor_ops::{l2_normalize_channels, pointwise_conv2d, reflect_conv2d};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::module::conv2d;
use burn::tensor::ops::{ConvOptions, PadMode};
use burn::tensor::{Distribution, Tensor};

#[derive(Module, Debug)]
pub struct AnyUpResBlock<B: Backend> {
    pub norm1: GroupNorm<B>,
    pub conv1: Conv2d<B>,
    pub norm2: GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub shortcut: Option<Conv2d<B>>,
    #[module(skip)]
    kernel_size: usize,
}

impl<B: Backend> AnyUpResBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        num_groups: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let conv = |channels| {
            Conv2dConfig::new(channels, [kernel_size.max(1), kernel_size.max(1)])
                .with_bias(false)
                .init(device)
        };
        let shortcut = (in_channels != out_channels).then(|| {
            Conv2dConfig::new([in_channels.max(1), out_channels.max(1)], [1, 1])
                .with_bias(false)
                .init(device)
        });
        Self {
            norm1: GroupNormConfig::new(num_groups.max(1), in_channels.max(1))
                .with_epsilon(epsilon)
                .init(device),
            conv1: conv([in_channels.max(1), out_channels.max(1)]),
            norm2: GroupNormConfig::new(num_groups.max(1), out_channels.max(1))
                .with_epsilon(epsilon)
                .init(device),
            conv2: conv([out_channels.max(1), out_channels.max(1)]),
            shortcut,
            kernel_size: kernel_size.max(1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.norm1.forward(x.clone());
        let y = activation::silu(y);
        let y = reflect_conv2d(&self.conv1, y, self.kernel_size);
        let y = self.norm2.forward(y);
        let y = activation::silu(y);
        let y = reflect_conv2d(&self.conv2, y, self.kernel_size);
        let shortcut = match &self.shortcut {
            Some(shortcut) => pointwise_conv2d(shortcut, x),
            None => x,
        };
        y + shortcut
    }
}

#[derive(Module, Debug)]
pub struct AnyUpConvEncoder<B: Backend> {
    pub pre: Conv2d<B>,
    pub blocks: Vec<AnyUpResBlock<B>>,
    #[module(skip)]
    pre_kernel_size: usize,
}

impl<B: Backend> AnyUpConvEncoder<B> {
    pub fn new(
        in_channels: usize,
        qk_dim: usize,
        pre_kernel_size: usize,
        layers: usize,
        num_groups: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let pre_kernel_size = pre_kernel_size.max(1);
        let blocks = (0..layers)
            .map(|_| AnyUpResBlock::new(qk_dim, qk_dim, 1, num_groups, epsilon, device))
            .collect();
        Self {
            pre: Conv2dConfig::new(
                [in_channels.max(1), qk_dim.max(1)],
                [pre_kernel_size, pre_kernel_size],
            )
            .with_bias(false)
            .init(device),
            blocks,
            pre_kernel_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = reflect_conv2d(&self.pre, x, self.pre_kernel_size);
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct LearnedFeatureUnification<B: Backend> {
    pub basis: Param<Tensor<B, 4>>,
    #[module(skip)]
    pub out_channels: usize,
    #[module(skip)]
    pub kernel_size: usize,
}

impl<B: Backend> LearnedFeatureUnification<B> {
    pub fn new(out_channels: usize, kernel_size: usize, device: &B::Device) -> Self {
        let out_channels = out_channels.max(1);
        let kernel_size = kernel_size.max(1);
        Self {
            basis: Param::from_tensor(Tensor::<B, 4>::random(
                [out_channels, 1, kernel_size, kernel_size],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
            out_channels,
            kernel_size,
        }
    }

    pub fn forward(&self, features: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = features.shape().dims::<4>();
        let device = features.device();
        let pad = self.kernel_size / 2;
        let x = if pad == 0 {
            features
        } else {
            features.pad((pad, pad, pad, pad), PadMode::Constant(0.0))
        };
        let basis = self.basis.val().repeat_dim(0, channels);
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], channels);
        let x = conv2d(x, basis, None, options)
            / depthwise_denominator::<B>(height, width, self.kernel_size, device);
        let x = x
            .reshape([batch, channels, self.out_channels, height, width])
            .swap_dims(1, 2);
        let attn = activation::softmax(x, 1);
        attn.mean_dim(2)
            .reshape([batch, self.out_channels, height, width])
    }
}

fn depthwise_denominator<B: Backend>(
    height: usize,
    width: usize,
    kernel_size: usize,
    device: B::Device,
) -> Tensor<B, 4> {
    let pad = kernel_size / 2;
    let mask = Tensor::<B, 4>::ones([1, 1, height, width], &device);
    let mask = if pad == 0 {
        mask
    } else {
        mask.pad((pad, pad, pad, pad), PadMode::Constant(0.0))
    };
    let weight = Tensor::<B, 4>::ones([1, 1, kernel_size, kernel_size], &device);
    conv2d(
        mask,
        weight,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    )
}

#[derive(Module, Debug)]
pub struct AnyUpFeatureEncoder<B: Backend> {
    pub pre: LearnedFeatureUnification<B>,
    pub blocks: Vec<AnyUpResBlock<B>>,
}

impl<B: Backend> AnyUpFeatureEncoder<B> {
    pub fn new(
        qk_dim: usize,
        lfu_dim: usize,
        kernel_size: usize,
        layers: usize,
        num_groups: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..layers)
            .map(|index| {
                let in_channels = if index == 0 { lfu_dim } else { qk_dim };
                AnyUpResBlock::new(in_channels, qk_dim, 1, num_groups, epsilon, device)
            })
            .collect();
        Self {
            pre: LearnedFeatureUnification::new(lfu_dim, kernel_size, device),
            blocks,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.pre.forward(l2_normalize_channels(x));
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

pub(crate) fn aggregation_encoder<B: Backend>(
    qk_dim: usize,
    num_groups: usize,
    epsilon: f64,
    device: &B::Device,
) -> AnyUpConvEncoder<B> {
    AnyUpConvEncoder::new(2 * qk_dim, qk_dim, 3, 2, num_groups, epsilon, device)
}
