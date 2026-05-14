use super::config::TttEncoderConfig;
use super::state::TttLayerState;
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Module, Debug)]
pub struct VJepaTttLayer<B: Backend> {
    pub target_proj: Option<Linear<B>>,
    pub temporal_conv: Conv1d<B>,
    pub out_proj: Linear<B>,
    #[module(skip)]
    dim: usize,
    #[module(skip)]
    chunk_tokens: usize,
    #[module(skip)]
    ttt_lr: f32,
    #[module(skip)]
    memory_scale: f64,
}

impl<B: Backend> VJepaTttLayer<B> {
    pub fn new(dim: usize, config: &TttEncoderConfig, device: &B::Device) -> Self {
        let dim = dim.max(1);
        let kernel = config.conv_kernel.max(1);
        let target_proj = config.use_projection.then(|| identity_linear(dim, device));
        let mut temporal_conv = Conv1dConfig::new(dim, dim, kernel)
            .with_groups(dim)
            .with_padding(PaddingConfig1d::Same)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        temporal_conv.weight = Param::from_tensor(depthwise_identity_kernel(dim, kernel, device));
        let mut out_proj = LinearConfig::new(dim, dim)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        out_proj.weight = Param::from_tensor(Tensor::<B, 2>::zeros([dim, dim], device));
        out_proj.bias = None;
        Self {
            target_proj,
            temporal_conv,
            out_proj,
            dim,
            chunk_tokens: config.chunk_tokens.max(1),
            ttt_lr: config.ttt_lr,
            memory_scale: (dim as f64).powf(-0.5),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, dim] = x.shape().dims::<3>();
        debug_assert_eq!(dim, self.dim);
        let device = x.device();
        let target = target.unwrap_or_else(|| x.clone()).detach();
        let target = self
            .temporal_conv
            .forward(target.swap_dims(1, 2))
            .swap_dims(1, 2);
        let target = if let Some(proj) = &self.target_proj {
            proj.forward(target)
        } else {
            target
        };

        let mut fast = state
            .fast_weight
            .take()
            .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, dim, dim], &device));
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.chunk_tokens));
        for start in (0..tokens).step_by(self.chunk_tokens) {
            let end = (start + self.chunk_tokens).min(tokens);
            let len = (end - start).max(1) as f64;
            let x_chunk = x.clone().slice_dim(1, start..end);
            let memory = x_chunk
                .clone()
                .matmul(fast.clone())
                .mul_scalar(self.memory_scale);
            chunks.push(self.out_proj.forward(memory));

            let target_chunk = target.clone().slice_dim(1, start..end);
            let delta = x_chunk.swap_dims(1, 2).matmul(target_chunk);
            fast = fast.mul_scalar(1.0 - self.ttt_lr as f64)
                + delta.mul_scalar(self.ttt_lr as f64 / len);
        }
        state.fast_weight = Some(fast);
        x + Tensor::cat(chunks, 1)
    }
}

fn identity_linear<B: Backend>(dim: usize, device: &B::Device) -> Linear<B> {
    let mut layer = LinearConfig::new(dim, dim)
        .with_bias(false)
        .with_initializer(Initializer::Zeros)
        .init(device);
    let mut values = vec![0.0f32; dim * dim];
    for index in 0..dim {
        values[index * dim + index] = 1.0;
    }
    layer.weight = Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(values, [dim, dim]),
        device,
    ));
    layer
}

fn depthwise_identity_kernel<B: Backend>(
    dim: usize,
    kernel: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut values = vec![0.0f32; dim * kernel];
    let center = kernel / 2;
    for channel in 0..dim {
        values[channel * kernel + center] = 1.0;
    }
    Tensor::<B, 3>::from_data(TensorData::new(values, [dim, 1, kernel]), device)
}
