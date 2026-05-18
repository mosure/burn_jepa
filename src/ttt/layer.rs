use super::config::{TttEncoderConfig, TttMemoryDynamics};
use super::state::TttLayerState;
use crate::VJepaMlp;
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation;
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
    #[module(skip)]
    memory_dynamics: TttMemoryDynamics,
    #[module(skip)]
    memory_alibi_half_lives: Vec<usize>,
    #[module(skip)]
    memory_alibi_read_weights: Vec<f32>,
    #[module(skip)]
    memory_alibi_update_weights: Vec<f32>,
    #[module(skip)]
    memory_clip_rms: f32,
}

#[derive(Module, Debug)]
pub struct VJepaInPlaceTttMlp<B: Backend> {
    pub target_proj: Option<Linear<B>>,
    pub temporal_conv: Conv1d<B>,
    #[module(skip)]
    embed_dim: usize,
    #[module(skip)]
    hidden_dim: usize,
    #[module(skip)]
    chunk_tokens: usize,
    #[module(skip)]
    ttt_lr: f32,
    #[module(skip)]
    memory_dynamics: TttMemoryDynamics,
    #[module(skip)]
    memory_alibi_half_lives: Vec<usize>,
    #[module(skip)]
    memory_alibi_read_weights: Vec<f32>,
    #[module(skip)]
    memory_alibi_update_weights: Vec<f32>,
    #[module(skip)]
    memory_clip_rms: f32,
}

#[derive(Clone, Debug)]
pub struct VJepaTttLayerProbe<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub memory_read: Tensor<B, 3>,
    pub adapter_delta: Tensor<B, 3>,
    pub fast_weight_before: Tensor<B, 3>,
    pub fast_weight_after: Tensor<B, 3>,
}

impl<B: Backend> VJepaInPlaceTttMlp<B> {
    pub fn new(
        embed_dim: usize,
        hidden_dim: usize,
        config: &TttEncoderConfig,
        device: &B::Device,
    ) -> Self {
        let embed_dim = embed_dim.max(1);
        let hidden_dim = hidden_dim.max(1);
        let kernel = config.conv_kernel.max(1);
        let target_proj = config
            .use_projection
            .then(|| identity_linear(embed_dim, device));
        let temporal_conv = Conv1dConfig::new(embed_dim, embed_dim, kernel)
            .with_groups(embed_dim)
            .with_padding(PaddingConfig1d::Same)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self {
            target_proj,
            temporal_conv,
            embed_dim,
            hidden_dim,
            chunk_tokens: config.chunk_tokens.max(1),
            ttt_lr: config.ttt_lr,
            memory_dynamics: config.memory_dynamics,
            memory_alibi_half_lives: config.resolved_memory_alibi_half_lives(),
            memory_alibi_read_weights: config.resolved_memory_alibi_read_weights(),
            memory_alibi_update_weights: config.resolved_memory_alibi_update_weights(),
            memory_clip_rms: config.memory_clip_rms,
        }
    }

    pub(crate) fn forward_mlp_with_options(
        &self,
        mlp: &VJepaMlp<B>,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
    ) -> Tensor<B, 3> {
        self.forward_mlp_impl(mlp, x, target, state, update_fast_weight, false)
            .0
    }

    pub(crate) fn forward_mlp_with_probe(
        &self,
        mlp: &VJepaMlp<B>,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
    ) -> (Tensor<B, 3>, VJepaTttLayerProbe<B>) {
        let (output, probe) =
            self.forward_mlp_impl(mlp, x, target, state, update_fast_weight, true);
        (
            output,
            probe.expect("in-place TTT probe should be available when requested"),
        )
    }

    fn forward_mlp_impl(
        &self,
        mlp: &VJepaMlp<B>,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [batch, _, dim] = x.shape().dims::<3>();
        debug_assert_eq!(dim, self.embed_dim);
        let h = activation::gelu(mlp.fc1.forward(x.clone()));
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
        let base = self.base_down_weight(mlp, batch);
        match self.memory_dynamics {
            TttMemoryDynamics::Ema => self.forward_ema(
                mlp,
                x,
                h,
                target,
                base,
                state,
                update_fast_weight,
                capture_probe,
            ),
            TttMemoryDynamics::MemoryAlibi => self.forward_memory_alibi(
                mlp,
                x,
                h,
                target,
                base,
                state,
                update_fast_weight,
                capture_probe,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_ema(
        &self,
        mlp: &VJepaMlp<B>,
        x: Tensor<B, 3>,
        h: Tensor<B, 3>,
        target: Tensor<B, 3>,
        base: Tensor<B, 3>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [batch, tokens, hidden] = h.shape().dims::<3>();
        let device = h.device();
        debug_assert_eq!(hidden, self.hidden_dim);
        let mut delta = state
            .fast_weight
            .take()
            .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, hidden, self.embed_dim], &device));
        let fast_before = capture_probe.then(|| base.clone() + delta.clone());
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.chunk_tokens));
        let mut fast_chunks = capture_probe.then(Vec::new);
        let mut base_chunks = capture_probe.then(Vec::new);
        for start in (0..tokens).step_by(self.chunk_tokens) {
            let end = (start + self.chunk_tokens).min(tokens);
            let len = (end - start).max(1) as f64;
            let h_chunk = h.clone().slice_dim(1, start..end);
            let fast = base.clone() + delta.clone();
            let output = self.linear_with_down_bias(mlp, h_chunk.clone().matmul(fast));
            if let Some(fast_chunks) = fast_chunks.as_mut() {
                fast_chunks.push(output.clone());
            }
            if let Some(base_chunks) = base_chunks.as_mut() {
                base_chunks
                    .push(self.linear_with_down_bias(mlp, h_chunk.clone().matmul(base.clone())));
            }
            chunks.push(output);

            if update_fast_weight {
                let target_chunk = target.clone().slice_dim(1, start..end);
                let update = h_chunk.swap_dims(1, 2).matmul(target_chunk);
                delta = delta.mul_scalar(1.0 - self.ttt_lr as f64)
                    + update.mul_scalar(self.ttt_lr as f64 / len);
                delta = self.maybe_clip_fast_delta(delta);
            }
        }
        state.fast_weight = Some(delta.clone());
        state.fast_weight_banks = None;
        let output = Tensor::cat(chunks, 1);
        let probe = fast_before.map(|fast_weight_before| {
            let memory_read = Tensor::cat(
                fast_chunks.expect("fast chunks should exist when probing"),
                1,
            );
            let base_read = Tensor::cat(
                base_chunks.expect("base chunks should exist when probing"),
                1,
            );
            VJepaTttLayerProbe {
                hidden: x,
                adapter_delta: memory_read.clone() - base_read,
                memory_read,
                fast_weight_before,
                fast_weight_after: base + delta,
            }
        });
        (output, probe)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_memory_alibi(
        &self,
        mlp: &VJepaMlp<B>,
        x: Tensor<B, 3>,
        h: Tensor<B, 3>,
        target: Tensor<B, 3>,
        base: Tensor<B, 3>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [batch, tokens, hidden] = h.shape().dims::<3>();
        let device = h.device();
        debug_assert_eq!(hidden, self.hidden_dim);
        let banks = self.memory_alibi_half_lives.len().max(1);
        let mut delta = state.fast_weight_banks.take().unwrap_or_else(|| {
            Tensor::<B, 4>::zeros([batch, banks, hidden, self.embed_dim], &device)
        });
        let fast_before =
            capture_probe.then(|| base.clone() + self.effective_fast_delta(delta.clone()));
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.chunk_tokens));
        let mut fast_chunks = capture_probe.then(Vec::new);
        let mut base_chunks = capture_probe.then(Vec::new);
        for start in (0..tokens).step_by(self.chunk_tokens) {
            let end = (start + self.chunk_tokens).min(tokens);
            let len = (end - start).max(1) as f64;
            let h_chunk = h.clone().slice_dim(1, start..end);
            let fast = base.clone() + self.effective_fast_delta(delta.clone());
            let output = self.linear_with_down_bias(mlp, h_chunk.clone().matmul(fast));
            if let Some(fast_chunks) = fast_chunks.as_mut() {
                fast_chunks.push(output.clone());
            }
            if let Some(base_chunks) = base_chunks.as_mut() {
                base_chunks
                    .push(self.linear_with_down_bias(mlp, h_chunk.clone().matmul(base.clone())));
            }
            chunks.push(output);

            if update_fast_weight {
                let target_chunk = target.clone().slice_dim(1, start..end);
                let update = h_chunk
                    .swap_dims(1, 2)
                    .matmul(target_chunk)
                    .mul_scalar(1.0 / len);
                let updated = (0..banks)
                    .map(|bank| {
                        let bank_delta = delta.clone().slice_dim(1, bank..bank + 1).reshape([
                            batch,
                            hidden,
                            self.embed_dim,
                        ]);
                        let half_life = self.memory_alibi_half_lives[bank].max(1) as f64;
                        let decay = 2.0f64.powf(-1.0 / half_life);
                        let weight =
                            self.ttt_lr as f64 * self.memory_alibi_update_weights[bank] as f64;
                        self.maybe_clip_fast_delta(
                            bank_delta.mul_scalar(decay) + update.clone().mul_scalar(weight),
                        )
                        .reshape([batch, 1, hidden, self.embed_dim])
                    })
                    .collect::<Vec<_>>();
                delta = Tensor::cat(updated, 1);
            }
        }
        state.fast_weight = None;
        state.fast_weight_banks = Some(delta.clone());
        let output = Tensor::cat(chunks, 1);
        let probe = fast_before.map(|fast_weight_before| {
            let memory_read = Tensor::cat(
                fast_chunks.expect("fast chunks should exist when probing"),
                1,
            );
            let base_read = Tensor::cat(
                base_chunks.expect("base chunks should exist when probing"),
                1,
            );
            VJepaTttLayerProbe {
                hidden: x,
                adapter_delta: memory_read.clone() - base_read,
                memory_read,
                fast_weight_before,
                fast_weight_after: base + self.effective_fast_delta(delta),
            }
        });
        (output, probe)
    }

    fn base_down_weight(&self, mlp: &VJepaMlp<B>, batch: usize) -> Tensor<B, 3> {
        mlp.fc2
            .weight
            .val()
            .reshape([1, self.hidden_dim, self.embed_dim])
            .repeat_dim(0, batch)
    }

    fn linear_with_down_bias(&self, mlp: &VJepaMlp<B>, output: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(bias) = &mlp.fc2.bias {
            let [batch, tokens, _] = output.shape().dims::<3>();
            output
                + bias
                    .val()
                    .reshape([1, 1, self.embed_dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, tokens)
        } else {
            output
        }
    }

    fn effective_fast_delta(&self, delta: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, banks, hidden, dim] = delta.shape().dims::<4>();
        let device = delta.device();
        let mut effective = Tensor::<B, 3>::zeros([batch, hidden, dim], &device);
        for bank in 0..banks {
            let weight = self
                .memory_alibi_read_weights
                .get(bank)
                .copied()
                .unwrap_or(1.0 / banks.max(1) as f32) as f64;
            let bank_delta = delta
                .clone()
                .slice_dim(1, bank..bank + 1)
                .reshape([batch, hidden, dim]);
            effective = effective + bank_delta.mul_scalar(weight);
        }
        effective
    }

    fn maybe_clip_fast_delta(&self, delta: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.memory_clip_rms <= 0.0 {
            return delta;
        }
        let clip = self.memory_clip_rms as f64;
        let rms = delta
            .clone()
            .powf_scalar(2.0)
            .mean()
            .add_scalar(1.0e-12)
            .sqrt();
        let scale = rms
            .div_scalar(clip.max(1.0e-6))
            .add_scalar(1.0)
            .recip()
            .reshape([1, 1, 1]);
        delta.mul(scale)
    }
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
            memory_dynamics: config.memory_dynamics,
            memory_alibi_half_lives: config.resolved_memory_alibi_half_lives(),
            memory_alibi_read_weights: config.resolved_memory_alibi_read_weights(),
            memory_alibi_update_weights: config.resolved_memory_alibi_update_weights(),
            memory_clip_rms: config.memory_clip_rms,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
    ) -> Tensor<B, 3> {
        self.forward_impl(x, target, state, true, false).0
    }

    pub(crate) fn forward_with_options(
        &self,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
    ) -> Tensor<B, 3> {
        self.forward_impl(x, target, state, update_fast_weight, false)
            .0
    }

    pub(crate) fn forward_with_probe(
        &self,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
    ) -> (Tensor<B, 3>, VJepaTttLayerProbe<B>) {
        let (output, probe) = self.forward_impl(x, target, state, update_fast_weight, true);
        (
            output,
            probe.expect("TTT layer probe should be available when requested"),
        )
    }

    fn forward_impl(
        &self,
        x: Tensor<B, 3>,
        target: Option<Tensor<B, 3>>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [_, _, dim] = x.shape().dims::<3>();
        debug_assert_eq!(dim, self.dim);
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
        match self.memory_dynamics {
            TttMemoryDynamics::Ema => {
                self.forward_ema(x, target, state, update_fast_weight, capture_probe)
            }
            TttMemoryDynamics::MemoryAlibi => {
                self.forward_memory_alibi(x, target, state, update_fast_weight, capture_probe)
            }
        }
    }

    fn forward_ema(
        &self,
        x: Tensor<B, 3>,
        target: Tensor<B, 3>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [batch, tokens, dim] = x.shape().dims::<3>();
        let device = x.device();
        let mut fast = state
            .fast_weight
            .take()
            .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, dim, dim], &device));
        let fast_before = capture_probe.then(|| fast.clone());
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.chunk_tokens));
        let mut memory_chunks = capture_probe.then(Vec::new);
        for start in (0..tokens).step_by(self.chunk_tokens) {
            let end = (start + self.chunk_tokens).min(tokens);
            let len = (end - start).max(1) as f64;
            let x_chunk = x.clone().slice_dim(1, start..end);
            let memory = x_chunk
                .clone()
                .matmul(fast.clone())
                .mul_scalar(self.memory_scale);
            if let Some(memory_chunks) = memory_chunks.as_mut() {
                memory_chunks.push(memory.clone());
            }
            chunks.push(self.out_proj.forward(memory));

            if update_fast_weight {
                let target_chunk = target.clone().slice_dim(1, start..end);
                let delta = x_chunk.swap_dims(1, 2).matmul(target_chunk);
                fast = fast.mul_scalar(1.0 - self.ttt_lr as f64)
                    + delta.mul_scalar(self.ttt_lr as f64 / len);
                fast = self.maybe_clip_fast_weight(fast);
            }
        }
        state.fast_weight = Some(fast.clone());
        state.fast_weight_banks = None;
        let adapter_delta = Tensor::cat(chunks, 1);
        let output = x.clone() + adapter_delta.clone();
        let probe = fast_before.map(|fast_weight_before| VJepaTttLayerProbe {
            hidden: x,
            memory_read: Tensor::cat(
                memory_chunks.expect("memory chunks should exist when probing"),
                1,
            ),
            adapter_delta,
            fast_weight_before,
            fast_weight_after: fast,
        });
        (output, probe)
    }

    fn forward_memory_alibi(
        &self,
        x: Tensor<B, 3>,
        target: Tensor<B, 3>,
        state: &mut TttLayerState<B>,
        update_fast_weight: bool,
        capture_probe: bool,
    ) -> (Tensor<B, 3>, Option<VJepaTttLayerProbe<B>>) {
        let [batch, tokens, dim] = x.shape().dims::<3>();
        let device = x.device();
        let banks = self.memory_alibi_half_lives.len().max(1);
        let mut fast = state
            .fast_weight_banks
            .take()
            .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, banks, dim, dim], &device));
        let fast_before = capture_probe.then(|| self.effective_fast_weight(fast.clone()));
        let mut chunks = Vec::with_capacity(tokens.div_ceil(self.chunk_tokens));
        let mut memory_chunks = capture_probe.then(Vec::new);
        for start in (0..tokens).step_by(self.chunk_tokens) {
            let end = (start + self.chunk_tokens).min(tokens);
            let len = (end - start).max(1) as f64;
            let x_chunk = x.clone().slice_dim(1, start..end);
            let mut memory = Tensor::<B, 3>::zeros([batch, end - start, dim], &device);
            for bank in 0..banks {
                let bank_fast = fast
                    .clone()
                    .slice_dim(1, bank..bank + 1)
                    .reshape([batch, dim, dim]);
                let read_weight = self.memory_alibi_read_weights[bank] as f64;
                memory = memory
                    + x_chunk
                        .clone()
                        .matmul(bank_fast)
                        .mul_scalar(self.memory_scale * read_weight);
            }
            if let Some(memory_chunks) = memory_chunks.as_mut() {
                memory_chunks.push(memory.clone());
            }
            chunks.push(self.out_proj.forward(memory));

            if update_fast_weight {
                let target_chunk = target.clone().slice_dim(1, start..end);
                let delta = x_chunk
                    .swap_dims(1, 2)
                    .matmul(target_chunk)
                    .mul_scalar(1.0 / len);
                let updated = (0..banks)
                    .map(|bank| {
                        let bank_fast = fast
                            .clone()
                            .slice_dim(1, bank..bank + 1)
                            .reshape([batch, dim, dim]);
                        let half_life = self.memory_alibi_half_lives[bank].max(1) as f64;
                        let decay = 2.0f64.powf(-1.0 / half_life);
                        let update =
                            self.ttt_lr as f64 * self.memory_alibi_update_weights[bank] as f64;
                        self.maybe_clip_fast_weight(
                            bank_fast.mul_scalar(decay) + delta.clone().mul_scalar(update),
                        )
                        .reshape([batch, 1, dim, dim])
                    })
                    .collect::<Vec<_>>();
                fast = Tensor::cat(updated, 1);
            }
        }
        state.fast_weight = None;
        state.fast_weight_banks = Some(fast.clone());
        let adapter_delta = Tensor::cat(chunks, 1);
        let output = x.clone() + adapter_delta.clone();
        let probe = fast_before.map(|fast_weight_before| VJepaTttLayerProbe {
            hidden: x,
            memory_read: Tensor::cat(
                memory_chunks.expect("memory chunks should exist when probing"),
                1,
            ),
            adapter_delta,
            fast_weight_before,
            fast_weight_after: self.effective_fast_weight(fast),
        });
        (output, probe)
    }

    fn effective_fast_weight(&self, fast: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, banks, dim, _] = fast.shape().dims::<4>();
        let device = fast.device();
        let mut effective = Tensor::<B, 3>::zeros([batch, dim, dim], &device);
        for bank in 0..banks {
            let weight = self
                .memory_alibi_read_weights
                .get(bank)
                .copied()
                .unwrap_or(1.0 / banks.max(1) as f32) as f64;
            let bank_fast = fast
                .clone()
                .slice_dim(1, bank..bank + 1)
                .reshape([batch, dim, dim]);
            effective = effective + bank_fast.mul_scalar(weight);
        }
        effective
    }

    fn maybe_clip_fast_weight(&self, fast: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.memory_clip_rms <= 0.0 {
            return fast;
        }
        let clip = self.memory_clip_rms as f64;
        let rms = fast
            .clone()
            .powf_scalar(2.0)
            .mean()
            .add_scalar(1.0e-12)
            .sqrt();
        let scale = rms
            .div_scalar(clip.max(1.0e-6))
            .add_scalar(1.0)
            .recip()
            .reshape([1, 1, 1]);
        fast.mul(scale)
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
