use crate::sparse::{AnyUpSparseOutput, AnyUpSparseOutputPlan};
use crate::tensor_ops::{
    flatten_nchw_to_nlc, gather_tokens, nearest_resize, nlc_to_nchw, pointwise_conv_tokens,
};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig2d, RmsNorm, RmsNormConfig};
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;

#[derive(Module, Debug)]
pub struct EfficientCrossAttention<B: Backend> {
    pub q_proj: Conv2d<B>,
    pub k_proj: Conv2d<B>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    scale: f32,
}

impl<B: Backend> EfficientCrossAttention<B> {
    pub fn new(qk_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let qk_dim = qk_dim.max(1);
        let num_heads = num_heads.max(1);
        let head_dim = (qk_dim / num_heads).max(1);
        let proj = || Conv2dConfig::new([qk_dim, qk_dim], [1, 1]).init(device);
        Self {
            q_proj: proj(),
            k_proj: proj(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    fn project_q(&self, q: Tensor<B, 4>) -> Tensor<B, 4> {
        self.q_proj.forward(q)
    }

    fn project_k(&self, k: Tensor<B, 4>) -> Tensor<B, 4> {
        self.k_proj.forward(k)
    }
}

#[derive(Module, Debug)]
pub struct EfficientCrossAttentionBlock<B: Backend> {
    pub cross_attn: EfficientCrossAttention<B>,
    pub norm_q: RmsNorm<B>,
    pub norm_k: RmsNorm<B>,
    pub conv: Conv2d<B>,
    #[module(skip)]
    pub window_ratio: f32,
}

impl<B: Backend> EfficientCrossAttentionBlock<B> {
    pub fn new(
        qk_dim: usize,
        num_heads: usize,
        window_ratio: f32,
        rms_norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            cross_attn: EfficientCrossAttention::new(qk_dim, num_heads, device),
            norm_q: RmsNormConfig::new(qk_dim)
                .with_epsilon(rms_norm_eps)
                .init(device),
            norm_k: RmsNormConfig::new(qk_dim)
                .with_epsilon(rms_norm_eps)
                .init(device),
            conv: Conv2dConfig::new([qk_dim, qk_dim], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            window_ratio,
        }
    }

    pub fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        let q = self.conv.forward(q);
        let [batch, _, hq, wq] = q.shape().dims::<4>();
        let [_, _, hk, wk] = k.shape().dims::<4>();
        let [_, value_dim, _, _] = v.shape().dims::<4>();

        let q = self.norm_q.forward(flatten_nchw_to_nlc(q));
        let k = self.norm_k.forward(flatten_nchw_to_nlc(k));
        let q = self.cross_attn.project_q(nlc_to_nchw(q, hq, wq));
        let k = self.cross_attn.project_k(nlc_to_nchw(k, hk, wk));
        let k = nearest_resize(k, [hq, wq]);
        let v = nearest_resize(v, [hq, wq]);

        let q = self.heads_nchw(q);
        let k = self.heads_nchw(k);
        let (kernel_h, kernel_w, dilation_h, dilation_w) =
            natten_window(hq, wq, hk, wk, self.window_ratio);
        let radius_h = kernel_h / 2;
        let radius_w = kernel_w / 2;
        let pad_h = radius_h * dilation_h;
        let pad_w = radius_w * dilation_w;
        let k_padded = k
            .reshape([
                batch * self.cross_attn.num_heads,
                self.cross_attn.head_dim,
                hq,
                wq,
            ])
            .pad((pad_w, pad_w, pad_h, pad_h), PadMode::Constant(0.0));
        let v_padded = v.pad((pad_w, pad_w, pad_h, pad_h), PadMode::Constant(0.0));
        let valid_padded = Tensor::<B, 4>::ones([1, 1, hq, wq], &v_padded.device())
            .pad((pad_w, pad_w, pad_h, pad_h), PadMode::Constant(0.0));

        let q_rows_per_chunk = q_chunk_size
            .map(|rows| rows.max(1) * (hq / hk).max(1))
            .unwrap_or(hq)
            .max(1);
        let mut chunks = Vec::new();
        let mut start_row = 0usize;
        while start_row < hq {
            let end_row = (start_row + q_rows_per_chunk).min(hq);
            let chunk_h = end_row - start_row;
            let q_chunk = q.clone().slice([
                0..batch,
                0..self.cross_attn.num_heads,
                0..self.cross_attn.head_dim,
                start_row..end_row,
                0..wq,
            ]);
            let mut logits = Vec::with_capacity(kernel_h * kernel_w);
            let mut shifted_values = Vec::with_capacity(kernel_h * kernel_w);
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let row_offset = ky as isize - radius_h as isize;
                    let col_offset = kx as isize - radius_w as isize;
                    let row_start = shifted_start(start_row, pad_h, row_offset, dilation_h);
                    let col_start = shifted_start(0, pad_w, col_offset, dilation_w);
                    let k_shift = k_padded
                        .clone()
                        .slice([
                            0..batch * self.cross_attn.num_heads,
                            0..self.cross_attn.head_dim,
                            row_start..row_start + chunk_h,
                            col_start..col_start + wq,
                        ])
                        .reshape([
                            batch,
                            self.cross_attn.num_heads,
                            self.cross_attn.head_dim,
                            chunk_h,
                            wq,
                        ]);
                    let valid = valid_padded.clone().slice([
                        0..1,
                        0..1,
                        row_start..row_start + chunk_h,
                        col_start..col_start + wq,
                    ]);
                    let invalid = valid
                        .clone()
                        .neg()
                        .add_scalar(1.0)
                        .mul_scalar(-1.0e9)
                        .unsqueeze_dim::<5>(2);
                    let score = (q_chunk.clone() * k_shift)
                        .sum_dim(2)
                        .mul_scalar(self.cross_attn.scale)
                        + invalid;
                    logits.push(score);
                    shifted_values.push(v_padded.clone().slice([
                        0..batch,
                        0..value_dim,
                        row_start..row_start + chunk_h,
                        col_start..col_start + wq,
                    ]));
                }
            }
            let attn = activation::softmax(Tensor::cat(logits, 2), 2)
                .mean_dim(1)
                .reshape([batch, kernel_h * kernel_w, chunk_h, wq]);
            let mut out = Tensor::<B, 4>::zeros([batch, value_dim, chunk_h, wq], &attn.device());
            for (offset, value) in shifted_values.into_iter().enumerate() {
                let weight = attn
                    .clone()
                    .slice([0..batch, offset..offset + 1, 0..chunk_h, 0..wq]);
                out = out + value * weight;
            }
            chunks.push(out);
            start_row = end_row;
        }

        Tensor::cat(chunks, 2)
    }

    pub fn forward_upstream_masked(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        let q = self.conv.forward(q);
        let [batch, _, hq, wq] = q.shape().dims::<4>();
        let [_, _, hk, wk] = k.shape().dims::<4>();
        let [_, value_dim, _, _] = v.shape().dims::<4>();
        let q_tokens = hq * wq;
        let k_tokens = hk * wk;

        let q = self.norm_q.forward(flatten_nchw_to_nlc(q));
        let k = self.norm_k.forward(flatten_nchw_to_nlc(k));
        let q = pointwise_conv_tokens(&self.cross_attn.q_proj, q)
            .reshape([
                batch,
                q_tokens,
                self.cross_attn.num_heads,
                self.cross_attn.head_dim,
            ])
            .permute([0, 2, 1, 3]);
        let k = pointwise_conv_tokens(&self.cross_attn.k_proj, k)
            .reshape([
                batch,
                k_tokens,
                self.cross_attn.num_heads,
                self.cross_attn.head_dim,
            ])
            .permute([0, 2, 1, 3]);
        let k_t = k.swap_dims(2, 3);
        let v = flatten_nchw_to_nlc(v);

        let chunk_size = q_chunk_size.unwrap_or(q_tokens).max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < q_tokens {
            let end = (start + chunk_size).min(q_tokens);
            let q_chunk = q.clone().slice([
                0..batch,
                0..self.cross_attn.num_heads,
                start..end,
                0..self.cross_attn.head_dim,
            ]);
            let mut logits = q_chunk
                .matmul(k_t.clone())
                .mul_scalar(self.cross_attn.scale);
            if self.window_ratio > 0.0 {
                let device = logits.device();
                logits = logits
                    + upstream_attention_bias::<B>(
                        start,
                        end,
                        [hq, wq],
                        [hk, wk],
                        self.window_ratio,
                        &device,
                    );
            }
            let attn =
                activation::softmax(logits, 3)
                    .mean_dim(1)
                    .reshape([batch, end - start, k_tokens]);
            chunks.push(attn.matmul(v.clone()));
            start = end;
        }

        nlc_to_nchw(Tensor::cat(chunks, 1), hq, wq).reshape([batch, value_dim, hq, wq])
    }

    pub fn forward_sparse(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        plan: &AnyUpSparseOutputPlan<B>,
    ) -> AnyUpSparseOutput<B> {
        let q = self.conv.forward(q);
        let [batch, _, hq, wq] = q.shape().dims::<4>();
        let [_, _, hk, wk] = k.shape().dims::<4>();
        let [_, value_dim, _, _] = v.shape().dims::<4>();
        debug_assert_eq!(plan.batch, batch);
        debug_assert_eq!(plan.output_size, [hq, wq]);
        debug_assert_eq!(plan.feature_size, [hk, wk]);

        let q = gather_tokens(flatten_nchw_to_nlc(q), plan.indices.clone());
        let q = self.norm_q.forward(q);
        let q = pointwise_conv_tokens(&self.cross_attn.q_proj, q);
        let q = q
            .reshape([
                batch,
                plan.sparse_len,
                self.cross_attn.num_heads,
                self.cross_attn.head_dim,
            ])
            .permute([0, 2, 3, 1]);

        let k = self.norm_k.forward(flatten_nchw_to_nlc(k));
        let k = pointwise_conv_tokens(&self.cross_attn.k_proj, k);
        let v = flatten_nchw_to_nlc(v);

        let mut logits = Vec::with_capacity(plan.window_len());
        let mut shifted_values = Vec::with_capacity(plan.window_len());
        for window in &plan.windows {
            let k_shift = gather_tokens(k.clone(), window.low_indices.clone())
                .reshape([
                    batch,
                    plan.sparse_len,
                    self.cross_attn.num_heads,
                    self.cross_attn.head_dim,
                ])
                .permute([0, 2, 3, 1]);
            let invalid = window
                .valid
                .clone()
                .neg()
                .add_scalar(1.0)
                .mul_scalar(-1.0e9)
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2);
            let score = (q.clone() * k_shift)
                .sum_dim(2)
                .mul_scalar(self.cross_attn.scale)
                + invalid;
            logits.push(score);
            shifted_values.push(gather_tokens(v.clone(), window.low_indices.clone()));
        }
        let attn = activation::softmax(Tensor::cat(logits, 2), 2).mean_dim(1);
        let mut out = Tensor::<B, 3>::zeros([batch, plan.sparse_len, value_dim], &v.device());
        for (offset, value) in shifted_values.into_iter().enumerate() {
            let weight = attn
                .clone()
                .slice([0..batch, 0..1, offset..offset + 1, 0..plan.sparse_len])
                .reshape([batch, plan.sparse_len, 1]);
            out = out + value * weight;
        }

        AnyUpSparseOutput {
            features: out,
            indices: plan.indices.clone(),
            output_size: plan.output_size,
        }
    }

    fn heads_nchw(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        let [batch, dim, height, width] = x.shape().dims::<4>();
        debug_assert_eq!(dim, self.cross_attn.num_heads * self.cross_attn.head_dim);
        x.reshape([
            batch,
            self.cross_attn.num_heads,
            self.cross_attn.head_dim,
            height,
            width,
        ])
    }
}

pub(crate) fn natten_window(
    hq: usize,
    wq: usize,
    hk: usize,
    wk: usize,
    ratio: f32,
) -> (usize, usize, usize, usize) {
    let dilation_h = (hq / hk).max(1);
    let dilation_w = (wq / wk).max(1);
    let (kernel_h, kernel_w) = if ratio > 0.0 && ratio < 0.5 {
        (
            odd(round_ties_to_even(2.0 * ratio as f64 * hk as f64).max(3)),
            odd(round_ties_to_even(2.0 * ratio as f64 * wk as f64).max(3)),
        )
    } else {
        (odd(hk), odd(wk))
    };
    (kernel_h, kernel_w, dilation_h, dilation_w)
}

fn odd(value: usize) -> usize {
    if value % 2 == 1 { value } else { value + 1 }
}

fn round_ties_to_even(value: f64) -> usize {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 || (floor as usize).is_multiple_of(2) {
        floor
    } else {
        floor + 1.0
    };
    rounded as usize
}

fn shifted_start(origin: usize, pad: usize, offset: isize, dilation: usize) -> usize {
    let start = origin as isize + pad as isize + offset * dilation as isize;
    debug_assert!(start >= 0);
    start as usize
}

fn upstream_attention_bias<B: Backend>(
    start: usize,
    end: usize,
    output_size: [usize; 2],
    feature_size: [usize; 2],
    ratio: f32,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [hq, wq] = output_size;
    let [hk, wk] = feature_size;
    let q_chunk = end.saturating_sub(start);
    let k_tokens = hk * wk;
    let mut values = Vec::with_capacity(q_chunk * k_tokens);
    for query_index in start..end {
        let row = query_index / wq;
        let col = query_index % wq;
        let row_pos = (row as f32 + 0.5) / hq as f32;
        let col_pos = (col as f32 + 0.5) / wq as f32;
        let row_start = ((row_pos - ratio).clamp(0.0, 1.0) * hk as f32).floor() as usize;
        let row_end = ((row_pos + ratio).clamp(0.0, 1.0) * hk as f32).ceil() as usize;
        let col_start = ((col_pos - ratio).clamp(0.0, 1.0) * wk as f32).floor() as usize;
        let col_end = ((col_pos + ratio).clamp(0.0, 1.0) * wk as f32).ceil() as usize;
        for key_row in 0..hk {
            for key_col in 0..wk {
                let allowed = key_row >= row_start
                    && key_row < row_end
                    && key_col >= col_start
                    && key_col < col_end;
                values.push(if allowed { 0.0 } else { -1.0e9 });
            }
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [q_chunk, k_tokens]), device)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
}

#[cfg(test)]
mod tests {
    use super::{natten_window, round_ties_to_even};

    #[test]
    fn natten_window_matches_upstream_rounding_and_dilation() {
        assert_eq!(round_ties_to_even(2.5), 2);
        assert_eq!(round_ties_to_even(3.5), 4);
        assert_eq!(natten_window(384, 384, 24, 24, 0.1), (5, 5, 16, 16));
        assert_eq!(natten_window(224, 224, 16, 16, 0.1), (3, 3, 14, 14));
    }
}
