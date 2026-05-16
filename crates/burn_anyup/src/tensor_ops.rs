use burn::nn::conv::Conv2d;
use burn::tensor::backend::Backend;
use burn::tensor::module::{adaptive_avg_pool2d, interpolate, linear};
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Int, Tensor, TensorData};

pub(crate) fn reflect_conv2d<B: Backend>(
    conv: &Conv2d<B>,
    x: Tensor<B, 4>,
    kernel_size: usize,
) -> Tensor<B, 4> {
    let pad = kernel_size / 2;
    if pad == 0 && is_pointwise_conv2d(conv) {
        return pointwise_conv2d(conv, x);
    }
    if pad == 0 {
        conv.forward(x)
    } else {
        conv.forward(x.pad((pad, pad, pad, pad), PadMode::Reflect))
    }
}

pub(crate) fn pointwise_conv2d<B: Backend>(conv: &Conv2d<B>, x: Tensor<B, 4>) -> Tensor<B, 4> {
    debug_assert!(is_pointwise_conv2d(conv));
    let [batch, _, height, width] = x.shape().dims::<4>();
    let [out_channels, in_channels, _, _] = conv.weight.dims();
    let weight = conv
        .weight
        .val()
        .reshape([out_channels, in_channels])
        .swap_dims(0, 1);
    let y = linear(
        flatten_nchw_to_nlc(x),
        weight,
        conv.bias.as_ref().map(|bias| bias.val()),
    );
    y.reshape([batch, height, width, out_channels])
        .permute([0, 3, 1, 2])
}

pub(crate) fn pointwise_conv_tokens<B: Backend>(conv: &Conv2d<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    debug_assert!(is_pointwise_conv2d(conv));
    let [out_channels, in_channels, _, _] = conv.weight.dims();
    let weight = conv
        .weight
        .val()
        .reshape([out_channels, in_channels])
        .swap_dims(0, 1);
    linear(x, weight, conv.bias.as_ref().map(|bias| bias.val()))
}

fn is_pointwise_conv2d<B: Backend>(conv: &Conv2d<B>) -> bool {
    conv.kernel_size == [1, 1]
        && conv.stride == [1, 1]
        && conv.dilation == [1, 1]
        && conv.groups == 1
}

pub(crate) fn gather_tokens<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let channels = tokens.shape().dims::<3>()[2];
    let gather_indices = indices.unsqueeze_dim::<3>(2).repeat_dim(2, channels);
    tokens.gather(1, gather_indices)
}

pub(crate) fn flatten_nchw_to_nlc<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, channels, height, width] = x.shape().dims::<4>();
    x.permute([0, 2, 3, 1])
        .reshape([batch, height * width, channels])
}

pub(crate) fn nlc_to_nchw<B: Backend>(
    x: Tensor<B, 3>,
    height: usize,
    width: usize,
) -> Tensor<B, 4> {
    let [batch, tokens, channels] = x.shape().dims::<3>();
    debug_assert_eq!(tokens, height * width);
    x.reshape([batch, height, width, channels])
        .permute([0, 3, 1, 2])
}

pub(crate) fn l2_normalize_channels<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let denom = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .add_scalar(1.0e-12)
        .sqrt();
    x / denom
}

pub(crate) fn adaptive_pool<B: Backend>(x: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4> {
    let [_, _, height, width] = x.shape().dims::<4>();
    if [height, width] == output_size {
        x
    } else {
        adaptive_avg_pool2d(x, output_size)
    }
}

pub(crate) fn nearest_resize<B: Backend>(x: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4> {
    let [_, _, height, width] = x.shape().dims::<4>();
    if [height, width] == output_size {
        x
    } else {
        interpolate(
            x,
            output_size,
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    }
}

pub(crate) fn coordinate_grid<B: Backend>(
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut values = Vec::with_capacity(height * width * 2);
    for row in 0..height {
        let y = if height > 1 {
            row as f32 / (height - 1) as f32
        } else {
            0.0
        };
        for col in 0..width {
            let x = if width > 1 {
                col as f32 / (width - 1) as f32
            } else {
                0.0
            };
            values.push(y);
            values.push(x);
        }
    }
    Tensor::<B, 3>::from_data(TensorData::new(values, [1, height * width, 2]), device)
}
