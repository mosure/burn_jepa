use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Module, Debug)]
pub struct AnyUpRoPE<B: Backend> {
    pub freqs: Param<Tensor<B, 2>>,
    #[module(skip)]
    pub dim: usize,
    #[module(skip)]
    pub theta: f32,
}

impl<B: Backend> AnyUpRoPE<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let dim = dim.max(4);
        Self {
            freqs: Param::from_tensor(Tensor::<B, 2>::from_data(
                TensorData::new(rope_frequencies(dim, 100.0), [2, dim]),
                device,
            )),
            dim,
            theta: 100.0,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let angle = coords.matmul(self.freqs.val().unsqueeze_dim::<3>(0));
        x.clone() * angle.clone().cos() + rotate_half(x) * angle.sin()
    }
}

pub fn rotate_half<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, tokens, dim] = x.shape().dims::<3>();
    let first = x.clone().slice_dim(2, 0..dim / 2);
    let second = x.slice_dim(2, dim / 2..dim);
    Tensor::cat(vec![second.neg(), first], 2).reshape([batch, tokens, dim])
}

fn rope_frequencies(dim: usize, theta: f32) -> Vec<f32> {
    let quarter = dim / 4;
    let mut freqs_1d = Vec::with_capacity(quarter * 2);
    for index in 0..quarter {
        let exponent = if quarter > 1 {
            -(index as f32) / (quarter - 1) as f32
        } else {
            0.0
        };
        freqs_1d.push(theta.powf(exponent));
    }
    let original = freqs_1d.clone();
    freqs_1d.extend(original);

    let mut freqs = vec![0.0; 2 * dim];
    for (index, value) in freqs_1d.iter().copied().enumerate() {
        freqs[index] = value * 2.0 * core::f32::consts::PI;
        freqs[dim + dim / 2 + index] = value * 2.0 * core::f32::consts::PI;
    }
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_frequency_layout_matches_anyup_2d_split() {
        let freqs = rope_frequencies(8, 100.0);
        assert!(freqs[0] > 0.0);
        assert_eq!(freqs[4], 0.0);
        assert_eq!(freqs[8], 0.0);
        assert!(freqs[12] > 0.0);
    }
}
