use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenGridShape {
    pub depth: usize,
    pub height: usize,
    pub width: usize,
}

impl TokenGridShape {
    pub const fn new(depth: usize, height: usize, width: usize) -> Self {
        Self {
            depth,
            height,
            width,
        }
    }

    pub const fn len(&self) -> usize {
        self.depth * self.height * self.width
    }

    pub const fn is_empty(&self) -> bool {
        self.depth == 0 || self.height == 0 || self.width == 0
    }

    pub const fn tokens_per_frame(&self) -> usize {
        self.height * self.width
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseTokenMask {
    indices: Vec<usize>,
    dense_len: usize,
}

impl SparseTokenMask {
    pub fn new(mut indices: Vec<usize>, dense_len: usize) -> Result<Self> {
        indices.sort_unstable();
        indices.dedup();
        ensure!(dense_len > 0, "dense token count must be nonzero");
        ensure!(
            indices.iter().all(|&index| index < dense_len),
            "sparse token index outside dense token range"
        );
        Ok(Self { indices, dense_len })
    }

    pub fn from_keep_ratio(dense_len: usize, keep_ratio: f32) -> Self {
        let keep = ((dense_len as f32) * keep_ratio.clamp(0.0, 1.0)).ceil() as usize;
        let keep = keep.max(1).min(dense_len.max(1));
        Self {
            indices: (0..keep).collect(),
            dense_len,
        }
    }

    pub fn evenly_spaced(dense_len: usize, keep: usize) -> Self {
        let keep = keep.max(1).min(dense_len.max(1));
        if keep == dense_len {
            return Self {
                indices: (0..dense_len).collect(),
                dense_len,
            };
        }
        let last = dense_len.saturating_sub(1);
        let indices = (0..keep)
            .map(|i| ((i * last) + (keep / 2)) / keep.max(1))
            .collect();
        Self::new(indices, dense_len).expect("generated mask is valid")
    }

    pub fn all(dense_len: usize) -> Self {
        Self {
            indices: (0..dense_len).collect(),
            dense_len,
        }
    }

    pub fn complement(&self) -> Self {
        let mut keep = vec![false; self.dense_len];
        for &index in &self.indices {
            keep[index] = true;
        }
        let indices = keep
            .into_iter()
            .enumerate()
            .filter_map(|(index, present)| (!present).then_some(index))
            .collect();
        Self {
            indices,
            dense_len: self.dense_len,
        }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn dense_len(&self) -> usize {
        self.dense_len
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn to_tensor<B: Backend>(&self, batch: usize, device: &B::Device) -> Tensor<B, 2, Int> {
        repeat_token_indices::<B>(&self.indices, batch, device)
    }
}

#[derive(Clone, Debug)]
pub struct SparseVideoTokens<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub indices: Tensor<B, 2, Int>,
    pub grid: TokenGridShape,
}

impl<B: Backend> SparseVideoTokens<B> {
    pub fn new(tokens: Tensor<B, 3>, indices: Tensor<B, 2, Int>, grid: TokenGridShape) -> Self {
        Self {
            tokens,
            indices,
            grid,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.tokens.shape().dims::<3>()
    }
}

pub fn dense_token_indices(grid: TokenGridShape) -> Vec<usize> {
    (0..grid.len()).collect()
}

pub fn complement_indices(indices: &[usize], dense_len: usize) -> Vec<usize> {
    SparseTokenMask::new(indices.to_vec(), dense_len)
        .expect("valid token indices")
        .complement()
        .indices
}

pub fn make_context_target_masks(
    grid: TokenGridShape,
    context_keep_ratio: f32,
) -> (SparseTokenMask, SparseTokenMask) {
    let context = SparseTokenMask::from_keep_ratio(grid.len(), context_keep_ratio);
    let target = context.complement();
    (context, target)
}

pub fn repeat_token_indices<B: Backend>(
    indices: &[usize],
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values: Vec<i64> = (0..batch)
        .flat_map(|_| indices.iter().map(|&index| index as i64))
        .collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, indices.len()]), device)
}

pub fn apply_token_mask<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let dim = tokens.shape().dims::<3>()[2];
    let gather_indices = indices.unsqueeze_dim::<3>(2).repeat_dim(2, dim);
    tokens.gather(1, gather_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32>;

    #[test]
    fn complement_preserves_sorted_indices() {
        let mask = SparseTokenMask::new(vec![3, 1, 1], 5).expect("mask");
        assert_eq!(mask.indices(), &[1, 3]);
        assert_eq!(mask.complement().indices(), &[0, 2, 4]);
    }

    #[test]
    fn token_mask_gathers_backend_values() {
        let device = Default::default();
        let data = TensorData::new(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
            [1, 6, 2],
        );
        let tokens = Tensor::<B, 3>::from_data(data, &device);
        let indices = repeat_token_indices::<B>(&[1, 4], 1, &device);
        let kept = apply_token_mask(tokens, indices)
            .into_data()
            .to_vec::<f32>()
            .expect("values");
        assert_eq!(kept, vec![2.0, 3.0, 8.0, 9.0]);
    }
}
