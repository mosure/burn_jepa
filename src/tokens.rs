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
        let indices = Self::evenly_spaced_indices(dense_len, keep);
        Self::new(indices, dense_len).expect("generated mask is valid")
    }

    pub fn evenly_spaced_indices(dense_len: usize, keep: usize) -> Vec<usize> {
        let keep = keep.max(1).min(dense_len.max(1));
        if keep == dense_len {
            return (0..dense_len).collect();
        }
        let last = dense_len.saturating_sub(1);
        (0..keep)
            .map(|i| ((i * last) + (keep / 2)) / keep.max(1))
            .collect()
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
pub enum SparseMaskBatch<B: Backend> {
    Uniform {
        mask: SparseTokenMask,
        indices: Tensor<B, 2, Int>,
        batch: usize,
    },
    FixedWidth {
        rows: Vec<Vec<usize>>,
        dense_len: usize,
        indices: Tensor<B, 2, Int>,
    },
    Ragged {
        rows: Vec<Vec<usize>>,
        dense_len: usize,
        indices: Tensor<B, 2, Int>,
        lengths: Vec<usize>,
    },
}

impl<B: Backend> SparseMaskBatch<B> {
    pub fn uniform(mask: SparseTokenMask, batch: usize, device: &B::Device) -> Result<Self> {
        ensure!(batch > 0, "sparse mask batch must be non-empty");
        let indices = mask.to_tensor::<B>(batch, device);
        Ok(Self::Uniform {
            mask,
            indices,
            batch,
        })
    }

    pub fn fixed_width(
        rows: Vec<Vec<usize>>,
        dense_len: usize,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(dense_len > 0, "dense token count must be nonzero");
        ensure!(!rows.is_empty(), "sparse mask batch must be non-empty");
        let width = rows[0].len();
        ensure!(width > 0, "fixed-width sparse mask rows must be non-empty");
        ensure!(
            rows.iter().all(|row| row.len() == width),
            "fixed-width sparse mask rows must have equal lengths"
        );
        let mut normalized = Vec::with_capacity(rows.len());
        for mut row in rows {
            row.sort_unstable();
            row.dedup();
            ensure!(
                row.len() == width,
                "fixed-width sparse mask row contains duplicate indices"
            );
            ensure!(
                row.iter().all(|&index| index < dense_len),
                "sparse token index outside dense token range"
            );
            normalized.push(row);
        }
        let values = normalized
            .iter()
            .flat_map(|row| row.iter().map(|&index| index as i64))
            .collect::<Vec<_>>();
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(values, [normalized.len(), width]),
            device,
        );
        Ok(Self::FixedWidth {
            rows: normalized,
            dense_len,
            indices,
        })
    }

    pub fn from_rows(rows: Vec<Vec<usize>>, dense_len: usize, device: &B::Device) -> Result<Self> {
        ensure!(!rows.is_empty(), "sparse mask batch must be non-empty");
        if rows.iter().all(|row| row == &rows[0]) {
            return Self::uniform(
                SparseTokenMask::new(rows[0].clone(), dense_len)?,
                rows.len(),
                device,
            );
        }
        let same_width = rows.iter().all(|row| row.len() == rows[0].len());
        if same_width {
            return Self::fixed_width(rows, dense_len, device);
        }
        Self::ragged(rows, dense_len, device)
    }

    pub fn ragged(rows: Vec<Vec<usize>>, dense_len: usize, device: &B::Device) -> Result<Self> {
        ensure!(dense_len > 0, "dense token count must be nonzero");
        ensure!(
            !rows.is_empty(),
            "ragged sparse mask batch must be non-empty"
        );
        let mut normalized = Vec::with_capacity(rows.len());
        let mut lengths = Vec::with_capacity(rows.len());
        for mut row in rows {
            row.sort_unstable();
            row.dedup();
            ensure!(!row.is_empty(), "ragged sparse mask rows must be non-empty");
            ensure!(
                row.iter().all(|&index| index < dense_len),
                "sparse token index outside dense token range"
            );
            lengths.push(row.len());
            normalized.push(row);
        }
        let max_width = lengths.iter().copied().max().unwrap_or(0);
        ensure!(max_width > 0, "ragged sparse mask rows must be non-empty");
        let padded = padded_rows_from_normalized(&normalized, max_width);
        let values = padded
            .iter()
            .flat_map(|row| row.iter().map(|&index| index as i64))
            .collect::<Vec<_>>();
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(values, [normalized.len(), max_width]),
            device,
        );
        Ok(Self::Ragged {
            rows: normalized,
            dense_len,
            indices,
            lengths,
        })
    }

    pub fn batch(&self) -> usize {
        match self {
            Self::Uniform { batch, .. } => *batch,
            Self::FixedWidth { rows, .. } => rows.len(),
            Self::Ragged { rows, .. } => rows.len(),
        }
    }

    pub fn dense_len(&self) -> usize {
        match self {
            Self::Uniform { mask, .. } => mask.dense_len(),
            Self::FixedWidth { dense_len, .. } => *dense_len,
            Self::Ragged { dense_len, .. } => *dense_len,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Uniform { mask, .. } => mask.len(),
            Self::FixedWidth { rows, .. } => rows[0].len(),
            Self::Ragged { lengths, .. } => lengths.iter().copied().max().unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform { .. })
    }

    pub fn is_ragged(&self) -> bool {
        matches!(self, Self::Ragged { .. })
    }

    pub fn valid_token_count(&self) -> usize {
        match self {
            Self::Uniform { mask, batch, .. } => mask.len() * *batch,
            Self::FixedWidth { rows, .. } => rows.iter().map(Vec::len).sum(),
            Self::Ragged { lengths, .. } => lengths.iter().sum(),
        }
    }

    pub fn uniform_mask(&self) -> Option<&SparseTokenMask> {
        match self {
            Self::Uniform { mask, .. } => Some(mask),
            Self::FixedWidth { .. } | Self::Ragged { .. } => None,
        }
    }

    pub fn indices(&self) -> Tensor<B, 2, Int> {
        match self {
            Self::Uniform { indices, .. }
            | Self::FixedWidth { indices, .. }
            | Self::Ragged { indices, .. } => indices.clone(),
        }
    }

    pub fn rows(&self) -> Vec<Vec<usize>> {
        match self {
            Self::Uniform { mask, batch, .. } => (0..*batch)
                .map(|_| mask.indices().to_vec())
                .collect::<Vec<_>>(),
            Self::FixedWidth { rows, .. } => rows.clone(),
            Self::Ragged { rows, .. } => rows.clone(),
        }
    }

    pub fn padded_rows(&self) -> Vec<Vec<usize>> {
        match self {
            Self::Uniform { mask, batch, .. } => (0..*batch)
                .map(|_| mask.indices().to_vec())
                .collect::<Vec<_>>(),
            Self::FixedWidth { rows, .. } => rows.clone(),
            Self::Ragged { rows, lengths, .. } => {
                padded_rows_from_normalized(rows, lengths.iter().copied().max().unwrap_or(0))
            }
        }
    }

    pub fn valid_token_mask(&self, device: &B::Device) -> Option<Tensor<B, 2>> {
        match self {
            Self::Ragged { lengths, .. } => {
                let width = lengths.iter().copied().max().unwrap_or(0);
                let values = lengths
                    .iter()
                    .flat_map(|&length| {
                        (0..width).map(move |index| if index < length { 1.0 } else { 0.0 })
                    })
                    .collect::<Vec<_>>();
                Some(Tensor::<B, 2>::from_data(
                    TensorData::new(values, [lengths.len(), width]),
                    device,
                ))
            }
            _ => None,
        }
    }

    pub fn first_mask(&self) -> Result<SparseTokenMask> {
        match self {
            Self::Uniform { mask, .. } => Ok(mask.clone()),
            Self::FixedWidth {
                rows, dense_len, ..
            }
            | Self::Ragged {
                rows, dense_len, ..
            } => SparseTokenMask::new(rows[0].clone(), *dense_len),
        }
    }
}

fn padded_rows_from_normalized(rows: &[Vec<usize>], width: usize) -> Vec<Vec<usize>> {
    rows.iter()
        .map(|row| {
            let mut padded = row.clone();
            if let Some(&pad) = padded.last() {
                padded.resize(width, pad);
            }
            padded
        })
        .collect()
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

pub fn target_mask_from_context(
    context: &SparseTokenMask,
    target_tokens: usize,
) -> Result<SparseTokenMask> {
    let dense_len = context.dense_len();
    let target_tokens = target_tokens
        .max(1)
        .min(dense_len.saturating_sub(context.len()).max(1));
    let mut blocked = vec![false; dense_len];
    for &index in context.indices() {
        blocked[index] = true;
    }
    let mut target = SparseTokenMask::evenly_spaced_indices(dense_len, target_tokens)
        .into_iter()
        .filter(|&index| !blocked[index])
        .collect::<Vec<_>>();
    for &index in &target {
        blocked[index] = true;
    }
    if target.len() < target_tokens {
        for (index, blocked) in blocked.iter_mut().enumerate() {
            if !*blocked {
                target.push(index);
                *blocked = true;
                if target.len() >= target_tokens {
                    break;
                }
            }
        }
    }
    SparseTokenMask::new(target, dense_len)
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

pub fn apply_mask_batch<B: Backend>(
    tokens: Tensor<B, 3>,
    mask: &SparseMaskBatch<B>,
) -> Tensor<B, 3> {
    apply_token_mask(tokens, mask.indices())
}

#[cfg(all(test, feature = "ndarray"))]
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
    fn target_mask_from_context_is_disjoint_and_fills_budget() {
        let context = SparseTokenMask::new(vec![0, 2, 4], 6).expect("context");
        let target = target_mask_from_context(&context, 2).expect("target");

        assert_eq!(target.len(), 2);
        assert!(
            target
                .indices()
                .iter()
                .all(|idx| !context.indices().contains(idx))
        );
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
