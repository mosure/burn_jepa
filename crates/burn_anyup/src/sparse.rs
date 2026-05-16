use crate::attention::natten_window;
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AnyUpSparseFeatureUpdateMode {
    #[default]
    AssignLatest,
    Ema {
        alpha: f32,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnyUpSparseFeatureAgeMode {
    #[default]
    ObservedFrames,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnyUpSparseFeatureMemoryWriteMode {
    #[default]
    Auto,
    ScatterNdAssign,
    ScatterAddDelta,
}

impl AnyUpSparseFeatureMemoryWriteMode {
    fn resolved<B: Backend>(self) -> Self {
        match self {
            Self::Auto if backend_is_cuda::<B>() => Self::ScatterAddDelta,
            Self::Auto => Self::ScatterNdAssign,
            mode => mode,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnyUpHighResFeatureMemoryConfig {
    pub update_mode: AnyUpSparseFeatureUpdateMode,
    pub age_mode: AnyUpSparseFeatureAgeMode,
    pub write_mode: AnyUpSparseFeatureMemoryWriteMode,
}

impl Default for AnyUpHighResFeatureMemoryConfig {
    fn default() -> Self {
        Self {
            update_mode: AnyUpSparseFeatureUpdateMode::AssignLatest,
            age_mode: AnyUpSparseFeatureAgeMode::ObservedFrames,
            write_mode: AnyUpSparseFeatureMemoryWriteMode::Auto,
        }
    }
}

impl AnyUpHighResFeatureMemoryConfig {
    fn normalized(mut self) -> Self {
        if let AnyUpSparseFeatureUpdateMode::Ema { alpha } = self.update_mode {
            self.update_mode = AnyUpSparseFeatureUpdateMode::Ema {
                alpha: alpha.clamp(0.0, 1.0),
            };
        }
        self
    }
}

#[derive(Debug)]
pub struct AnyUpSparseOutput<B: Backend> {
    pub features: Tensor<B, 3>,
    pub indices: Tensor<B, 2, Int>,
    pub output_size: [usize; 2],
}

impl<B: Backend> Clone for AnyUpSparseOutput<B> {
    fn clone(&self) -> Self {
        Self {
            features: self.features.clone(),
            indices: self.indices.clone(),
            output_size: self.output_size,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AnyUpSparseWindow<B: Backend> {
    pub(crate) low_indices: Tensor<B, 2, Int>,
    pub(crate) valid: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct AnyUpSparseOutputPlan<B: Backend> {
    pub batch: usize,
    pub sparse_len: usize,
    pub output_size: [usize; 2],
    pub feature_size: [usize; 2],
    pub indices: Tensor<B, 2, Int>,
    pub(crate) windows: Vec<AnyUpSparseWindow<B>>,
}

impl<B: Backend> AnyUpSparseOutputPlan<B> {
    pub fn new(
        indices: Vec<usize>,
        output_size: [usize; 2],
        feature_size: [usize; 2],
        batch: usize,
        window_ratio: f32,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(batch > 0, "sparse AnyUp plan batch must be nonzero");
        Self::from_rows(
            (0..batch).map(|_| indices.clone()).collect(),
            output_size,
            feature_size,
            window_ratio,
            device,
        )
    }

    pub fn from_rows(
        rows: Vec<Vec<usize>>,
        output_size: [usize; 2],
        feature_size: [usize; 2],
        window_ratio: f32,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(!rows.is_empty(), "sparse AnyUp plan batch must be nonempty");
        ensure!(
            output_size[0] > 0 && output_size[1] > 0,
            "sparse AnyUp output size must be nonempty"
        );
        ensure!(
            feature_size[0] > 0 && feature_size[1] > 0,
            "sparse AnyUp feature size must be nonempty"
        );
        let dense_len = output_size[0] * output_size[1];
        let sparse_len = rows[0].len();
        ensure!(sparse_len > 0, "sparse AnyUp rows must be nonempty");
        let rows = rows
            .into_iter()
            .map(|mut row| {
                row.sort_unstable();
                row.dedup();
                ensure!(
                    row.len() == sparse_len,
                    "sparse AnyUp rows must have the same width and no duplicates"
                );
                ensure!(
                    row.iter().all(|&index| index < dense_len),
                    "sparse AnyUp output index outside dense output range"
                );
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = rows.len();
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                rows.iter()
                    .flat_map(|row| row.iter().map(|&index| index as i64))
                    .collect::<Vec<_>>(),
                [batch, sparse_len],
            ),
            device,
        );
        let windows = sparse_windows(
            &rows,
            output_size,
            feature_size,
            window_ratio,
            batch,
            sparse_len,
            device,
        );
        Ok(Self {
            batch,
            sparse_len,
            output_size,
            feature_size,
            indices,
            windows,
        })
    }

    pub fn window_len(&self) -> usize {
        self.windows.len()
    }
}

fn sparse_windows<B: Backend>(
    rows: &[Vec<usize>],
    output_size: [usize; 2],
    feature_size: [usize; 2],
    window_ratio: f32,
    batch: usize,
    sparse_len: usize,
    device: &B::Device,
) -> Vec<AnyUpSparseWindow<B>> {
    let [hq, wq] = output_size;
    let [hk, wk] = feature_size;
    let (kernel_h, kernel_w, dilation_h, dilation_w) = natten_window(hq, wq, hk, wk, window_ratio);
    let radius_h = kernel_h / 2;
    let radius_w = kernel_w / 2;
    let mut windows = Vec::with_capacity(kernel_h * kernel_w);
    for ky in 0..kernel_h {
        for kx in 0..kernel_w {
            let row_offset = ky as isize - radius_h as isize;
            let col_offset = kx as isize - radius_w as isize;
            let mut low_indices = Vec::with_capacity(batch * sparse_len);
            let mut valid = Vec::with_capacity(batch * sparse_len);
            for row in rows {
                for &index in row {
                    let query_row = index / wq;
                    let query_col = index % wq;
                    let shifted_row = query_row as isize + row_offset * dilation_h as isize;
                    let shifted_col = query_col as isize + col_offset * dilation_w as isize;
                    let in_bounds = shifted_row >= 0
                        && shifted_col >= 0
                        && (shifted_row as usize) < hq
                        && (shifted_col as usize) < wq;
                    if in_bounds {
                        let low_row = shifted_row as usize * hk / hq;
                        let low_col = shifted_col as usize * wk / wq;
                        low_indices.push((low_row * wk + low_col) as i64);
                        valid.push(1.0);
                    } else {
                        low_indices.push(0);
                        valid.push(0.0);
                    }
                }
            }
            windows.push(AnyUpSparseWindow {
                low_indices: Tensor::<B, 2, Int>::from_data(
                    TensorData::new(low_indices, [batch, sparse_len]),
                    device,
                ),
                valid: Tensor::<B, 2>::from_data(
                    TensorData::new(valid, [batch, sparse_len]),
                    device,
                ),
            });
        }
    }
    windows
}

#[derive(Clone, Debug)]
struct AnyUpSparseUpdatePlan<B: Backend> {
    batch: usize,
    sparse_len: usize,
    dense_len: usize,
    scatter_rows: Tensor<B, 3, Int>,
}

impl<B: Backend> AnyUpSparseUpdatePlan<B> {
    fn new(batch: usize, sparse_len: usize, dense_len: usize, device: &B::Device) -> Self {
        let scatter_rows = Tensor::<B, 1, Int>::arange(0..batch as i64, device)
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, sparse_len)
            .unsqueeze_dim::<3>(2);
        Self {
            batch,
            sparse_len,
            dense_len,
            scatter_rows,
        }
    }

    fn matches(&self, batch: usize, sparse_len: usize, dense_len: usize) -> bool {
        self.batch == batch && self.sparse_len == sparse_len && self.dense_len == dense_len
    }

    fn scatter_indices(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3, Int> {
        Tensor::cat(
            vec![self.scatter_rows.clone(), indices.unsqueeze_dim::<3>(2)],
            2,
        )
    }
}

#[derive(Clone, Debug)]
pub struct AnyUpHighResFeatureMemory<B: Backend> {
    config: AnyUpHighResFeatureMemoryConfig,
    batch: usize,
    output_size: [usize; 2],
    channels: usize,
    device: B::Device,
    features: Tensor<B, 3>,
    observed: Tensor<B, 2>,
    age_frames: Tensor<B, 2>,
    step: usize,
    last_updated_tokens: usize,
    update_plan: Option<AnyUpSparseUpdatePlan<B>>,
}

#[derive(Debug)]
pub struct AnyUpHighResFeatureMemoryOutput<B: Backend> {
    pub features: Tensor<B, 3>,
    pub observed: Tensor<B, 2>,
    pub age_frames: Tensor<B, 2>,
    pub output_size: [usize; 2],
    pub step: usize,
    pub updated_tokens: usize,
    pub dense_tokens: usize,
}

impl<B: Backend> AnyUpHighResFeatureMemory<B> {
    pub fn new(
        config: AnyUpHighResFeatureMemoryConfig,
        batch: usize,
        output_size: [usize; 2],
        channels: usize,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(batch > 0, "AnyUp feature memory batch must be nonzero");
        ensure!(
            output_size[0] > 0 && output_size[1] > 0,
            "AnyUp feature memory output size must be nonempty"
        );
        ensure!(
            channels > 0,
            "AnyUp feature memory channels must be nonzero"
        );
        let dense_tokens = output_size[0] * output_size[1];
        Ok(Self {
            config: config.normalized(),
            batch,
            output_size,
            channels,
            device: device.clone(),
            features: Tensor::<B, 3>::zeros([batch, dense_tokens, channels], device),
            observed: Tensor::<B, 2>::zeros([batch, dense_tokens], device),
            age_frames: Tensor::<B, 2>::zeros([batch, dense_tokens], device),
            step: 0,
            last_updated_tokens: 0,
            update_plan: None,
        })
    }

    pub fn snapshot(&self) -> AnyUpHighResFeatureMemoryOutput<B> {
        self.output(self.last_updated_tokens)
    }

    pub fn reset(&mut self) {
        let dense_tokens = self.dense_tokens();
        self.features =
            Tensor::<B, 3>::zeros([self.batch, dense_tokens, self.channels], &self.device);
        self.observed = Tensor::<B, 2>::zeros([self.batch, dense_tokens], &self.device);
        self.age_frames = Tensor::<B, 2>::zeros([self.batch, dense_tokens], &self.device);
        self.step = 0;
        self.last_updated_tokens = 0;
        self.update_plan = None;
    }

    pub fn update_sparse_output(
        &mut self,
        output: AnyUpSparseOutput<B>,
    ) -> Result<AnyUpHighResFeatureMemoryOutput<B>> {
        ensure!(
            output.output_size == self.output_size,
            "sparse AnyUp output size does not match high-res feature memory"
        );
        self.update_tokens(output.features, output.indices)
    }

    pub fn update_tokens(
        &mut self,
        tokens: Tensor<B, 3>,
        indices: Tensor<B, 2, Int>,
    ) -> Result<AnyUpHighResFeatureMemoryOutput<B>> {
        let [batch, sparse_len, channels] = tokens.shape().dims::<3>();
        ensure!(
            batch == self.batch,
            "sparse AnyUp update batch does not match high-res feature memory"
        );
        ensure!(
            channels == self.channels,
            "sparse AnyUp update channels do not match high-res feature memory"
        );
        let [index_batch, index_len] = indices.shape().dims::<2>();
        ensure!(
            index_batch == batch && index_len == sparse_len,
            "sparse AnyUp update indices must match token shape"
        );
        ensure!(sparse_len > 0, "sparse AnyUp update must be nonempty");
        let write_mode = self.config.write_mode.resolved::<B>();
        match self.config.update_mode {
            AnyUpSparseFeatureUpdateMode::AssignLatest => {
                self.update_features(
                    tokens,
                    indices.clone(),
                    sparse_len,
                    channels,
                    write_mode,
                    None,
                );
            }
            AnyUpSparseFeatureUpdateMode::Ema { alpha } => {
                let previous_features =
                    gather_memory_tokens(self.features.clone(), indices.clone());
                let observed = gather_memory_tokens(
                    self.observed.clone().unsqueeze_dim::<3>(2),
                    indices.clone(),
                )
                .repeat_dim(2, channels);
                let blended = previous_features.clone().mul_scalar(1.0 - alpha)
                    + tokens.clone().mul_scalar(alpha);
                let first_observation = observed.clone().mul_scalar(-1.0) + 1.0;
                let update_values = blended * observed + tokens * first_observation;
                self.update_features(
                    update_values,
                    indices.clone(),
                    sparse_len,
                    channels,
                    write_mode,
                    Some(previous_features),
                );
            }
        }
        if self.step > 0 {
            match self.config.age_mode {
                AnyUpSparseFeatureAgeMode::ObservedFrames => {
                    self.age_frames = self.age_frames.clone() + self.observed.clone();
                }
            }
        }
        self.update_observation_metadata(indices, batch, sparse_len, write_mode);
        self.step += 1;
        self.last_updated_tokens = batch * sparse_len;
        Ok(self.output(self.last_updated_tokens))
    }

    pub fn dense_tokens(&self) -> usize {
        self.output_size[0] * self.output_size[1]
    }

    fn scatter_indices(
        &mut self,
        indices: Tensor<B, 2, Int>,
        sparse_len: usize,
    ) -> Tensor<B, 3, Int> {
        let dense_len = self.dense_tokens();
        if self
            .update_plan
            .as_ref()
            .is_none_or(|plan| !plan.matches(self.batch, sparse_len, dense_len))
        {
            self.update_plan = Some(AnyUpSparseUpdatePlan::new(
                self.batch,
                sparse_len,
                dense_len,
                &self.device,
            ));
        }
        self.update_plan
            .as_ref()
            .expect("AnyUp update plan initialized")
            .scatter_indices(indices)
    }

    fn update_features(
        &mut self,
        values: Tensor<B, 3>,
        indices: Tensor<B, 2, Int>,
        sparse_len: usize,
        channels: usize,
        write_mode: AnyUpSparseFeatureMemoryWriteMode,
        previous_values: Option<Tensor<B, 3>>,
    ) {
        match write_mode {
            AnyUpSparseFeatureMemoryWriteMode::Auto => unreachable!("write mode is resolved"),
            AnyUpSparseFeatureMemoryWriteMode::ScatterNdAssign => {
                let scatter_indices = self.scatter_indices(indices, sparse_len);
                self.features = self.features.clone().scatter_nd(
                    scatter_indices,
                    values,
                    IndexingUpdateOp::Assign,
                );
            }
            AnyUpSparseFeatureMemoryWriteMode::ScatterAddDelta => {
                let feature_indices = indices
                    .clone()
                    .unsqueeze_dim::<3>(2)
                    .repeat_dim(2, channels);
                let previous = previous_values
                    .unwrap_or_else(|| gather_memory_tokens(self.features.clone(), indices));
                self.features = self.features.clone().scatter(
                    1,
                    feature_indices,
                    values - previous,
                    IndexingUpdateOp::Add,
                );
            }
        }
    }

    fn update_observation_metadata(
        &mut self,
        indices: Tensor<B, 2, Int>,
        batch: usize,
        sparse_len: usize,
        write_mode: AnyUpSparseFeatureMemoryWriteMode,
    ) {
        match write_mode {
            AnyUpSparseFeatureMemoryWriteMode::Auto => unreachable!("write mode is resolved"),
            AnyUpSparseFeatureMemoryWriteMode::ScatterNdAssign => {
                let scatter_indices = self.scatter_indices(indices, sparse_len);
                self.observed = self.observed.clone().scatter_nd(
                    scatter_indices.clone(),
                    Tensor::<B, 2>::ones([batch, sparse_len], &self.device),
                    IndexingUpdateOp::Assign,
                );
                self.age_frames = self.age_frames.clone().scatter_nd(
                    scatter_indices,
                    Tensor::<B, 2>::zeros([batch, sparse_len], &self.device),
                    IndexingUpdateOp::Assign,
                );
            }
            AnyUpSparseFeatureMemoryWriteMode::ScatterAddDelta => {
                let observed_delta = Tensor::<B, 2>::ones([batch, sparse_len], &self.device)
                    - self.observed.clone().gather(1, indices.clone());
                self.observed = self.observed.clone().scatter(
                    1,
                    indices.clone(),
                    observed_delta,
                    IndexingUpdateOp::Add,
                );
                let age_delta = self
                    .age_frames
                    .clone()
                    .gather(1, indices.clone())
                    .mul_scalar(-1.0);
                self.age_frames =
                    self.age_frames
                        .clone()
                        .scatter(1, indices, age_delta, IndexingUpdateOp::Add);
            }
        }
    }

    fn output(&self, updated_tokens: usize) -> AnyUpHighResFeatureMemoryOutput<B> {
        AnyUpHighResFeatureMemoryOutput {
            features: self.features.clone(),
            observed: self.observed.clone(),
            age_frames: self.age_frames.clone(),
            output_size: self.output_size,
            step: self.step,
            updated_tokens,
            dense_tokens: self.dense_tokens(),
        }
    }
}

pub fn sparse_low_features_to_nchw<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
    feature_size: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    ensure!(
        feature_size[0] > 0 && feature_size[1] > 0,
        "sparse AnyUp low-res feature size must be nonempty"
    );
    let [batch, sparse_len, channels] = tokens.shape().dims::<3>();
    let [index_batch, index_len] = indices.shape().dims::<2>();
    ensure!(
        index_batch == batch && index_len == sparse_len,
        "sparse AnyUp low-res indices must match token shape"
    );
    let tokens = tokens + Tensor::<B, 3>::zeros([batch, sparse_len, channels], device);
    let dense_len = feature_size[0] * feature_size[1];
    let rows = Tensor::<B, 1, Int>::arange(0..batch as i64, device)
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, sparse_len)
        .unsqueeze_dim::<3>(2);
    let scatter_indices = Tensor::cat(vec![rows, indices.unsqueeze_dim::<3>(2)], 2);
    let dense = Tensor::<B, 3>::zeros([batch, dense_len, channels], device).scatter_nd(
        scatter_indices,
        tokens,
        IndexingUpdateOp::Assign,
    );
    Ok(dense
        .reshape([batch, feature_size[0], feature_size[1], channels])
        .permute([0, 3, 1, 2]))
}

fn gather_memory_tokens<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let channels = tokens.shape().dims::<3>()[2];
    tokens.gather(1, indices.unsqueeze_dim::<3>(2).repeat_dim(2, channels))
}

fn backend_is_cuda<B: Backend>() -> bool {
    let name = std::any::type_name::<B>();
    name.contains("Cuda") || name.contains("cuda") || name.contains("CUDA")
}
