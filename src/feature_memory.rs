use crate::{SparseTokenMask, TokenGridShape, VJepaEncoderOutput, apply_token_mask};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterframeJepaFeatureUpdateMode {
    #[default]
    AssignLatest,
    Ema {
        alpha: f32,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterframeJepaFeatureAgeMode {
    #[default]
    ObservedFrames,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterframeJepaFeatureMemoryConfig {
    pub update_mode: InterframeJepaFeatureUpdateMode,
    pub age_mode: InterframeJepaFeatureAgeMode,
}

impl Default for InterframeJepaFeatureMemoryConfig {
    fn default() -> Self {
        Self {
            update_mode: InterframeJepaFeatureUpdateMode::AssignLatest,
            age_mode: InterframeJepaFeatureAgeMode::ObservedFrames,
        }
    }
}

impl InterframeJepaFeatureMemoryConfig {
    fn normalized(mut self) -> Self {
        if let InterframeJepaFeatureUpdateMode::Ema { alpha } = self.update_mode {
            self.update_mode = InterframeJepaFeatureUpdateMode::Ema {
                alpha: alpha.clamp(0.0, 1.0),
            };
        }
        self
    }
}

#[derive(Debug)]
pub struct InterframeJepaFeatureMemoryOutput<B: Backend> {
    pub features: Tensor<B, 3>,
    pub observed: Tensor<B, 2>,
    pub age_frames: Tensor<B, 2>,
    pub grid: TokenGridShape,
    pub step: usize,
    pub updated_tokens: usize,
    pub dense_tokens: usize,
}

pub fn jepa_feature_tokens_to_nchw<B: Backend>(
    tokens: Tensor<B, 3>,
    grid: TokenGridShape,
) -> Result<Tensor<B, 4>> {
    ensure!(
        grid.depth == 1,
        "NCHW feature view requires a single-frame token grid"
    );
    let [batch, token_count, channels] = tokens.shape().dims::<3>();
    ensure!(
        token_count == grid.len(),
        "token count does not match feature grid"
    );
    Ok(tokens
        .reshape([batch, grid.height, grid.width, channels])
        .permute([0, 3, 1, 2]))
}

impl<B: Backend> InterframeJepaFeatureMemoryOutput<B> {
    pub fn features_nchw(&self) -> Result<Tensor<B, 4>> {
        jepa_feature_tokens_to_nchw(self.features.clone(), self.grid)
    }
}

#[derive(Clone, Debug)]
struct InterframeJepaFeatureUpdatePlan<B: Backend> {
    batch: usize,
    token_count: usize,
    dense_tokens: usize,
    embed_dim: usize,
    scatter_rows: Tensor<B, 3, Int>,
    observed_values: Tensor<B, 2>,
    reset_age_values: Tensor<B, 2>,
}

impl<B: Backend> InterframeJepaFeatureUpdatePlan<B> {
    fn new(
        batch: usize,
        token_count: usize,
        dense_tokens: usize,
        embed_dim: usize,
        device: &B::Device,
    ) -> Self {
        let rows = Tensor::<B, 1, Int>::arange(0..batch as i64, device)
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, token_count)
            .unsqueeze_dim::<3>(2);
        Self {
            batch,
            token_count,
            dense_tokens,
            embed_dim,
            scatter_rows: rows,
            observed_values: Tensor::<B, 2>::ones([batch, token_count], device),
            reset_age_values: Tensor::<B, 2>::zeros([batch, token_count], device),
        }
    }

    fn matches(
        &self,
        batch: usize,
        token_count: usize,
        dense_tokens: usize,
        embed_dim: usize,
    ) -> bool {
        self.batch == batch
            && self.token_count == token_count
            && self.dense_tokens == dense_tokens
            && self.embed_dim == embed_dim
    }

    fn scatter_indices(&self, token_indices: Tensor<B, 2, Int>) -> Tensor<B, 3, Int> {
        Tensor::cat(
            vec![
                self.scatter_rows.clone(),
                token_indices.unsqueeze_dim::<3>(2),
            ],
            2,
        )
    }
}

#[derive(Clone, Debug)]
pub struct InterframeJepaFeatureMemory<B: Backend> {
    config: InterframeJepaFeatureMemoryConfig,
    batch: usize,
    grid: TokenGridShape,
    embed_dim: usize,
    device: B::Device,
    features: Tensor<B, 3>,
    observed: Tensor<B, 2>,
    age_frames: Tensor<B, 2>,
    step: usize,
    last_updated_tokens: usize,
    update_plan: Option<InterframeJepaFeatureUpdatePlan<B>>,
}

impl<B: Backend> InterframeJepaFeatureMemory<B> {
    pub fn new(
        config: InterframeJepaFeatureMemoryConfig,
        batch: usize,
        grid: TokenGridShape,
        embed_dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(batch > 0, "feature memory batch must be nonzero");
        ensure!(!grid.is_empty(), "feature memory grid must be nonempty");
        ensure!(embed_dim > 0, "feature memory embed dim must be nonzero");
        let dense_tokens = grid.len();
        Ok(Self {
            config: config.normalized(),
            batch,
            grid,
            embed_dim,
            device: device.clone(),
            features: Tensor::<B, 3>::zeros([batch, dense_tokens, embed_dim], device),
            observed: Tensor::<B, 2>::zeros([batch, dense_tokens], device),
            age_frames: Tensor::<B, 2>::zeros([batch, dense_tokens], device),
            step: 0,
            last_updated_tokens: 0,
            update_plan: None,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn grid(&self) -> TokenGridShape {
        self.grid
    }

    pub fn dense_tokens(&self) -> usize {
        self.grid.len()
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn config(&self) -> InterframeJepaFeatureMemoryConfig {
        self.config
    }

    pub fn snapshot(&self) -> InterframeJepaFeatureMemoryOutput<B> {
        self.output(self.last_updated_tokens)
    }

    pub fn update_from_encoder_output(
        &mut self,
        output: VJepaEncoderOutput<B>,
    ) -> Result<InterframeJepaFeatureMemoryOutput<B>> {
        self.update_tokens(output.tokens, output.token_indices, output.grid)
    }

    pub fn update_masked_tokens(
        &mut self,
        tokens: Tensor<B, 3>,
        mask: &SparseTokenMask,
        grid: TokenGridShape,
    ) -> Result<InterframeJepaFeatureMemoryOutput<B>> {
        ensure!(
            mask.dense_len() == self.dense_tokens(),
            "sparse mask dense length must match feature memory grid"
        );
        let token_indices = mask.to_tensor::<B>(self.batch, &self.device);
        self.update_tokens(tokens, token_indices, grid)
    }

    pub fn update_tokens(
        &mut self,
        tokens: Tensor<B, 3>,
        token_indices: Tensor<B, 2, Int>,
        grid: TokenGridShape,
    ) -> Result<InterframeJepaFeatureMemoryOutput<B>> {
        ensure!(
            grid == self.grid,
            "encoder output grid does not match feature memory grid"
        );
        let [batch, token_count, embed_dim] = tokens.shape().dims::<3>();
        ensure!(
            batch == self.batch,
            "encoder output batch does not match feature memory batch"
        );
        ensure!(
            embed_dim == self.embed_dim,
            "encoder output dim does not match feature memory dim"
        );
        let [index_batch, index_count] = token_indices.shape().dims::<2>();
        ensure!(
            index_batch == batch && index_count == token_count,
            "token index shape must match sparse encoder token shape"
        );
        ensure!(
            token_count > 0,
            "sparse update must include at least one token"
        );

        let (scatter_indices, observed_values, reset_age_values) =
            self.scatter_update_plan(token_indices.clone(), token_count);
        let update_values = match self.config.update_mode {
            InterframeJepaFeatureUpdateMode::AssignLatest => tokens,
            InterframeJepaFeatureUpdateMode::Ema { alpha } => {
                let previous = apply_token_mask(self.features.clone(), token_indices.clone());
                let observed =
                    apply_token_mask(self.observed.clone().unsqueeze_dim::<3>(2), token_indices)
                        .repeat_dim(2, embed_dim);
                let blended = previous.mul_scalar(1.0 - alpha) + tokens.clone().mul_scalar(alpha);
                let first_observation = observed.clone().mul_scalar(-1.0) + 1.0;
                blended * observed + tokens * first_observation
            }
        };
        self.features = self.features.clone().scatter_nd(
            scatter_indices.clone(),
            update_values,
            IndexingUpdateOp::Assign,
        );

        if self.step > 0 {
            match self.config.age_mode {
                InterframeJepaFeatureAgeMode::ObservedFrames => {
                    self.age_frames = self.age_frames.clone() + self.observed.clone();
                }
            }
        }
        self.observed = self.observed.clone().scatter_nd(
            scatter_indices.clone(),
            observed_values,
            IndexingUpdateOp::Assign,
        );
        self.age_frames = self.age_frames.clone().scatter_nd(
            scatter_indices,
            reset_age_values,
            IndexingUpdateOp::Assign,
        );
        self.step += 1;
        self.last_updated_tokens = batch * token_count;
        Ok(self.output(self.last_updated_tokens))
    }

    pub fn reset(&mut self) {
        let dense_tokens = self.dense_tokens();
        self.features =
            Tensor::<B, 3>::zeros([self.batch, dense_tokens, self.embed_dim], &self.device);
        self.observed = Tensor::<B, 2>::zeros([self.batch, dense_tokens], &self.device);
        self.age_frames = Tensor::<B, 2>::zeros([self.batch, dense_tokens], &self.device);
        self.step = 0;
        self.last_updated_tokens = 0;
        self.update_plan = None;
    }

    pub fn reset_rows(&mut self, rows: Tensor<B, 1, Int>) -> Result<()> {
        let [row_count] = rows.shape().dims::<1>();
        if row_count == 0 {
            return Ok(());
        }
        let dense_tokens = self.dense_tokens();
        let metadata_indices = self.row_token_indices(rows.clone(), row_count);
        let feature_indices = self.row_token_feature_indices(rows, row_count);
        let feature_values =
            Tensor::<B, 3>::zeros([row_count, dense_tokens, self.embed_dim], &self.device);
        self.features = self.features.clone().scatter_nd(
            feature_indices,
            feature_values,
            IndexingUpdateOp::Assign,
        );
        let metadata_values = Tensor::<B, 2>::zeros([row_count, dense_tokens], &self.device);
        self.observed = self.observed.clone().scatter_nd(
            metadata_indices.clone(),
            metadata_values.clone(),
            IndexingUpdateOp::Assign,
        );
        self.age_frames = self.age_frames.clone().scatter_nd(
            metadata_indices,
            metadata_values,
            IndexingUpdateOp::Assign,
        );
        Ok(())
    }

    pub fn reset_row_indices(&mut self, rows: &[usize]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        ensure!(
            rows.iter().all(|&row| row < self.batch),
            "feature memory row reset index outside batch"
        );
        let row_values = rows.iter().map(|&row| row as i64).collect::<Vec<_>>();
        let row_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(row_values, [rows.len()]), &self.device);
        self.reset_rows(row_tensor)
    }

    fn scatter_update_plan(
        &mut self,
        token_indices: Tensor<B, 2, Int>,
        token_count: usize,
    ) -> (Tensor<B, 3, Int>, Tensor<B, 2>, Tensor<B, 2>) {
        if self.update_plan.as_ref().is_none_or(|plan| {
            !plan.matches(self.batch, token_count, self.dense_tokens(), self.embed_dim)
        }) {
            self.update_plan = Some(InterframeJepaFeatureUpdatePlan::new(
                self.batch,
                token_count,
                self.dense_tokens(),
                self.embed_dim,
                &self.device,
            ));
        }
        let plan = self.update_plan.as_ref().expect("update plan initialized");
        (
            plan.scatter_indices(token_indices),
            plan.observed_values.clone(),
            plan.reset_age_values.clone(),
        )
    }

    fn row_token_indices(&self, rows: Tensor<B, 1, Int>, row_count: usize) -> Tensor<B, 3, Int> {
        let dense_tokens = self.dense_tokens();
        let row_indices = rows
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, dense_tokens)
            .unsqueeze_dim::<3>(2);
        let token_indices = Tensor::<B, 1, Int>::arange(0..dense_tokens as i64, &self.device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, row_count)
            .unsqueeze_dim::<3>(2);
        Tensor::cat(vec![row_indices, token_indices], 2)
    }

    fn row_token_feature_indices(
        &self,
        rows: Tensor<B, 1, Int>,
        row_count: usize,
    ) -> Tensor<B, 4, Int> {
        let dense_tokens = self.dense_tokens();
        let row_indices = rows
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, dense_tokens)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, self.embed_dim)
            .unsqueeze_dim::<4>(3);
        let token_indices = Tensor::<B, 1, Int>::arange(0..dense_tokens as i64, &self.device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, row_count)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, self.embed_dim)
            .unsqueeze_dim::<4>(3);
        let feature_indices = Tensor::<B, 1, Int>::arange(0..self.embed_dim as i64, &self.device)
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, row_count)
            .repeat_dim(1, dense_tokens)
            .unsqueeze_dim::<4>(3);
        Tensor::cat(vec![row_indices, token_indices, feature_indices], 3)
    }

    fn output(&self, updated_tokens: usize) -> InterframeJepaFeatureMemoryOutput<B> {
        InterframeJepaFeatureMemoryOutput {
            features: self.features.clone(),
            observed: self.observed.clone(),
            age_frames: self.age_frames.clone(),
            grid: self.grid,
            step: self.step,
            updated_tokens,
            dense_tokens: self.dense_tokens(),
        }
    }
}
