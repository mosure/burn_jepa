use crate::{
    SparseImageTokenGrid, SparsePredictorPlan, SparseTokenMask, TokenGridShape, VJepaConfig,
    VJepaPredictor, VJepaPredictorOutput, sparse_mask_from_frame_token_indices,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalSparseJepaConfig {
    pub keyframe_interval: usize,
    pub feature_blend: f32,
}

impl Default for TemporalSparseJepaConfig {
    fn default() -> Self {
        Self {
            keyframe_interval: 16,
            feature_blend: 1.0,
        }
    }
}

impl TemporalSparseJepaConfig {
    pub fn with_keyframe_interval(mut self, keyframe_interval: usize) -> Self {
        self.keyframe_interval = keyframe_interval.max(1);
        self
    }

    pub fn with_feature_blend(mut self, feature_blend: f32) -> Self {
        self.feature_blend = feature_blend.clamp(0.0, 1.0);
        self
    }

    pub fn normalized(self) -> Self {
        Self {
            keyframe_interval: self.keyframe_interval.max(1),
            feature_blend: self.feature_blend.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug)]
pub struct TemporalSparseJepaOutput<B: Backend> {
    pub features: Tensor<B, 3>,
    pub predictor: VJepaPredictorOutput<B>,
    pub keyframe: bool,
    pub reused_predictor_plan: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalSparseMaskConfig {
    pub keyframe_interval: usize,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub dilation: usize,
}

impl TemporalSparseMaskConfig {
    pub fn new(context_tokens: usize, target_tokens: usize) -> Self {
        Self {
            keyframe_interval: 16,
            context_tokens,
            target_tokens,
            dilation: 0,
        }
    }

    pub fn with_keyframe_interval(mut self, keyframe_interval: usize) -> Self {
        self.keyframe_interval = keyframe_interval.max(1);
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    fn normalized(self) -> Self {
        Self {
            keyframe_interval: self.keyframe_interval.max(1),
            context_tokens: self.context_tokens.max(1),
            target_tokens: self.target_tokens.max(1),
            dilation: self.dilation,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TemporalSparseMaskOutput {
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
    pub keyframe: bool,
}

#[derive(Debug)]
pub struct TemporalSparseMaskState {
    config: TemporalSparseMaskConfig,
    step: usize,
}

impl TemporalSparseMaskState {
    pub fn new(config: TemporalSparseMaskConfig) -> Self {
        Self {
            config: config.normalized(),
            step: 0,
        }
    }

    pub fn config(&self) -> TemporalSparseMaskConfig {
        self.config
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn next_is_keyframe(&self) -> bool {
        self.step % self.config.keyframe_interval.max(1) == 0
    }

    pub fn reset(&mut self) {
        self.step = 0;
    }

    pub fn next_from_frame_tokens(
        &mut self,
        grid: TokenGridShape,
        tubelet_size: usize,
        image_grid: SparseImageTokenGrid,
        frame_tokens: &[Vec<usize>],
    ) -> Result<TemporalSparseMaskOutput> {
        let keyframe = self.next_is_keyframe();
        let context_mask = sparse_mask_from_frame_token_indices(
            grid,
            tubelet_size,
            image_grid,
            frame_tokens,
            self.config.dilation,
            self.config.context_tokens,
        )?;
        ensure!(
            context_mask.len() < grid.len(),
            "temporal context mask must leave at least one target token"
        );
        let target_mask = target_mask_for_context(&context_mask, self.config.target_tokens)?;
        self.step = self.step.saturating_add(1);
        Ok(TemporalSparseMaskOutput {
            context_mask,
            target_mask,
            keyframe,
        })
    }
}

#[derive(Debug)]
pub struct TemporalSparseJepaState<B: Backend> {
    config: TemporalSparseJepaConfig,
    step: usize,
    cached_features: Option<CachedSparseFeatures<B>>,
    cached_plan: Option<CachedPredictorPlan<B>>,
}

#[derive(Debug)]
struct CachedSparseFeatures<B: Backend> {
    tokens: Tensor<B, 3>,
    mask: SparseTokenMask,
    grid: TokenGridShape,
    batch: usize,
}

#[derive(Debug)]
struct CachedPredictorPlan<B: Backend> {
    plan: SparsePredictorPlan<B>,
    context_mask: SparseTokenMask,
    target_mask: SparseTokenMask,
    grid: TokenGridShape,
    batch: usize,
}

impl<B: Backend> TemporalSparseJepaState<B> {
    pub fn new(config: TemporalSparseJepaConfig) -> Self {
        Self {
            config: config.normalized(),
            step: 0,
            cached_features: None,
            cached_plan: None,
        }
    }

    pub fn config(&self) -> TemporalSparseJepaConfig {
        self.config
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn next_is_keyframe(&self) -> bool {
        self.is_keyframe()
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.cached_features = None;
        self.cached_plan = None;
    }

    pub fn forward_predictor(
        &mut self,
        config: &VJepaConfig,
        predictor: &VJepaPredictor<B>,
        context_tokens: Tensor<B, 3>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
        grid: TokenGridShape,
        mask_index: usize,
    ) -> Result<TemporalSparseJepaOutput<B>> {
        ensure!(
            context_mask.dense_len() == grid.len(),
            "context mask dense token count must match temporal grid"
        );
        ensure!(
            target_mask.dense_len() == grid.len(),
            "target mask dense token count must match temporal grid"
        );
        let [batch, context_len, _dim] = context_tokens.shape().dims::<3>();
        ensure!(
            context_len == context_mask.len(),
            "context token length must match context mask"
        );

        let keyframe = self.is_keyframe();
        let features = self.update_features(context_tokens, context_mask, grid, batch, keyframe);
        let reused_predictor_plan =
            self.ensure_predictor_plan(config, context_mask, target_mask, grid, batch)?;
        let plan = &self
            .cached_plan
            .as_ref()
            .expect("predictor plan should be cached")
            .plan;
        let predictor = predictor.forward_sparse_with_plan(features.clone(), plan, mask_index)?;
        self.step = self.step.saturating_add(1);

        Ok(TemporalSparseJepaOutput {
            features,
            predictor,
            keyframe,
            reused_predictor_plan,
        })
    }

    fn is_keyframe(&self) -> bool {
        self.step % self.config.keyframe_interval.max(1) == 0
    }

    fn update_features(
        &mut self,
        context_tokens: Tensor<B, 3>,
        context_mask: &SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
        keyframe: bool,
    ) -> Tensor<B, 3> {
        let alpha = self.config.feature_blend.clamp(0.0, 1.0);
        let can_blend = !keyframe
            && alpha < 1.0
            && self
                .cached_features
                .as_ref()
                .is_some_and(|cached| cached.matches(context_mask, grid, batch));
        let tokens = if can_blend {
            let cached = self
                .cached_features
                .as_ref()
                .expect("cached features checked above");
            cached.tokens.clone().mul_scalar(1.0 - alpha) + context_tokens.mul_scalar(alpha)
        } else {
            context_tokens
        };
        self.cached_features = Some(CachedSparseFeatures {
            tokens: tokens.clone(),
            mask: context_mask.clone(),
            grid,
            batch,
        });
        tokens
    }

    fn ensure_predictor_plan(
        &mut self,
        config: &VJepaConfig,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
    ) -> Result<bool> {
        if self.cached_plan.as_ref().is_some_and(|cached| {
            cached.context_mask == *context_mask
                && cached.target_mask == *target_mask
                && cached.grid == grid
                && cached.batch == batch
        }) {
            return Ok(true);
        }

        let device = self
            .cached_features
            .as_ref()
            .expect("features are updated before predictor plan")
            .tokens
            .device();
        let plan = SparsePredictorPlan::new(
            config,
            context_mask.clone(),
            target_mask.clone(),
            grid,
            batch,
            &device,
        )?;
        self.cached_plan = Some(CachedPredictorPlan {
            plan,
            context_mask: context_mask.clone(),
            target_mask: target_mask.clone(),
            grid,
            batch,
        });
        Ok(false)
    }
}

impl<B: Backend> CachedSparseFeatures<B> {
    fn matches(&self, mask: &SparseTokenMask, grid: TokenGridShape, batch: usize) -> bool {
        self.mask == *mask && self.grid == grid && self.batch == batch
    }
}

fn target_mask_for_context(
    context: &SparseTokenMask,
    target_tokens: usize,
) -> Result<SparseTokenMask> {
    let dense_len = context.dense_len();
    let target_tokens = target_tokens
        .max(1)
        .min(dense_len.saturating_sub(context.len()).max(1));
    let context_set = context.indices().iter().copied().collect::<BTreeSet<_>>();
    let mut target = SparseTokenMask::evenly_spaced(dense_len, target_tokens)
        .indices()
        .iter()
        .copied()
        .filter(|index| !context_set.contains(index))
        .collect::<Vec<_>>();
    if target.len() < target_tokens {
        for index in 0..dense_len {
            if !context_set.contains(&index) && !target.contains(&index) {
                target.push(index);
                if target.len() >= target_tokens {
                    break;
                }
            }
        }
    }
    SparseTokenMask::new(target, dense_len)
}
