#[cfg(feature = "sparse-patchify-wgpu")]
use crate::SparsePatchifyPlan;
use crate::{
    SparseImageTokenGrid, SparsePredictorPlan, SparseTokenMask, TokenGridShape, VJepa2_1Model,
    VJepaConfig, VJepaEncoderOutput, VJepaPredictor, VJepaPredictorOutput,
    sparse_mask_from_frame_token_indices,
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

pub struct TemporalSparsePredictorInput<'a, B: Backend> {
    pub config: &'a VJepaConfig,
    pub predictor: &'a VJepaPredictor<B>,
    pub context_tokens: Tensor<B, 3>,
    pub context_mask: &'a SparseTokenMask,
    pub target_mask: &'a SparseTokenMask,
    pub grid: TokenGridShape,
    pub mask_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalSparseJepaStreamConfig {
    pub keyframe_interval: usize,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub dilation: usize,
    pub feature_blend: f32,
    pub dense_keyframe_refresh: bool,
    pub image_grid: SparseImageTokenGrid,
}

impl TemporalSparseJepaStreamConfig {
    pub fn new(
        context_tokens: usize,
        target_tokens: usize,
        image_grid: SparseImageTokenGrid,
    ) -> Self {
        Self {
            keyframe_interval: 16,
            context_tokens,
            target_tokens,
            dilation: 0,
            feature_blend: 1.0,
            dense_keyframe_refresh: false,
            image_grid,
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

    pub fn with_feature_blend(mut self, feature_blend: f32) -> Self {
        self.feature_blend = feature_blend.clamp(0.0, 1.0);
        self
    }

    pub fn with_dense_keyframe_refresh(mut self, dense_keyframe_refresh: bool) -> Self {
        self.dense_keyframe_refresh = dense_keyframe_refresh;
        self
    }

    fn normalized(self) -> Self {
        Self {
            keyframe_interval: self.keyframe_interval.max(1),
            context_tokens: self.context_tokens.max(1),
            target_tokens: self.target_tokens.max(1),
            dilation: self.dilation,
            feature_blend: self.feature_blend.clamp(0.0, 1.0),
            dense_keyframe_refresh: self.dense_keyframe_refresh,
            image_grid: self.image_grid,
        }
    }

    fn mask_config(self) -> TemporalSparseMaskConfig {
        TemporalSparseMaskConfig::new(self.context_tokens, self.target_tokens)
            .with_keyframe_interval(self.keyframe_interval)
            .with_dilation(self.dilation)
    }

    fn jepa_config(self) -> TemporalSparseJepaConfig {
        TemporalSparseJepaConfig::default()
            .with_keyframe_interval(self.keyframe_interval)
            .with_feature_blend(self.feature_blend)
    }
}

#[derive(Debug)]
pub struct TemporalSparseJepaStreamOutput<B: Backend> {
    pub masks: TemporalSparseMaskOutput,
    pub context: VJepaEncoderOutput<B>,
    pub temporal: TemporalSparseJepaOutput<B>,
    pub dense_keyframe: Option<VJepaEncoderOutput<B>>,
    pub reused_patchify_plan: bool,
}

#[derive(Debug)]
pub struct TemporalSparseJepaStream<B: Backend> {
    config: TemporalSparseJepaStreamConfig,
    mask_state: TemporalSparseMaskState,
    jepa_state: TemporalSparseJepaState<B>,
    #[cfg(feature = "sparse-patchify-wgpu")]
    cached_patchify_plan: Option<CachedSparsePatchifyPlan<B>>,
}

impl<B: Backend> TemporalSparseJepaStream<B> {
    pub fn new(config: TemporalSparseJepaStreamConfig) -> Self {
        let config = config.normalized();
        Self {
            mask_state: TemporalSparseMaskState::new(config.mask_config()),
            jepa_state: TemporalSparseJepaState::new(config.jepa_config()),
            #[cfg(feature = "sparse-patchify-wgpu")]
            cached_patchify_plan: None,
            config,
        }
    }

    pub fn config(&self) -> TemporalSparseJepaStreamConfig {
        self.config
    }

    pub fn step(&self) -> usize {
        self.jepa_state.step()
    }

    pub fn next_is_keyframe(&self) -> bool {
        self.jepa_state.next_is_keyframe()
    }

    pub fn reset(&mut self) {
        self.mask_state.reset();
        self.jepa_state.reset();
        self.reset_patchify_plan();
    }

    pub fn forward_frame_tokens(
        &mut self,
        model: &VJepa2_1Model<B>,
        video: Tensor<B, 5>,
        frame_tokens: &[Vec<usize>],
        mask_index: usize,
    ) -> Result<TemporalSparseJepaStreamOutput<B>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            channels == model.config().in_channels,
            "temporal stream video channel count must match V-JEPA config"
        );
        let tubelet_size = model.config().tubelet_size.max(1);
        let patch_size = model.config().patch_size.max(1);
        ensure!(
            frames >= tubelet_size && height >= patch_size && width >= patch_size,
            "temporal stream video must contain at least one V-JEPA tubelet patch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet_size,
            height / patch_size,
            width / patch_size,
        );
        ensure!(
            !grid.is_empty(),
            "temporal stream video produced an empty token grid"
        );
        let masks = self.mask_state.next_from_frame_tokens(
            grid,
            tubelet_size,
            self.config.image_grid,
            frame_tokens,
        )?;
        let dense_keyframe = if self.config.dense_keyframe_refresh && masks.keyframe {
            let dense = model.encode_video(video.clone(), None);
            ensure!(
                dense.grid == grid,
                "temporal stream dense keyframe grid changed during encode"
            );
            Some(dense)
        } else {
            None
        };
        let context = model.encode_video(video, Some(&masks.context_mask));
        ensure!(
            context.grid == grid,
            "temporal stream encoder grid changed during sparse encode"
        );
        ensure!(
            context.tokens.shape().dims::<3>()[0] == batch,
            "temporal stream encoder batch changed during sparse encode"
        );
        let temporal = self
            .jepa_state
            .forward_predictor(TemporalSparsePredictorInput {
                config: model.config(),
                predictor: &model.predictor,
                context_tokens: context.tokens.clone(),
                context_mask: &masks.context_mask,
                target_mask: &masks.target_mask,
                grid: context.grid,
                mask_index,
            })?;
        ensure!(
            masks.keyframe == temporal.keyframe,
            "temporal stream mask and predictor keyframe state diverged"
        );

        Ok(TemporalSparseJepaStreamOutput {
            masks,
            context,
            temporal,
            dense_keyframe,
            reused_patchify_plan: false,
        })
    }

    pub fn forward_masks(
        &mut self,
        model: &VJepa2_1Model<B>,
        video: Tensor<B, 5>,
        context_mask: SparseTokenMask,
        target_mask: SparseTokenMask,
        mask_index: usize,
    ) -> Result<TemporalSparseJepaStreamOutput<B>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            channels == model.config().in_channels,
            "temporal stream video channel count must match V-JEPA config"
        );
        let tubelet_size = model.config().tubelet_size.max(1);
        let patch_size = model.config().patch_size.max(1);
        ensure!(
            frames >= tubelet_size && height >= patch_size && width >= patch_size,
            "temporal stream video must contain at least one V-JEPA tubelet patch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet_size,
            height / patch_size,
            width / patch_size,
        );
        ensure!(
            !grid.is_empty(),
            "temporal stream video produced an empty token grid"
        );
        let masks = self
            .mask_state
            .next_from_masks(grid, context_mask, target_mask)?;
        let dense_keyframe = if self.config.dense_keyframe_refresh && masks.keyframe {
            let dense = model.encode_video(video.clone(), None);
            ensure!(
                dense.grid == grid,
                "temporal stream dense keyframe grid changed during encode"
            );
            Some(dense)
        } else {
            None
        };
        let context = model.encode_video(video, Some(&masks.context_mask));
        ensure!(
            context.grid == grid,
            "temporal stream encoder grid changed during sparse encode"
        );
        ensure!(
            context.tokens.shape().dims::<3>()[0] == batch,
            "temporal stream encoder batch changed during sparse encode"
        );
        let temporal = self
            .jepa_state
            .forward_predictor(TemporalSparsePredictorInput {
                config: model.config(),
                predictor: &model.predictor,
                context_tokens: context.tokens.clone(),
                context_mask: &masks.context_mask,
                target_mask: &masks.target_mask,
                grid: context.grid,
                mask_index,
            })?;
        ensure!(
            masks.keyframe == temporal.keyframe,
            "temporal stream mask and predictor keyframe state diverged"
        );

        Ok(TemporalSparseJepaStreamOutput {
            masks,
            context,
            temporal,
            dense_keyframe,
            reused_patchify_plan: false,
        })
    }

    #[cfg(feature = "sparse-patchify-wgpu")]
    fn ensure_patchify_plan(
        &mut self,
        context_mask: &SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
        device: &B::Device,
    ) -> Result<bool> {
        if self
            .cached_patchify_plan
            .as_ref()
            .is_some_and(|cached| cached.matches(context_mask, grid, batch))
        {
            return Ok(true);
        }
        self.cached_patchify_plan = Some(CachedSparsePatchifyPlan {
            plan: SparsePatchifyPlan::new(context_mask.clone(), grid, batch, device)?,
            context_mask: context_mask.clone(),
            grid,
            batch,
        });
        Ok(false)
    }

    #[cfg(feature = "sparse-patchify-wgpu")]
    fn reset_patchify_plan(&mut self) {
        self.cached_patchify_plan = None;
    }

    #[cfg(not(feature = "sparse-patchify-wgpu"))]
    fn reset_patchify_plan(&mut self) {}
}

#[cfg(feature = "sparse-patchify-wgpu")]
#[derive(Debug)]
struct CachedSparsePatchifyPlan<B: Backend> {
    plan: SparsePatchifyPlan<B>,
    context_mask: SparseTokenMask,
    grid: TokenGridShape,
    batch: usize,
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl<B: Backend> CachedSparsePatchifyPlan<B> {
    fn matches(&self, mask: &SparseTokenMask, grid: TokenGridShape, batch: usize) -> bool {
        self.context_mask == *mask && self.grid == grid && self.batch == batch
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl TemporalSparseJepaStream<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn forward_frame_tokens_sparse_patchify_wgpu(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        frame_tokens: &[Vec<usize>],
        mask_index: usize,
    ) -> Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            channels == model.config().in_channels,
            "temporal stream video channel count must match V-JEPA config"
        );
        let device = video.device();
        let tubelet_size = model.config().tubelet_size.max(1);
        let patch_size = model.config().patch_size.max(1);
        ensure!(
            frames >= tubelet_size && height >= patch_size && width >= patch_size,
            "temporal stream video must contain at least one V-JEPA tubelet patch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet_size,
            height / patch_size,
            width / patch_size,
        );
        ensure!(
            !grid.is_empty(),
            "temporal stream video produced an empty token grid"
        );
        let masks = self.mask_state.next_from_frame_tokens(
            grid,
            tubelet_size,
            self.config.image_grid,
            frame_tokens,
        )?;
        let dense_keyframe = if self.config.dense_keyframe_refresh && masks.keyframe {
            let dense = model.encode_video(video.clone(), None);
            ensure!(
                dense.grid == grid,
                "temporal stream dense keyframe grid changed during encode"
            );
            Some(dense)
        } else {
            None
        };
        let reused_patchify_plan =
            self.ensure_patchify_plan(&masks.context_mask, grid, batch, &device)?;
        let plan = &self
            .cached_patchify_plan
            .as_ref()
            .expect("sparse patchify plan should be cached")
            .plan;
        let context = model.encode_video_sparse_patchify_wgpu(video, plan)?;
        ensure!(
            context.grid == grid,
            "temporal stream sparse patchify grid changed during encode"
        );
        ensure!(
            context.tokens.shape().dims::<3>()[0] == batch,
            "temporal stream encoder batch changed during sparse patchify encode"
        );
        let temporal = self
            .jepa_state
            .forward_predictor(TemporalSparsePredictorInput {
                config: model.config(),
                predictor: &model.predictor,
                context_tokens: context.tokens.clone(),
                context_mask: &masks.context_mask,
                target_mask: &masks.target_mask,
                grid: context.grid,
                mask_index,
            })?;
        ensure!(
            masks.keyframe == temporal.keyframe,
            "temporal stream mask and predictor keyframe state diverged"
        );

        Ok(TemporalSparseJepaStreamOutput {
            masks,
            context,
            temporal,
            dense_keyframe,
            reused_patchify_plan,
        })
    }

    pub fn forward_masks_sparse_patchify_wgpu(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        context_mask: SparseTokenMask,
        target_mask: SparseTokenMask,
        mask_index: usize,
    ) -> Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            channels == model.config().in_channels,
            "temporal stream video channel count must match V-JEPA config"
        );
        let device = video.device();
        let tubelet_size = model.config().tubelet_size.max(1);
        let patch_size = model.config().patch_size.max(1);
        ensure!(
            frames >= tubelet_size && height >= patch_size && width >= patch_size,
            "temporal stream video must contain at least one V-JEPA tubelet patch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet_size,
            height / patch_size,
            width / patch_size,
        );
        ensure!(
            !grid.is_empty(),
            "temporal stream video produced an empty token grid"
        );
        let masks = self
            .mask_state
            .next_from_masks(grid, context_mask, target_mask)?;
        let dense_keyframe = if self.config.dense_keyframe_refresh && masks.keyframe {
            let dense = model.encode_video(video.clone(), None);
            ensure!(
                dense.grid == grid,
                "temporal stream dense keyframe grid changed during encode"
            );
            Some(dense)
        } else {
            None
        };
        let reused_patchify_plan =
            self.ensure_patchify_plan(&masks.context_mask, grid, batch, &device)?;
        let plan = &self
            .cached_patchify_plan
            .as_ref()
            .expect("sparse patchify plan should be cached")
            .plan;
        let context = model.encode_video_sparse_patchify_wgpu(video, plan)?;
        ensure!(
            context.grid == grid,
            "temporal stream sparse patchify grid changed during encode"
        );
        ensure!(
            context.tokens.shape().dims::<3>()[0] == batch,
            "temporal stream encoder batch changed during sparse patchify encode"
        );
        let temporal = self
            .jepa_state
            .forward_predictor(TemporalSparsePredictorInput {
                config: model.config(),
                predictor: &model.predictor,
                context_tokens: context.tokens.clone(),
                context_mask: &masks.context_mask,
                target_mask: &masks.target_mask,
                grid: context.grid,
                mask_index,
            })?;
        ensure!(
            masks.keyframe == temporal.keyframe,
            "temporal stream mask and predictor keyframe state diverged"
        );

        Ok(TemporalSparseJepaStreamOutput {
            masks,
            context,
            temporal,
            dense_keyframe,
            reused_patchify_plan,
        })
    }
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
        self.step
            .is_multiple_of(self.config.keyframe_interval.max(1))
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

    pub fn next_from_masks(
        &mut self,
        grid: TokenGridShape,
        context_mask: SparseTokenMask,
        target_mask: SparseTokenMask,
    ) -> Result<TemporalSparseMaskOutput> {
        validate_temporal_masks(&context_mask, &target_mask, grid)?;
        let keyframe = self.next_is_keyframe();
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
        input: TemporalSparsePredictorInput<'_, B>,
    ) -> Result<TemporalSparseJepaOutput<B>> {
        ensure!(
            input.context_mask.dense_len() == input.grid.len(),
            "context mask dense token count must match temporal grid"
        );
        ensure!(
            input.target_mask.dense_len() == input.grid.len(),
            "target mask dense token count must match temporal grid"
        );
        let [batch, context_len, _dim] = input.context_tokens.shape().dims::<3>();
        ensure!(
            context_len == input.context_mask.len(),
            "context token length must match context mask"
        );

        let keyframe = self.is_keyframe();
        let features = self.update_features(
            input.context_tokens,
            input.context_mask,
            input.grid,
            batch,
            keyframe,
        );
        let reused_predictor_plan = self.ensure_predictor_plan(
            input.config,
            input.context_mask,
            input.target_mask,
            input.grid,
            batch,
        )?;
        let plan = &self
            .cached_plan
            .as_ref()
            .expect("predictor plan should be cached")
            .plan;
        let predictor =
            input
                .predictor
                .forward_sparse_with_plan(features.clone(), plan, input.mask_index)?;
        self.step = self.step.saturating_add(1);

        Ok(TemporalSparseJepaOutput {
            features,
            predictor,
            keyframe,
            reused_predictor_plan,
        })
    }

    fn is_keyframe(&self) -> bool {
        self.step
            .is_multiple_of(self.config.keyframe_interval.max(1))
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

fn validate_temporal_masks(
    context: &SparseTokenMask,
    target: &SparseTokenMask,
    grid: TokenGridShape,
) -> Result<()> {
    ensure!(
        context.dense_len() == grid.len(),
        "context mask dense token count must match temporal grid"
    );
    ensure!(
        target.dense_len() == grid.len(),
        "target mask dense token count must match temporal grid"
    );
    ensure!(
        context.len() < grid.len(),
        "temporal context mask must leave at least one target token"
    );
    ensure!(
        !context.is_empty(),
        "temporal context mask must contain at least one token"
    );
    ensure!(
        !target.is_empty(),
        "temporal target mask must contain at least one token"
    );
    let context_set = context.indices().iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        target
            .indices()
            .iter()
            .all(|index| !context_set.contains(index)),
        "temporal context and target masks must not overlap"
    );
    Ok(())
}
