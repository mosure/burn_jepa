#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use crate::SparsePatchifyBatchPlan;
use crate::{
    FeaturePcaConfig, FeaturePcaProjector, FeaturePcaUpdateConfig, FeaturePcaUpdateScheduler,
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    InterframeJepaFeatureMemoryOutput, SparseMaskBatch, SparseTokenMask, TokenGridShape,
    VJepa2_1Model, VJepaConfig, VJepaEncoderOutput, jepa_feature_tokens_to_nchw,
};
use anyhow::{Result, bail, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_anyup::{AnyUp, AnyUpImageGrid};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaMeasurementConfig {
    pub enabled: bool,
    pub sync_backend: bool,
}

impl SparseJepaAnyUpPcaMeasurementConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            sync_backend: false,
        }
    }

    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            sync_backend: false,
        }
    }

    pub const fn enabled_with_backend_sync() -> Self {
        Self {
            enabled: true,
            sync_backend: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseJepaAnyUpPcaEncodePath {
    #[default]
    DensePatchEmbed,
    SparsePatchify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFrameNode {
    LowResPca,
    HighResAnyUpPca,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFrameRequest {
    pub low_res_pca: bool,
    pub high_res_pca: bool,
    pub high_res_features: bool,
}

impl FeatureFrameRequest {
    pub const fn none() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: false,
            high_res_features: false,
        }
    }

    pub const fn low_res() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: false,
            high_res_features: false,
        }
    }

    pub const fn high_res_pca() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: true,
            high_res_features: false,
        }
    }

    pub const fn full_pca() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: true,
            high_res_features: false,
        }
    }

    pub const fn high_res() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: true,
            high_res_features: true,
        }
    }

    pub const fn full() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: true,
            high_res_features: true,
        }
    }

    pub const fn includes(self, node: FeatureFrameNode) -> bool {
        match node {
            FeatureFrameNode::LowResPca => self.low_res_pca,
            FeatureFrameNode::HighResAnyUpPca => self.high_res_pca,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFrameSchedule {
    pub low_res_pca_every: Option<u64>,
    pub high_res_pca_every: Option<u64>,
}

impl Default for FeatureFrameSchedule {
    fn default() -> Self {
        Self {
            low_res_pca_every: None,
            high_res_pca_every: Some(1),
        }
    }
}

impl FeatureFrameSchedule {
    pub const fn every_frame_full() -> Self {
        Self {
            low_res_pca_every: Some(1),
            high_res_pca_every: Some(1),
        }
    }

    pub const fn low_res_every_frame() -> Self {
        Self {
            low_res_pca_every: Some(1),
            high_res_pca_every: None,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            low_res_pca_every: None,
            high_res_pca_every: None,
        }
    }

    pub fn request_for(&self, ids: &[SparseJepaAnyUpPcaFrameId]) -> FeatureFrameRequest {
        FeatureFrameRequest {
            low_res_pca: scheduled_for_any(ids, self.low_res_pca_every),
            high_res_pca: scheduled_for_any(ids, self.high_res_pca_every),
            high_res_features: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaStageMetrics {
    pub measured: bool,
    pub sync_backend: bool,
    pub encode_path: SparseJepaAnyUpPcaEncodePath,
    pub frame_count: usize,
    pub dense_tokens_per_frame: usize,
    pub sparse_width: usize,
    pub valid_sparse_tokens: usize,
    pub output_pixels: usize,
    pub encode_us: u64,
    pub cache_update_us: u64,
    pub token_view_us: u64,
    pub anyup_context_us: u64,
    pub anyup_decode_us: u64,
    pub low_res_pca_project_us: u64,
    pub pca_update_us: u64,
    pub pca_online_us: u64,
    pub pca_project_us: u64,
    pub pca_update_applied: bool,
    pub pca_update_tokens: usize,
    pub total_us: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaPipelineConfig {
    pub memory: InterframeJepaFeatureMemoryConfig,
    pub pca: FeaturePcaConfig,
    pub pca_update: FeaturePcaUpdateConfig,
    pub anyup_q_chunk_size: Option<usize>,
    pub update_pca_online: bool,
    pub measurement: SparseJepaAnyUpPcaMeasurementConfig,
}

impl Default for SparseJepaAnyUpPcaPipelineConfig {
    fn default() -> Self {
        Self {
            memory: InterframeJepaFeatureMemoryConfig::default(),
            pca: FeaturePcaConfig::default(),
            pca_update: FeaturePcaUpdateConfig::disabled(),
            anyup_q_chunk_size: Some(16),
            update_pca_online: false,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
        }
    }
}

impl SparseJepaAnyUpPcaPipelineConfig {
    pub fn validate(&self) -> Result<()> {
        self.pca.validate()?;
        self.pca_update.validate()?;
        self.effective_pca_update().validate()?;
        ensure!(
            self.anyup_q_chunk_size.is_none_or(|chunk| chunk > 0),
            "AnyUp q chunk size must be positive when set"
        );
        Ok(())
    }

    pub fn effective_pca_update(&self) -> FeaturePcaUpdateConfig {
        if self.pca_update.enabled() || !self.update_pca_online {
            self.pca_update.clone()
        } else {
            FeaturePcaUpdateConfig::rolling_low_res_every(1)
        }
    }
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaOutput<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res_features: Tensor<B, 4>,
    pub high_res_features: Tensor<B, 4>,
    pub pca_display: Tensor<B, 4>,
    pub mask: SparseTokenMask,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaStepBatchOutput<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res_features: Tensor<B, 4>,
    pub high_res_features: Tensor<B, 4>,
    pub pca_display: Tensor<B, 4>,
    pub mask: SparseMaskBatch<B>,
}

#[derive(Debug)]
pub struct LowResFrameArtifacts<B: Backend> {
    pub features: Tensor<B, 4>,
    pub pca_display: Option<Tensor<B, 4>>,
}

#[derive(Debug)]
pub struct HighResFrameArtifacts<B: Backend> {
    pub features: Option<Tensor<B, 4>>,
    pub pca_display: Option<Tensor<B, 4>>,
}

#[derive(Debug)]
pub struct FeatureFrameBatch<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res: LowResFrameArtifacts<B>,
    pub high_res: Option<HighResFrameArtifacts<B>>,
    pub mask: SparseMaskBatch<B>,
}

impl<B: Backend> FeatureFrameBatch<B> {
    pub fn has_low_res_pca(&self) -> bool {
        self.low_res.pca_display.is_some()
    }

    pub fn has_high_res_pca(&self) -> bool {
        self.high_res
            .as_ref()
            .is_some_and(|high_res| high_res.pca_display.is_some())
    }

    pub fn has_high_res_features(&self) -> bool {
        self.high_res
            .as_ref()
            .is_some_and(|high_res| high_res.features.is_some())
    }

    fn into_full_output(self) -> Result<SparseJepaAnyUpPcaStepBatchOutput<B>> {
        let Some(high_res) = self.high_res else {
            bail!("full high-res output was not requested for this frame batch");
        };
        Ok(SparseJepaAnyUpPcaStepBatchOutput {
            encoded: self.encoded,
            token_cache: self.token_cache,
            low_res_features: self.low_res.features,
            high_res_features: high_res
                .features
                .ok_or_else(|| anyhow::anyhow!("high-res features were not requested"))?,
            pca_display: high_res
                .pca_display
                .ok_or_else(|| anyhow::anyhow!("high-res PCA display was not requested"))?,
            mask: self.mask,
        })
    }
}

#[derive(Debug)]
pub struct MeasuredFeatureFrameBatch<B: Backend> {
    pub output: FeatureFrameBatch<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaMeasuredOutput<B: Backend> {
    pub output: SparseJepaAnyUpPcaOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaMeasuredBatchOutput<B: Backend> {
    pub output: SparseJepaAnyUpPcaStepBatchOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

pub struct SparseJepaAnyUpPcaPipeline<B: Backend> {
    jepa: VJepa2_1Model<B>,
    anyup: AnyUp<B>,
    anyup_image_grid: AnyUpImageGrid<B>,
    token_memory: InterframeJepaFeatureMemory<B>,
    pca: FeaturePcaProjector<B>,
    pca_update_scheduler: FeaturePcaUpdateScheduler,
    config: SparseJepaAnyUpPcaPipelineConfig,
    batch: usize,
    image_size: [usize; 2],
    grid: TokenGridShape,
    device: B::Device,
}

impl<B: Backend> SparseJepaAnyUpPcaPipeline<B> {
    pub fn new(
        jepa: VJepa2_1Model<B>,
        anyup: AnyUp<B>,
        jepa_config: &VJepaConfig,
        config: SparseJepaAnyUpPcaPipelineConfig,
        batch: usize,
        image_size: [usize; 2],
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(batch > 0, "pipeline batch must be nonzero");
        ensure!(
            image_size[0] > 0 && image_size[1] > 0,
            "pipeline image size must be nonzero"
        );
        ensure!(
            image_size[0].is_multiple_of(jepa_config.patch_size.max(1))
                && image_size[1].is_multiple_of(jepa_config.patch_size.max(1)),
            "image size must be divisible by V-JEPA patch size"
        );
        let grid = TokenGridShape::new(
            1,
            image_size[0] / jepa_config.patch_size.max(1),
            image_size[1] / jepa_config.patch_size.max(1),
        );
        let token_memory = InterframeJepaFeatureMemory::new(
            config.memory,
            batch,
            grid,
            jepa_config.encoder.embed_dim,
            device,
        )?;
        let pca = FeaturePcaProjector::identity(
            jepa_config.encoder.embed_dim,
            config.pca.clone(),
            device,
        )?;
        let pca_update_scheduler = FeaturePcaUpdateScheduler::new(config.effective_pca_update())?;
        let anyup_image_grid = anyup.prepare_image_grid(image_size, device);
        Ok(Self {
            jepa,
            anyup,
            anyup_image_grid,
            token_memory,
            pca,
            pca_update_scheduler,
            config,
            batch,
            image_size,
            grid,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &SparseJepaAnyUpPcaPipelineConfig {
        &self.config
    }

    pub fn jepa(&self) -> &VJepa2_1Model<B> {
        &self.jepa
    }

    pub fn anyup(&self) -> &AnyUp<B> {
        &self.anyup
    }

    pub fn token_memory(&self) -> &InterframeJepaFeatureMemory<B> {
        &self.token_memory
    }

    pub fn token_memory_mut(&mut self) -> &mut InterframeJepaFeatureMemory<B> {
        &mut self.token_memory
    }

    pub fn pca(&self) -> &FeaturePcaProjector<B> {
        &self.pca
    }

    pub fn pca_mut(&mut self) -> &mut FeaturePcaProjector<B> {
        &mut self.pca
    }

    pub fn pca_update_scheduler(&self) -> &FeaturePcaUpdateScheduler {
        &self.pca_update_scheduler
    }

    pub fn grid(&self) -> TokenGridShape {
        self.grid
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn image_size(&self) -> [usize; 2] {
        self.image_size
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn reset(&mut self) {
        self.token_memory.reset();
        self.pca_update_scheduler.reset();
    }

    pub fn step_image_keep_ratio(
        &mut self,
        image: Tensor<B, 4>,
        keep_ratio: f32,
    ) -> Result<SparseJepaAnyUpPcaOutput<B>> {
        let mask = SparseTokenMask::from_keep_ratio(self.grid.len(), keep_ratio);
        self.step_image_with_mask(image, &mask)
    }

    pub fn step_image_with_mask(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<B>> {
        Ok(self.step_image_with_mask_measured(image, mask)?.output)
    }

    pub fn step_image_with_mask_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<B>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured =
            self.step_image_with_mask_batch_measured(image, batch_mask, self.config.measurement)?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
        request: FeatureFrameRequest,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        self.step_image_with_mask_batch_nodes_measured(
            image,
            batch_mask,
            request,
            self.config.measurement,
        )
    }

    pub fn step_image_with_mask_batch(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<B>> {
        Ok(self
            .step_image_with_mask_batch_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    fn validate_batch_step_input(
        &self,
        image: &Tensor<B, 4>,
        mask: &SparseMaskBatch<B>,
    ) -> Result<[usize; 4]> {
        ensure!(
            mask.dense_len() == self.grid.len(),
            "sparse token mask dense length must match pipeline grid"
        );
        ensure!(
            !mask.is_empty(),
            "sparse token mask must include at least one token"
        );
        ensure!(
            !mask.is_ragged(),
            "high-res pipeline requires uniform or fixed-width mask batches; group variable-width masks before this stage"
        );
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(batch == self.batch, "image batch does not match pipeline");
        ensure!(channels == 3, "pipeline expects RGB image input");
        ensure!(
            [height, width] == self.image_size,
            "image spatial size does not match pipeline"
        );
        Ok([batch, channels, height, width])
    }

    fn initial_stage_metrics(
        &self,
        batch: usize,
        height: usize,
        width: usize,
        mask: &SparseMaskBatch<B>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
        encode_path: SparseJepaAnyUpPcaEncodePath,
    ) -> SparseJepaAnyUpPcaStageMetrics {
        SparseJepaAnyUpPcaStageMetrics {
            measured: measurement.enabled,
            sync_backend: measurement.sync_backend,
            encode_path,
            frame_count: batch,
            dense_tokens_per_frame: self.grid.len(),
            sparse_width: mask.len(),
            valid_sparse_tokens: mask.valid_token_count(),
            output_pixels: batch * height * width,
            ..SparseJepaAnyUpPcaStageMetrics::default()
        }
    }

    fn finish_encoded_batch_nodes(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        encoded: VJepaEncoderOutput<B>,
        request: FeatureFrameRequest,
        mut timer: StageTimer,
        mut metrics: SparseJepaAnyUpPcaStageMetrics,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let token_cache = self.token_memory.update_tokens(
            encoded.tokens.clone(),
            encoded.token_indices.clone(),
            encoded.grid,
        )?;
        metrics.cache_update_us = timer.mark::<B>(&self.device)?;
        let low_res_features =
            jepa_feature_tokens_to_nchw(token_cache.features.clone(), self.grid)?;
        metrics.token_view_us = timer.mark::<B>(&self.device)?;

        let pca_update = self
            .pca_update_scheduler
            .observe_batch(metrics.frame_count, self.grid.len());
        if pca_update.update {
            let iterations = self.pca_update_scheduler.config().iterations_per_update;
            self.pca.update_rolling_masked_tokens_iterations(
                token_cache.features.clone(),
                token_cache.observed.clone(),
                iterations,
            )?;
            metrics.pca_update_us = timer.mark::<B>(&self.device)?;
            metrics.pca_online_us = metrics.pca_update_us;
            metrics.pca_update_applied = true;
            metrics.pca_update_tokens = metrics.frame_count * self.grid.len();
        }

        let mut low_res_pca_components = None;
        let low_res_pca_display = if request.low_res_pca {
            let pca_components = self.pca.project_nchw(low_res_features.clone())?;
            let pca = self.pca.display_nchw(pca_components.clone())?;
            metrics.low_res_pca_project_us = timer.mark::<B>(&self.device)?;
            low_res_pca_components = Some(pca_components);
            Some(pca)
        } else {
            None
        };

        let high_res = if request.high_res_pca || request.high_res_features {
            let context = self.anyup.prepare_image_context_with_grid(
                image,
                &self.anyup_image_grid,
                Some(self.image_size),
                [self.grid.height, self.grid.width],
            );
            metrics.anyup_context_us = timer.mark::<B>(&self.device)?;

            let (features, pca_display) = if request.high_res_features {
                let features = self.anyup.upsample_with_context(
                    &context,
                    low_res_features.clone(),
                    self.config.anyup_q_chunk_size,
                );
                metrics.anyup_decode_us = timer.mark::<B>(&self.device)?;
                let pca_display = if request.high_res_pca {
                    let pca_display = self.pca.project_nchw_display(features.clone())?;
                    metrics.pca_project_us = timer.mark::<B>(&self.device)?;
                    Some(pca_display)
                } else {
                    None
                };
                (Some(features), pca_display)
            } else {
                let pca_values = if let Some(pca_components) = low_res_pca_components {
                    pca_components
                } else {
                    let pca_values = self.pca.project_nchw(low_res_features.clone())?;
                    metrics.pca_project_us = metrics
                        .pca_project_us
                        .saturating_add(timer.mark::<B>(&self.device)?);
                    pca_values
                };
                let pca_values = self.anyup.upsample_values_with_context(
                    &context,
                    low_res_features.clone(),
                    pca_values,
                    self.config.anyup_q_chunk_size,
                );
                metrics.anyup_decode_us = timer.mark::<B>(&self.device)?;
                let pca_display = self.pca.display_nchw(pca_values)?;
                metrics.pca_project_us = metrics
                    .pca_project_us
                    .saturating_add(timer.mark::<B>(&self.device)?);
                (None, Some(pca_display))
            };
            Some(HighResFrameArtifacts {
                features,
                pca_display,
            })
        } else {
            None
        };

        metrics.total_us = timer.total_us();
        Ok(MeasuredFeatureFrameBatch {
            output: FeatureFrameBatch {
                encoded,
                token_cache,
                low_res: LowResFrameArtifacts {
                    features: low_res_features,
                    pca_display: low_res_pca_display,
                },
                high_res,
                mask,
            },
            metrics,
        })
    }

    fn finish_encoded_batch_step(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        encoded: VJepaEncoderOutput<B>,
        timer: StageTimer,
        metrics: SparseJepaAnyUpPcaStageMetrics,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<B>> {
        let measured = self.finish_encoded_batch_nodes(
            image,
            mask,
            encoded,
            FeatureFrameRequest::high_res(),
            timer,
            metrics,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::DensePatchEmbed,
        );
        let mut timer = StageTimer::new(measurement);

        let encoded = self.jepa.encode_image_batch(image.clone(), mask.clone())?;
        metrics.encode_us = timer.mark::<B>(&self.device)?;
        self.finish_encoded_batch_nodes(image, mask, encoded, request, timer, metrics)
    }

    pub fn step_image_with_mask_batch_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<B>> {
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::DensePatchEmbed,
        );
        let mut timer = StageTimer::new(measurement);

        let encoded = self.jepa.encode_image_batch(image.clone(), mask.clone())?;
        metrics.encode_us = timer.mark::<B>(&self.device)?;
        self.finish_encoded_batch_step(image, mask, encoded, timer, metrics)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparseJepaAnyUpPcaPipeline<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn step_image_with_mask_sparse_patchify_wgpu(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        Ok(self
            .step_image_with_mask_sparse_patchify_wgpu_measured(image, mask)?
            .output)
    }

    pub fn step_image_with_mask_sparse_patchify_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured = self.step_image_with_mask_batch_sparse_patchify_wgpu_measured(
            image,
            batch_mask,
            self.config.measurement,
        )?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_sparse_patchify_wgpu(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        Ok(self
            .step_image_with_mask_batch_sparse_patchify_wgpu_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    pub fn step_image_with_mask_batch_sparse_patchify_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        let patchify_plan = SparsePatchifyBatchPlan::new(mask.clone(), self.grid, &self.device)?;
        self.step_image_with_sparse_patchify_plan_wgpu_measured(image, &patchify_plan, measurement)
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        let measured = self.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
            image,
            patchify_plan,
            FeatureFrameRequest::high_res(),
            measurement,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        let mut timer = StageTimer::new(measurement);
        let encoded = self
            .jepa
            .encode_image_sparse_patchify_wgpu_batch(image.clone(), patchify_plan)?;
        metrics.encode_us = timer.mark::<burn_flex_gmm::wgpu::DefaultWgpuBackend>(&self.device)?;
        self.finish_encoded_batch_nodes(image, mask, encoded, request, timer, metrics)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparseJepaAnyUpPcaPipeline<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn step_image_with_mask_sparse_patchify_cuda(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        Ok(self
            .step_image_with_mask_sparse_patchify_cuda_measured(image, mask)?
            .output)
    }

    pub fn step_image_with_mask_sparse_patchify_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured = self.step_image_with_mask_batch_sparse_patchify_cuda_measured(
            image,
            batch_mask,
            self.config.measurement,
        )?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_sparse_patchify_cuda(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        Ok(self
            .step_image_with_mask_batch_sparse_patchify_cuda_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    pub fn step_image_with_mask_batch_sparse_patchify_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::cuda::DefaultCudaBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        let patchify_plan = SparsePatchifyBatchPlan::new(mask.clone(), self.grid, &self.device)?;
        self.step_image_with_sparse_patchify_plan_cuda_measured(image, &patchify_plan, measurement)
    }

    pub fn step_image_with_sparse_patchify_plan_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        let measured = self.step_image_with_sparse_patchify_plan_cuda_nodes_measured(
            image,
            patchify_plan,
            FeatureFrameRequest::high_res(),
            measurement,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_sparse_patchify_plan_cuda_nodes_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        let mut timer = StageTimer::new(measurement);
        let encoded = self
            .jepa
            .encode_image_sparse_patchify_cuda_batch(image.clone(), patchify_plan)?;
        metrics.encode_us = timer.mark::<burn_flex_gmm::cuda::DefaultCudaBackend>(&self.device)?;
        self.finish_encoded_batch_nodes(image, mask, encoded, request, timer, metrics)
    }
}

struct StageTimer {
    measurement: SparseJepaAnyUpPcaMeasurementConfig,
    start: Option<Instant>,
    last: Option<Instant>,
}

impl StageTimer {
    fn new(measurement: SparseJepaAnyUpPcaMeasurementConfig) -> Self {
        let now = measurement.enabled.then(Instant::now);
        Self {
            measurement,
            start: now,
            last: now,
        }
    }

    fn mark<B: Backend>(&mut self, device: &B::Device) -> Result<u64> {
        if !self.measurement.enabled {
            return Ok(0);
        }
        if self.measurement.sync_backend {
            B::sync(device)?;
        }
        let now = Instant::now();
        let elapsed = self
            .last
            .replace(now)
            .map(|last| micros_u64(now.duration_since(last).as_micros()))
            .unwrap_or(0);
        Ok(elapsed)
    }

    fn total_us(&self) -> u64 {
        self.start
            .map(|start| micros_u64(start.elapsed().as_micros()))
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaFrameId {
    pub stream_id: u64,
    pub sequence: u64,
    pub capture_time_nanos: u64,
}

pub struct SparseJepaAnyUpPcaFrameInput<B: Backend> {
    pub id: SparseJepaAnyUpPcaFrameId,
    pub image: Tensor<B, 4>,
    pub mask: SparseTokenMask,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseJepaAnyUpPcaBackpressurePolicy {
    #[default]
    RejectNewest,
    DropOldest,
    OverwriteNewest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaStreamConfig {
    pub queue_capacity: usize,
    pub batch_size: usize,
    pub backpressure: SparseJepaAnyUpPcaBackpressurePolicy,
    pub schedule: FeatureFrameSchedule,
    pub require_monotonic_stream_sequences: bool,
    pub measurement: SparseJepaAnyUpPcaMeasurementConfig,
}

impl Default for SparseJepaAnyUpPcaStreamConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 2,
            batch_size: 1,
            backpressure: SparseJepaAnyUpPcaBackpressurePolicy::RejectNewest,
            schedule: FeatureFrameSchedule::default(),
            require_monotonic_stream_sequences: true,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
        }
    }
}

impl SparseJepaAnyUpPcaStreamConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.batch_size > 0, "stream batch size must be nonzero");
        ensure!(
            self.queue_capacity >= self.batch_size,
            "stream queue capacity must be at least the batch size"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseJepaAnyUpPcaQueueReport {
    pub accepted: bool,
    pub dropped_frame: Option<SparseJepaAnyUpPcaFrameId>,
    pub overwritten_frame: Option<SparseJepaAnyUpPcaFrameId>,
    pub queued_frames: usize,
    pub capacity: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseJepaAnyUpPcaQueuedFrameTiming {
    pub id: SparseJepaAnyUpPcaFrameId,
    pub queue_wait_us: u64,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaStreamBatchOutput<B: Backend> {
    pub frame_ids: Vec<SparseJepaAnyUpPcaFrameId>,
    pub frame_timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    pub output: SparseJepaAnyUpPcaStepBatchOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
    pub queued_after: usize,
    pub dropped_frames_total: usize,
}

#[derive(Debug)]
pub struct FeatureFrameStreamOutput<B: Backend> {
    pub frame_ids: Vec<SparseJepaAnyUpPcaFrameId>,
    pub frame_timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    pub output: FeatureFrameBatch<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
    pub request: FeatureFrameRequest,
    pub queued_after: usize,
    pub dropped_frames_total: usize,
    pub overwritten_frames_total: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseJepaAnyUpPcaStreamStats {
    pub queued_frames: usize,
    pub dropped_frames: usize,
    pub overwritten_frames: usize,
    pub emitted_batches: usize,
    pub emitted_frames: usize,
}

pub struct SparseJepaAnyUpPcaStream<B: Backend> {
    pipeline: SparseJepaAnyUpPcaPipeline<B>,
    config: SparseJepaAnyUpPcaStreamConfig,
    pending: VecDeque<QueuedFrame<B>>,
    last_seen_by_stream: BTreeMap<u64, u64>,
    cached_mask_batch: Option<CachedSparseMaskBatch<B>>,
    #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
    cached_sparse_patchify_plan: Option<CachedSparsePatchifyBatchPlan<B>>,
    dropped_frames: usize,
    overwritten_frames: usize,
    emitted_batches: usize,
    emitted_frames: usize,
}

struct QueuedFrame<B: Backend> {
    input: SparseJepaAnyUpPcaFrameInput<B>,
    queued_at: Instant,
}

struct DequeuedBatch<B: Backend> {
    ids: Vec<SparseJepaAnyUpPcaFrameId>,
    timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    image_batch: Tensor<B, 4>,
    mask_batch: SparseMaskBatch<B>,
}

struct CachedSparseMaskBatch<B: Backend> {
    dense_len: usize,
    rows: Vec<Vec<usize>>,
    mask: SparseMaskBatch<B>,
}

#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
struct CachedSparsePatchifyBatchPlan<B: Backend> {
    grid: TokenGridShape,
    rows: Vec<Vec<usize>>,
    plan: SparsePatchifyBatchPlan<B>,
}

impl<B: Backend> SparseJepaAnyUpPcaStream<B> {
    pub fn new(
        pipeline: SparseJepaAnyUpPcaPipeline<B>,
        config: SparseJepaAnyUpPcaStreamConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            pipeline.batch() == config.batch_size,
            "stream batch size must match the pipeline batch"
        );
        Ok(Self {
            pipeline,
            config,
            pending: VecDeque::new(),
            last_seen_by_stream: BTreeMap::new(),
            cached_mask_batch: None,
            #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
            cached_sparse_patchify_plan: None,
            dropped_frames: 0,
            overwritten_frames: 0,
            emitted_batches: 0,
            emitted_frames: 0,
        })
    }

    pub fn pipeline(&self) -> &SparseJepaAnyUpPcaPipeline<B> {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut SparseJepaAnyUpPcaPipeline<B> {
        &mut self.pipeline
    }

    pub fn config(&self) -> &SparseJepaAnyUpPcaStreamConfig {
        &self.config
    }

    pub fn queued_frames(&self) -> usize {
        self.pending.len()
    }

    pub fn dropped_frames(&self) -> usize {
        self.dropped_frames
    }

    pub fn can_process_batch(&self) -> bool {
        self.pending.len() >= self.config.batch_size
    }

    pub fn stats(&self) -> SparseJepaAnyUpPcaStreamStats {
        SparseJepaAnyUpPcaStreamStats {
            queued_frames: self.pending.len(),
            dropped_frames: self.dropped_frames,
            overwritten_frames: self.overwritten_frames,
            emitted_batches: self.emitted_batches,
            emitted_frames: self.emitted_frames,
        }
    }

    pub fn enqueue(
        &mut self,
        input: SparseJepaAnyUpPcaFrameInput<B>,
    ) -> Result<SparseJepaAnyUpPcaQueueReport> {
        ensure!(
            input.mask.dense_len() == self.pipeline.grid().len(),
            "frame sparse mask dense length must match pipeline grid"
        );
        ensure!(
            input.image.shape().dims::<4>()[0] == 1,
            "stream frame inputs must have batch size 1"
        );
        if self.config.require_monotonic_stream_sequences
            && let Some(&last) = self.last_seen_by_stream.get(&input.id.stream_id)
        {
            ensure!(
                input.id.sequence > last,
                "frame sequence must increase monotonically per stream"
            );
        }

        let mut dropped_frame = None;
        let mut overwritten_frame = None;
        if self.pending.len() >= self.config.queue_capacity {
            match self.config.backpressure {
                SparseJepaAnyUpPcaBackpressurePolicy::RejectNewest => {
                    bail!("high-res pipeline queue is full; apply backpressure or use drop_oldest");
                }
                SparseJepaAnyUpPcaBackpressurePolicy::DropOldest => {
                    let dropped = self
                        .pending
                        .pop_front()
                        .expect("queue is nonempty when full");
                    dropped_frame = Some(dropped.input.id);
                    self.dropped_frames += 1;
                }
                SparseJepaAnyUpPcaBackpressurePolicy::OverwriteNewest => {
                    let overwritten = self
                        .pending
                        .pop_back()
                        .expect("queue is nonempty when full");
                    overwritten_frame = Some(overwritten.input.id);
                    dropped_frame = overwritten_frame;
                    self.dropped_frames += 1;
                    self.overwritten_frames += 1;
                }
            }
        }

        self.last_seen_by_stream
            .insert(input.id.stream_id, input.id.sequence);
        self.pending.push_back(QueuedFrame {
            input,
            queued_at: Instant::now(),
        });
        Ok(SparseJepaAnyUpPcaQueueReport {
            accepted: true,
            dropped_frame,
            overwritten_frame,
            queued_frames: self.pending.len(),
            capacity: self.config.queue_capacity,
        })
    }

    pub fn process_next_ready(&mut self) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<B>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let measured = self.pipeline.step_image_with_mask_batch_measured(
            dequeued.image_batch,
            dequeued.mask_batch,
            self.config.measurement,
        )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_nodes(&mut self) -> Result<Option<FeatureFrameStreamOutput<B>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let measured = self.pipeline.step_image_with_mask_batch_nodes_measured(
            dequeued.image_batch,
            dequeued.mask_batch,
            request,
            self.config.measurement,
        )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    fn dequeue_ready_batch(&mut self) -> Result<Option<DequeuedBatch<B>>> {
        if self.pending.len() < self.config.batch_size {
            return Ok(None);
        }
        let sparse_width = self
            .pending
            .front()
            .map(|frame| frame.input.mask.len())
            .unwrap_or(0);
        ensure!(
            self.pending
                .iter()
                .take(self.config.batch_size)
                .all(|frame| frame.input.mask.len() == sparse_width),
            "front in-flight batch has variable sparse mask widths; use batch_size=1 or group masks by token budget"
        );

        let now = Instant::now();
        let mut ids = Vec::with_capacity(self.config.batch_size);
        let mut timings = Vec::with_capacity(self.config.batch_size);
        let mut images = Vec::with_capacity(self.config.batch_size);
        let mut rows = Vec::with_capacity(self.config.batch_size);
        for _ in 0..self.config.batch_size {
            let queued = self
                .pending
                .pop_front()
                .expect("ready batch has enough queued frames");
            ids.push(queued.input.id);
            timings.push(SparseJepaAnyUpPcaQueuedFrameTiming {
                id: queued.input.id,
                queue_wait_us: micros_u64(now.duration_since(queued.queued_at).as_micros()),
            });
            rows.push(queued.input.mask.indices().to_vec());
            images.push(queued.input.image);
        }

        let image_batch = Tensor::cat(images, 0);
        let dense_len = self.pipeline.grid().len();
        let device = image_batch.device();
        let mask_batch = self.cached_sparse_mask_batch(rows, dense_len, &device)?;
        ensure!(
            !mask_batch.is_ragged(),
            "stream produced a ragged sparse mask batch; use fixed token budgets for batched in-flight processing"
        );
        Ok(Some(DequeuedBatch {
            ids,
            timings,
            image_batch,
            mask_batch,
        }))
    }

    fn finish_dequeued_batch(
        &mut self,
        ids: Vec<SparseJepaAnyUpPcaFrameId>,
        timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
        measured: SparseJepaAnyUpPcaMeasuredBatchOutput<B>,
    ) -> SparseJepaAnyUpPcaStreamBatchOutput<B> {
        self.emitted_batches += 1;
        self.emitted_frames += ids.len();
        SparseJepaAnyUpPcaStreamBatchOutput {
            frame_ids: ids,
            frame_timings: timings,
            output: measured.output,
            metrics: measured.metrics,
            queued_after: self.pending.len(),
            dropped_frames_total: self.dropped_frames,
        }
    }

    fn finish_dequeued_nodes(
        &mut self,
        ids: Vec<SparseJepaAnyUpPcaFrameId>,
        timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
        request: FeatureFrameRequest,
        measured: MeasuredFeatureFrameBatch<B>,
    ) -> FeatureFrameStreamOutput<B> {
        self.emitted_batches += 1;
        self.emitted_frames += ids.len();
        FeatureFrameStreamOutput {
            frame_ids: ids,
            frame_timings: timings,
            output: measured.output,
            metrics: measured.metrics,
            request,
            queued_after: self.pending.len(),
            dropped_frames_total: self.dropped_frames,
            overwritten_frames_total: self.overwritten_frames,
        }
    }

    fn cached_sparse_mask_batch(
        &mut self,
        rows: Vec<Vec<usize>>,
        dense_len: usize,
        device: &B::Device,
    ) -> Result<SparseMaskBatch<B>> {
        let reuse = self
            .cached_mask_batch
            .as_ref()
            .is_some_and(|cached| cached.dense_len == dense_len && cached.rows == rows);
        if !reuse {
            let mask = SparseMaskBatch::from_rows(rows, dense_len, device)?;
            self.cached_mask_batch = Some(CachedSparseMaskBatch {
                dense_len,
                rows: mask.rows(),
                mask,
            });
        }
        Ok(self
            .cached_mask_batch
            .as_ref()
            .expect("sparse mask batch cache is initialized")
            .mask
            .clone())
    }

    #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
    fn cached_sparse_patchify_batch_plan(
        &mut self,
        mask: &SparseMaskBatch<B>,
        device: &B::Device,
    ) -> Result<SparsePatchifyBatchPlan<B>> {
        let grid = self.pipeline.grid();
        let rows = mask.rows();
        let reuse = self
            .cached_sparse_patchify_plan
            .as_ref()
            .is_some_and(|cached| cached.grid == grid && cached.rows == rows);
        if !reuse {
            self.cached_sparse_patchify_plan = Some(CachedSparsePatchifyBatchPlan {
                grid,
                rows,
                plan: SparsePatchifyBatchPlan::new(mask.clone(), grid, device)?,
            });
        }
        Ok(self
            .cached_sparse_patchify_plan
            .as_ref()
            .expect("sparse patchify plan cache is initialized")
            .plan
            .clone())
    }

    pub fn process_all_ready(&mut self) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<B>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_nodes(&mut self) -> Result<Vec<FeatureFrameStreamOutput<B>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparseJepaAnyUpPcaStream<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn process_next_ready_sparse_patchify_wgpu(
        &mut self,
    ) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_wgpu_measured(
                dequeued.image_batch,
                &patchify_plan,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_sparse_patchify_wgpu_nodes(
        &mut self,
    ) -> Result<Option<FeatureFrameStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
                dequeued.image_batch,
                &patchify_plan,
                request,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    pub fn process_all_ready_sparse_patchify_wgpu(
        &mut self,
    ) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_wgpu()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_sparse_patchify_wgpu_nodes(
        &mut self,
    ) -> Result<Vec<FeatureFrameStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_wgpu_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparseJepaAnyUpPcaStream<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn process_next_ready_sparse_patchify_cuda(
        &mut self,
    ) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>>
    {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_cuda_measured(
                dequeued.image_batch,
                &patchify_plan,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_sparse_patchify_cuda_nodes(
        &mut self,
    ) -> Result<Option<FeatureFrameStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_cuda_nodes_measured(
                dequeued.image_batch,
                &patchify_plan,
                request,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    pub fn process_all_ready_sparse_patchify_cuda(
        &mut self,
    ) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>>
    {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_cuda()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_sparse_patchify_cuda_nodes(
        &mut self,
    ) -> Result<Vec<FeatureFrameStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_cuda_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

pub type FeatureFrameMeasureConfig = SparseJepaAnyUpPcaMeasurementConfig;
pub type FeatureFrameEncodePath = SparseJepaAnyUpPcaEncodePath;
pub type FeatureFrameMetrics = SparseJepaAnyUpPcaStageMetrics;
pub type FeatureFramePipelineConfig = SparseJepaAnyUpPcaPipelineConfig;
pub type FeatureFramePipeline<B> = SparseJepaAnyUpPcaPipeline<B>;
pub type FrameId = SparseJepaAnyUpPcaFrameId;
pub type FeatureFrameInput<B> = SparseJepaAnyUpPcaFrameInput<B>;
pub type FrameQueuePolicy = SparseJepaAnyUpPcaBackpressurePolicy;
pub type FrameStreamConfig = SparseJepaAnyUpPcaStreamConfig;
pub type FrameQueueReport = SparseJepaAnyUpPcaQueueReport;
pub type FrameQueueTiming = SparseJepaAnyUpPcaQueuedFrameTiming;
pub type FeatureFrameStream<B> = SparseJepaAnyUpPcaStream<B>;
pub type FeatureFrameStreamStats = SparseJepaAnyUpPcaStreamStats;

fn scheduled_for_any(ids: &[SparseJepaAnyUpPcaFrameId], every: Option<u64>) -> bool {
    let Some(every) = every.filter(|&every| every > 0) else {
        return false;
    };
    ids.iter().any(|id| id.sequence % every == 0)
}

fn micros_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
