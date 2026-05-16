use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePcaDisplayMode {
    SignedUnit,
    #[default]
    SemanticRgb,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FeaturePcaConfig {
    pub output_channels: usize,
    pub epsilon: f64,
    pub display_scale: f64,
    pub display_mode: FeaturePcaDisplayMode,
    pub display_momentum: f64,
    pub display_std_floor: f64,
    pub display_clip_sigma: f64,
    pub online_learning_rate: f64,
    pub mean_momentum: f64,
}

impl Default for FeaturePcaConfig {
    fn default() -> Self {
        Self {
            output_channels: 3,
            epsilon: 1.0e-6,
            display_scale: 1.0,
            display_mode: FeaturePcaDisplayMode::SemanticRgb,
            display_momentum: 0.2,
            display_std_floor: 1.0e-3,
            display_clip_sigma: 2.5,
            online_learning_rate: 0.35,
            mean_momentum: 0.25,
        }
    }
}

impl FeaturePcaConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.output_channels > 0,
            "PCA output channel count must be nonzero"
        );
        ensure!(self.epsilon > 0.0, "PCA epsilon must be positive");
        ensure!(
            self.display_scale > 0.0,
            "PCA display scale must be positive"
        );
        ensure!(
            (0.0..=1.0).contains(&self.display_momentum),
            "PCA display momentum must be in [0, 1]"
        );
        ensure!(
            self.display_std_floor > 0.0,
            "PCA display std floor must be positive"
        );
        ensure!(
            self.display_clip_sigma > 0.0,
            "PCA display clip sigma must be positive"
        );
        ensure!(
            (0.0..=1.0).contains(&self.online_learning_rate),
            "PCA online learning rate must be in [0, 1]"
        );
        ensure!(
            (0.0..=1.0).contains(&self.mean_momentum),
            "PCA mean momentum must be in [0, 1]"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePcaUpdateMode {
    #[default]
    Disabled,
    RollingOja,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FeaturePcaUpdateConfig {
    pub mode: FeaturePcaUpdateMode,
    pub every_n_frames: u64,
    pub warmup_frames: u64,
    pub min_tokens_per_update: usize,
    pub iterations_per_update: usize,
    pub sample_window_frames: usize,
    pub min_sample_frames: usize,
}

impl Default for FeaturePcaUpdateConfig {
    fn default() -> Self {
        Self {
            mode: FeaturePcaUpdateMode::Disabled,
            every_n_frames: 1,
            warmup_frames: 0,
            min_tokens_per_update: 1,
            iterations_per_update: 1,
            sample_window_frames: 1,
            min_sample_frames: 1,
        }
    }
}

impl FeaturePcaUpdateConfig {
    pub const fn disabled() -> Self {
        Self {
            mode: FeaturePcaUpdateMode::Disabled,
            every_n_frames: 1,
            warmup_frames: 0,
            min_tokens_per_update: 1,
            iterations_per_update: 1,
            sample_window_frames: 1,
            min_sample_frames: 1,
        }
    }

    pub const fn rolling_low_res_every(every_n_frames: u64) -> Self {
        let sample_frames = if every_n_frames < 2 {
            2
        } else {
            every_n_frames as usize
        };
        Self {
            mode: FeaturePcaUpdateMode::RollingOja,
            every_n_frames,
            warmup_frames: 0,
            min_tokens_per_update: 1,
            iterations_per_update: 4,
            sample_window_frames: sample_frames,
            min_sample_frames: sample_frames,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.every_n_frames > 0,
            "PCA update cadence must be positive"
        );
        ensure!(
            self.min_tokens_per_update > 0,
            "PCA update minimum token count must be positive"
        );
        ensure!(
            self.iterations_per_update > 0,
            "PCA update iterations must be positive"
        );
        ensure!(
            self.sample_window_frames > 0,
            "PCA update sample window must be positive"
        );
        ensure!(
            self.min_sample_frames > 0,
            "PCA update minimum sample frames must be positive"
        );
        ensure!(
            self.min_sample_frames <= self.sample_window_frames,
            "PCA update minimum sample frames cannot exceed sample window"
        );
        Ok(())
    }

    pub const fn enabled(&self) -> bool {
        matches!(self.mode, FeaturePcaUpdateMode::RollingOja)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FeaturePcaUpdateDecision {
    pub update: bool,
    pub observed_frames: u64,
    pub update_index: u64,
}

#[derive(Clone, Debug)]
pub struct FeaturePcaUpdateScheduler {
    config: FeaturePcaUpdateConfig,
    observed_frames: u64,
    next_update_frame: u64,
    update_count: u64,
}

impl FeaturePcaUpdateScheduler {
    pub fn new(config: FeaturePcaUpdateConfig) -> Result<Self> {
        config.validate()?;
        let next_update_frame = config.warmup_frames + config.every_n_frames;
        Ok(Self {
            config,
            observed_frames: 0,
            next_update_frame,
            update_count: 0,
        })
    }

    pub fn config(&self) -> &FeaturePcaUpdateConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.observed_frames = 0;
        self.next_update_frame = self.config.warmup_frames + self.config.every_n_frames;
        self.update_count = 0;
    }

    pub fn observed_frames(&self) -> u64 {
        self.observed_frames
    }

    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    pub fn observe_batch(
        &mut self,
        frame_count: usize,
        tokens_per_frame: usize,
    ) -> FeaturePcaUpdateDecision {
        if frame_count == 0 {
            return FeaturePcaUpdateDecision {
                update: false,
                observed_frames: self.observed_frames,
                update_index: self.update_count,
            };
        }

        self.observed_frames = self.observed_frames.saturating_add(frame_count as u64);
        let min_observed_frames = self
            .config
            .warmup_frames
            .saturating_add(self.config.min_sample_frames as u64);
        if !self.config.enabled()
            || tokens_per_frame < self.config.min_tokens_per_update
            || self.observed_frames < min_observed_frames
            || self.observed_frames < self.next_update_frame
        {
            return FeaturePcaUpdateDecision {
                update: false,
                observed_frames: self.observed_frames,
                update_index: self.update_count,
            };
        }

        while self.next_update_frame <= self.observed_frames {
            self.next_update_frame = self
                .next_update_frame
                .saturating_add(self.config.every_n_frames);
        }
        self.update_count = self.update_count.saturating_add(1);
        FeaturePcaUpdateDecision {
            update: true,
            observed_frames: self.observed_frames,
            update_index: self.update_count,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeaturePcaProjector<B: Backend> {
    config: FeaturePcaConfig,
    feature_dim: usize,
    components: Tensor<B, 3>,
    mean: Tensor<B, 3>,
    display_center: Tensor<B, 3>,
    display_spread: Tensor<B, 3>,
}

impl<B: Backend> FeaturePcaProjector<B> {
    pub fn identity(
        feature_dim: usize,
        config: FeaturePcaConfig,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(feature_dim > 0, "PCA feature dim must be nonzero");
        let mut values = vec![0.0f32; feature_dim * config.output_channels];
        for channel in 0..feature_dim.min(config.output_channels) {
            values[channel * config.output_channels + channel] = 1.0;
        }
        Ok(Self {
            components: Tensor::<B, 3>::from_data(
                TensorData::new(values, [1, feature_dim, config.output_channels]),
                device,
            ),
            mean: Tensor::<B, 3>::zeros([1, 1, feature_dim], device),
            display_center: Tensor::<B, 3>::zeros([1, 1, config.output_channels], device),
            display_spread: Tensor::<B, 3>::ones([1, 1, config.output_channels], device),
            config,
            feature_dim,
        })
    }

    pub fn from_components(
        components: Tensor<B, 3>,
        mean: Tensor<B, 3>,
        config: FeaturePcaConfig,
    ) -> Result<Self> {
        config.validate()?;
        let [component_batch, feature_dim, output_channels] = components.shape().dims::<3>();
        let [mean_batch, mean_tokens, mean_dim] = mean.shape().dims::<3>();
        ensure!(
            component_batch == 1,
            "PCA components must have shape [1, feature_dim, output_channels]"
        );
        ensure!(
            output_channels == config.output_channels,
            "PCA component output channels must match config"
        );
        ensure!(
            mean_batch == 1 && mean_tokens == 1 && mean_dim == feature_dim,
            "PCA mean must have shape [1, 1, feature_dim]"
        );
        Ok(Self {
            display_center: Tensor::<B, 3>::zeros([1, 1, output_channels], &components.device()),
            display_spread: Tensor::<B, 3>::ones([1, 1, output_channels], &components.device()),
            config,
            feature_dim,
            components,
            mean,
        })
    }

    pub fn config(&self) -> &FeaturePcaConfig {
        &self.config
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn output_channels(&self) -> usize {
        self.config.output_channels
    }

    pub fn components(&self) -> Tensor<B, 3> {
        self.components.clone()
    }

    pub fn mean(&self) -> Tensor<B, 3> {
        self.mean.clone()
    }

    pub fn display_center(&self) -> Tensor<B, 3> {
        self.display_center.clone()
    }

    pub fn display_spread(&self) -> Tensor<B, 3> {
        self.display_spread.clone()
    }

    pub fn project_tokens(&self, tokens: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [_, _, feature_dim] = tokens.shape().dims::<3>();
        ensure!(
            feature_dim == self.feature_dim,
            "PCA token feature dim does not match projector"
        );
        Ok((tokens - self.mean.clone()).matmul(self.components.clone()))
    }

    pub fn project_tokens_display(&self, tokens: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.display_tokens(self.project_tokens(tokens)?)
    }

    pub fn project_nchw(&self, features: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
        let [batch, channels, height, width] = features.shape().dims::<4>();
        ensure!(
            channels == self.feature_dim,
            "PCA NCHW feature channel count does not match projector"
        );
        let tokens = features
            .permute([0, 2, 3, 1])
            .reshape([batch, height * width, channels]);
        let projected = self.project_tokens(tokens)?;
        Ok(projected
            .reshape([batch, height, width, self.config.output_channels])
            .permute([0, 3, 1, 2]))
    }

    pub fn project_nchw_display(&self, features: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
        self.display_nchw(self.project_nchw(features)?)
    }

    pub fn display_nchw(&self, components: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
        let [_, channels, _, _] = components.shape().dims::<4>();
        ensure!(
            channels == self.config.output_channels,
            "PCA display component channel count does not match projector output channels"
        );
        Ok(match self.config.display_mode {
            FeaturePcaDisplayMode::SignedUnit => self.signed_unit_display(components),
            FeaturePcaDisplayMode::SemanticRgb => self.semantic_nchw_display(components),
        })
    }

    pub fn update_rolling_tokens(&mut self, tokens: Tensor<B, 3>) -> Result<()> {
        let [_, token_count, feature_dim] = tokens.shape().dims::<3>();
        ensure!(
            feature_dim == self.feature_dim,
            "PCA token feature dim does not match projector"
        );
        ensure!(
            token_count > 0,
            "PCA online update requires at least one token"
        );

        let mean_update = tokens.clone().mean_dim(1).mean_dim(0);
        self.mean = self
            .mean
            .clone()
            .mul_scalar(1.0 - self.config.mean_momentum)
            + mean_update.mul_scalar(self.config.mean_momentum);

        let centered = tokens - self.mean.clone();
        let projected = centered.clone().matmul(self.components.clone());
        let update = centered
            .clone()
            .swap_dims(1, 2)
            .matmul(projected)
            .mean_dim(0)
            .div_scalar(token_count as f64);
        let components = self
            .components
            .clone()
            .mul_scalar(1.0 - self.config.online_learning_rate)
            + update.mul_scalar(self.config.online_learning_rate);
        self.components = self.stabilize_components(components);
        let projected_stats = centered.matmul(self.components.clone());
        self.update_display_stats_from_projected(projected_stats, None)?;
        Ok(())
    }

    pub fn update_rolling_tokens_iterations(
        &mut self,
        tokens: Tensor<B, 3>,
        iterations: usize,
    ) -> Result<()> {
        ensure!(iterations > 0, "PCA update iterations must be positive");
        for _ in 0..iterations {
            self.update_rolling_tokens(tokens.clone())?;
        }
        Ok(())
    }

    pub fn update_rolling_masked_tokens(
        &mut self,
        tokens: Tensor<B, 3>,
        weights: Tensor<B, 2>,
    ) -> Result<()> {
        let [batch, token_count, feature_dim] = tokens.shape().dims::<3>();
        let [weight_batch, weight_tokens] = weights.shape().dims::<2>();
        ensure!(
            feature_dim == self.feature_dim,
            "PCA token feature dim does not match projector"
        );
        ensure!(
            token_count > 0,
            "PCA rolling update requires at least one token"
        );
        ensure!(
            weight_batch == batch && weight_tokens == token_count,
            "PCA update weights must match token batch and token count"
        );

        let stats_weights = weights.clone();
        let weights = weights.unsqueeze_dim::<3>(2);
        let denom = weights
            .clone()
            .sum_dim(1)
            .sum_dim(0)
            .add_scalar(self.config.epsilon);
        let mean_denom = denom.clone().repeat_dim(2, self.feature_dim);
        let mean_update = (tokens.clone() * weights.clone()).sum_dim(1).sum_dim(0) / mean_denom;
        self.mean = self
            .mean
            .clone()
            .mul_scalar(1.0 - self.config.mean_momentum)
            + mean_update.mul_scalar(self.config.mean_momentum);

        let centered = tokens.clone() - self.mean.clone();
        let projected = centered.clone().matmul(self.components.clone());
        let update_denom = denom
            .repeat_dim(1, self.feature_dim)
            .repeat_dim(2, self.config.output_channels);
        let update = (centered * weights)
            .swap_dims(1, 2)
            .matmul(projected)
            .sum_dim(0)
            / update_denom;
        let components = self
            .components
            .clone()
            .mul_scalar(1.0 - self.config.online_learning_rate)
            + update.mul_scalar(self.config.online_learning_rate);
        self.components = self.stabilize_components(components);
        let projected_stats = (tokens - self.mean.clone()).matmul(self.components.clone());
        self.update_display_stats_from_projected(projected_stats, Some(stats_weights))?;
        Ok(())
    }

    pub fn update_rolling_masked_tokens_iterations(
        &mut self,
        tokens: Tensor<B, 3>,
        weights: Tensor<B, 2>,
        iterations: usize,
    ) -> Result<()> {
        ensure!(iterations > 0, "PCA update iterations must be positive");
        for _ in 0..iterations {
            self.update_rolling_masked_tokens(tokens.clone(), weights.clone())?;
        }
        Ok(())
    }

    pub fn update_rolling_nchw(&mut self, features: Tensor<B, 4>) -> Result<()> {
        let [batch, channels, height, width] = features.shape().dims::<4>();
        ensure!(
            channels == self.feature_dim,
            "PCA NCHW feature channel count does not match projector"
        );
        self.update_rolling_tokens(features.permute([0, 2, 3, 1]).reshape([
            batch,
            height * width,
            channels,
        ]))
    }

    pub fn update_rolling_nchw_iterations(
        &mut self,
        features: Tensor<B, 4>,
        iterations: usize,
    ) -> Result<()> {
        ensure!(iterations > 0, "PCA update iterations must be positive");
        for _ in 0..iterations {
            self.update_rolling_nchw(features.clone())?;
        }
        Ok(())
    }

    pub fn update_online_tokens(&mut self, tokens: Tensor<B, 3>) -> Result<()> {
        self.update_rolling_tokens(tokens)
    }

    pub fn update_online_nchw(&mut self, features: Tensor<B, 4>) -> Result<()> {
        self.update_rolling_nchw(features)
    }

    fn stabilize_components(&self, components: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut basis: Vec<Tensor<B, 3>> = Vec::with_capacity(self.config.output_channels);
        for channel in 0..self.config.output_channels {
            let mut vector =
                components
                    .clone()
                    .slice([0..1, 0..self.feature_dim, channel..channel + 1]);
            for previous in &basis {
                let coefficient = (vector.clone() * previous.clone()).sum_dim(1);
                vector = vector - previous.clone() * coefficient.repeat_dim(1, self.feature_dim);
            }
            basis.push(self.normalize_component(vector));
        }
        Tensor::cat(basis, 2)
    }

    fn normalize_component(&self, component: Tensor<B, 3>) -> Tensor<B, 3> {
        let denom = component
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .add_scalar(self.config.epsilon)
            .sqrt()
            .repeat_dim(1, self.feature_dim);
        component / denom
    }

    fn display_tokens(&self, components: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [_, _, channels] = components.shape().dims::<3>();
        ensure!(
            channels == self.config.output_channels,
            "PCA display token channel count does not match projector output channels"
        );
        Ok(match self.config.display_mode {
            FeaturePcaDisplayMode::SignedUnit => self.signed_unit_display(components),
            FeaturePcaDisplayMode::SemanticRgb => self.semantic_tokens_display(components),
        })
    }

    fn update_display_stats_from_projected(
        &mut self,
        projected: Tensor<B, 3>,
        weights: Option<Tensor<B, 2>>,
    ) -> Result<()> {
        if self.config.display_mode != FeaturePcaDisplayMode::SemanticRgb
            || self.config.display_momentum == 0.0
        {
            return Ok(());
        }
        let [batch, token_count, channels] = projected.shape().dims::<3>();
        ensure!(
            channels == self.config.output_channels,
            "PCA projected display channel count does not match projector output channels"
        );
        ensure!(
            token_count > 0,
            "PCA display statistics update requires at least one token"
        );
        let weights = match weights {
            Some(weights) => {
                let [weight_batch, weight_tokens] = weights.shape().dims::<2>();
                ensure!(
                    weight_batch == batch && weight_tokens == token_count,
                    "PCA display statistics weights must match projected token shape"
                );
                weights
            }
            None => Tensor::<B, 2>::ones([batch, token_count], &projected.device()),
        };
        let weights = weights.unsqueeze_dim::<3>(2);
        let denom = weights
            .clone()
            .sum_dim(1)
            .sum_dim(0)
            .add_scalar(self.config.epsilon)
            .repeat_dim(2, channels);
        let center = (projected.clone() * weights.clone()).sum_dim(1).sum_dim(0) / denom.clone();
        let centered = projected - center.clone().repeat_dim(1, token_count);
        let spread = ((centered.powf_scalar(2.0) * weights).sum_dim(1).sum_dim(0) / denom)
            .add_scalar(self.config.epsilon)
            .sqrt()
            .add_scalar(self.config.display_std_floor);

        self.display_center = self
            .display_center
            .clone()
            .mul_scalar(1.0 - self.config.display_momentum)
            + center.mul_scalar(self.config.display_momentum);
        self.display_spread = self
            .display_spread
            .clone()
            .mul_scalar(1.0 - self.config.display_momentum)
            + spread.mul_scalar(self.config.display_momentum);
        Ok(())
    }

    fn semantic_tokens_display(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, token_count, _] = x.shape().dims::<3>();
        let centered = x - self.display_center.clone().repeat_dim(1, token_count);
        let spread = self
            .display_spread
            .clone()
            .repeat_dim(1, token_count)
            .add_scalar(self.config.display_std_floor);
        self.signed_unit_display(
            (centered / spread)
                .div_scalar(self.config.display_clip_sigma)
                .mul_scalar(self.config.display_scale),
        )
    }

    fn semantic_nchw_display(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.shape().dims::<4>();
        let center = self
            .display_center
            .clone()
            .reshape([1, channels, 1, 1])
            .repeat_dim(0, batch)
            .repeat_dim(2, height)
            .repeat_dim(3, width);
        let spread = self
            .display_spread
            .clone()
            .reshape([1, channels, 1, 1])
            .repeat_dim(0, batch)
            .repeat_dim(2, height)
            .repeat_dim(3, width)
            .add_scalar(self.config.display_std_floor);
        self.signed_unit_display(
            ((x - center) / spread)
                .div_scalar(self.config.display_clip_sigma)
                .mul_scalar(self.config.display_scale),
        )
    }

    fn signed_unit_display<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let scaled = x.mul_scalar(self.config.display_scale);
        let denom = scaled
            .clone()
            .powf_scalar(2.0)
            .add_scalar(self.config.epsilon)
            .sqrt()
            .add_scalar(1.0);
        (scaled / denom).mul_scalar(0.5).add_scalar(0.5)
    }
}
