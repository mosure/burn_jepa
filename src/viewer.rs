use crate::{
    FeaturePcaUpdateConfig, FeaturePcaUpdateMode, SparseJepaAnyUpPcaMeasurementConfig,
    SparseJepaPatchDiffSparsityConfig, SparseTokenMask, TokenGridShape, TttRuntimeStateConfig,
    coords_to_token_index, patch_diff_context_mask_from_scores,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt, str::FromStr};

/// Minimum square input resolution used by the live feature-frame viewer policy.
pub const MIN_PIPELINE_IMAGE_SIZE: usize = 256;
/// Viewer image sizes are rounded to this multiple so they remain patch-grid aligned.
pub const PIPELINE_IMAGE_SIZE_MULTIPLE: usize = 16;
/// Default square input resolution for live sparse V-JEPA feature viewing.
pub const DEFAULT_IMAGE_SIZE: usize = 512;
/// Default maximum dynamic patch-diff context density.
pub const DEFAULT_CONTEXT_DENSITY: f32 = 1.0;
/// Default patch-diff quality, mapped to `threshold = 1 - quality`.
pub const DEFAULT_PATCH_DIFF_QUALITY: f32 = 0.97;
/// Default dynamic patch-diff minimum context density for near-static frames.
pub const DEFAULT_MIN_CONTEXT_DENSITY: f32 = 0.0;
/// Default first-frame cache bootstrap density.
pub const DEFAULT_BOOTSTRAP_CONTEXT_DENSITY: f32 = 1.0;
/// Default patch-diff score threshold.
pub const DEFAULT_PATCH_DIFF_THRESHOLD: f32 = 1.0 - DEFAULT_PATCH_DIFF_QUALITY;
/// Density at which a sparse write mask is promoted to the dense ordered path.
pub const DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY: f32 = 0.60;
/// Default sparse encode bucket width used to reduce GPU shape churn.
pub const DEFAULT_SPARSE_MASK_BUCKET_TOKENS: usize = 256;
/// Whether patch-diff refresh adds bounded drift-prevention tokens by default.
pub const DEFAULT_PATCH_DIFF_REFRESH_ENABLED: bool = true;
/// Default residual decay for subthreshold patch-diff evidence.
pub const DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY: f32 = 0.92;
/// Default normalized residual value required before a below-threshold patch is refreshed.
pub const DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER: f32 = 1.0;
/// Default maximum density used by subthreshold refresh tokens.
pub const DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY: f32 = 0.04;
/// Default stale-token interval for age-priority refresh.
pub const DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES: u64 = 90;
/// Default maximum density used by age-priority refresh tokens.
pub const DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY: f32 = 0.01;
/// Default maximum density used by deterministic blue-noise refresh tokens.
pub const DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY: f32 = 0.005;
/// Default total refresh density cap across all patch-diff refresh modes.
pub const DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY: f32 = 0.05;
/// Whether live viewers should prewarm bucketed sparse token widths.
pub const DEFAULT_PREWARM_SHAPE_BUCKETS: bool = true;
/// Default AnyUp query chunk size used by the portable high-resolution decode path.
pub const DEFAULT_ANYUP_CHUNK_SIZE: usize = 16;
/// Default cadence for low-resolution PCA basis updates.
pub const DEFAULT_PCA_UPDATE_EVERY: u64 = 1;
/// Default rolling frame window used to fit low-resolution PCA weights.
pub const DEFAULT_PCA_SAMPLE_WINDOW_FRAMES: usize = 16;
/// Minimum frames required before fitting the first rolling PCA basis.
pub const DEFAULT_PCA_MIN_SAMPLE_FRAMES: usize = 2;
/// Default Oja iterations per rolling PCA update.
pub const DEFAULT_PCA_UPDATE_ITERATIONS: usize = 4;
/// Default high-resolution AnyUp PCA cadence; zero keeps AnyUp off the hot path.
pub const DEFAULT_HIGH_RES_PCA_EVERY: u64 = 0;

const PATCH_DIFF_DENSE_FAST_PATH_MAX_SAMPLES: usize = 64;

/// Runtime route for V-JEPA patch embedding in live feature-frame pipelines.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFrameEncodeRoute {
    /// Use sparse patchify for non-dense masks when a backend implementation is available.
    #[default]
    Auto,
    /// Always use the standard dense patch embedding and gather sparse encoder tokens after it.
    DensePatchEmbed,
    /// Force sparse pixel-skip patchify. Requires a backend feature such as `sparse-patchify-wgpu`.
    SparsePatchify,
}

impl FeatureFrameEncodeRoute {
    /// Stable command-line/config string for this route.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DensePatchEmbed => "dense-patch",
            Self::SparsePatchify => "sparse-patchify",
        }
    }

    /// Accepted parse aliases for this route.
    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "auto",
            "dense-patch",
            "dense-patch-embed",
            "dense",
            "sparse-patchify",
            "sparse",
            "flex-gmm",
        ]
    }
}

impl fmt::Display for FeatureFrameEncodeRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for FeatureFrameEncodeRoute {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "dense-patch" | "dense-patch-embed" | "dense" | "dense-patchify" => {
                Ok(Self::DensePatchEmbed)
            }
            "sparse-patchify" | "sparse" | "flex-gmm" | "flex_gmm" => Ok(Self::SparsePatchify),
            other => Err(format!(
                "unsupported JEPA encode path `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

/// How a threshold-selected sparse write mask is expanded before V-JEPA encoding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFrameSparseEncodeMode {
    /// Encode exactly the same tokens that will be written into the feature cache.
    Exact,
    /// Keep cache writes exact, but widen encoder context to stable token-width buckets.
    #[default]
    BucketedContext,
}

impl FeatureFrameSparseEncodeMode {
    /// Stable command-line/config string for this mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::BucketedContext => "bucketed-context",
        }
    }

    /// Accepted parse aliases for this mode.
    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "exact",
            "semantic",
            "write-mask",
            "write_mask",
            "bucketed-context",
            "bucketed_context",
            "bucketed",
            "widened-context",
            "widened_context",
            "widened",
        ]
    }
}

impl fmt::Display for FeatureFrameSparseEncodeMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for FeatureFrameSparseEncodeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "exact" | "semantic" | "write-mask" | "write_mask" => Ok(Self::Exact),
            "bucketed-context" | "bucketed_context" | "bucketed" | "bucket" | "widened-context"
            | "widened_context" | "widened" => Ok(Self::BucketedContext),
            other => Err(format!(
                "unsupported sparse encode mode `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

/// Bounded drift-prevention policy for patch-diff sparse cache writes.
///
/// Threshold hits are always kept first. These refresh modes only fill unused
/// context budget, so they avoid hiding real motion while still refreshing slow
/// semantic changes that never cross the instantaneous patch-diff threshold.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PatchDiffRefreshConfig {
    /// Enables all configured refresh modes.
    pub enabled: bool,
    /// Accumulate repeated below-threshold scores until they merit a refresh.
    pub subthreshold_enabled: bool,
    /// Per-frame decay applied to subthreshold evidence.
    pub subthreshold_decay: f32,
    /// Multiplier applied to normalized below-threshold scores.
    pub subthreshold_gain: f32,
    /// Accumulated evidence required before selecting a below-threshold token.
    pub subthreshold_trigger: f32,
    /// Per-frame density cap for subthreshold refresh tokens.
    pub subthreshold_max_density: f32,
    /// Refresh tokens that have not been overwritten for many frames.
    pub age_refresh_enabled: bool,
    /// Frames since last write before a token is eligible for age refresh.
    pub age_refresh_interval_frames: u64,
    /// Per-frame density cap for age-priority refresh tokens.
    pub age_refresh_max_density: f32,
    /// Enables deterministic blue-noise-like refresh probes.
    pub blue_noise_enabled: bool,
    /// Per-frame density cap for deterministic blue-noise refresh tokens.
    pub blue_noise_refresh_density: f32,
    /// Seed for deterministic blue-noise refresh ordering.
    pub blue_noise_seed: u64,
    /// Total per-frame density cap across all refresh modes.
    pub max_extra_density: f32,
}

impl Default for PatchDiffRefreshConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_PATCH_DIFF_REFRESH_ENABLED,
            subthreshold_enabled: true,
            subthreshold_decay: DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY,
            subthreshold_gain: 1.0,
            subthreshold_trigger: DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER,
            subthreshold_max_density: DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY,
            age_refresh_enabled: true,
            age_refresh_interval_frames: DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES,
            age_refresh_max_density: DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY,
            blue_noise_enabled: true,
            blue_noise_refresh_density: DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY,
            blue_noise_seed: 0x9e37_79b9_7f4a_7c15,
            max_extra_density: DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY,
        }
    }
}

impl PatchDiffRefreshConfig {
    /// Returns a config that preserves legacy instantaneous patch-diff behavior.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            subthreshold_enabled: false,
            subthreshold_decay: DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY,
            subthreshold_gain: 1.0,
            subthreshold_trigger: DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER,
            subthreshold_max_density: 0.0,
            age_refresh_enabled: false,
            age_refresh_interval_frames: DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES,
            age_refresh_max_density: 0.0,
            blue_noise_enabled: false,
            blue_noise_refresh_density: 0.0,
            blue_noise_seed: 0x9e37_79b9_7f4a_7c15,
            max_extra_density: 0.0,
        }
    }

    /// True when any refresh mode can add tokens.
    pub fn can_add_tokens(&self) -> bool {
        self.enabled
            && self.max_extra_density > 0.0
            && ((self.subthreshold_enabled && self.subthreshold_max_density > 0.0)
                || (self.age_refresh_enabled
                    && self.age_refresh_interval_frames > 0
                    && self.age_refresh_max_density > 0.0)
                || (self.blue_noise_enabled && self.blue_noise_refresh_density > 0.0))
    }
}

/// Stateful patch-diff refresh accumulator used by live interframe token caches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchDiffRefreshState {
    dense_len: usize,
    frame_index: u64,
    subthreshold_residual: Vec<f32>,
    age_frames: Vec<u32>,
}

impl PatchDiffRefreshState {
    /// Clears accumulated residuals and token ages.
    pub fn reset(&mut self) {
        self.dense_len = 0;
        self.frame_index = 0;
        self.subthreshold_residual.clear();
        self.age_frames.clear();
    }

    /// Current refresh frame index, useful for diagnostics and tests.
    pub const fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Returns the accumulated residual for one token.
    pub fn subthreshold_residual(&self, index: usize) -> Option<f32> {
        self.subthreshold_residual.get(index).copied()
    }

    /// Returns the age in frames for one token.
    pub fn age_frames(&self, index: usize) -> Option<u32> {
        self.age_frames.get(index).copied()
    }

    /// Records a patch-diff frame whose final cache write mask is already known.
    pub fn observe_write_mask(&mut self, mask: &SparseTokenMask, grid: TokenGridShape) {
        self.begin_frame(grid);
        self.reset_selected(mask.indices());
    }

    /// Builds patch-diff masks from scores and adds bounded refresh tokens.
    pub fn masks_from_scores(
        &mut self,
        scores: Vec<f32>,
        grid: TokenGridShape,
        sparsity: &SparseJepaPatchDiffSparsityConfig,
        config: &FeatureFrameViewerConfig,
    ) -> Result<FeatureFrameSparseMasks> {
        ensure!(
            scores.len() == grid.len(),
            "patch-diff refresh received unexpected score length"
        );
        if !config.patch_diff_refresh.can_add_tokens() {
            let mask = patch_diff_context_mask_from_scores(scores, grid, sparsity)?;
            self.begin_frame(grid);
            let masks = finalize_patch_diff_masks(mask, grid, config);
            self.reset_selected(masks.write_mask.indices());
            return Ok(masks);
        }
        let base_mask = patch_diff_context_mask_from_scores(scores.clone(), grid, sparsity)?;
        self.begin_frame(grid);
        let refreshed = self.augment_mask(
            base_mask,
            &scores,
            grid,
            sparsity,
            &config.patch_diff_refresh,
        )?;
        let masks = finalize_patch_diff_masks(refreshed, grid, config);
        self.reset_selected(masks.write_mask.indices());
        Ok(masks)
    }

    fn begin_frame(&mut self, grid: TokenGridShape) {
        if self.dense_len != grid.len() {
            self.dense_len = grid.len();
            self.frame_index = 0;
            self.subthreshold_residual = vec![0.0; grid.len()];
            self.age_frames = vec![0; grid.len()];
        }
        self.frame_index = self.frame_index.saturating_add(1);
        for age in &mut self.age_frames {
            *age = age.saturating_add(1);
        }
    }

    fn reset_selected(&mut self, indices: &[usize]) {
        for &index in indices {
            if let Some(residual) = self.subthreshold_residual.get_mut(index) {
                *residual = 0.0;
            }
            if let Some(age) = self.age_frames.get_mut(index) {
                *age = 0;
            }
        }
    }

    fn augment_mask(
        &mut self,
        base_mask: SparseTokenMask,
        scores: &[f32],
        grid: TokenGridShape,
        sparsity: &SparseJepaPatchDiffSparsityConfig,
        config: &PatchDiffRefreshConfig,
    ) -> Result<SparseTokenMask> {
        if base_mask.len() >= grid.len() {
            return Ok(base_mask);
        }
        let max_context = sparsity.context_tokens.max(1).min(grid.len());
        let mut remaining = max_context.saturating_sub(base_mask.len());
        remaining = remaining.min(density_tokens(
            grid.len(),
            config.max_extra_density,
            DensityRound::Floor,
            0.0,
        ));
        if remaining == 0 {
            self.update_subthreshold_residual(scores, base_mask.indices(), sparsity, config);
            return Ok(base_mask);
        }

        let mut selected = vec![false; grid.len()];
        let mut indices = base_mask.indices().to_vec();
        for &index in &indices {
            selected[index] = true;
        }
        self.update_subthreshold_residual(scores, &indices, sparsity, config);

        if config.subthreshold_enabled && config.subthreshold_max_density > 0.0 {
            let budget = remaining.min(density_tokens(
                grid.len(),
                config.subthreshold_max_density,
                DensityRound::Floor,
                0.0,
            ));
            let added = self.add_ranked_candidates(&mut selected, &mut indices, budget, |index| {
                let residual = self.subthreshold_residual[index];
                (residual >= config.subthreshold_trigger.max(1.0e-6)).then_some((
                    residual,
                    self.refresh_rank(index, grid, config.blue_noise_seed),
                ))
            });
            remaining = remaining.saturating_sub(added);
        }

        if remaining > 0
            && config.age_refresh_enabled
            && config.age_refresh_interval_frames > 0
            && config.age_refresh_max_density > 0.0
        {
            let interval = config.age_refresh_interval_frames.min(u32::MAX as u64) as u32;
            let budget = remaining.min(density_tokens(
                grid.len(),
                config.age_refresh_max_density,
                DensityRound::Floor,
                0.0,
            ));
            let added = self.add_ranked_candidates(&mut selected, &mut indices, budget, |index| {
                let age = self.age_frames[index];
                (age >= interval).then_some((
                    age as f32,
                    self.refresh_rank(index, grid, config.blue_noise_seed),
                ))
            });
            remaining = remaining.saturating_sub(added);
        }

        if remaining > 0 && config.blue_noise_enabled && config.blue_noise_refresh_density > 0.0 {
            let budget = remaining.min(density_tokens(
                grid.len(),
                config.blue_noise_refresh_density,
                DensityRound::Floor,
                0.0,
            ));
            self.add_ranked_candidates(&mut selected, &mut indices, budget, |index| {
                Some((0.0, self.refresh_rank(index, grid, config.blue_noise_seed)))
            });
        }

        SparseTokenMask::new(indices, grid.len())
    }

    fn update_subthreshold_residual(
        &mut self,
        scores: &[f32],
        selected_indices: &[usize],
        sparsity: &SparseJepaPatchDiffSparsityConfig,
        config: &PatchDiffRefreshConfig,
    ) {
        let mut selected = vec![false; scores.len()];
        for &index in selected_indices {
            if let Some(slot) = selected.get_mut(index) {
                *slot = true;
            }
        }
        let decay = config.subthreshold_decay.clamp(0.0, 1.0);
        let threshold = sparsity.threshold.max(1.0e-6);
        for (index, residual) in self.subthreshold_residual.iter_mut().enumerate() {
            if selected[index] || scores[index] >= sparsity.threshold {
                *residual = 0.0;
            } else {
                let normalized = (scores[index] / threshold).clamp(0.0, 1.0);
                *residual = *residual * decay + normalized * config.subthreshold_gain.max(0.0);
            }
        }
    }

    fn add_ranked_candidates<F>(
        &self,
        selected: &mut [bool],
        indices: &mut Vec<usize>,
        budget: usize,
        mut rank: F,
    ) -> usize
    where
        F: FnMut(usize) -> Option<(f32, u64)>,
    {
        if budget == 0 {
            return 0;
        }
        let mut candidates = Vec::new();
        for (index, present) in selected.iter().enumerate() {
            if *present {
                continue;
            }
            if let Some((priority, tie_breaker)) = rank(index) {
                candidates.push((index, priority, tie_breaker));
            }
        }
        let compare = |left: &(usize, f32, u64), right: &(usize, f32, u64)| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.0.cmp(&right.0))
        };
        if candidates.len() > budget {
            candidates.select_nth_unstable_by(budget, compare);
            candidates.truncate(budget);
        }
        candidates.sort_by(compare);
        let mut added = 0usize;
        for (index, _, _) in candidates {
            selected[index] = true;
            indices.push(index);
            added += 1;
        }
        added
    }

    fn refresh_rank(&self, index: usize, grid: TokenGridShape, seed: u64) -> u64 {
        let row = index / grid.width.max(1);
        let col = index % grid.width.max(1);
        mix64(
            seed ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (col as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                ^ self.frame_index.wrapping_mul(0x94d0_49bb_1331_11eb),
        )
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Shared live feature-frame policy used by viewers, examples, benches, and downstream clients.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FeatureFrameViewerConfig {
    /// V-JEPA encode route.
    pub encode_path: FeatureFrameEncodeRoute,
    /// Requested square input size before policy rounding.
    pub image_size: usize,
    /// Maximum fraction of dense tokens that patch-diff may use.
    pub context_density: f32,
    /// Minimum fraction of dense tokens to keep when patch-diff detects little motion.
    pub min_context_density: f32,
    /// Fraction of dense tokens used to initialize the cache on the first frame.
    pub bootstrap_context_density: f32,
    /// Patch-diff score threshold.
    pub patch_diff_threshold: f32,
    /// Density at which sparse masks are promoted to dense ordered masks.
    pub patch_diff_dense_fallback_density: f32,
    /// Sparse encode widening mode.
    pub sparse_encode_mode: FeatureFrameSparseEncodeMode,
    /// Token bucket width used with [`FeatureFrameSparseEncodeMode::BucketedContext`].
    pub sparse_mask_bucket_tokens: usize,
    /// Bounded refresh policy for slow/stale patch-diff tokens.
    pub patch_diff_refresh: PatchDiffRefreshConfig,
    /// Prewarm bucketed sparse widths during startup.
    pub prewarm_shape_buckets: bool,
    /// AnyUp query chunk size.
    pub anyup_q_chunk_size: usize,
    /// Rolling PCA update cadence.
    pub pca_update_every: u64,
    /// Rolling PCA sample window size in frames.
    pub pca_sample_window_frames: usize,
    /// Minimum sample frames before fitting the first PCA basis.
    pub pca_min_sample_frames: usize,
    /// Oja iterations per PCA update.
    pub pca_update_iterations: usize,
    /// High-resolution AnyUp PCA cadence; zero disables the high-res worker by default.
    pub high_res_pca_every: u64,
    /// Runtime TTT fast-memory policy used by the viewer pipeline.
    pub ttt_runtime: TttRuntimeStateConfig,
    /// Collect stage timings.
    pub measure_stages: bool,
    /// Synchronize the backend around measurements.
    pub sync_measurements: bool,
}

impl Default for FeatureFrameViewerConfig {
    fn default() -> Self {
        Self {
            encode_path: FeatureFrameEncodeRoute::Auto,
            image_size: DEFAULT_IMAGE_SIZE,
            context_density: DEFAULT_CONTEXT_DENSITY,
            min_context_density: DEFAULT_MIN_CONTEXT_DENSITY,
            bootstrap_context_density: DEFAULT_BOOTSTRAP_CONTEXT_DENSITY,
            patch_diff_threshold: DEFAULT_PATCH_DIFF_THRESHOLD,
            patch_diff_dense_fallback_density: DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY,
            sparse_encode_mode: FeatureFrameSparseEncodeMode::BucketedContext,
            sparse_mask_bucket_tokens: DEFAULT_SPARSE_MASK_BUCKET_TOKENS,
            patch_diff_refresh: PatchDiffRefreshConfig::default(),
            prewarm_shape_buckets: DEFAULT_PREWARM_SHAPE_BUCKETS,
            anyup_q_chunk_size: DEFAULT_ANYUP_CHUNK_SIZE,
            pca_update_every: DEFAULT_PCA_UPDATE_EVERY,
            pca_sample_window_frames: DEFAULT_PCA_SAMPLE_WINDOW_FRAMES,
            pca_min_sample_frames: DEFAULT_PCA_MIN_SAMPLE_FRAMES,
            pca_update_iterations: DEFAULT_PCA_UPDATE_ITERATIONS,
            high_res_pca_every: DEFAULT_HIGH_RES_PCA_EVERY,
            ttt_runtime: TttRuntimeStateConfig::default(),
            measure_stages: true,
            sync_measurements: false,
        }
    }
}

impl FeatureFrameViewerConfig {
    /// Returns the square image size after enforcing minimum size and patch multiple.
    pub fn pipeline_image_size(&self) -> usize {
        self.image_size
            .max(MIN_PIPELINE_IMAGE_SIZE)
            .div_ceil(PIPELINE_IMAGE_SIZE_MULTIPLE)
            * PIPELINE_IMAGE_SIZE_MULTIPLE
    }

    /// Converts a context density to a bounded token count.
    pub fn context_tokens(&self, dense_tokens: usize) -> usize {
        density_tokens(
            dense_tokens,
            self.context_density,
            DensityRound::Round,
            0.01,
        )
    }

    /// Converts the minimum context density to a bounded token count.
    pub fn min_context_tokens(&self, dense_tokens: usize) -> usize {
        density_tokens(
            dense_tokens,
            self.min_context_density,
            DensityRound::Ceil,
            0.0,
        )
    }

    /// Converts the first-frame bootstrap density to a bounded token count.
    pub fn bootstrap_context_tokens(&self, dense_tokens: usize) -> usize {
        density_tokens(
            dense_tokens,
            self.bootstrap_context_density,
            DensityRound::Ceil,
            0.0,
        )
    }

    /// Returns the quality value corresponding to the patch-diff threshold.
    pub fn patch_diff_quality(&self) -> f32 {
        (1.0 - self.patch_diff_threshold).clamp(0.0, 1.0)
    }

    /// True when sparse masks are widened into stable context buckets before encoding.
    pub const fn uses_bucketed_sparse_encode(&self) -> bool {
        matches!(
            self.sparse_encode_mode,
            FeatureFrameSparseEncodeMode::BucketedContext
        )
    }

    /// Shared rolling PCA update config for live low-resolution feature visualization.
    pub fn pca_update_config(&self) -> FeaturePcaUpdateConfig {
        let sample_window_frames = self.pca_sample_window_frames.max(2);
        FeaturePcaUpdateConfig {
            mode: FeaturePcaUpdateMode::RollingOja,
            every_n_frames: self.pca_update_every.max(1),
            warmup_frames: 0,
            min_tokens_per_update: 1,
            iterations_per_update: self.pca_update_iterations.max(1),
            sample_window_frames,
            min_sample_frames: self.pca_min_sample_frames.clamp(1, sample_window_frames),
        }
    }

    /// Shared measurement config for the feature-frame pipeline.
    pub fn measurement_config(&self) -> SparseJepaAnyUpPcaMeasurementConfig {
        if self.measure_stages {
            SparseJepaAnyUpPcaMeasurementConfig {
                enabled: true,
                sync_backend: self.sync_measurements,
            }
        } else {
            SparseJepaAnyUpPcaMeasurementConfig::disabled()
        }
    }

    /// Shared adaptive patch-diff sparsity config for the provided token grid.
    pub fn patch_diff_sparsity_config(
        &self,
        grid: TokenGridShape,
    ) -> SparseJepaPatchDiffSparsityConfig {
        patch_diff_sparsity_config(self, grid)
    }
}

#[derive(Clone, Copy)]
enum DensityRound {
    Floor,
    Round,
    Ceil,
}

fn density_tokens(
    dense_tokens: usize,
    density: f32,
    round: DensityRound,
    min_density: f32,
) -> usize {
    let density = density.clamp(min_density, 1.0);
    let tokens = match round {
        DensityRound::Floor => (dense_tokens as f32 * density).floor(),
        DensityRound::Round => (dense_tokens as f32 * density).round(),
        DensityRound::Ceil => (dense_tokens as f32 * density).ceil(),
    } as usize;
    tokens.clamp(1, dense_tokens.max(1))
}

/// Resolves the CLI threshold/quality pair into a clamped patch-diff threshold.
pub fn patch_diff_threshold_from_quality(threshold: f32, quality: Option<f32>) -> f32 {
    quality
        .map(|quality| 1.0 - quality.clamp(0.0, 1.0))
        .unwrap_or(threshold)
        .clamp(0.0, 1.0)
}

/// Distinct sparse masks for feature-cache writes and encoder context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureFrameSparseMasks {
    /// Exact tokens to overwrite in the low-resolution feature cache.
    pub write_mask: SparseTokenMask,
    /// Tokens supplied to the V-JEPA encoder. This may be wider than `write_mask`.
    pub encode_mask: SparseTokenMask,
}

impl FeatureFrameSparseMasks {
    /// Uses the same mask for encoder context and cache writes.
    pub fn same(mask: SparseTokenMask) -> Self {
        Self {
            write_mask: mask.clone(),
            encode_mask: mask,
        }
    }
}

/// Promotes high-density sparse masks to a dense ordered mask for faster dense-path updates.
pub fn patch_diff_dense_fallback(
    mask: SparseTokenMask,
    grid: TokenGridShape,
    fallback_density: f32,
) -> SparseTokenMask {
    let fallback_tokens = (grid.len() as f32 * fallback_density.clamp(0.0, 1.0)).ceil() as usize;
    if mask.len() == grid.len()
        || (fallback_tokens < grid.len() && mask.len() >= fallback_tokens.max(1))
    {
        SparseTokenMask::all(grid.len())
    } else {
        mask
    }
}

/// Pads sparse masks to stable token-width buckets while preserving all selected tokens.
pub fn bucket_sparse_mask(
    mask: SparseTokenMask,
    grid: TokenGridShape,
    bucket_tokens: usize,
) -> SparseTokenMask {
    if bucket_tokens <= 1 || grid.len() < 256 || mask.is_dense_ordered() {
        return mask;
    }
    let effective_bucket = bucket_tokens.min((grid.len() / 4).max(1));
    mask.padded_to_multiple(effective_bucket)
}

/// Applies dense fallback and sparse encode widening to a threshold-selected patch-diff mask.
pub fn finalize_patch_diff_masks(
    mask: SparseTokenMask,
    grid: TokenGridShape,
    config: &FeatureFrameViewerConfig,
) -> FeatureFrameSparseMasks {
    let write_mask =
        patch_diff_dense_fallback(mask, grid, config.patch_diff_dense_fallback_density);
    let encode_mask = match config.sparse_encode_mode {
        FeatureFrameSparseEncodeMode::Exact => write_mask.clone(),
        FeatureFrameSparseEncodeMode::BucketedContext => {
            bucket_sparse_mask(write_mask.clone(), grid, config.sparse_mask_bucket_tokens)
        }
    };
    FeatureFrameSparseMasks {
        write_mask,
        encode_mask,
    }
}

/// Applies dense fallback and encode widening, returning only the encoder mask.
pub fn finalize_patch_diff_mask(
    mask: SparseTokenMask,
    grid: TokenGridShape,
    config: &FeatureFrameViewerConfig,
) -> SparseTokenMask {
    finalize_patch_diff_masks(mask, grid, config).encode_mask
}

/// Returns the bucket widths that should be warmed before running a live bucketed viewer.
pub fn shape_prewarm_masks(
    grid: TokenGridShape,
    config: &FeatureFrameViewerConfig,
) -> Vec<SparseTokenMask> {
    if !config.uses_bucketed_sparse_encode() {
        return Vec::new();
    }
    let mut masks = Vec::new();
    let mut seen = Vec::<(usize, bool)>::new();
    let mut push_mask = |mask: SparseTokenMask| {
        let mask = finalize_patch_diff_mask(mask, grid, config);
        let key = (mask.len(), mask.is_dense_ordered());
        if !seen.contains(&key) {
            seen.push(key);
            masks.push(mask);
        }
    };

    let bucket = if !config.uses_bucketed_sparse_encode()
        || config.sparse_mask_bucket_tokens <= 1
        || grid.len() < 256
    {
        0
    } else {
        config
            .sparse_mask_bucket_tokens
            .min((grid.len() / 4).max(1))
    };
    if bucket > 0 {
        let mut width = bucket;
        while width < grid.len() {
            push_mask(SparseTokenMask::evenly_spaced(grid.len(), width));
            width = width.saturating_add(bucket);
        }
    }
    push_mask(SparseTokenMask::all(grid.len()));
    masks
}

/// Builds the adaptive patch-diff sparsity policy used by live feature-frame pipelines.
pub fn patch_diff_sparsity_config(
    config: &FeatureFrameViewerConfig,
    grid: TokenGridShape,
) -> SparseJepaPatchDiffSparsityConfig {
    let max_context_tokens = config.context_tokens(grid.len());
    let min_context_tokens = config
        .min_context_tokens(grid.len())
        .min(max_context_tokens);
    let target_tokens = grid.len().saturating_sub(max_context_tokens).max(1);
    SparseJepaPatchDiffSparsityConfig::adaptive_threshold(
        config.patch_diff_threshold,
        min_context_tokens,
        max_context_tokens,
        target_tokens,
    )
}

/// True when patch-diff policy can skip scoring and directly use a dense ordered mask.
pub fn patch_diff_can_use_dense_fast_path(
    config: &SparseJepaPatchDiffSparsityConfig,
    grid: TokenGridShape,
) -> bool {
    config.threshold <= 0.0
        && config.dilation == 0
        && config.allow_full_context
        && config.context_tokens >= grid.len()
}

/// Center-prior bootstrap mask used before a previous frame exists.
pub fn center_prior_mask(grid: TokenGridShape, context_tokens: usize) -> Result<SparseTokenMask> {
    let center_row = grid.height.saturating_sub(1) as f32 * 0.5;
    let center_col = grid.width.saturating_sub(1) as f32 * 0.5;
    let mut scores = Vec::with_capacity(grid.len());
    for row in 0..grid.height {
        for col in 0..grid.width {
            let dr = row as f32 - center_row;
            let dc = col as f32 - center_col;
            let dist = dr * dr + dc * dc;
            scores.push((coords_to_token_index(0, row, col, grid), dist));
        }
    }
    scores.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    SparseTokenMask::new(
        scores
            .into_iter()
            .take(context_tokens.max(1).min(grid.len()))
            .map(|(index, _)| index)
            .collect(),
        grid.len(),
    )
}

/// Per-frame global lighting statistics for RGBA patch-diff scoring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbaPatchDiffFrameStats {
    /// Mean RGB delta in normalized `[0, 1]` space.
    pub channel_delta: [f32; 3],
    /// Mean luma delta in normalized `[0, 1]` space.
    pub luma_delta: f32,
    /// True when patch scores should subtract a global lighting shift.
    pub compensate_global_lighting: bool,
}

/// Computes global lighting statistics for two RGBA frames.
pub fn rgba_patch_diff_frame_stats(prev: &[u8], current: &[u8]) -> RgbaPatchDiffFrameStats {
    let pixels = (prev.len().min(current.len()) / 4).max(1);
    let mut channel_delta = [0.0f32; 3];
    let mut luma_delta = 0.0f32;
    let mut raw_abs = 0.0f32;
    for index in 0..pixels {
        let offset = index * 4;
        let prev_rgb = rgba_rgb(prev, offset);
        let current_rgb = rgba_rgb(current, offset);
        let mut pixel_abs = 0.0f32;
        for channel in 0..3 {
            let delta = current_rgb[channel] - prev_rgb[channel];
            channel_delta[channel] += delta;
            pixel_abs += delta.abs();
        }
        raw_abs += pixel_abs / 3.0;
        luma_delta += rgb_luma(current_rgb) - rgb_luma(prev_rgb);
    }
    for channel in &mut channel_delta {
        *channel /= pixels as f32;
    }
    luma_delta /= pixels as f32;

    let mut residual_abs = 0.0f32;
    for index in 0..pixels {
        let offset = index * 4;
        let prev_rgb = rgba_rgb(prev, offset);
        let current_rgb = rgba_rgb(current, offset);
        let mut pixel_abs = 0.0f32;
        for channel in 0..3 {
            let delta = current_rgb[channel] - prev_rgb[channel] - channel_delta[channel];
            pixel_abs += delta.abs();
        }
        residual_abs += pixel_abs / 3.0;
    }
    let raw_abs = raw_abs / pixels as f32;
    let residual_abs = residual_abs / pixels as f32;
    let compensate_global_lighting = raw_abs > 1.0e-4 && residual_abs <= raw_abs * 0.35;

    RgbaPatchDiffFrameStats {
        channel_delta,
        luma_delta,
        compensate_global_lighting,
    }
}

/// Computes patch-diff scores for a square RGBA frame pair.
pub fn patch_diff_scores_from_rgba(
    prev: &[u8],
    current: &[u8],
    width: usize,
    patch_size: usize,
    grid: TokenGridShape,
) -> Result<Vec<f32>> {
    ensure!(
        grid.depth == 1,
        "RGBA patch-diff expects a single-frame grid"
    );
    ensure!(patch_size > 0, "RGBA patch-diff patch size must be nonzero");
    ensure!(
        width == grid.width * patch_size,
        "RGBA patch-diff width must match the token grid"
    );
    let expected_len = width * grid.height * patch_size * 4;
    ensure!(
        prev.len() >= expected_len && current.len() >= expected_len,
        "RGBA patch-diff buffers are smaller than the token grid"
    );

    let stats = rgba_patch_diff_frame_stats(prev, current);
    let mut scores = vec![0.0f32; grid.len()];
    for row in 0..grid.height {
        for col in 0..grid.width {
            scores[coords_to_token_index(0, row, col, grid)] =
                rgba_patch_diff_score(prev, current, width, patch_size, row, col, stats);
        }
    }
    Ok(scores)
}

/// Samples an RGBA frame pair to decide whether scoring can route directly to dense.
pub fn patch_diff_sampled_dense_fast_path_from_rgba(
    prev: &[u8],
    current: &[u8],
    width: usize,
    patch_size: usize,
    grid: TokenGridShape,
    config: &SparseJepaPatchDiffSparsityConfig,
    fallback_density: f32,
) -> bool {
    let fallback_density = fallback_density.clamp(0.0, 1.0);
    if fallback_density >= 1.0
        || config.threshold <= 0.0
        || config.dilation != 0
        || !config.allow_full_context
        || config.context_tokens < grid.len()
    {
        return false;
    }
    let stats = rgba_patch_diff_frame_stats(prev, current);
    let stride = ((grid.len() as f32 / PATCH_DIFF_DENSE_FAST_PATH_MAX_SAMPLES as f32)
        .sqrt()
        .ceil() as usize)
        .max(1);
    let mut sampled = 0usize;
    let mut active = 0usize;
    for row in (0..grid.height).step_by(stride) {
        for col in (0..grid.width).step_by(stride) {
            sampled += 1;
            let score = rgba_patch_diff_score(prev, current, width, patch_size, row, col, stats);
            if score >= config.threshold {
                active += 1;
            }
        }
    }
    let required = (sampled as f32 * fallback_density).ceil() as usize;
    sampled > 0 && active >= required.max(1)
}

fn rgba_patch_diff_score(
    prev: &[u8],
    current: &[u8],
    width: usize,
    patch_size: usize,
    row: usize,
    col: usize,
    stats: RgbaPatchDiffFrameStats,
) -> f32 {
    let mut diff_sum = 0.0f32;
    for y in row * patch_size..(row + 1) * patch_size {
        for x in col * patch_size..(col + 1) * patch_size {
            let offset = (y * width + x) * 4;
            diff_sum += rgba_patch_diff_pixel_score(prev, current, offset, stats);
        }
    }
    diff_sum / (patch_size * patch_size) as f32
}

fn rgba_patch_diff_pixel_score(
    prev: &[u8],
    current: &[u8],
    offset: usize,
    stats: RgbaPatchDiffFrameStats,
) -> f32 {
    let prev_rgb = rgba_rgb(prev, offset);
    let current_rgb = rgba_rgb(current, offset);
    let mut rgb_abs = 0.0f32;
    for channel in 0..3 {
        let mut delta = current_rgb[channel] - prev_rgb[channel];
        if stats.compensate_global_lighting {
            delta -= stats.channel_delta[channel];
        }
        rgb_abs += delta.abs();
    }
    let rgb_abs = rgb_abs / 3.0;

    let prev_luma = rgb_luma(prev_rgb);
    let current_luma = rgb_luma(current_rgb);
    let mut luma_delta = current_luma - prev_luma;
    if stats.compensate_global_lighting {
        luma_delta -= stats.luma_delta;
    }
    let luma_abs = luma_delta.abs();
    let relative_luma = (luma_abs / prev_luma.max(current_luma).max(0.10)).min(1.0) * 0.25;
    let chroma = rgb_chroma_diff(prev_rgb, current_rgb) * 0.5;
    rgb_abs.max(luma_abs).max(relative_luma).max(chroma)
}

fn rgba_rgb(rgba: &[u8], offset: usize) -> [f32; 3] {
    [
        rgba[offset] as f32 / 255.0,
        rgba[offset + 1] as f32 / 255.0,
        rgba[offset + 2] as f32 / 255.0,
    ]
}

fn rgb_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn rgb_chroma_diff(prev: [f32; 3], current: [f32; 3]) -> f32 {
    let prev_chroma = rgb_chroma(prev);
    let current_chroma = rgb_chroma(current);
    ((current_chroma[0] - prev_chroma[0]).abs()
        + (current_chroma[1] - prev_chroma[1]).abs()
        + (current_chroma[2] - prev_chroma[2]).abs())
        / 3.0
}

fn rgb_chroma(rgb: [f32; 3]) -> [f32; 3] {
    let denom = (rgb[0] + rgb[1] + rgb[2]).max(0.10);
    [rgb[0] / denom, rgb[1] / denom, rgb[2] / denom]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_matches_live_viewer_expectations() {
        let config = FeatureFrameViewerConfig::default();
        assert_eq!(config.pipeline_image_size(), DEFAULT_IMAGE_SIZE);
        assert_eq!(config.min_context_tokens(1024), 1);
        assert!((config.patch_diff_quality() - DEFAULT_PATCH_DIFF_QUALITY).abs() <= f32::EPSILON);
        assert_eq!(
            config.sparse_encode_mode,
            FeatureFrameSparseEncodeMode::BucketedContext
        );
        assert!(config.patch_diff_refresh.enabled);
        assert!(config.patch_diff_refresh.subthreshold_enabled);
        assert!(config.prewarm_shape_buckets);
        assert!(
            config.pca_update_iterations > 1,
            "live PCA should converge from the identity basis quickly enough for viewer/docs output"
        );
    }

    #[test]
    fn image_size_rounding_preserves_patch_alignment() {
        let config = FeatureFrameViewerConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE + 1,
            ..FeatureFrameViewerConfig::default()
        };
        assert_eq!(
            config.pipeline_image_size() % PIPELINE_IMAGE_SIZE_MULTIPLE,
            0
        );
        assert!(config.pipeline_image_size() > MIN_PIPELINE_IMAGE_SIZE);
    }

    #[test]
    fn bucketed_masks_keep_writes_exact_and_widen_encode_context() {
        let grid = TokenGridShape::new(1, 32, 32);
        let changed =
            SparseTokenMask::new(vec![coords_to_token_index(0, 15, 16, grid)], grid.len())
                .expect("mask");

        let masks =
            finalize_patch_diff_masks(changed.clone(), grid, &FeatureFrameViewerConfig::default());

        assert_eq!(masks.write_mask, changed);
        assert_eq!(masks.encode_mask.len(), DEFAULT_SPARSE_MASK_BUCKET_TOKENS);
        assert!(
            masks
                .encode_mask
                .indices()
                .contains(&coords_to_token_index(0, 15, 16, grid))
        );
    }

    #[test]
    fn prewarm_masks_cover_bucket_widths_once() {
        let grid = TokenGridShape::new(1, 32, 32);
        let masks = shape_prewarm_masks(grid, &FeatureFrameViewerConfig::default());
        let widths: Vec<_> = masks.iter().map(SparseTokenMask::len).collect();

        assert_eq!(widths, vec![256, 512, 1024]);
        assert!(masks.last().expect("dense mask").is_dense_ordered());
    }

    #[test]
    fn rgba_patch_diff_ignores_uniform_lighting_shift() {
        let grid = TokenGridShape::new(1, 4, 4);
        let prev = vec![80u8; 64 * 64 * 4];
        let current = vec![110u8; 64 * 64 * 4];

        let scores = patch_diff_scores_from_rgba(&prev, &current, 64, 16, grid).expect("scores");

        assert!(
            scores
                .iter()
                .all(|score| *score < DEFAULT_PATCH_DIFF_THRESHOLD)
        );
    }

    #[test]
    fn subthreshold_patch_diff_accumulates_slow_motion() {
        let grid = TokenGridShape::new(1, 4, 4);
        let slow_index = coords_to_token_index(0, 2, 1, grid);
        let config = FeatureFrameViewerConfig {
            context_density: 0.25,
            min_context_density: 0.0,
            patch_diff_threshold: 0.10,
            patch_diff_dense_fallback_density: 1.0,
            sparse_encode_mode: FeatureFrameSparseEncodeMode::Exact,
            patch_diff_refresh: PatchDiffRefreshConfig {
                subthreshold_decay: 1.0,
                subthreshold_trigger: 1.0,
                subthreshold_max_density: 0.25,
                age_refresh_enabled: false,
                blue_noise_enabled: false,
                max_extra_density: 0.25,
                ..PatchDiffRefreshConfig::default()
            },
            ..FeatureFrameViewerConfig::default()
        };
        let sparsity = patch_diff_sparsity_config(&config, grid);
        let mut state = PatchDiffRefreshState::default();
        let mut scores = vec![0.0f32; grid.len()];
        scores[0] = 0.09;
        scores[slow_index] = 0.04;

        let first = state
            .masks_from_scores(scores.clone(), grid, &sparsity, &config)
            .expect("first mask");
        assert!(!first.write_mask.indices().contains(&slow_index));

        let second = state
            .masks_from_scores(scores.clone(), grid, &sparsity, &config)
            .expect("second mask");
        assert!(!second.write_mask.indices().contains(&slow_index));

        let third = state
            .masks_from_scores(scores, grid, &sparsity, &config)
            .expect("third mask");
        assert!(
            third.write_mask.indices().contains(&slow_index),
            "repeated below-threshold motion should eventually refresh the token"
        );
        assert_eq!(state.subthreshold_residual(slow_index), Some(0.0));
        assert_eq!(state.age_frames(slow_index), Some(0));
    }

    #[test]
    fn subthreshold_refresh_fills_only_unused_context_budget() {
        let grid = TokenGridShape::new(1, 4, 4);
        let config = FeatureFrameViewerConfig {
            context_density: 2.0 / 16.0,
            min_context_density: 0.0,
            patch_diff_threshold: 0.10,
            patch_diff_dense_fallback_density: 1.0,
            sparse_encode_mode: FeatureFrameSparseEncodeMode::Exact,
            patch_diff_refresh: PatchDiffRefreshConfig {
                subthreshold_decay: 1.0,
                subthreshold_trigger: 0.5,
                subthreshold_max_density: 1.0,
                age_refresh_enabled: false,
                blue_noise_enabled: false,
                max_extra_density: 1.0,
                ..PatchDiffRefreshConfig::default()
            },
            ..FeatureFrameViewerConfig::default()
        };
        let sparsity = patch_diff_sparsity_config(&config, grid);
        let mut state = PatchDiffRefreshState::default();
        let mut scores = vec![0.06f32; grid.len()];
        scores[0] = 0.09;

        let output = state
            .masks_from_scores(scores, grid, &sparsity, &config)
            .expect("mask");

        assert_eq!(
            output.write_mask.len(),
            2,
            "refresh tokens must not exceed the configured context budget"
        );
    }

    #[test]
    fn age_priority_refresh_selects_stale_tokens() {
        let grid = TokenGridShape::new(1, 4, 4);
        let config = FeatureFrameViewerConfig {
            context_density: 0.50,
            min_context_density: 0.0,
            patch_diff_threshold: 0.10,
            patch_diff_dense_fallback_density: 1.0,
            sparse_encode_mode: FeatureFrameSparseEncodeMode::Exact,
            patch_diff_refresh: PatchDiffRefreshConfig {
                subthreshold_enabled: false,
                age_refresh_interval_frames: 2,
                age_refresh_max_density: 0.25,
                blue_noise_enabled: false,
                max_extra_density: 0.25,
                ..PatchDiffRefreshConfig::default()
            },
            ..FeatureFrameViewerConfig::default()
        };
        let sparsity = patch_diff_sparsity_config(&config, grid);
        let mut state = PatchDiffRefreshState::default();
        let mut scores = vec![0.0f32; grid.len()];
        scores[0] = 0.09;

        let first = state
            .masks_from_scores(scores.clone(), grid, &sparsity, &config)
            .expect("first mask");
        assert_eq!(first.write_mask.len(), 1);

        let second = state
            .masks_from_scores(scores, grid, &sparsity, &config)
            .expect("second mask");
        assert!(
            second.write_mask.len() > first.write_mask.len(),
            "stale age-priority refresh should add bounded cache writes"
        );
    }

    #[test]
    fn blue_noise_refresh_is_deterministic_for_same_state() {
        let grid = TokenGridShape::new(1, 4, 4);
        let config = FeatureFrameViewerConfig {
            context_density: 0.50,
            min_context_density: 0.0,
            patch_diff_threshold: 0.10,
            patch_diff_dense_fallback_density: 1.0,
            sparse_encode_mode: FeatureFrameSparseEncodeMode::Exact,
            patch_diff_refresh: PatchDiffRefreshConfig {
                subthreshold_enabled: false,
                age_refresh_enabled: false,
                blue_noise_refresh_density: 0.25,
                max_extra_density: 0.25,
                blue_noise_seed: 7,
                ..PatchDiffRefreshConfig::default()
            },
            ..FeatureFrameViewerConfig::default()
        };
        let sparsity = patch_diff_sparsity_config(&config, grid);
        let mut left = PatchDiffRefreshState::default();
        let mut right = PatchDiffRefreshState::default();
        let scores = vec![0.0f32; grid.len()];

        let left_mask = left
            .masks_from_scores(scores.clone(), grid, &sparsity, &config)
            .expect("left mask");
        let right_mask = right
            .masks_from_scores(scores, grid, &sparsity, &config)
            .expect("right mask");

        assert_eq!(left_mask.write_mask, right_mask.write_mask);
    }
}
