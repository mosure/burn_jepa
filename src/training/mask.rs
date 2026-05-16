use crate::{
    JepaSampleMetadata, SparseImageTokenGrid, SparseJepaAutogazeSparsityConfig,
    SparseJepaPatchDiffSparsityConfig, SparseJepaSparsityDriverConfig, SparseTokenMask,
    TokenGridShape, VJepaConfig, resolve_sparsity_driver_masks,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingMaskConfig {
    KeepRatio {
        context_keep_ratio: f32,
    },
    FullFrame {
        target_tokens: usize,
    },
    RandomSparse {
        context_tokens: usize,
        target_tokens: usize,
        #[serde(default)]
        seed: u64,
    },
    TemporalUniformSparse {
        context_tokens: usize,
        target_tokens: usize,
    },
    AutogazeSparse {
        image_grid: TrainingImageTokenGrid,
        context_tokens: usize,
        target_tokens: usize,
        #[serde(
            default,
            skip_serializing_if = "TrainingAutogazeTokenSource::is_default"
        )]
        source: TrainingAutogazeTokenSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        frame_tokens: Option<Vec<Vec<usize>>>,
        #[serde(default)]
        dilation: usize,
    },
    PatchDiff {
        threshold: f32,
        context_tokens: usize,
        target_tokens: usize,
        #[serde(default)]
        dilation: usize,
    },
    PrecomputedMasks {
        context_indices: Vec<usize>,
        target_indices: Vec<usize>,
    },
    ManifestPrecomputedMasks,
}

impl Default for TrainingMaskConfig {
    fn default() -> Self {
        Self::KeepRatio {
            context_keep_ratio: 0.75,
        }
    }
}

impl TrainingMaskConfig {
    pub fn keep_ratio(context_keep_ratio: f32) -> Self {
        Self::KeepRatio { context_keep_ratio }
    }

    pub fn full_frame(target_tokens: usize) -> Self {
        Self::FullFrame { target_tokens }
    }

    pub fn autogaze_center_prior(
        image_grid: TrainingImageTokenGrid,
        context_tokens: usize,
        target_tokens: usize,
    ) -> Self {
        Self::AutogazeSparse {
            image_grid,
            context_tokens,
            target_tokens,
            source: TrainingAutogazeTokenSource::default(),
            frame_tokens: None,
            dilation: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::KeepRatio { context_keep_ratio } => ensure!(
                context_keep_ratio.is_finite()
                    && *context_keep_ratio > 0.0
                    && *context_keep_ratio < 1.0,
                "training.mask.keep_ratio context_keep_ratio must be finite and in (0, 1)"
            ),
            Self::FullFrame { target_tokens } => ensure!(
                *target_tokens > 0,
                "training.mask.full_frame target_tokens must be nonzero"
            ),
            Self::RandomSparse {
                context_tokens,
                target_tokens,
                ..
            }
            | Self::TemporalUniformSparse {
                context_tokens,
                target_tokens,
            } => {
                ensure!(
                    *context_tokens > 0,
                    "training.mask sparse context_tokens must be nonzero"
                );
                ensure!(
                    *target_tokens > 0,
                    "training.mask sparse target_tokens must be nonzero"
                );
            }
            Self::AutogazeSparse {
                image_grid,
                context_tokens,
                target_tokens,
                source,
                frame_tokens,
                ..
            } => {
                ensure!(
                    !image_grid.is_empty(),
                    "training.mask.autogaze_sparse image_grid must be non-empty"
                );
                ensure!(
                    *context_tokens > 0,
                    "training.mask.autogaze_sparse context_tokens must be nonzero"
                );
                ensure!(
                    *target_tokens > 0,
                    "training.mask.autogaze_sparse target_tokens must be nonzero"
                );
                source.validate(*image_grid)?;
                if let Some(frame_tokens) = frame_tokens {
                    validate_frame_tokens(image_grid.len(), frame_tokens)?;
                }
            }
            Self::PatchDiff {
                threshold,
                context_tokens,
                target_tokens,
                ..
            } => {
                ensure!(
                    threshold.is_finite() && *threshold >= 0.0,
                    "training.mask.patch_diff threshold must be finite and non-negative"
                );
                ensure!(
                    *context_tokens > 0,
                    "training.mask.patch_diff context_tokens must be nonzero"
                );
                ensure!(
                    *target_tokens > 0,
                    "training.mask.patch_diff target_tokens must be nonzero"
                );
            }
            Self::PrecomputedMasks {
                context_indices,
                target_indices,
            } => {
                ensure!(
                    !context_indices.is_empty(),
                    "training.mask.precomputed_masks context_indices must be non-empty"
                );
                ensure!(
                    !target_indices.is_empty(),
                    "training.mask.precomputed_masks target_indices must be non-empty"
                );
            }
            Self::ManifestPrecomputedMasks => {}
        }
        Ok(())
    }

    pub fn resolve_masks<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
        grid: TokenGridShape,
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        self.validate()?;
        ensure!(
            !matches!(self, Self::ManifestPrecomputedMasks),
            "training.mask.manifest_precomputed_masks requires sample metadata"
        );
        let driver = self.to_sparsity_driver(grid, model_config)?;
        resolve_sparsity_driver_masks(&driver, video, model_config, grid)
    }

    pub fn resolve_masks_with_metadata<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
        grid: TokenGridShape,
        metadata: &[JepaSampleMetadata],
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        if matches!(self, Self::ManifestPrecomputedMasks) {
            let (context_mask, target_mask) = manifest_precomputed_masks(metadata, grid)?;
            return Ok((context_mask, target_mask));
        }
        self.resolve_masks(video, model_config, grid)
    }

    fn to_sparsity_driver(
        &self,
        grid: TokenGridShape,
        model_config: &VJepaConfig,
    ) -> Result<SparseJepaSparsityDriverConfig> {
        Ok(match self {
            Self::KeepRatio { context_keep_ratio } => SparseJepaSparsityDriverConfig::KeepRatio {
                context_keep_ratio: *context_keep_ratio,
            },
            Self::FullFrame { target_tokens } => SparseJepaSparsityDriverConfig::FullFrame {
                target_tokens: *target_tokens,
            },
            Self::RandomSparse {
                context_tokens,
                target_tokens,
                seed,
            } => {
                let (context_mask, target_mask) =
                    random_sparse_masks(grid, *context_tokens, *target_tokens, *seed)?;
                SparseJepaSparsityDriverConfig::PrecomputedMasks {
                    context_mask,
                    target_mask,
                }
            }
            Self::TemporalUniformSparse {
                context_tokens,
                target_tokens,
            } => {
                let (context_mask, target_mask) =
                    temporal_uniform_sparse_masks(grid, *context_tokens, *target_tokens)?;
                SparseJepaSparsityDriverConfig::PrecomputedMasks {
                    context_mask,
                    target_mask,
                }
            }
            Self::AutogazeSparse {
                image_grid,
                context_tokens,
                target_tokens,
                source,
                frame_tokens,
                dilation,
            } => {
                let frame_tokens = resolve_autogaze_frame_tokens(
                    grid,
                    model_config.tubelet_size,
                    *context_tokens,
                    source,
                    frame_tokens.as_deref(),
                )?;
                SparseJepaSparsityDriverConfig::AutogazeSparse(SparseJepaAutogazeSparsityConfig {
                    image_grid: (*image_grid).into(),
                    frame_tokens,
                    context_tokens: *context_tokens,
                    target_tokens: *target_tokens,
                    dilation: *dilation,
                })
            }
            Self::PatchDiff {
                threshold,
                context_tokens,
                target_tokens,
                dilation,
            } => SparseJepaSparsityDriverConfig::PatchDiff(SparseJepaPatchDiffSparsityConfig {
                threshold: *threshold,
                context_tokens: *context_tokens,
                target_tokens: *target_tokens,
                dilation: *dilation,
                min_context_tokens: 1,
                fill_to_context_tokens: true,
                allow_full_context: false,
            }),
            Self::PrecomputedMasks {
                context_indices,
                target_indices,
            } => SparseJepaSparsityDriverConfig::PrecomputedMasks {
                context_mask: SparseTokenMask::new(context_indices.clone(), grid.len())?,
                target_mask: SparseTokenMask::new(target_indices.clone(), grid.len())?,
            },
            Self::ManifestPrecomputedMasks => {
                unreachable!("manifest precomputed masks are resolved before driver conversion")
            }
        })
    }
}

fn manifest_precomputed_masks(
    metadata: &[JepaSampleMetadata],
    grid: TokenGridShape,
) -> Result<(SparseTokenMask, SparseTokenMask)> {
    ensure!(
        !metadata.is_empty(),
        "training.mask.manifest_precomputed_masks requires batch metadata"
    );
    let first_context = metadata[0]
        .precomputed_context_indices
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("manifest row is missing precomputed_context_indices"))?;
    let first_target = metadata[0]
        .precomputed_target_indices
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("manifest row is missing precomputed_target_indices"))?;
    ensure!(
        metadata.iter().all(|row| {
            row.precomputed_context_indices.as_ref() == Some(first_context)
                && row.precomputed_target_indices.as_ref() == Some(first_target)
        }),
        "manifest precomputed masks vary within this batch; use batch_size=1 or group windows by mask"
    );
    Ok((
        SparseTokenMask::new(first_context.clone(), grid.len())?,
        SparseTokenMask::new(first_target.clone(), grid.len())?,
    ))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingAutogazeTokenSource {
    CenterPrior {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_per_frame: Option<usize>,
    },
    FrameTokens {
        frame_tokens: Vec<Vec<usize>>,
    },
}

impl Default for TrainingAutogazeTokenSource {
    fn default() -> Self {
        Self::CenterPrior {
            tokens_per_frame: None,
        }
    }
}

impl TrainingAutogazeTokenSource {
    pub fn is_default(&self) -> bool {
        matches!(
            self,
            Self::CenterPrior {
                tokens_per_frame: None
            }
        )
    }

    fn validate(&self, image_grid: TrainingImageTokenGrid) -> Result<()> {
        match self {
            Self::CenterPrior { tokens_per_frame } => {
                if let Some(tokens_per_frame) = tokens_per_frame {
                    ensure!(
                        *tokens_per_frame > 0,
                        "training.mask.autogaze_sparse.source.center_prior tokens_per_frame must be nonzero"
                    );
                    ensure!(
                        *tokens_per_frame <= image_grid.len(),
                        "training.mask.autogaze_sparse.source.center_prior tokens_per_frame exceeds image grid"
                    );
                }
                Ok(())
            }
            Self::FrameTokens { frame_tokens } => {
                validate_frame_tokens(image_grid.len(), frame_tokens)
            }
        }
    }
}

pub fn center_prior_frame_tokens(
    grid: TokenGridShape,
    tubelet_size: usize,
    context_tokens: usize,
    tokens_per_frame: Option<usize>,
) -> Vec<Vec<usize>> {
    let image_tokens = grid.height * grid.width;
    let frames = grid.depth.max(1) * tubelet_size.max(1);
    let tokens_per_frame = tokens_per_frame
        .unwrap_or_else(|| context_tokens.div_ceil(frames))
        .max(1)
        .min(image_tokens.max(1));
    let center_row = (grid.height.saturating_sub(1)) as f32 * 0.5;
    let center_col = (grid.width.saturating_sub(1)) as f32 * 0.5;
    let mut ranked = (0..image_tokens)
        .map(|index| {
            let row = index / grid.width.max(1);
            let col = index % grid.width.max(1);
            let dr = row as f32 - center_row;
            let dc = col as f32 - center_col;
            (dr.mul_add(dr, dc * dc), index)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let frame_tokens = ranked
        .iter()
        .take(tokens_per_frame)
        .map(|&(_, index)| index)
        .collect::<Vec<_>>();
    vec![frame_tokens; frames]
}

fn resolve_autogaze_frame_tokens(
    grid: TokenGridShape,
    tubelet_size: usize,
    context_tokens: usize,
    source: &TrainingAutogazeTokenSource,
    frame_tokens: Option<&[Vec<usize>]>,
) -> Result<Vec<Vec<usize>>> {
    match (source, frame_tokens) {
        (source, Some(frame_tokens)) if source.is_default() => Ok(frame_tokens.to_vec()),
        (TrainingAutogazeTokenSource::CenterPrior { tokens_per_frame }, _) => Ok(
            center_prior_frame_tokens(grid, tubelet_size, context_tokens, *tokens_per_frame),
        ),
        (TrainingAutogazeTokenSource::FrameTokens { frame_tokens }, _) => Ok(frame_tokens.clone()),
    }
}

fn validate_frame_tokens(image_tokens: usize, frame_tokens: &[Vec<usize>]) -> Result<()> {
    ensure!(
        !frame_tokens.is_empty(),
        "training.mask.autogaze_sparse frame tokens must be non-empty"
    );
    ensure!(
        frame_tokens
            .iter()
            .flatten()
            .all(|&token| token < image_tokens),
        "training.mask.autogaze_sparse frame tokens contain an image token outside image_grid"
    );
    Ok(())
}

fn random_sparse_masks(
    grid: TokenGridShape,
    context_tokens: usize,
    target_tokens: usize,
    seed: u64,
) -> Result<(SparseTokenMask, SparseTokenMask)> {
    ensure!(
        grid.len() > 1,
        "random sparse mask requires at least two dense tokens"
    );
    let context_tokens = context_tokens.max(1).min(grid.len() - 1);
    let target_tokens = target_tokens
        .max(1)
        .min(grid.len().saturating_sub(context_tokens));
    let mut keyed = (0..grid.len())
        .map(|index| (splitmix64(seed ^ index as u64), index))
        .collect::<Vec<_>>();
    keyed.sort_unstable_by_key(|&(key, index)| (key, index));
    let context = keyed
        .iter()
        .take(context_tokens)
        .map(|&(_, index)| index)
        .collect::<Vec<_>>();
    let target = keyed
        .iter()
        .skip(context_tokens)
        .take(target_tokens)
        .map(|&(_, index)| index)
        .collect::<Vec<_>>();
    Ok((
        SparseTokenMask::new(context, grid.len())?,
        SparseTokenMask::new(target, grid.len())?,
    ))
}

fn temporal_uniform_sparse_masks(
    grid: TokenGridShape,
    context_tokens: usize,
    target_tokens: usize,
) -> Result<(SparseTokenMask, SparseTokenMask)> {
    ensure!(
        grid.len() > 1,
        "temporal uniform sparse mask requires at least two dense tokens"
    );
    let context_tokens = context_tokens.max(1).min(grid.len() - 1);
    let unblocked = vec![false; grid.len()];
    let context = temporal_uniform_indices(grid, context_tokens, &unblocked);
    let mut blocked = vec![false; grid.len()];
    for &index in &context {
        blocked[index] = true;
    }
    let target_tokens = target_tokens
        .max(1)
        .min(grid.len().saturating_sub(context.len()).max(1));
    let target = temporal_uniform_indices(grid, target_tokens, &blocked);
    Ok((
        SparseTokenMask::new(context, grid.len())?,
        SparseTokenMask::new(target, grid.len())?,
    ))
}

fn temporal_uniform_indices(grid: TokenGridShape, budget: usize, blocked: &[bool]) -> Vec<usize> {
    let budget = budget.max(1).min(grid.len());
    let tubelets = grid.depth.max(1);
    let frame_tokens = grid.tokens_per_frame().max(1);
    let mut selected = vec![false; grid.len()];
    let mut indices = Vec::with_capacity(budget);
    for tubelet in 0..tubelets {
        let available = (0..frame_tokens)
            .map(|spatial| tubelet * frame_tokens + spatial)
            .filter(|&index| index < grid.len() && !blocked.get(index).copied().unwrap_or(true))
            .collect::<Vec<_>>();
        if available.is_empty() {
            continue;
        }
        let quota = budget / tubelets + usize::from(tubelet < budget % tubelets);
        let quota = quota.min(available.len());
        for index in evenly_spaced_from_available(&available, quota) {
            if !selected[index] {
                selected[index] = true;
                indices.push(index);
            }
        }
    }
    if indices.len() < budget {
        let available = (0..grid.len())
            .filter(|&index| !blocked[index] && !selected[index])
            .collect::<Vec<_>>();
        for index in evenly_spaced_from_available(&available, budget - indices.len()) {
            if !selected[index] {
                selected[index] = true;
                indices.push(index);
            }
        }
    }
    indices
}

fn evenly_spaced_from_available(available: &[usize], keep: usize) -> Vec<usize> {
    let keep = keep.min(available.len());
    if keep == 0 {
        return Vec::new();
    }
    if keep == available.len() {
        return available.to_vec();
    }
    let last = available.len().saturating_sub(1);
    (0..keep)
        .map(|i| available[((i * last) + (keep / 2)) / keep.max(1)])
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingImageTokenGrid {
    pub height: usize,
    pub width: usize,
}

impl TrainingImageTokenGrid {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub const fn len(&self) -> usize {
        self.height * self.width
    }

    pub const fn is_empty(&self) -> bool {
        self.height == 0 || self.width == 0
    }
}

impl From<TrainingImageTokenGrid> for SparseImageTokenGrid {
    fn from(value: TrainingImageTokenGrid) -> Self {
        SparseImageTokenGrid::new(value.height, value.width)
    }
}

impl From<SparseImageTokenGrid> for TrainingImageTokenGrid {
    fn from(value: SparseImageTokenGrid) -> Self {
        Self::new(value.height, value.width)
    }
}
