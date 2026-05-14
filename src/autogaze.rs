use crate::{
    SparseImageTokenGrid, SparseTokenMask, TemporalSparseJepaStreamConfig, TokenGridShape,
    sparse_mask_from_frame_token_indices, sparse_mask_from_frame_token_pairs,
    target_mask_from_context,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_autogaze::{AutoGazeGenerateOutput, AutoGazePipeline, AutoGazeStreamingCache};

const DEFAULT_AUTOGAZE_SPARSE_TOP_K_OVERFETCH: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutogazeSparseJepaWindowConfig {
    pub frames: usize,
    pub tubelet_size: usize,
    pub patch_size: usize,
    pub image_height: usize,
    pub image_width: usize,
    pub autogaze_tokens_per_frame: usize,
    pub image_grid: SparseImageTokenGrid,
    pub context_density: f32,
    pub target_tokens: usize,
    pub max_gaze_tokens_each_frame: usize,
    pub dilation: usize,
    pub top_k_overfetch: f32,
}

impl AutogazeSparseJepaWindowConfig {
    pub fn new(
        frames: usize,
        tubelet_size: usize,
        patch_size: usize,
        image_height: usize,
        image_width: usize,
        autogaze_tokens_per_frame: usize,
        context_density: f32,
        target_tokens: usize,
        max_gaze_tokens_each_frame: usize,
    ) -> Self {
        Self {
            frames: frames.max(1),
            tubelet_size: tubelet_size.max(1),
            patch_size: patch_size.max(1),
            image_height,
            image_width,
            autogaze_tokens_per_frame: autogaze_tokens_per_frame.max(1),
            image_grid: autogaze_image_token_grid(autogaze_tokens_per_frame),
            context_density,
            target_tokens: target_tokens.max(1),
            max_gaze_tokens_each_frame: max_gaze_tokens_each_frame.max(1),
            dilation: 0,
            top_k_overfetch: DEFAULT_AUTOGAZE_SPARSE_TOP_K_OVERFETCH,
        }
    }

    pub fn with_image_grid(mut self, image_grid: SparseImageTokenGrid) -> Self {
        self.image_grid = image_grid;
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn with_top_k_overfetch(mut self, top_k_overfetch: f32) -> Self {
        self.top_k_overfetch = top_k_overfetch.max(1.0);
        self
    }

    pub fn build(self) -> Result<AutogazeSparseJepaWindowPlan> {
        ensure!(
            self.frames.is_multiple_of(self.tubelet_size),
            "window frames must be divisible by V-JEPA tubelet size"
        );
        ensure!(
            self.image_height.is_multiple_of(self.patch_size),
            "window height must be divisible by V-JEPA patch size"
        );
        ensure!(
            self.image_width.is_multiple_of(self.patch_size),
            "window width must be divisible by V-JEPA patch size"
        );
        ensure!(
            !self.image_grid.is_empty(),
            "AutoGaze sparse image-token grid must be non-empty"
        );
        let grid = TokenGridShape::new(
            self.frames / self.tubelet_size,
            self.image_height / self.patch_size,
            self.image_width / self.patch_size,
        );
        ensure!(
            !grid.is_empty(),
            "V-JEPA sparse window grid must be non-empty"
        );
        let context_tokens = autogaze_sparse_context_tokens(grid, self.context_density);
        ensure!(
            context_tokens < grid.len(),
            "context token budget must leave at least one target token"
        );
        let target_tokens = self.target_tokens.min(grid.len() - context_tokens).max(1);
        let top_k = autogaze_sparse_top_k_for_context_with_overfetch(
            grid,
            self.image_grid,
            self.frames,
            context_tokens,
            self.max_gaze_tokens_each_frame,
            self.top_k_overfetch,
        );
        let generation_budget =
            autogaze_sparse_generation_budget(self.max_gaze_tokens_each_frame, top_k);
        let projection = AutogazeSparseJepaProjectionConfig::new(
            self.frames,
            self.tubelet_size,
            self.autogaze_tokens_per_frame,
            context_tokens,
            target_tokens,
        )
        .with_image_grid(self.image_grid)
        .with_dilation(self.dilation);
        let stream = projection.temporal_stream_config();
        Ok(AutogazeSparseJepaWindowPlan {
            grid,
            image_grid: self.image_grid,
            context_tokens,
            target_tokens,
            top_k,
            generation_budget,
            projection,
            stream,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutogazeSparseJepaWindowPlan {
    pub grid: TokenGridShape,
    pub image_grid: SparseImageTokenGrid,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub top_k: usize,
    pub generation_budget: usize,
    pub projection: AutogazeSparseJepaProjectionConfig,
    pub stream: TemporalSparseJepaStreamConfig,
}

impl AutogazeSparseJepaWindowPlan {
    pub fn project_generated_tokens(
        &self,
        generated: &AutoGazeGenerateOutput,
    ) -> Result<AutogazeSparseJepaProjection> {
        project_autogaze_generated_tokens(generated, self.grid, self.projection, self.top_k)
    }

    pub fn project_generated_masks(
        &self,
        generated: &AutoGazeGenerateOutput,
    ) -> Result<AutogazeSparseJepaMasks> {
        project_autogaze_generated_masks(generated, self.grid, self.projection, self.top_k)
    }

    pub fn generate<B: Backend>(
        &self,
        autogaze: &AutoGazePipeline<B>,
        video: Tensor<B, 5>,
    ) -> AutoGazeGenerateOutput {
        autogaze.generate_with_limit(video, self.generation_budget)
    }

    pub fn generate_streaming<B: Backend>(
        &self,
        autogaze: &AutoGazePipeline<B>,
        video: Tensor<B, 5>,
        cache: &mut AutoGazeStreamingCache<B>,
    ) -> AutoGazeGenerateOutput {
        generate_autogaze_streaming_with_budget(autogaze, video, cache, self.generation_budget)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutogazeSparseJepaProjectionConfig {
    pub frames: usize,
    pub tubelet_size: usize,
    pub autogaze_tokens_per_frame: usize,
    pub image_grid: SparseImageTokenGrid,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub dilation: usize,
}

impl AutogazeSparseJepaProjectionConfig {
    pub fn new(
        frames: usize,
        tubelet_size: usize,
        autogaze_tokens_per_frame: usize,
        context_tokens: usize,
        target_tokens: usize,
    ) -> Self {
        Self {
            frames: frames.max(1),
            tubelet_size: tubelet_size.max(1),
            autogaze_tokens_per_frame: autogaze_tokens_per_frame.max(1),
            image_grid: autogaze_image_token_grid(autogaze_tokens_per_frame),
            context_tokens: context_tokens.max(1),
            target_tokens: target_tokens.max(1),
            dilation: 0,
        }
    }

    pub fn with_image_grid(mut self, image_grid: SparseImageTokenGrid) -> Self {
        self.image_grid = image_grid;
        self
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    pub fn temporal_stream_config(self) -> TemporalSparseJepaStreamConfig {
        TemporalSparseJepaStreamConfig::new(
            self.context_tokens,
            self.target_tokens,
            self.image_grid,
        )
        .with_dilation(self.dilation)
    }
}

#[derive(Clone, Debug)]
pub struct AutogazeSparseJepaProjection {
    pub frame_tokens: Vec<Vec<usize>>,
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
}

#[derive(Clone, Debug)]
pub struct AutogazeSparseJepaMasks {
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
}

pub fn project_autogaze_generated_tokens(
    generated: &AutoGazeGenerateOutput,
    grid: TokenGridShape,
    config: AutogazeSparseJepaProjectionConfig,
    top_k: usize,
) -> Result<AutogazeSparseJepaProjection> {
    ensure!(!grid.is_empty(), "V-JEPA token grid must be non-empty");
    ensure!(
        config.context_tokens < grid.len(),
        "context token budget must leave at least one target token"
    );
    let frame_tokens = autogaze_frame_tokens(
        generated,
        config.frames,
        top_k,
        config.autogaze_tokens_per_frame,
    );
    let context_mask = sparse_mask_from_frame_token_indices(
        grid,
        config.tubelet_size,
        config.image_grid,
        &frame_tokens,
        config.dilation,
        config.context_tokens,
    )?;
    let target_mask = target_mask_from_context(&context_mask, config.target_tokens)?;
    Ok(AutogazeSparseJepaProjection {
        frame_tokens,
        context_mask,
        target_mask,
    })
}

pub fn project_autogaze_generated_masks(
    generated: &AutoGazeGenerateOutput,
    grid: TokenGridShape,
    config: AutogazeSparseJepaProjectionConfig,
    top_k: usize,
) -> Result<AutogazeSparseJepaMasks> {
    ensure!(!grid.is_empty(), "V-JEPA token grid must be non-empty");
    ensure!(
        config.context_tokens < grid.len(),
        "context token budget must leave at least one target token"
    );
    let context_mask = sparse_mask_from_frame_token_pairs(
        grid,
        config.tubelet_size,
        config.image_grid,
        autogaze_frame_token_pairs(
            generated,
            config.frames,
            top_k,
            config.autogaze_tokens_per_frame,
        ),
        config.dilation,
        config.context_tokens,
    )?;
    let target_mask = target_mask_from_context(&context_mask, config.target_tokens)?;
    Ok(AutogazeSparseJepaMasks {
        context_mask,
        target_mask,
    })
}

pub fn autogaze_frame_tokens(
    generated: &AutoGazeGenerateOutput,
    frames: usize,
    top_k: usize,
    tokens_per_frame: usize,
) -> Vec<Vec<usize>> {
    let mut frame_tokens = vec![Vec::new(); frames];
    for (frame, token) in autogaze_frame_token_pairs(generated, frames, top_k, tokens_per_frame) {
        frame_tokens[frame].push(token);
    }
    frame_tokens
}

pub fn autogaze_frame_token_pairs(
    generated: &AutoGazeGenerateOutput,
    frames: usize,
    top_k: usize,
    tokens_per_frame: usize,
) -> AutogazeFrameTokenPairs<'_> {
    AutogazeFrameTokenPairs {
        generated,
        frames,
        top_k: top_k.max(1),
        tokens_per_frame: tokens_per_frame.max(1),
        frame_idx: 0,
        cursor: 0,
        local_idx: 0,
        emitted_in_frame: 0,
    }
}

#[derive(Clone, Debug)]
pub struct AutogazeFrameTokenPairs<'a> {
    generated: &'a AutoGazeGenerateOutput,
    frames: usize,
    top_k: usize,
    tokens_per_frame: usize,
    frame_idx: usize,
    cursor: usize,
    local_idx: usize,
    emitted_in_frame: usize,
}

impl Iterator for AutogazeFrameTokenPairs<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let tokens = self.generated.gazing_pos.first();
        let padded = self.generated.if_padded_gazing.first();
        while self.frame_idx < self.frames {
            let frame_len = self
                .generated
                .num_gazing_each_frame
                .get(self.frame_idx)
                .copied()
                .unwrap_or(0);
            while self.local_idx < frame_len && self.emitted_in_frame < self.top_k {
                let token_index = self.cursor + self.local_idx;
                self.local_idx += 1;
                if padded
                    .and_then(|flags| flags.get(token_index))
                    .copied()
                    .unwrap_or(true)
                {
                    continue;
                }
                let Some(raw_token) = tokens.and_then(|tokens| tokens.get(token_index)).copied()
                else {
                    continue;
                };
                let token = raw_token - (self.frame_idx * self.tokens_per_frame) as i64;
                if token < 0 {
                    continue;
                }
                let token = token as usize;
                if token < self.tokens_per_frame {
                    self.emitted_in_frame += 1;
                    return Some((self.frame_idx, token));
                }
            }
            self.cursor += frame_len;
            self.frame_idx += 1;
            self.local_idx = 0;
            self.emitted_in_frame = 0;
        }
        None
    }
}

pub fn autogaze_image_token_grid(tokens_per_frame: usize) -> SparseImageTokenGrid {
    let side = (tokens_per_frame as f32).sqrt() as usize;
    if side * side == tokens_per_frame {
        SparseImageTokenGrid::new(side, side)
    } else {
        SparseImageTokenGrid::new(1, tokens_per_frame.max(1))
    }
}

pub fn autogaze_sparse_top_k_for_context(
    grid: TokenGridShape,
    image_grid: SparseImageTokenGrid,
    frames: usize,
    context_tokens: usize,
    max_gaze_tokens_each_frame: usize,
) -> usize {
    autogaze_sparse_top_k_for_context_with_overfetch(
        grid,
        image_grid,
        frames,
        context_tokens,
        max_gaze_tokens_each_frame,
        DEFAULT_AUTOGAZE_SPARSE_TOP_K_OVERFETCH,
    )
}

pub fn autogaze_sparse_top_k_for_context_with_overfetch(
    grid: TokenGridShape,
    image_grid: SparseImageTokenGrid,
    frames: usize,
    context_tokens: usize,
    max_gaze_tokens_each_frame: usize,
    overfetch: f32,
) -> usize {
    let max_gaze_tokens_each_frame = max_gaze_tokens_each_frame.max(1);
    if grid.is_empty() || image_grid.is_empty() || frames == 0 || context_tokens == 0 {
        return 1.min(max_gaze_tokens_each_frame);
    }

    let sparse_tokens_per_autogaze_token =
        grid.tokens_per_frame() as f32 / image_grid.len().max(1) as f32;
    let sparse_tokens_per_frame_budget =
        (frames as f32 * sparse_tokens_per_autogaze_token.max(f32::EPSILON)).max(1.0);
    let top_k = ((context_tokens as f32 / sparse_tokens_per_frame_budget) * overfetch.max(1.0))
        .ceil() as usize;
    top_k.max(1).min(max_gaze_tokens_each_frame)
}

pub fn autogaze_sparse_context_tokens(grid: TokenGridShape, density: f32) -> usize {
    let tokens = ((grid.len() as f32) * density.clamp(0.0, 1.0)).ceil() as usize;
    tokens.max(1).min(grid.len().max(1))
}

pub fn autogaze_sparse_generation_budget(max_gaze_tokens_each_frame: usize, top_k: usize) -> usize {
    top_k.max(1).min(max_gaze_tokens_each_frame.max(1))
}

pub fn generate_autogaze_streaming_with_budget<B: Backend>(
    autogaze: &AutoGazePipeline<B>,
    video: Tensor<B, 5>,
    cache: &mut AutoGazeStreamingCache<B>,
    generation_budget: usize,
) -> AutoGazeGenerateOutput {
    autogaze
        .model()
        .generate_streaming_with_task_loss_requirement_and_coverage_stop(
            video,
            cache,
            generation_budget.max(1),
            autogaze.task_loss_requirement(),
            autogaze.generation_coverage_stop_ratio(),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autogaze_projection_uses_generated_token_ids_without_trace_decoding() {
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 3, 4, 7]],
            num_gazing_each_frame: vec![2, 2],
            if_padded_gazing: vec![vec![false, false, false, false]],
            confidences: vec![vec![1.0; 4]],
        };
        let config = AutogazeSparseJepaProjectionConfig::new(2, 1, 4, 2, 1);
        let projection =
            project_autogaze_generated_tokens(&generated, TokenGridShape::new(2, 2, 2), config, 1)
                .expect("project sparse autogaze tokens");

        assert_eq!(projection.frame_tokens, vec![vec![0], vec![0]]);
        assert_eq!(projection.context_mask.indices(), &[0, 4]);
        assert_eq!(projection.target_mask.len(), 1);
        assert!(
            !projection
                .context_mask
                .indices()
                .contains(&projection.target_mask.indices()[0])
        );
    }

    #[test]
    fn autogaze_frame_token_pairs_match_grouped_frame_tokens() {
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 1, 4, 5, 9, 11]],
            num_gazing_each_frame: vec![2, 2, 2],
            if_padded_gazing: vec![vec![false, true, false, false, false, false]],
            confidences: vec![vec![1.0; 6]],
        };

        assert_eq!(
            autogaze_frame_token_pairs(&generated, 3, 2, 4).collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 3)]
        );
        assert_eq!(
            autogaze_frame_tokens(&generated, 3, 2, 4),
            vec![vec![0], vec![0, 1], vec![1, 3]]
        );
    }

    #[test]
    fn direct_mask_projection_matches_token_projection_without_frame_allocations() {
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 3, 4, 7]],
            num_gazing_each_frame: vec![2, 2],
            if_padded_gazing: vec![vec![false, false, false, false]],
            confidences: vec![vec![1.0; 4]],
        };
        let config = AutogazeSparseJepaProjectionConfig::new(2, 1, 4, 2, 1);
        let grid = TokenGridShape::new(2, 2, 2);

        let tokens =
            project_autogaze_generated_tokens(&generated, grid, config, 1).expect("tokens");
        let masks = project_autogaze_generated_masks(&generated, grid, config, 1).expect("masks");

        assert_eq!(masks.context_mask, tokens.context_mask);
        assert_eq!(masks.target_mask, tokens.target_mask);
    }

    #[test]
    fn non_square_autogaze_connector_grid_falls_back_to_one_row() {
        assert_eq!(
            autogaze_image_token_grid(6),
            SparseImageTokenGrid::new(1, 6)
        );
    }

    #[test]
    fn sparse_top_k_accounts_for_clip_frames_and_projection_fanout() {
        let image_grid = SparseImageTokenGrid::new(14, 14);

        assert_eq!(
            autogaze_sparse_top_k_for_context(
                TokenGridShape::new(2, 14, 14),
                image_grid,
                4,
                40,
                32,
            ),
            10
        );
        assert_eq!(
            autogaze_sparse_top_k_for_context(
                TokenGridShape::new(2, 45, 80),
                image_grid,
                4,
                72,
                32,
            ),
            1
        );
        assert_eq!(
            autogaze_sparse_top_k_for_context(
                TokenGridShape::new(2, 45, 80),
                image_grid,
                4,
                720,
                32,
            ),
            10
        );
    }

    #[test]
    fn sparse_window_plan_consolidates_density_top_k_and_stream_config() {
        let plan = AutogazeSparseJepaWindowConfig::new(4, 2, 16, 720, 1280, 196, 0.01, 64, 32)
            .build()
            .expect("window plan");

        assert_eq!(plan.grid, TokenGridShape::new(2, 45, 80));
        assert_eq!(plan.context_tokens, 72);
        assert_eq!(plan.target_tokens, 64);
        assert_eq!(plan.top_k, 1);
        assert_eq!(plan.generation_budget, 1);
        assert_eq!(plan.projection.context_tokens, plan.context_tokens);
        assert_eq!(plan.projection.target_tokens, plan.target_tokens);
        assert_eq!(plan.stream.context_tokens, plan.context_tokens);
        assert_eq!(plan.stream.target_tokens, plan.target_tokens);
    }

    #[test]
    fn sparse_window_plan_projects_generated_tokens_and_masks() {
        let generated = AutoGazeGenerateOutput {
            gazing_pos: vec![vec![0, 3, 4, 7]],
            num_gazing_each_frame: vec![2, 2],
            if_padded_gazing: vec![vec![false, false, false, false]],
            confidences: vec![vec![1.0; 4]],
        };
        let plan = AutogazeSparseJepaWindowConfig::new(2, 1, 1, 2, 2, 4, 0.5, 1, 4)
            .build()
            .expect("window plan");

        let tokens = plan.project_generated_tokens(&generated).expect("tokens");
        let masks = plan.project_generated_masks(&generated).expect("masks");

        assert_eq!(tokens.context_mask, masks.context_mask);
        assert_eq!(tokens.target_mask, masks.target_mask);
        assert_eq!(tokens.frame_tokens, vec![vec![0, 3], vec![0, 3]]);
    }

    #[test]
    fn sparse_generation_budget_never_expands_to_pipeline_max() {
        assert_eq!(autogaze_sparse_generation_budget(32, 2), 2);
        assert_eq!(autogaze_sparse_generation_budget(32, 64), 32);
        assert_eq!(autogaze_sparse_generation_budget(0, 0), 1);
    }
}
