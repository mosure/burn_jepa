use crate::config::VJepaConfig;
use crate::positional::sparse_3d_sincos_pos_embed;
#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use crate::sparse_patchify::{SparsePatchifyBatchPlan, SparsePatchifyPlan};
use crate::tokens::{
    SparseMaskBatch, SparseTokenMask, TokenGridShape, apply_mask_batch, apply_token_mask,
    repeat_token_indices,
};
use anyhow::{Result, ensure};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct TokenSequencePosition<B: Backend> {
    pub indices: Tensor<B, 2, Int>,
    pub rope_sin: Option<Tensor<B, 3>>,
    pub rope_cos: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct PatchEmbed2d<B: Backend> {
    pub proj: Conv2d<B>,
    #[module(skip)]
    pub patch_size: usize,
}

impl<B: Backend> PatchEmbed2d<B> {
    pub fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        device: &B::Device,
    ) -> Self {
        let patch_size = patch_size.max(1);
        Self {
            proj: Conv2dConfig::new(
                [in_channels.max(1), embed_dim.max(1)],
                [patch_size, patch_size],
            )
            .with_stride([patch_size, patch_size])
            .init(device),
            patch_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.proj.forward(x);
        let [batch, dim, height, width] = x.shape().dims::<4>();
        x.reshape([batch, dim, height * width]).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbed3d<B: Backend> {
    pub proj: Conv3d<B>,
    #[module(skip)]
    pub patch_size: usize,
    #[module(skip)]
    pub tubelet_size: usize,
}

impl<B: Backend> PatchEmbed3d<B> {
    pub fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        tubelet_size: usize,
        device: &B::Device,
    ) -> Self {
        let patch_size = patch_size.max(1);
        let tubelet_size = tubelet_size.max(1);
        Self {
            proj: Conv3dConfig::new(
                [in_channels.max(1), embed_dim.max(1)],
                [tubelet_size, patch_size, patch_size],
            )
            .with_stride([tubelet_size, patch_size, patch_size])
            .init(device),
            patch_size,
            tubelet_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 3> {
        let x = self.proj.forward(x);
        let [batch, dim, depth, height, width] = x.shape().dims::<5>();
        x.reshape([batch, dim, depth * height * width])
            .swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct VJepaSelfAttention<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    scale: f32,
    #[module(skip)]
    use_rope: bool,
}

impl<B: Backend> VJepaSelfAttention<B> {
    pub fn new(embed_dim: usize, num_heads: usize, use_rope: bool, device: &B::Device) -> Self {
        let embed_dim = embed_dim.max(1);
        let num_heads = num_heads.max(1);
        let head_dim = (embed_dim / num_heads).max(1);
        Self {
            qkv: LinearConfig::new(embed_dim, embed_dim * 3).init(device),
            proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            use_rope,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: Option<&TokenSequencePosition<B>>,
    ) -> Tensor<B, 3> {
        let [batch, tokens, dim] = x.shape().dims::<3>();
        let qkv = self.qkv.forward(x);
        let mut q = qkv
            .clone()
            .slice_dim(2, 0..dim)
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let mut k = qkv
            .clone()
            .slice_dim(2, dim..dim * 2)
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = qkv
            .slice_dim(2, dim * 2..dim * 3)
            .reshape([batch, tokens, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        if self.use_rope
            && let Some(positions) = positions
            && let (Some(sin), Some(cos)) = (&positions.rope_sin, &positions.rope_cos)
        {
            q = apply_rotary(q, sin.clone(), cos.clone());
            k = apply_rotary(k, sin.clone(), cos.clone());
        }

        let attn = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let attn = activation::softmax(attn, 3);
        let out = attn
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([batch, tokens, dim]);
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct VJepaMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> VJepaMlp<B> {
    pub fn new(embed_dim: usize, mlp_ratio: f32, device: &B::Device) -> Self {
        let hidden = ((embed_dim as f32) * mlp_ratio.max(1.0)).round() as usize;
        Self {
            fc1: LinearConfig::new(embed_dim.max(1), hidden.max(1)).init(device),
            fc2: LinearConfig::new(hidden.max(1), embed_dim.max(1)).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(activation::gelu(self.fc1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: VJepaSelfAttention<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: VJepaMlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        norm_eps: f64,
        use_rope: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(embed_dim)
                .with_epsilon(norm_eps)
                .init(device),
            attn: VJepaSelfAttention::new(embed_dim, num_heads, use_rope, device),
            norm2: LayerNormConfig::new(embed_dim)
                .with_epsilon(norm_eps)
                .init(device),
            mlp: VJepaMlp::new(embed_dim, mlp_ratio, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: Option<&TokenSequencePosition<B>>,
    ) -> Tensor<B, 3> {
        let y = self.attn.forward(self.norm1.forward(x.clone()), positions);
        let x = x + y;
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }
}

#[derive(Debug)]
pub struct VJepaEncoderOutput<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub hierarchical: Vec<Tensor<B, 3>>,
    pub captured_layers: Vec<usize>,
    pub token_indices: Tensor<B, 2, Int>,
    pub grid: TokenGridShape,
}

#[derive(Clone, Debug)]
pub struct SparseEncoderPlan<B: Backend> {
    pub mask: SparseTokenMask,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub video: bool,
    pub positions: TokenSequencePosition<B>,
    pub position_embed: Option<Tensor<B, 3>>,
}

impl<B: Backend> SparseEncoderPlan<B> {
    pub fn new(
        config: &VJepaConfig,
        mask: SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
        video: bool,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(
            mask.dense_len() == grid.len(),
            "encoder plan mask dense token count must match token grid"
        );
        ensure!(batch > 0, "encoder plan batch must be nonzero");
        let dim = config.encoder.embed_dim;
        let position_embed = (!config.encoder.use_rope)
            .then(|| position_tensor::<B>(mask.indices(), grid, dim, batch, device));
        let positions = token_sequence_position::<B>(
            mask.indices(),
            grid,
            config.encoder.embed_dim / config.encoder.num_heads.max(1),
            batch,
            device,
            config.encoder.use_rope,
            config.encoder.interpolate_rope,
            config.patch_size,
        );
        Ok(Self {
            mask,
            grid,
            batch,
            video,
            positions,
            position_embed,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SparseEncoderBatchPlan<B: Backend> {
    pub mask: SparseMaskBatch<B>,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub video: bool,
    pub positions: TokenSequencePosition<B>,
    pub position_embed: Option<Tensor<B, 3>>,
}

impl<B: Backend> SparseEncoderBatchPlan<B> {
    pub fn new(
        config: &VJepaConfig,
        mask: SparseMaskBatch<B>,
        grid: TokenGridShape,
        video: bool,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(
            mask.dense_len() == grid.len(),
            "encoder batch plan mask dense token count must match token grid"
        );
        ensure!(mask.batch() > 0, "encoder batch plan batch must be nonzero");
        let batch = mask.batch();
        let token_count = mask.len();
        let dim = config.encoder.embed_dim;
        let rows = mask.padded_rows();
        let position_embed =
            (!config.encoder.use_rope).then(|| position_tensor_rows::<B>(&rows, grid, dim, device));
        let positions = token_sequence_position_rows::<B>(
            &rows,
            grid,
            config.encoder.embed_dim / config.encoder.num_heads.max(1),
            device,
            config.encoder.use_rope,
            config.encoder.interpolate_rope,
            config.patch_size,
        );
        ensure!(
            positions.indices.shape().dims::<2>() == [batch, token_count],
            "encoder batch plan positions do not match mask shape"
        );
        Ok(Self {
            mask,
            grid,
            batch,
            video,
            positions,
            position_embed,
        })
    }
}

#[derive(Module, Debug)]
pub struct VJepaEncoder<B: Backend> {
    pub patch_embed: PatchEmbed3d<B>,
    pub image_patch_embed: PatchEmbed3d<B>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub norms_block: Vec<LayerNorm<B>>,
    pub video_mod_embed: Param<Tensor<B, 2>>,
    pub image_mod_embed: Param<Tensor<B, 2>>,
    #[module(skip)]
    config: VJepaConfig,
    #[module(skip)]
    hierarchical_layers: Vec<usize>,
}

impl<B: Backend> VJepaEncoder<B> {
    pub fn new(config: &VJepaConfig, device: &B::Device) -> Self {
        let encoder = &config.encoder;
        let patch_embed = PatchEmbed3d::new(
            config.in_channels,
            encoder.embed_dim,
            config.patch_size,
            config.tubelet_size,
            device,
        );
        let image_patch_embed = PatchEmbed3d::new(
            config.in_channels,
            encoder.embed_dim,
            config.patch_size,
            1,
            device,
        );
        let blocks = (0..encoder.depth.max(1))
            .map(|_| {
                TransformerBlock::new(
                    encoder.embed_dim,
                    encoder.num_heads,
                    encoder.mlp_ratio,
                    encoder.layer_norm_eps,
                    encoder.use_rope,
                    device,
                )
            })
            .collect();
        let hierarchical_layers = encoder.hierarchical_layers();
        let norms_block = hierarchical_layers
            .iter()
            .map(|_| {
                LayerNormConfig::new(encoder.embed_dim)
                    .with_epsilon(encoder.layer_norm_eps)
                    .init(device)
            })
            .collect();
        Self {
            patch_embed,
            image_patch_embed,
            blocks,
            norms_block,
            video_mod_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [1, encoder.embed_dim],
                Distribution::Normal(0.0, 1.0e-6),
                device,
            )),
            image_mod_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [1, encoder.embed_dim],
                Distribution::Normal(0.0, 1.0e-6),
                device,
            )),
            config: config.clone(),
            hierarchical_layers,
        }
    }

    pub fn forward_video(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
    ) -> VJepaEncoderOutput<B> {
        let [batch, _channels, frames, height, width] = video.shape().dims::<5>();
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        let tokens = self.patch_embed.forward(video);
        self.forward_tokens_capture_layers(
            tokens,
            batch,
            grid,
            mask,
            true,
            &self.hierarchical_layers,
        )
    }

    pub fn forward_video_capture_layers(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
        capture_layers: &[usize],
    ) -> VJepaEncoderOutput<B> {
        let [batch, _channels, frames, height, width] = video.shape().dims::<5>();
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        let tokens = self.patch_embed.forward(video);
        self.forward_tokens_capture_layers(tokens, batch, grid, mask, true, capture_layers)
    }

    pub fn forward_video_with_plan(
        &self,
        video: Tensor<B, 5>,
        plan: &SparseEncoderPlan<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            batch == plan.batch,
            "video batch does not match sparse encoder plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "video channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(grid == plan.grid, "video token grid does not match plan");
        ensure!(plan.video, "video encoder path requires a video plan");
        let tokens = self.patch_embed.forward(video);
        let tokens = if plan.mask.len() < plan.grid.len() {
            apply_token_mask(tokens, plan.positions.indices.clone())
        } else {
            tokens
        };
        self.forward_sparse_tokens_with_plan(tokens, plan)
    }

    pub fn forward_image(
        &self,
        image: Tensor<B, 4>,
        mask: Option<&SparseTokenMask>,
    ) -> VJepaEncoderOutput<B> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        let image = image.reshape([batch, channels, 1, height, width]);
        let tokens = self.image_patch_embed.forward(image);
        self.forward_tokens_capture_layers(
            tokens,
            batch,
            grid,
            mask,
            false,
            &self.hierarchical_layers,
        )
    }

    pub fn forward_image_with_mask_batch(
        &self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == mask.batch(),
            "image batch does not match sparse mask batch"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len(),
            "sparse mask dense length must match image token grid"
        );
        let device = image.device();
        let image = image.reshape([batch, channels, 1, height, width]);
        let tokens = self.image_patch_embed.forward(image);
        let tokens = if mask.len() < grid.len() {
            apply_mask_batch(tokens, &mask)
        } else {
            tokens
        };
        let plan = SparseEncoderBatchPlan::new(&self.config, mask, grid, false, &device)?;
        self.forward_sparse_tokens_with_batch_plan(tokens, &plan)
    }

    fn forward_tokens_capture_layers(
        &self,
        tokens: Tensor<B, 3>,
        batch: usize,
        grid: TokenGridShape,
        mask: Option<&SparseTokenMask>,
        video: bool,
        capture_layers: &[usize],
    ) -> VJepaEncoderOutput<B> {
        let device = tokens.device();
        let mask = mask
            .cloned()
            .unwrap_or_else(|| SparseTokenMask::all(grid.len()));
        let plan = SparseEncoderPlan::new(&self.config, mask, grid, batch, video, &device)
            .expect("encoder plan from valid model grid");
        let tokens = if plan.mask.len() < grid.len() {
            apply_token_mask(tokens, plan.positions.indices.clone())
        } else {
            tokens
        };
        self.forward_sparse_tokens_with_plan_capture_layers(tokens, &plan, capture_layers)
            .expect("encoder plan matches generated token shape")
    }

    pub fn forward_sparse_tokens(
        &self,
        tokens: Tensor<B, 3>,
        batch: usize,
        grid: TokenGridShape,
        active_indices: &[usize],
        video: bool,
    ) -> VJepaEncoderOutput<B> {
        let device = tokens.device();
        let mask = SparseTokenMask::new(active_indices.to_vec(), grid.len())
            .expect("active sparse encoder indices must match token grid");
        let plan = SparseEncoderPlan::new(&self.config, mask, grid, batch, video, &device)
            .expect("sparse encoder plan");
        self.forward_sparse_tokens_with_plan(tokens, &plan)
            .expect("sparse encoder plan matches token shape")
    }

    pub fn forward_sparse_tokens_with_plan(
        &self,
        tokens: Tensor<B, 3>,
        plan: &SparseEncoderPlan<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_sparse_tokens_with_plan_capture_layers(tokens, plan, &self.hierarchical_layers)
    }

    pub fn forward_sparse_tokens_with_plan_capture_layers(
        &self,
        mut tokens: Tensor<B, 3>,
        plan: &SparseEncoderPlan<B>,
        capture_layers: &[usize],
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, token_count, dim] = tokens.shape().dims::<3>();
        ensure!(
            batch == plan.batch,
            "encoder token batch does not match plan"
        );
        ensure!(
            token_count == plan.mask.len(),
            "encoder token count does not match plan"
        );
        ensure!(
            dim == self.config.encoder.embed_dim,
            "encoder token dimension does not match config"
        );
        if let Some(position_embed) = &plan.position_embed {
            tokens = tokens + position_embed.clone();
        }
        if self.config.encoder.modality_embedding {
            let embed = if plan.video {
                self.video_mod_embed.val()
            } else {
                self.image_mod_embed.val()
            }
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, token_count);
            tokens = tokens + embed;
        }

        let capture_layers = normalized_capture_layers(capture_layers, self.blocks.len());
        let mut hierarchical = Vec::with_capacity(capture_layers.len());
        let mut x = tokens;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, Some(&plan.positions));
            if capture_layers.binary_search(&layer_index).is_ok() {
                hierarchical.push(layer_norm_for_capture(
                    &self.norms_block,
                    &self.hierarchical_layers,
                    layer_index,
                    x.clone(),
                ));
            }
        }
        let tokens = if let Some(norm) = self.norms_block.last() {
            norm.forward(x)
        } else {
            x
        };
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical,
            captured_layers: capture_layers,
            token_indices: plan.positions.indices.clone(),
            grid: plan.grid,
        })
    }

    pub fn forward_sparse_tokens_with_batch_plan(
        &self,
        mut tokens: Tensor<B, 3>,
        plan: &SparseEncoderBatchPlan<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, token_count, dim] = tokens.shape().dims::<3>();
        ensure!(
            batch == plan.batch,
            "encoder token batch does not match batch plan"
        );
        ensure!(
            token_count == plan.mask.len(),
            "encoder token count does not match batch plan"
        );
        ensure!(
            dim == self.config.encoder.embed_dim,
            "encoder token dimension does not match config"
        );
        if let Some(position_embed) = &plan.position_embed {
            tokens = tokens + position_embed.clone();
        }
        if self.config.encoder.modality_embedding {
            let embed = if plan.video {
                self.video_mod_embed.val()
            } else {
                self.image_mod_embed.val()
            }
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, token_count);
            tokens = tokens + embed;
        }

        let capture_layers =
            normalized_capture_layers(&self.hierarchical_layers, self.blocks.len());
        let mut hierarchical = Vec::with_capacity(capture_layers.len());
        let mut x = tokens;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, Some(&plan.positions));
            if capture_layers.binary_search(&layer_index).is_ok() {
                hierarchical.push(layer_norm_for_capture(
                    &self.norms_block,
                    &self.hierarchical_layers,
                    layer_index,
                    x.clone(),
                ));
            }
        }
        let tokens = if let Some(norm) = self.norms_block.last() {
            norm.forward(x)
        } else {
            x
        };
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical,
            captured_layers: capture_layers,
            token_indices: plan.positions.indices.clone(),
            grid: plan.grid,
        })
    }
}

fn normalized_capture_layers(layers: &[usize], depth: usize) -> Vec<usize> {
    let mut layers = layers
        .iter()
        .copied()
        .filter(|&layer| layer < depth)
        .collect::<Vec<_>>();
    layers.sort_unstable();
    layers.dedup();
    layers
}

fn layer_norm_for_capture<B: Backend>(
    norms: &[LayerNorm<B>],
    norm_layers: &[usize],
    layer_index: usize,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    if let Some(norm_index) = norm_layers.iter().position(|&index| index == layer_index) {
        norms[norm_index].forward(x)
    } else {
        x
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaEncoder<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn sparse_patchify_image_wgpu(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn sparse_patchify_image_wgpu_batch(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify batch plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify batch plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_image_sparse_patchify_wgpu(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let encoder_plan = SparseEncoderPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            plan.batch,
            false,
            &image.device(),
        )?;
        self.forward_image_sparse_patchify_wgpu_with_plan(image, plan, &encoder_plan)
    }

    pub fn forward_image_sparse_patchify_wgpu_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            encoder_plan.mask == patchify_plan.mask
                && encoder_plan.grid == patchify_plan.grid
                && encoder_plan.batch == patchify_plan.batch
                && !encoder_plan.video,
            "sparse patchify and sparse encoder plans must match"
        );
        let tokens = self.sparse_patchify_image_wgpu(image, patchify_plan)?;
        self.forward_sparse_tokens_with_plan(tokens, encoder_plan)
    }

    pub fn forward_image_sparse_patchify_wgpu_batch(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let encoder_plan = SparseEncoderBatchPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            false,
            &image.device(),
        )?;
        self.forward_image_sparse_patchify_wgpu_batch_with_plan(image, plan, &encoder_plan)
    }

    pub fn forward_image_sparse_patchify_wgpu_batch_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        encoder_plan: &SparseEncoderBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            encoder_plan.mask.rows() == patchify_plan.mask.rows()
                && encoder_plan.grid == patchify_plan.grid
                && encoder_plan.batch == patchify_plan.batch
                && !encoder_plan.video,
            "sparse patchify and sparse encoder batch plans must match"
        );
        let tokens = self.sparse_patchify_image_wgpu_batch(image, patchify_plan)?;
        self.forward_sparse_tokens_with_batch_plan(tokens, encoder_plan)
    }

    pub fn sparse_patchify_video_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            batch == plan.batch,
            "video batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "video channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "video token grid does not match sparse patchify plan"
        );
        let device = video.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames,
            height,
            width,
            tubelet_size: self.config.tubelet_size,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            video,
            plan.coords.clone(),
            self.patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_video_sparse_patchify_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let encoder_plan = SparseEncoderPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            plan.batch,
            true,
            &video.device(),
        )?;
        self.forward_video_sparse_patchify_wgpu_with_plan(video, plan, &encoder_plan)
    }

    pub fn forward_video_sparse_patchify_wgpu_with_plan(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            encoder_plan.mask == plan.mask
                && encoder_plan.grid == plan.grid
                && encoder_plan.batch == plan.batch
                && encoder_plan.video,
            "sparse patchify and sparse encoder plans must match"
        );
        let tokens = self.sparse_patchify_video_wgpu(video, plan)?;
        self.forward_sparse_tokens_with_plan(tokens, encoder_plan)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaEncoder<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn sparse_patchify_image_cuda(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::cuda::DefaultCudaBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn sparse_patchify_image_cuda_batch(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify batch plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify batch plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::cuda::DefaultCudaBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_image_sparse_patchify_cuda(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let encoder_plan = SparseEncoderPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            plan.batch,
            false,
            &image.device(),
        )?;
        self.forward_image_sparse_patchify_cuda_with_plan(image, plan, &encoder_plan)
    }

    pub fn forward_image_sparse_patchify_cuda_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            encoder_plan.mask == patchify_plan.mask
                && encoder_plan.grid == patchify_plan.grid
                && encoder_plan.batch == patchify_plan.batch
                && !encoder_plan.video,
            "sparse patchify and sparse encoder plans must match"
        );
        let tokens = self.sparse_patchify_image_cuda(image, patchify_plan)?;
        self.forward_sparse_tokens_with_plan(tokens, encoder_plan)
    }

    pub fn forward_image_sparse_patchify_cuda_batch(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let encoder_plan = SparseEncoderBatchPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            false,
            &image.device(),
        )?;
        self.forward_image_sparse_patchify_cuda_batch_with_plan(image, plan, &encoder_plan)
    }

    pub fn forward_image_sparse_patchify_cuda_batch_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        encoder_plan: &SparseEncoderBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            encoder_plan.mask.rows() == patchify_plan.mask.rows()
                && encoder_plan.grid == patchify_plan.grid
                && encoder_plan.batch == patchify_plan.batch
                && !encoder_plan.video,
            "sparse patchify and sparse encoder batch plans must match"
        );
        let tokens = self.sparse_patchify_image_cuda_batch(image, patchify_plan)?;
        self.forward_sparse_tokens_with_batch_plan(tokens, encoder_plan)
    }

    pub fn sparse_patchify_video_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        ensure!(
            batch == plan.batch,
            "video batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "video channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "video token grid does not match sparse patchify plan"
        );
        let device = video.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames,
            height,
            width,
            tubelet_size: self.config.tubelet_size,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::cuda::DefaultCudaBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
            &patchify_config,
            video,
            plan.coords.clone(),
            self.patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_video_sparse_patchify_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let encoder_plan = SparseEncoderPlan::new(
            &self.config,
            plan.mask.clone(),
            plan.grid,
            plan.batch,
            true,
            &video.device(),
        )?;
        self.forward_video_sparse_patchify_cuda_with_plan(video, plan, &encoder_plan)
    }

    pub fn forward_video_sparse_patchify_cuda_with_plan(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            encoder_plan.mask == plan.mask
                && encoder_plan.grid == plan.grid
                && encoder_plan.batch == plan.batch
                && encoder_plan.video,
            "sparse patchify and sparse encoder plans must match"
        );
        let tokens = self.sparse_patchify_video_cuda(video, plan)?;
        self.forward_sparse_tokens_with_plan(tokens, encoder_plan)
    }
}

#[derive(Debug)]
pub struct VJepaPredictorOutput<B: Backend> {
    pub target_predictions: Tensor<B, 3>,
    pub context_predictions: Option<Tensor<B, 3>>,
    pub sequence_tokens: Tensor<B, 3>,
    pub sequence_indices: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct SparsePredictorPlan<B: Backend> {
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub sort_order: Tensor<B, 2, Int>,
    pub reverse_order: Tensor<B, 2, Int>,
    pub sequence_indices: Tensor<B, 2, Int>,
    pub positions: TokenSequencePosition<B>,
    pub context_position_embed: Option<Tensor<B, 3>>,
    pub target_position_embed: Option<Tensor<B, 3>>,
}

impl<B: Backend> SparsePredictorPlan<B> {
    pub fn new(
        config: &VJepaConfig,
        context_mask: SparseTokenMask,
        target_mask: SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(
            context_mask.dense_len() == target_mask.dense_len(),
            "context and target masks must share a dense token range"
        );
        ensure!(
            context_mask.dense_len() == grid.len(),
            "mask dense token count must match predictor token grid"
        );
        let merged_indices = context_mask
            .indices()
            .iter()
            .chain(target_mask.indices().iter())
            .copied()
            .collect::<Vec<_>>();
        let sort_order = argsort(&merged_indices);
        let reverse_order = inverse_permutation(&sort_order);
        let sorted_indices = sort_order
            .iter()
            .map(|&index| merged_indices[index])
            .collect::<Vec<_>>();
        let position_grid = if config.predictor.use_rope {
            config.token_grid()
        } else {
            grid
        };
        let positions = token_sequence_position::<B>(
            &sorted_indices,
            position_grid,
            config.predictor.embed_dim / config.predictor.num_heads.max(1),
            batch,
            device,
            config.predictor.use_rope,
            false,
            config.patch_size,
        );
        let context_position_embed = (!config.predictor.use_rope).then(|| {
            position_tensor::<B>(
                context_mask.indices(),
                grid,
                config.predictor.embed_dim,
                batch,
                device,
            )
        });
        let target_position_embed = (!config.predictor.use_rope).then(|| {
            position_tensor::<B>(
                target_mask.indices(),
                grid,
                config.predictor.embed_dim,
                batch,
                device,
            )
        });
        Ok(Self {
            context_mask,
            target_mask,
            grid,
            batch,
            sort_order: repeat_token_indices::<B>(&sort_order, batch, device),
            reverse_order: repeat_token_indices::<B>(&reverse_order, batch, device),
            sequence_indices: repeat_token_indices::<B>(&merged_indices, batch, device),
            positions,
            context_position_embed,
            target_position_embed,
        })
    }
}

#[derive(Module, Debug)]
pub struct VJepaPredictor<B: Backend> {
    pub predictor_embed: Linear<B>,
    pub mask_tokens: Vec<Param<Tensor<B, 2>>>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub norm: LayerNorm<B>,
    pub target_proj: Linear<B>,
    pub context_proj: Option<Linear<B>>,
    pub video_mod_embed: Param<Tensor<B, 2>>,
    pub image_mod_embed: Param<Tensor<B, 2>>,
    #[module(skip)]
    config: VJepaConfig,
    #[module(skip)]
    return_all_tokens: bool,
}

impl<B: Backend> VJepaPredictor<B> {
    pub fn new(config: &VJepaConfig, device: &B::Device) -> Self {
        let predictor = &config.predictor;
        let encoder_dim = config.encoder.embed_dim;
        let pred_dim = predictor.embed_dim;
        let output_dim = predictor.output_dim.unwrap_or(encoder_dim);
        Self {
            predictor_embed: LinearConfig::new(encoder_dim, pred_dim).init(device),
            mask_tokens: (0..predictor.num_mask_tokens.max(1))
                .map(|_| Param::from_tensor(Tensor::<B, 2>::zeros([1, pred_dim], device)))
                .collect(),
            blocks: (0..predictor.depth.max(1))
                .map(|_| {
                    TransformerBlock::new(
                        pred_dim,
                        predictor.num_heads,
                        predictor.mlp_ratio,
                        predictor.layer_norm_eps,
                        predictor.use_rope,
                        device,
                    )
                })
                .collect(),
            norm: LayerNormConfig::new(pred_dim)
                .with_epsilon(predictor.layer_norm_eps)
                .init(device),
            target_proj: LinearConfig::new(pred_dim, output_dim).init(device),
            context_proj: predictor
                .return_all_tokens
                .then(|| LinearConfig::new(pred_dim, output_dim).init(device)),
            video_mod_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [1, pred_dim],
                Distribution::Normal(0.0, 1.0e-6),
                device,
            )),
            image_mod_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [1, pred_dim],
                Distribution::Normal(0.0, 1.0e-6),
                device,
            )),
            config: config.clone(),
            return_all_tokens: predictor.return_all_tokens,
        }
    }

    pub fn forward_sparse(
        &self,
        context_tokens: Tensor<B, 3>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
        grid: TokenGridShape,
        mask_index: usize,
    ) -> Result<VJepaPredictorOutput<B>> {
        let batch = context_tokens.shape().dims::<3>()[0];
        let device = context_tokens.device();
        let plan = SparsePredictorPlan::new(
            &self.config,
            context_mask.clone(),
            target_mask.clone(),
            grid,
            batch,
            &device,
        )?;
        self.forward_sparse_with_plan(context_tokens, &plan, mask_index)
    }

    pub fn forward_sparse_with_plan(
        &self,
        context_tokens: Tensor<B, 3>,
        plan: &SparsePredictorPlan<B>,
        mask_index: usize,
    ) -> Result<VJepaPredictorOutput<B>> {
        let context_mask = &plan.context_mask;
        let target_mask = &plan.target_mask;
        let [batch, context_len, _encoder_dim] = context_tokens.shape().dims::<3>();
        ensure!(
            batch == plan.batch,
            "context token batch does not match sparse predictor plan"
        );
        ensure!(
            context_len == context_mask.len(),
            "context token shape does not match context mask"
        );
        let mut context = self.predictor_embed.forward(context_tokens);
        if let Some(position_embed) = &plan.context_position_embed {
            context = context + position_embed.clone();
        }

        let target_len = target_mask.len();
        let token = self.mask_tokens[mask_index % self.mask_tokens.len()]
            .val()
            .reshape([1, 1, self.config.predictor.embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, target_len);
        let target = if let Some(position_embed) = &plan.target_position_embed {
            token + position_embed.clone()
        } else {
            token
        };
        let mut sequence = Tensor::cat(vec![context, target], 1);
        sequence = gather_with_indices(sequence, plan.sort_order.clone());
        if self.config.predictor.modality_embedding {
            let token_count = context_len + target_len;
            let embed = self
                .video_mod_embed
                .val()
                .reshape([1, 1, self.config.predictor.embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, token_count);
            sequence = sequence + embed;
        }

        for block in &self.blocks {
            sequence = block.forward(sequence, Some(&plan.positions));
        }
        sequence = self.norm.forward(sequence);
        sequence = gather_with_indices(sequence, plan.reverse_order.clone());
        let context_predictions = self.return_all_tokens.then(|| {
            let context = sequence.clone().slice_dim(1, 0..context_len);
            self.context_proj
                .as_ref()
                .expect("context projection")
                .forward(context)
        });
        let target_predictions = self.target_proj.forward(
            sequence
                .clone()
                .slice_dim(1, context_len..context_len + target_len),
        );
        Ok(VJepaPredictorOutput {
            target_predictions,
            context_predictions,
            sequence_tokens: sequence,
            sequence_indices: plan.sequence_indices.clone(),
        })
    }
}

#[derive(Debug)]
pub struct DensePredictionOutput<B: Backend> {
    pub predictions: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
    pub target_indices: Tensor<B, 2, Int>,
}

#[derive(Debug)]
pub struct SparseVJepaForwardOutput<B: Backend> {
    pub context: VJepaEncoderOutput<B>,
    pub target: VJepaEncoderOutput<B>,
    pub predictor: VJepaPredictorOutput<B>,
}

#[derive(Module, Debug)]
pub struct VJepa2_1Model<B: Backend> {
    pub encoder: VJepaEncoder<B>,
    pub predictor: VJepaPredictor<B>,
    #[module(skip)]
    config: VJepaConfig,
}

impl<B: Backend> VJepa2_1Model<B> {
    pub fn new(config: &VJepaConfig, device: &B::Device) -> Self {
        Self {
            encoder: VJepaEncoder::new(config, device),
            predictor: VJepaPredictor::new(config, device),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &VJepaConfig {
        &self.config
    }

    pub fn encode_video(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
    ) -> VJepaEncoderOutput<B> {
        self.encoder.forward_video(video, mask)
    }

    pub fn encode_image(
        &self,
        image: Tensor<B, 4>,
        mask: Option<&SparseTokenMask>,
    ) -> VJepaEncoderOutput<B> {
        self.encoder.forward_image(image, mask)
    }

    pub fn encode_image_batch(
        &self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder.forward_image_with_mask_batch(image, mask)
    }

    pub fn encode_video_with_plan(
        &self,
        video: Tensor<B, 5>,
        plan: &SparseEncoderPlan<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder.forward_video_with_plan(video, plan)
    }

    pub fn forward_sparse(
        &self,
        video: Tensor<B, 5>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
    ) -> Result<SparseVJepaForwardOutput<B>> {
        let context = self
            .encoder
            .forward_video(video.clone(), Some(context_mask));
        let target = self.encoder.forward_video(video, Some(target_mask));
        let predictor = self.predictor.forward_sparse(
            context.tokens.clone(),
            context_mask,
            target_mask,
            context.grid,
            0,
        )?;
        Ok(SparseVJepaForwardOutput {
            context,
            target,
            predictor,
        })
    }

    pub fn predict_dense_targets(
        &self,
        video: Tensor<B, 5>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
    ) -> Result<DensePredictionOutput<B>> {
        let out = self.forward_sparse(video, context_mask, target_mask)?;
        Ok(DensePredictionOutput {
            predictions: out.predictor.target_predictions,
            targets: out.target.tokens,
            target_indices: out.target.token_indices,
        })
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn encode_image_sparse_patchify_wgpu(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder.forward_image_sparse_patchify_wgpu(image, plan)
    }

    pub fn encode_image_sparse_patchify_wgpu_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder.forward_image_sparse_patchify_wgpu_with_plan(
            image,
            patchify_plan,
            encoder_plan,
        )
    }

    pub fn encode_image_sparse_patchify_wgpu_batch(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder
            .forward_image_sparse_patchify_wgpu_batch(image, plan)
    }

    pub fn encode_video_sparse_patchify_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder.forward_video_sparse_patchify_wgpu(video, plan)
    }

    pub fn encode_video_sparse_patchify_wgpu_with_plan(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder.forward_video_sparse_patchify_wgpu_with_plan(
            video,
            patchify_plan,
            encoder_plan,
        )
    }

    pub fn forward_sparse_patchify_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        context_plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        target_plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<SparseVJepaForwardOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            context_plan.grid == target_plan.grid,
            "context and target sparse patchify plans must share a grid"
        );
        ensure!(
            context_plan.batch == target_plan.batch,
            "context and target sparse patchify plans must share a batch"
        );
        let context = self
            .encoder
            .forward_video_sparse_patchify_wgpu(video.clone(), context_plan)?;
        let target = self
            .encoder
            .forward_video_sparse_patchify_wgpu(video, target_plan)?;
        let predictor = self.predictor.forward_sparse(
            context.tokens.clone(),
            &context_plan.mask,
            &target_plan.mask,
            context.grid,
            0,
        )?;
        Ok(SparseVJepaForwardOutput {
            context,
            target,
            predictor,
        })
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepa2_1Model<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn encode_image_sparse_patchify_cuda(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder.forward_image_sparse_patchify_cuda(image, plan)
    }

    pub fn encode_image_sparse_patchify_cuda_with_plan(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder.forward_image_sparse_patchify_cuda_with_plan(
            image,
            patchify_plan,
            encoder_plan,
        )
    }

    pub fn encode_image_sparse_patchify_cuda_batch(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder
            .forward_image_sparse_patchify_cuda_batch(image, plan)
    }

    pub fn encode_video_sparse_patchify_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder.forward_video_sparse_patchify_cuda(video, plan)
    }

    pub fn encode_video_sparse_patchify_cuda_with_plan(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        patchify_plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        encoder_plan: &SparseEncoderPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder.forward_video_sparse_patchify_cuda_with_plan(
            video,
            patchify_plan,
            encoder_plan,
        )
    }

    pub fn forward_sparse_patchify_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        context_plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        target_plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<SparseVJepaForwardOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            context_plan.grid == target_plan.grid,
            "context and target sparse patchify plans must share a grid"
        );
        ensure!(
            context_plan.batch == target_plan.batch,
            "context and target sparse patchify plans must share a batch"
        );
        let context = self
            .encoder
            .forward_video_sparse_patchify_cuda(video.clone(), context_plan)?;
        let target = self
            .encoder
            .forward_video_sparse_patchify_cuda(video, target_plan)?;
        let predictor = self.predictor.forward_sparse(
            context.tokens.clone(),
            &context_plan.mask,
            &target_plan.mask,
            context.grid,
            0,
        )?;
        Ok(SparseVJepaForwardOutput {
            context,
            target,
            predictor,
        })
    }
}

fn position_tensor<B: Backend>(
    indices: &[usize],
    grid: TokenGridShape,
    dim: usize,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let values = sparse_3d_sincos_pos_embed(dim, grid, indices);
    Tensor::<B, 3>::from_data(TensorData::new(values, [1, indices.len(), dim]), device)
        .repeat_dim(0, batch)
}

fn position_tensor_rows<B: Backend>(
    rows: &[Vec<usize>],
    grid: TokenGridShape,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let token_count = rows.first().map(Vec::len).unwrap_or(0);
    let values = rows
        .iter()
        .flat_map(|indices| sparse_3d_sincos_pos_embed(dim, grid, indices))
        .collect::<Vec<_>>();
    Tensor::<B, 3>::from_data(
        TensorData::new(values, [rows.len(), token_count, dim]),
        device,
    )
}

fn token_sequence_position<B: Backend>(
    indices: &[usize],
    grid: TokenGridShape,
    head_dim: usize,
    batch: usize,
    device: &B::Device,
    use_rope: bool,
    interpolate_rope: bool,
    patch_size: usize,
) -> TokenSequencePosition<B> {
    let (rope_sin, rope_cos) = if use_rope {
        let (sin, cos) = rotary_sin_cos_tensors::<B>(
            indices,
            grid,
            head_dim,
            batch,
            device,
            interpolate_rope,
            patch_size,
        );
        (Some(sin), Some(cos))
    } else {
        (None, None)
    };
    TokenSequencePosition {
        indices: repeat_token_indices::<B>(indices, batch, device),
        rope_sin,
        rope_cos,
    }
}

fn token_sequence_position_rows<B: Backend>(
    rows: &[Vec<usize>],
    grid: TokenGridShape,
    head_dim: usize,
    device: &B::Device,
    use_rope: bool,
    interpolate_rope: bool,
    patch_size: usize,
) -> TokenSequencePosition<B> {
    let batch = rows.len();
    let token_count = rows.first().map(Vec::len).unwrap_or(0);
    let values = rows
        .iter()
        .flat_map(|row| row.iter().map(|&index| index as i64))
        .collect::<Vec<_>>();
    let indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, token_count]), device);
    let (rope_sin, rope_cos) = if use_rope {
        let (sin, cos) = rotary_sin_cos_tensors_rows::<B>(
            rows,
            grid,
            head_dim,
            device,
            interpolate_rope,
            patch_size,
        );
        (Some(sin), Some(cos))
    } else {
        (None, None)
    };
    TokenSequencePosition {
        indices,
        rope_sin,
        rope_cos,
    }
}

fn rotary_sin_cos_tensors<B: Backend>(
    indices: &[usize],
    grid: TokenGridShape,
    head_dim: usize,
    batch: usize,
    device: &B::Device,
    interpolate_rope: bool,
    patch_size: usize,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let axis_dim = 2 * ((head_dim / 3) / 2);
    let mut sin = Vec::with_capacity(indices.len() * head_dim);
    let mut cos = Vec::with_capacity(indices.len() * head_dim);
    for &index in indices {
        let (frame, row, col) = crate::token_index_to_coords(index, grid);
        let row = rope_spatial_position(row, grid.height, patch_size, interpolate_rope);
        let col = rope_spatial_position(col, grid.width, patch_size, interpolate_rope);
        append_rotary_axis(&mut sin, &mut cos, axis_dim, frame as f32);
        append_rotary_axis(&mut sin, &mut cos, axis_dim, row);
        append_rotary_axis(&mut sin, &mut cos, axis_dim, col);
        let used = axis_dim * 3;
        for _ in used..head_dim {
            sin.push(0.0);
            cos.push(1.0);
        }
    }
    let sin = Tensor::<B, 3>::from_data(TensorData::new(sin, [1, indices.len(), head_dim]), device)
        .repeat_dim(0, batch);
    let cos = Tensor::<B, 3>::from_data(TensorData::new(cos, [1, indices.len(), head_dim]), device)
        .repeat_dim(0, batch);
    (sin, cos)
}

fn rotary_sin_cos_tensors_rows<B: Backend>(
    rows: &[Vec<usize>],
    grid: TokenGridShape,
    head_dim: usize,
    device: &B::Device,
    interpolate_rope: bool,
    patch_size: usize,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let axis_dim = 2 * ((head_dim / 3) / 2);
    let token_count = rows.first().map(Vec::len).unwrap_or(0);
    let mut sin = Vec::with_capacity(rows.len() * token_count * head_dim);
    let mut cos = Vec::with_capacity(rows.len() * token_count * head_dim);
    for row in rows {
        for &index in row {
            let (frame, row, col) = crate::token_index_to_coords(index, grid);
            let row = rope_spatial_position(row, grid.height, patch_size, interpolate_rope);
            let col = rope_spatial_position(col, grid.width, patch_size, interpolate_rope);
            append_rotary_axis(&mut sin, &mut cos, axis_dim, frame as f32);
            append_rotary_axis(&mut sin, &mut cos, axis_dim, row);
            append_rotary_axis(&mut sin, &mut cos, axis_dim, col);
            let used = axis_dim * 3;
            for _ in used..head_dim {
                sin.push(0.0);
                cos.push(1.0);
            }
        }
    }
    let sin = Tensor::<B, 3>::from_data(
        TensorData::new(sin, [rows.len(), token_count, head_dim]),
        device,
    );
    let cos = Tensor::<B, 3>::from_data(
        TensorData::new(cos, [rows.len(), token_count, head_dim]),
        device,
    );
    (sin, cos)
}

fn rope_spatial_position(
    index: usize,
    axis_len: usize,
    patch_size: usize,
    interpolate_rope: bool,
) -> f32 {
    if !interpolate_rope || axis_len <= 1 {
        return index as f32;
    }
    let pretrained = rope_pretrained_grid_size(patch_size);
    index as f32 * (pretrained - 1.0) / (axis_len as f32 - 1.0)
}

fn rope_pretrained_grid_size(patch_size: usize) -> f32 {
    match patch_size {
        14 => 18.0,
        16 => 16.0,
        patch => 256.0 / patch.max(1) as f32,
    }
}

fn append_rotary_axis(sin: &mut Vec<f32>, cos: &mut Vec<f32>, dim: usize, pos: f32) {
    let half = dim / 2;
    for i in 0..half {
        let omega = 1.0 / 10000_f32.powf(i as f32 / half.max(1) as f32);
        let angle = pos * omega;
        let s = angle.sin();
        let c = angle.cos();
        sin.push(s);
        sin.push(s);
        cos.push(c);
        cos.push(c);
    }
}

fn apply_rotary<B: Backend>(x: Tensor<B, 4>, sin: Tensor<B, 3>, cos: Tensor<B, 3>) -> Tensor<B, 4> {
    let sin = sin.unsqueeze_dim::<4>(1);
    let cos = cos.unsqueeze_dim::<4>(1);
    x.clone() * cos + rotate_half_pairs(x) * sin
}

fn rotate_half_pairs<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, tokens, dim] = x.shape().dims::<4>();
    debug_assert!(dim.is_multiple_of(2));
    let paired = x.reshape([batch, heads, tokens, dim / 2, 2]);
    let first = paired.clone().slice_dim(4, 0..1);
    let second = paired.slice_dim(4, 1..2);
    Tensor::cat(vec![second.neg(), first], 4).reshape([batch, heads, tokens, dim])
}

fn gather_with_indices<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    apply_token_mask(tokens, indices)
}

fn argsort(values: &[usize]) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| values[index]);
    order
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; order.len()];
    for (sorted_pos, &original_pos) in order.iter().enumerate() {
        inverse[original_pos] = sorted_pos;
    }
    inverse
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::make_context_target_masks;

    type B = burn::backend::NdArray<f32>;

    #[test]
    fn tiny_model_sparse_forward_shapes() {
        let device = Default::default();
        let config = VJepaConfig::tiny_for_tests();
        let model = VJepa2_1Model::<B>::new(&config, &device);
        let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
        let (context, target) = make_context_target_masks(config.token_grid(), 0.5);
        let out = model
            .forward_sparse(video, &context, &target)
            .expect("sparse forward");
        assert_eq!(out.context.tokens.shape().dims::<3>(), [1, 4, 32]);
        assert_eq!(out.target.tokens.shape().dims::<3>(), [1, 4, 32]);
        assert_eq!(
            out.predictor.target_predictions.shape().dims::<3>(),
            [1, 4, 32]
        );
    }

    #[test]
    fn rope_interpolation_matches_vjepa21_spatial_scale() {
        assert_eq!(rope_pretrained_grid_size(16), 16.0);
        assert_eq!(rope_spatial_position(23, 24, 16, true), 15.0);
        assert_eq!(rope_spatial_position(1, 24, 16, false), 1.0);
        assert_eq!(rope_spatial_position(0, 1, 16, true), 0.0);
    }

    #[test]
    fn rope_axis_uses_upstream_pairwise_frequency_repeat() {
        let mut sin = Vec::new();
        let mut cos = Vec::new();
        append_rotary_axis(&mut sin, &mut cos, 8, 1.0);
        assert_eq!(sin.len(), 8);
        assert_eq!(cos.len(), 8);
        assert_eq!(sin[0], sin[1]);
        assert_ne!(sin[0], sin[2]);
        assert_eq!(cos[6], cos[7]);
    }
}
