use crate::attention::EfficientCrossAttentionBlock;
use crate::config::AnyUpConfig;
use crate::layers::{AnyUpConvEncoder, AnyUpFeatureEncoder, aggregation_encoder};
use crate::rope::AnyUpRoPE;
use crate::sparse::{AnyUpSparseOutput, AnyUpSparseOutputPlan, sparse_low_features_to_nchw};
use crate::tensor_ops::{adaptive_pool, coordinate_grid, flatten_nchw_to_nlc, nlc_to_nchw};
use anyhow::Result;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct AnyUp<B: Backend> {
    pub image_encoder: AnyUpConvEncoder<B>,
    pub key_encoder: AnyUpConvEncoder<B>,
    pub query_encoder: AnyUpConvEncoder<B>,
    pub key_features_encoder: AnyUpFeatureEncoder<B>,
    pub cross_decode: EfficientCrossAttentionBlock<B>,
    pub aggregation: AnyUpConvEncoder<B>,
    pub rope: AnyUpRoPE<B>,
    #[module(skip)]
    pub config: AnyUpConfig,
}

#[derive(Debug)]
pub struct AnyUpImageContext<B: Backend> {
    pub query: Tensor<B, 4>,
    pub image_key: Tensor<B, 4>,
    pub output_size: [usize; 2],
    pub feature_size: [usize; 2],
}

#[derive(Debug)]
pub struct AnyUpImageGrid<B: Backend> {
    pub coords: Tensor<B, 3>,
    pub image_size: [usize; 2],
}

impl<B: Backend> Clone for AnyUpImageContext<B> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            image_key: self.image_key.clone(),
            output_size: self.output_size,
            feature_size: self.feature_size,
        }
    }
}

impl<B: Backend> Clone for AnyUpImageGrid<B> {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords.clone(),
            image_size: self.image_size,
        }
    }
}

impl<B: Backend> AnyUp<B> {
    pub fn new(config: AnyUpConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            image_encoder: AnyUpConvEncoder::new(
                config.input_dim,
                config.qk_dim,
                config.kernel_size,
                2,
                config.group_norm_groups,
                config.group_norm_eps,
                device,
            ),
            key_encoder: AnyUpConvEncoder::new(
                config.qk_dim,
                config.qk_dim,
                1,
                2,
                config.group_norm_groups,
                config.group_norm_eps,
                device,
            ),
            query_encoder: AnyUpConvEncoder::new(
                config.qk_dim,
                config.qk_dim,
                1,
                2,
                config.group_norm_groups,
                config.group_norm_eps,
                device,
            ),
            key_features_encoder: AnyUpFeatureEncoder::new(
                config.qk_dim,
                config.lfu_dim(),
                config.kernel_size_lfu,
                2,
                config.group_norm_groups,
                config.group_norm_eps,
                device,
            ),
            cross_decode: EfficientCrossAttentionBlock::new(
                config.qk_dim,
                config.num_heads,
                config.window_ratio,
                config.rms_norm_eps,
                device,
            ),
            aggregation: aggregation_encoder(
                config.qk_dim,
                config.group_norm_groups,
                config.group_norm_eps,
                device,
            ),
            rope: AnyUpRoPE::new(config.qk_dim, device),
            config,
        })
    }

    pub fn forward(
        &self,
        image: Tensor<B, 4>,
        features: Tensor<B, 4>,
        output_size: Option<[usize; 2]>,
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        let image_shape = image.shape().dims::<4>();
        let output_size = output_size.unwrap_or([image_shape[2], image_shape[3]]);
        let encoded = self.encode_image(image);
        self.upsample(encoded, features, output_size, q_chunk_size)
    }

    pub fn encode_image(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, height, width] = image.shape().dims::<4>();
        let grid = self.prepare_image_grid([height, width], &image.device());
        self.encode_image_with_grid(image, &grid)
    }

    pub fn prepare_image_grid(
        &self,
        image_size: [usize; 2],
        device: &B::Device,
    ) -> AnyUpImageGrid<B> {
        let [height, width] = image_size;
        AnyUpImageGrid {
            coords: coordinate_grid::<B>(height, width, device),
            image_size,
        }
    }

    pub fn encode_image_with_grid(
        &self,
        image: Tensor<B, 4>,
        grid: &AnyUpImageGrid<B>,
    ) -> Tensor<B, 4> {
        let encoded = self.image_encoder.forward(image);
        let [_, _, height, width] = encoded.shape().dims::<4>();
        debug_assert_eq!(grid.image_size, [height, width]);
        let encoded = flatten_nchw_to_nlc(encoded);
        let encoded = self.rope.forward(encoded, grid.coords.clone());
        nlc_to_nchw(encoded, height, width)
    }

    pub fn upsample(
        &self,
        encoded_image: Tensor<B, 4>,
        features: Tensor<B, 4>,
        output_size: [usize; 2],
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        let [_, _, low_h, low_w] = features.shape().dims::<4>();
        let context = self.prepare_encoded_context(encoded_image, output_size, [low_h, low_w]);
        self.upsample_with_context(&context, features, q_chunk_size)
    }

    pub fn prepare_image_context(
        &self,
        image: Tensor<B, 4>,
        output_size: Option<[usize; 2]>,
        feature_size: [usize; 2],
    ) -> AnyUpImageContext<B> {
        let image_shape = image.shape().dims::<4>();
        let output_size = output_size.unwrap_or([image_shape[2], image_shape[3]]);
        let encoded = self.encode_image(image);
        self.prepare_encoded_context(encoded, output_size, feature_size)
    }

    pub fn prepare_image_context_with_grid(
        &self,
        image: Tensor<B, 4>,
        grid: &AnyUpImageGrid<B>,
        output_size: Option<[usize; 2]>,
        feature_size: [usize; 2],
    ) -> AnyUpImageContext<B> {
        let image_shape = image.shape().dims::<4>();
        let output_size = output_size.unwrap_or([image_shape[2], image_shape[3]]);
        let encoded = self.encode_image_with_grid(image, grid);
        self.prepare_encoded_context(encoded, output_size, feature_size)
    }

    pub fn prepare_encoded_context(
        &self,
        encoded_image: Tensor<B, 4>,
        output_size: [usize; 2],
        feature_size: [usize; 2],
    ) -> AnyUpImageContext<B> {
        AnyUpImageContext {
            query: adaptive_pool(
                self.query_encoder.forward(encoded_image.clone()),
                output_size,
            ),
            image_key: adaptive_pool(self.key_encoder.forward(encoded_image), feature_size),
            output_size,
            feature_size,
        }
    }

    pub fn upsample_with_context(
        &self,
        context: &AnyUpImageContext<B>,
        features: Tensor<B, 4>,
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        self.upsample_values_with_context(context, features.clone(), features, q_chunk_size)
    }

    pub fn upsample_values_with_context(
        &self,
        context: &AnyUpImageContext<B>,
        key_features: Tensor<B, 4>,
        values: Tensor<B, 4>,
        q_chunk_size: Option<usize>,
    ) -> Tensor<B, 4> {
        let [_, _, low_h, low_w] = key_features.shape().dims::<4>();
        debug_assert_eq!(context.feature_size, [low_h, low_w]);
        let [_, _, value_h, value_w] = values.shape().dims::<4>();
        debug_assert_eq!([value_h, value_w], [low_h, low_w]);
        let k_feat = self.key_features_encoder.forward(key_features);
        let k = Tensor::cat(vec![context.image_key.clone(), k_feat], 1);
        let k = self.aggregation.forward(k);
        self.cross_decode
            .forward(context.query.clone(), k, values, q_chunk_size)
    }

    pub fn upsample_sparse_with_context(
        &self,
        context: &AnyUpImageContext<B>,
        features: Tensor<B, 4>,
        plan: &AnyUpSparseOutputPlan<B>,
    ) -> Result<AnyUpSparseOutput<B>> {
        let [batch, _, low_h, low_w] = features.shape().dims::<4>();
        anyhow::ensure!(
            plan.batch == batch,
            "sparse AnyUp plan batch does not match feature batch"
        );
        anyhow::ensure!(
            context.output_size == plan.output_size,
            "sparse AnyUp plan output size does not match image context"
        );
        anyhow::ensure!(
            context.feature_size == [low_h, low_w] && plan.feature_size == [low_h, low_w],
            "sparse AnyUp plan feature size does not match feature grid"
        );
        let k_feat = self.key_features_encoder.forward(features.clone());
        let k = Tensor::cat(vec![context.image_key.clone(), k_feat], 1);
        let k = self.aggregation.forward(k);
        Ok(self
            .cross_decode
            .forward_sparse(context.query.clone(), k, features, plan))
    }

    pub fn upsample_sparse_low_features_with_context(
        &self,
        context: &AnyUpImageContext<B>,
        sparse_features: Tensor<B, 3>,
        low_indices: Tensor<B, 2, burn::tensor::Int>,
        plan: &AnyUpSparseOutputPlan<B>,
    ) -> Result<AnyUpSparseOutput<B>> {
        let features = sparse_low_features_to_nchw(
            sparse_features,
            low_indices,
            plan.feature_size,
            &context.query.device(),
        )?;
        self.upsample_sparse_with_context(context, features, plan)
    }
}
