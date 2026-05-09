use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VJepaModelVariant {
    VitBase384,
    VitLarge384,
    VitGiant384,
    VitGigantic384,
}

impl VJepaModelVariant {
    pub const fn encoder_width(self) -> usize {
        match self {
            Self::VitBase384 => 768,
            Self::VitLarge384 => 1024,
            Self::VitGiant384 => 1408,
            Self::VitGigantic384 => 1664,
        }
    }

    pub const fn encoder_depth(self) -> usize {
        match self {
            Self::VitBase384 => 12,
            Self::VitLarge384 => 24,
            Self::VitGiant384 => 40,
            Self::VitGigantic384 => 48,
        }
    }

    pub const fn encoder_heads(self) -> usize {
        match self {
            Self::VitBase384 => 12,
            Self::VitLarge384 => 16,
            Self::VitGiant384 => 16,
            Self::VitGigantic384 => 16,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VJepaConfig {
    pub model_type: String,
    pub variant: VJepaModelVariant,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_frames: usize,
    pub tubelet_size: usize,
    pub in_channels: usize,
    pub encoder: VJepaEncoderConfig,
    pub predictor: VJepaPredictorConfig,
    pub preprocess: VJepaPreprocessConfig,
}

impl Default for VJepaConfig {
    fn default() -> Self {
        let variant = VJepaModelVariant::VitBase384;
        Self {
            model_type: "vjepa2_1".to_string(),
            variant,
            image_size: 384,
            patch_size: 16,
            num_frames: 64,
            tubelet_size: 2,
            in_channels: 3,
            encoder: VJepaEncoderConfig::from_variant(variant),
            predictor: VJepaPredictorConfig::default(),
            preprocess: VJepaPreprocessConfig::default(),
        }
    }
}

impl VJepaConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("read config from {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse config {}", path.display()))
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            model_type: "vjepa2_1_tiny".to_string(),
            variant: VJepaModelVariant::VitBase384,
            image_size: 32,
            patch_size: 16,
            num_frames: 4,
            tubelet_size: 2,
            in_channels: 3,
            encoder: VJepaEncoderConfig {
                embed_dim: 32,
                depth: 2,
                num_heads: 4,
                mlp_ratio: 2.0,
                layer_norm_eps: 1.0e-6,
                use_rope: true,
                interpolate_rope: true,
                modality_embedding: true,
                n_output_distillation: 1,
            },
            predictor: VJepaPredictorConfig {
                embed_dim: 24,
                depth: 2,
                num_heads: 4,
                mlp_ratio: 2.0,
                num_mask_tokens: 2,
                output_dim: Some(32),
                return_all_tokens: true,
                layer_norm_eps: 1.0e-6,
                use_rope: true,
            },
            preprocess: VJepaPreprocessConfig::default(),
        }
    }

    pub fn with_variant(variant: VJepaModelVariant) -> Self {
        Self {
            variant,
            encoder: VJepaEncoderConfig::from_variant(variant),
            ..Self::default()
        }
    }

    pub const fn grid_height(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub const fn grid_width(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub const fn grid_depth(&self) -> usize {
        self.num_frames / self.tubelet_size
    }

    pub const fn num_patches(&self) -> usize {
        self.grid_depth() * self.grid_height() * self.grid_width()
    }

    pub const fn token_grid(&self) -> crate::TokenGridShape {
        crate::TokenGridShape::new(self.grid_depth(), self.grid_height(), self.grid_width())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VJepaEncoderConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub layer_norm_eps: f64,
    pub use_rope: bool,
    pub interpolate_rope: bool,
    pub modality_embedding: bool,
    pub n_output_distillation: usize,
}

impl VJepaEncoderConfig {
    pub fn from_variant(variant: VJepaModelVariant) -> Self {
        Self {
            embed_dim: variant.encoder_width(),
            depth: variant.encoder_depth(),
            num_heads: variant.encoder_heads(),
            ..Self::default()
        }
    }

    pub fn hierarchical_layers(&self) -> Vec<usize> {
        let all = match self.depth {
            12 => vec![2, 5, 8, 11],
            24 => vec![5, 11, 17, 23],
            40 => vec![9, 19, 29, 39],
            48 => vec![11, 23, 37, 47],
            depth => vec![depth.saturating_sub(1)],
        };
        let keep = self.n_output_distillation.max(1).min(all.len());
        all[all.len() - keep..].to_vec()
    }
}

impl Default for VJepaEncoderConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
            interpolate_rope: true,
            modality_embedding: true,
            n_output_distillation: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VJepaPredictorConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub num_mask_tokens: usize,
    pub output_dim: Option<usize>,
    pub return_all_tokens: bool,
    pub layer_norm_eps: f64,
    pub use_rope: bool,
}

impl Default for VJepaPredictorConfig {
    fn default() -> Self {
        Self {
            embed_dim: 384,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            num_mask_tokens: 8,
            output_dim: None,
            return_all_tokens: true,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VJepaPreprocessConfig {
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for VJepaPreprocessConfig {
    fn default() -> Self {
        Self {
            rescale_factor: 1.0 / 255.0,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vjepa_2_1_base_shape_matches_upstream_defaults() {
        let config = VJepaConfig::default();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.num_frames, 64);
        assert_eq!(config.tubelet_size, 2);
        assert_eq!(config.grid_depth(), 32);
        assert_eq!(config.grid_height(), 24);
        assert_eq!(config.num_patches(), 18_432);
        assert_eq!(config.encoder.hierarchical_layers(), vec![11]);
    }

    #[test]
    fn tiny_shape_stays_small_for_ci() {
        let config = VJepaConfig::tiny_for_tests();
        assert_eq!(config.num_patches(), 8);
        assert_eq!(config.encoder.hierarchical_layers(), vec![1]);
    }
}
