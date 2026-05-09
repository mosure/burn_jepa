use bevy::prelude::*;
use burn::tensor::Tensor;
use burn_jepa::{VJepaConfig, VJepaPipeline, make_context_target_masks};

type ExampleBackend = burn::backend::NdArray<f32>;

#[derive(Clone, Debug, Resource)]
pub struct BevyBurnJepaConfig {
    pub context_keep_ratio: f32,
}

impl Default for BevyBurnJepaConfig {
    fn default() -> Self {
        Self {
            context_keep_ratio: 0.5,
        }
    }
}

#[derive(Clone, Debug, Default, Resource)]
pub struct BevyBurnJepaStatus {
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub embedding_dim: usize,
}

pub struct BevyBurnJepaPlugin;

impl Plugin for BevyBurnJepaPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BevyBurnJepaConfig>()
            .init_resource::<BevyBurnJepaStatus>()
            .add_systems(Startup, run_sparse_smoke);
    }
}

pub fn run_once() -> BevyBurnJepaStatus {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BevyBurnJepaPlugin);
    app.update();
    app.world().resource::<BevyBurnJepaStatus>().clone()
}

fn run_sparse_smoke(
    config_resource: Res<BevyBurnJepaConfig>,
    mut status: ResMut<BevyBurnJepaStatus>,
) {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<ExampleBackend>::random(config.clone(), &device);
    let video = Tensor::<ExampleBackend, 5>::zeros([1, 3, 4, 32, 32], &device);
    let (context, target) =
        make_context_target_masks(config.token_grid(), config_resource.context_keep_ratio);
    let output = pipeline
        .model()
        .predict_dense_targets(video, &context, &target)
        .expect("tiny V-JEPA forward");
    let shape = output.predictions.shape().dims::<3>();
    status.context_tokens = context.len();
    status.target_tokens = target.len();
    status.embedding_dim = shape[2];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bevy_plugin_runs_sparse_smoke_once() {
        let status = run_once();
        assert_eq!(status.context_tokens, 4);
        assert_eq!(status.target_tokens, 4);
        assert_eq!(status.embedding_dim, 32);
    }
}
