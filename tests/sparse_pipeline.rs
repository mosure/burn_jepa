use burn::tensor::Tensor;
use burn_jepa::{
    SparseJepaTensorPipeline, SparseJepaTensorPipelineConfig, TensorVideoInput, VJepaConfig,
    VJepaPipeline, VecOutputNode, make_context_target_masks,
};

type B = burn::backend::NdArray<f32>;

#[test]
fn sparse_pipeline_runs_one_packet() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let input = TensorVideoInput::new(video);
    let output = VecOutputNode::new();
    let mut graph = SparseJepaTensorPipeline::new(pipeline, input, output).with_config(
        SparseJepaTensorPipelineConfig {
            context_keep_ratio: 0.5,
        },
    );
    assert!(graph.run_next(&device).expect("run packet"));
    let output = graph.into_output();
    assert_eq!(output.packets.len(), 1);
}

#[test]
fn dense_prediction_output_matches_target_mask() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let (context, target) = make_context_target_masks(config.token_grid(), 0.5);
    let output = pipeline
        .model()
        .predict_dense_targets(video, &context, &target)
        .expect("predict");
    assert_eq!(output.predictions.shape().dims::<3>()[1], target.len());
    assert_eq!(output.targets.shape().dims::<3>()[1], target.len());
}
