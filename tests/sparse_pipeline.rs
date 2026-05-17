use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    SparseImageTokenGrid, SparseJepaAutogazeSparsityConfig, SparseJepaPatchDiffSparsityConfig,
    SparseJepaSparsityDriverConfig, SparseJepaTensorPipeline, SparseJepaTensorPipelineConfig,
    TensorVideoInput, VJepaConfig, VJepaPipeline, VecOutputNode, coords_to_token_index,
    make_context_target_masks,
};

#[cfg(any(feature = "ndarray", not(feature = "wgpu")))]
type B = burn::backend::NdArray<f32>;
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
type B = burn::backend::Wgpu<f32, i32>;

#[test]
fn sparse_pipeline_runs_one_packet() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let input = TensorVideoInput::new(video);
    let output = VecOutputNode::new();
    let mut graph = SparseJepaTensorPipeline::new(pipeline, input, output)
        .with_config(SparseJepaTensorPipelineConfig::keep_ratio(0.5));
    assert!(graph.run_next(&device).expect("run packet"));
    let output = graph.into_output();
    assert_eq!(output.packets.len(), 1);
}

#[test]
fn sparse_pipeline_full_frame_driver_uses_dense_context_budget() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let input = TensorVideoInput::new(video);
    let output = VecOutputNode::new();
    let mut graph = SparseJepaTensorPipeline::new(pipeline, input, output).with_config(
        SparseJepaTensorPipelineConfig::default()
            .with_sparsity_driver(SparseJepaSparsityDriverConfig::full_frame(2)),
    );

    assert!(graph.run_next(&device).expect("run packet"));
    let output = graph.into_output();
    let packet = output.packets.first().expect("packet");
    assert_eq!(packet.target_mask.len(), 2);
    assert_eq!(
        packet.context_mask.len() + packet.target_mask.len(),
        config.token_grid().len()
    );
}

#[test]
fn vjepa_pipeline_accepts_sparsity_driver_config() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config, &device)
        .with_sparsity_driver(SparseJepaSparsityDriverConfig::full_frame(2));
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let output = pipeline.predict_video_dense(video).expect("predict");

    assert_eq!(output.predictions.shape().dims::<3>()[1], 2);
    assert_eq!(output.targets.shape().dims::<3>()[1], 2);
}

#[test]
fn sparse_pipeline_autogaze_driver_projects_frame_tokens() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 32, 32], &device);
    let input = TensorVideoInput::new(video);
    let output = VecOutputNode::new();
    let frame_tokens = vec![vec![], vec![1], vec![2], vec![]];
    let driver = SparseJepaSparsityDriverConfig::AutogazeSparse(
        SparseJepaAutogazeSparsityConfig::new(SparseImageTokenGrid::new(2, 2), frame_tokens, 4, 2),
    );
    let mut graph = SparseJepaTensorPipeline::new(pipeline, input, output)
        .with_config(SparseJepaTensorPipelineConfig::default().with_sparsity_driver(driver));

    assert!(graph.run_next(&device).expect("run packet"));
    let output = graph.into_output();
    let packet = output.packets.first().expect("packet");
    assert_eq!(packet.context_mask.len(), 4);
    assert!(packet.context_mask.indices().contains(&1));
    assert!(packet.context_mask.indices().contains(&6));
    assert_eq!(packet.target_mask.len(), 2);
}

#[test]
fn sparse_pipeline_patch_diff_driver_selects_changed_patch() {
    let (config, packet) = patch_diff_packet(0.1);

    assert_eq!(
        packet.context_mask.indices(),
        &[coords_to_token_index(0, 1, 1, config.token_grid())]
    );
    assert_eq!(packet.target_mask.len(), 2);
}

#[test]
fn sparse_pipeline_patch_diff_fast_topk_selects_changed_patch() {
    let (config, packet) = patch_diff_packet(0.0);

    assert_eq!(
        packet.context_mask.indices(),
        &[coords_to_token_index(0, 1, 1, config.token_grid())]
    );
    assert_eq!(packet.target_mask.len(), 2);
}

fn patch_diff_packet(threshold: f32) -> (VJepaConfig, burn_jepa::SparseJepaPacket<B>) {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);
    let mut values = vec![0.0; 3 * 4 * 32 * 32];
    for channel in 0..3 {
        for y in 16..32 {
            for x in 16..32 {
                let offset = (((channel * 4 + 1) * 32 + y) * 32) + x;
                values[offset] = 1.0;
            }
        }
    }
    let video = Tensor::<B, 5>::from_data(TensorData::new(values, [1, 3, 4, 32, 32]), &device);
    let input = TensorVideoInput::new(video);
    let output = VecOutputNode::new();
    let driver = SparseJepaSparsityDriverConfig::PatchDiff(SparseJepaPatchDiffSparsityConfig::new(
        threshold, 1, 2,
    ));
    let mut graph = SparseJepaTensorPipeline::new(pipeline, input, output)
        .with_config(SparseJepaTensorPipelineConfig::default().with_sparsity_driver(driver));

    assert!(graph.run_next(&device).expect("run packet"));
    let output = graph.into_output();
    (config, output.packets.into_iter().next().expect("packet"))
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
