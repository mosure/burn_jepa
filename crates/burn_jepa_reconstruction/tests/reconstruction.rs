use burn::tensor::{Tensor, TensorData};
use burn_jepa_reconstruction::{
    JepaReconstructionConfig, JepaReconstructionDecoder, JepaReconstructionTrainConfig,
    fit_reconstruction_decoder, reconstruction_color_moment_loss, reconstruction_gradient_mse,
    reconstruction_psnr_scalar,
};

type B = burn::backend::NdArray<f32>;

#[test]
fn decoder_returns_image_at_patch_scaled_resolution() {
    let device = Default::default();
    let config = JepaReconstructionConfig::tiny_for_tests();
    let decoder = JepaReconstructionDecoder::<B>::new(config.clone(), &device)
        .expect("reconstruction decoder");
    let features = Tensor::<B, 4>::ones([1, config.input_dim, 3, 5], &device);

    let output = decoder.forward(features);

    assert_eq!(output.shape().dims::<4>(), [1, 3, 12, 20]);
    assert!(values(output).iter().all(|value| value.is_finite()));
}

#[test]
fn residual_blocks_per_scale_do_not_change_output_scale() {
    let device = Default::default();
    let mut config = JepaReconstructionConfig::tiny_for_tests();
    config.residual_blocks_per_scale = 3;
    let decoder = JepaReconstructionDecoder::<B>::new(config.clone(), &device)
        .expect("reconstruction decoder");
    let features = Tensor::<B, 4>::ones([1, config.input_dim, 3, 5], &device);

    let output = decoder.forward(features);

    assert_eq!(config.output_scale(), 4);
    assert_eq!(output.shape().dims::<4>(), [1, 3, 12, 20]);
}

#[test]
fn psnr_is_higher_for_closer_reconstruction() {
    let device = Default::default();
    let target = Tensor::<B, 4>::ones([1, 3, 4, 4], &device);
    let good = Tensor::<B, 4>::ones([1, 3, 4, 4], &device).mul_scalar(0.98);
    let bad = Tensor::<B, 4>::zeros([1, 3, 4, 4], &device);

    let good_psnr = reconstruction_psnr_scalar(good, target.clone(), 1.0).expect("good psnr");
    let bad_psnr = reconstruction_psnr_scalar(bad, target, 1.0).expect("bad psnr");

    assert!(good_psnr > bad_psnr);
}

#[test]
fn gradient_loss_penalizes_blurry_reconstruction() {
    let device = Default::default();
    let target = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..8 * 8)
                .flat_map(|index| {
                    let x = index % 8;
                    let value = if x < 4 { 0.0 } else { 1.0 };
                    [value, value, value]
                })
                .collect::<Vec<_>>(),
            [1, 3, 8, 8],
        ),
        &device,
    );
    let sharp = target.clone();
    let blurry = Tensor::<B, 4>::ones([1, 3, 8, 8], &device).mul_scalar(0.5);

    let sharp_loss = scalar(reconstruction_gradient_mse(sharp, target.clone()));
    let blurry_loss = scalar(reconstruction_gradient_mse(blurry, target));

    assert!(blurry_loss > sharp_loss);
}

#[test]
fn color_moment_loss_penalizes_washed_out_contrast() {
    let device = Default::default();
    let target = Tensor::<B, 4>::from_data(
        TensorData::new(
            (0..3 * 8 * 8)
                .map(|index| if index % 2 == 0 { 0.15 } else { 0.85 })
                .collect::<Vec<_>>(),
            [1, 3, 8, 8],
        ),
        &device,
    );
    let good = target.clone();
    let washed = Tensor::<B, 4>::ones([1, 3, 8, 8], &device).mul_scalar(0.5);

    let good_loss = scalar(reconstruction_color_moment_loss(good, target.clone()));
    let washed_loss = scalar(reconstruction_color_moment_loss(washed, target));

    assert!(washed_loss > good_loss);
}

#[test]
fn training_step_reduces_tiny_oracle_loss() {
    type AB = burn::backend::Autodiff<B>;

    let device = Default::default();
    let mut decoder = JepaReconstructionConfig::tiny_for_tests();
    decoder.hidden_dim = 8;
    decoder.residual_blocks_per_scale = 1;
    let config = JepaReconstructionTrainConfig {
        decoder: decoder.clone(),
        steps: 4,
        learning_rate: 1.0e-3,
        weight_decay: 0.0,
        l1_loss_weight: 0.02,
        gradient_loss_weight: 0.05,
        color_loss_weight: 0.02,
        log_interval: 1,
    };
    let features = Tensor::<AB, 4>::from_data(
        TensorData::new(
            (0..decoder.input_dim * 2 * 2)
                .map(|index| (index as f32).sin() * 0.05)
                .collect::<Vec<_>>(),
            [1, decoder.input_dim, 2, 2],
        ),
        &device,
    );
    let target = Tensor::<AB, 4>::from_data(
        TensorData::new(
            (0..3 * 8 * 8)
                .map(|index| 0.5 + (index as f32).cos() * 0.1)
                .collect::<Vec<_>>(),
            [1, 3, 8, 8],
        ),
        &device,
    );

    let (_decoder, report) =
        fit_reconstruction_decoder(config, features, target, &device).expect("fit decoder");

    assert_eq!(report.steps, 4);
    assert!(report.initial_loss.is_some());
    assert!(report.final_loss.is_some());
    assert!(report.best_loss.is_some());
    assert!(report.final_loss.unwrap() <= report.initial_loss.unwrap());
    assert!(report.best_loss.unwrap() <= report.initial_loss.unwrap());
}

fn values(tensor: Tensor<B, 4>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn scalar(tensor: Tensor<B, 1>) -> f32 {
    tensor
        .to_data()
        .to_vec::<f32>()
        .expect("tensor value")
        .into_iter()
        .next()
        .expect("scalar")
}
