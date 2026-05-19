use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

pub fn reconstruction_mse<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    (reconstructed - target).powf_scalar(2.0).mean()
}

pub fn reconstruction_psnr<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
    peak: f64,
) -> Tensor<B, 1> {
    let mse = reconstruction_mse(reconstructed, target).add_scalar(1.0e-12);
    let ratio = mse.recip().mul_scalar(peak * peak);
    ratio.log().mul_scalar(10.0 / std::f64::consts::LN_10)
}

pub fn reconstruction_psnr_scalar<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
    peak: f64,
) -> Option<f64> {
    tensor_scalar(reconstruction_psnr(reconstructed, target, peak).detach())
}

fn tensor_scalar<B: Backend>(tensor: Tensor<B, 1>) -> Option<f64> {
    let data: TensorData = tensor.to_data();
    data.to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .map(f64::from)
}
