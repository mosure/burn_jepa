use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

pub fn reconstruction_mse<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    (reconstructed - target).powf_scalar(2.0).mean()
}

pub fn reconstruction_l1<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    (reconstructed - target).abs().mean()
}

pub fn reconstruction_gradient_mse<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let [batch, channels, height, width] = reconstructed.shape().dims::<4>();
    let device = reconstructed.device();
    let mut terms = Vec::with_capacity(2);
    if width > 1 {
        let recon_dx = reconstructed
            .clone()
            .slice([0..batch, 0..channels, 0..height, 1..width])
            - reconstructed
                .clone()
                .slice([0..batch, 0..channels, 0..height, 0..width - 1]);
        let target_dx = target
            .clone()
            .slice([0..batch, 0..channels, 0..height, 1..width])
            - target
                .clone()
                .slice([0..batch, 0..channels, 0..height, 0..width - 1]);
        terms.push((recon_dx - target_dx).powf_scalar(2.0).mean());
    }
    if height > 1 {
        let recon_dy = reconstructed
            .clone()
            .slice([0..batch, 0..channels, 1..height, 0..width])
            - reconstructed
                .clone()
                .slice([0..batch, 0..channels, 0..height - 1, 0..width]);
        let target_dy = target
            .clone()
            .slice([0..batch, 0..channels, 1..height, 0..width])
            - target
                .clone()
                .slice([0..batch, 0..channels, 0..height - 1, 0..width]);
        terms.push((recon_dy - target_dy).powf_scalar(2.0).mean());
    }
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let count = terms.len().max(1) as f64;
    for term in terms {
        total = total + term;
    }
    total / count
}

pub fn reconstruction_color_moment_loss<B: Backend>(
    reconstructed: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let recon_mean = channel_mean(reconstructed.clone());
    let target_mean = channel_mean(target.clone());
    let mean_loss = (recon_mean.clone() - target_mean.clone())
        .powf_scalar(2.0)
        .mean();

    let recon_std = channel_std(reconstructed, recon_mean);
    let target_std = channel_std(target, target_mean);
    let std_loss = (recon_std - target_std).powf_scalar(2.0).mean();
    mean_loss + std_loss
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

fn channel_mean<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    tensor.mean_dim(3).mean_dim(2).mean_dim(0)
}

fn channel_std<B: Backend>(tensor: Tensor<B, 4>, mean: Tensor<B, 4>) -> Tensor<B, 4> {
    (tensor - mean)
        .powf_scalar(2.0)
        .mean_dim(3)
        .mean_dim(2)
        .mean_dim(0)
        .add_scalar(1.0e-8)
        .sqrt()
}
