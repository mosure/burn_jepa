use super::config::BurnJepaTrainConfig;
use crate::{VJepa2_1Model, VJepaConfig, VJepaLoadOptions};
use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use std::path::PathBuf;

pub(super) fn load_student_model<B: Backend>(
    config: &BurnJepaTrainConfig,
    device: &B::Device,
) -> Result<VJepa2_1Model<B>> {
    load_model_from_config(
        config.model.checkpoint_dir.as_ref(),
        config.model.config_path.as_ref(),
        config.model.weights_name.as_deref(),
        device,
    )
}

pub(super) fn load_teacher_model<B: Backend>(
    config: &BurnJepaTrainConfig,
    device: &B::Device,
) -> Result<VJepa2_1Model<B>> {
    load_model_from_config(
        config
            .model
            .teacher_checkpoint_dir
            .as_ref()
            .or(config.model.checkpoint_dir.as_ref()),
        config.model.config_path.as_ref(),
        config.model.weights_name.as_deref(),
        device,
    )
    .map(Module::no_grad)
}

fn load_model_from_config<B: Backend>(
    checkpoint_dir: Option<&PathBuf>,
    config_path: Option<&PathBuf>,
    weights_name: Option<&str>,
    device: &B::Device,
) -> Result<VJepa2_1Model<B>> {
    if let Some(checkpoint_dir) = checkpoint_dir {
        let mut options = VJepaLoadOptions::default();
        if let Some(weights_name) = weights_name {
            options.weights_name = weights_name.to_string();
        }
        let (model, _config, _report) = options.load_model(checkpoint_dir, device)?;
        return Ok(model);
    }
    let config = if let Some(config_path) = config_path {
        VJepaConfig::from_json_file(config_path)?
    } else {
        VJepaConfig::tiny_for_tests()
    };
    Ok(VJepa2_1Model::new(&config, device))
}
