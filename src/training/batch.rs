use super::config::TrainingBatchingMode;
use crate::{
    JepaDataset, JepaDatasetConfig, JepaSampleMetadata, JepaTensorBatch, VJepaConfig,
    load_jepa_tensor_batch,
};
use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub(super) fn load_training_batch<B: Backend>(
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    device: &B::Device,
    start_index: usize,
    batch_size: usize,
) -> Result<JepaTensorBatch<B>> {
    load_training_batch_with_policy(
        dataset,
        dataset_config,
        model_config,
        device,
        start_index,
        batch_size,
        TrainingBatchingMode::Sequential,
    )
}

pub(super) fn load_training_batch_with_policy<B: Backend>(
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    device: &B::Device,
    start_index: usize,
    batch_size: usize,
    batching: TrainingBatchingMode,
) -> Result<JepaTensorBatch<B>> {
    let indices = sample_indices(dataset, start_index, batch_size, batching)?;
    let mut students = Vec::with_capacity(batch_size);
    let mut teachers = Vec::with_capacity(batch_size);
    let mut metadata = Vec::with_capacity(batch_size);
    for index in indices {
        let sample = dataset.sample(index)?;
        let batch = load_jepa_tensor_batch::<B>(&sample, dataset_config, model_config, device)?;
        students.push(batch.student);
        teachers.push(batch.teacher);
        metadata.extend(batch.metadata);
    }
    Ok(JepaTensorBatch {
        student: Tensor::cat(students, 0),
        teacher: Tensor::cat(teachers, 0),
        metadata,
    })
}

fn sample_indices(
    dataset: &dyn JepaDataset,
    start_index: usize,
    batch_size: usize,
    batching: TrainingBatchingMode,
) -> Result<Vec<usize>> {
    let batch_size = batch_size.max(1);
    if batching == TrainingBatchingMode::Sequential || dataset.is_empty() {
        return Ok((0..batch_size).map(|offset| start_index + offset).collect());
    }

    let anchor = dataset.sample(start_index)?;
    let anchor_metadata = anchor.metadata().cloned().unwrap_or_default();
    let Some(anchor_key) = mask_batch_key(&anchor_metadata, batching) else {
        return Ok((0..batch_size).map(|offset| start_index + offset).collect());
    };
    let mut indices = vec![start_index];
    let scan_len = dataset.len().max(batch_size);
    for offset in 1..=scan_len {
        if indices.len() >= batch_size {
            break;
        }
        let index = start_index + offset;
        let sample = dataset.sample(index)?;
        let metadata = sample.metadata().cloned().unwrap_or_default();
        if mask_batch_key(&metadata, batching).as_ref() == Some(&anchor_key) {
            indices.push(index);
        }
    }
    if indices.len() < batch_size {
        for offset in 1..batch_size {
            let index = start_index + offset;
            if !indices.contains(&index) {
                indices.push(index);
            }
            if indices.len() >= batch_size {
                break;
            }
        }
    }
    Ok(indices)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MaskBatchKey {
    Uniform {
        context: Vec<usize>,
        target: Vec<usize>,
    },
    FixedWidth {
        context_len: usize,
        target_len: usize,
    },
}

fn mask_batch_key(
    metadata: &JepaSampleMetadata,
    batching: TrainingBatchingMode,
) -> Option<MaskBatchKey> {
    let context = metadata.precomputed_context_indices.as_ref()?;
    let target = metadata.precomputed_target_indices.as_ref()?;
    match batching {
        TrainingBatchingMode::Sequential => None,
        TrainingBatchingMode::GroupUniformMasks => Some(MaskBatchKey::Uniform {
            context: context.clone(),
            target: target.clone(),
        }),
        TrainingBatchingMode::FixedWidthMasks => Some(MaskBatchKey::FixedWidth {
            context_len: context.len(),
            target_len: target.len(),
        }),
    }
}
