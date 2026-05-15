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
    let planner = TrainingBatchPlanner::new(dataset, batching)?;
    planner.load_batch(
        dataset,
        dataset_config,
        model_config,
        device,
        start_index,
        batch_size,
    )
}

#[derive(Clone, Debug)]
pub(super) struct TrainingBatchPlanner {
    batching: TrainingBatchingMode,
    packed_streams: Option<Vec<Vec<usize>>>,
}

impl TrainingBatchPlanner {
    pub(super) fn new(dataset: &dyn JepaDataset, batching: TrainingBatchingMode) -> Result<Self> {
        let packed_streams = (batching == TrainingBatchingMode::PackedStreams)
            .then(|| stream_index_groups(dataset))
            .transpose()?;
        Ok(Self {
            batching,
            packed_streams,
        })
    }

    pub(super) fn load_batch<B: Backend>(
        &self,
        dataset: &dyn JepaDataset,
        dataset_config: &JepaDatasetConfig,
        model_config: &VJepaConfig,
        device: &B::Device,
        start_index: usize,
        batch_size: usize,
    ) -> Result<JepaTensorBatch<B>> {
        let indices = self.indices(dataset, start_index, batch_size)?;
        load_training_batch_from_indices(dataset, dataset_config, model_config, device, &indices)
    }

    fn indices(
        &self,
        dataset: &dyn JepaDataset,
        start_index: usize,
        batch_size: usize,
    ) -> Result<Vec<usize>> {
        if let Some(groups) = &self.packed_streams {
            return Ok(packed_stream_indices(groups, start_index, batch_size));
        }
        sample_indices(dataset, start_index, batch_size, self.batching)
    }
}

fn load_training_batch_from_indices<B: Backend>(
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    device: &B::Device,
    indices: &[usize],
) -> Result<JepaTensorBatch<B>> {
    let mut students = Vec::with_capacity(indices.len());
    let mut teachers = Vec::with_capacity(indices.len());
    let mut metadata = Vec::with_capacity(indices.len());
    for &index in indices {
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
    if batching == TrainingBatchingMode::PackedStreams {
        let groups = stream_index_groups(dataset)?;
        return Ok(packed_stream_indices(&groups, start_index, batch_size));
    }
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

fn packed_stream_indices(
    groups: &[Vec<usize>],
    start_index: usize,
    requested_batch_size: usize,
) -> Vec<usize> {
    let batch_size = requested_batch_size.max(1).min(groups.len().max(1));
    if groups.is_empty() {
        return Vec::new();
    }
    let step = start_index / requested_batch_size.max(1);
    let slot = step * batch_size;
    // Flatten stream/window coordinates in round-robin order. A stream may be
    // absent for several optimizer steps when there are more streams than batch
    // rows, but its next sampled window is still monotonic and the TTT state is
    // restored by stream key.
    (0..batch_size)
        .map(|lane| {
            let position = slot + lane;
            let stream = position % groups.len();
            let window = position / groups.len();
            let group = &groups[stream];
            group[window % group.len()]
        })
        .collect()
}

fn stream_index_groups(dataset: &dyn JepaDataset) -> Result<Vec<Vec<usize>>> {
    let mut order = Vec::<StreamBatchKey>::new();
    let mut groups = std::collections::BTreeMap::<StreamBatchKey, Vec<usize>>::new();
    for index in 0..dataset.len() {
        let sample = dataset.sample(index)?;
        let metadata = sample.metadata().cloned().unwrap_or_default();
        let key = stream_batch_key(&metadata).ok_or_else(|| {
            anyhow::anyhow!(
                "training.batching=\"packed_streams\" requires clip_id or source metadata on row {}",
                index + 1
            )
        })?;
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(index);
    }
    Ok(order
        .into_iter()
        .filter_map(|key| groups.remove(&key))
        .collect())
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct StreamBatchKey {
    clip_id: Option<String>,
    domain: Option<String>,
    source: Option<String>,
}

fn stream_batch_key(metadata: &JepaSampleMetadata) -> Option<StreamBatchKey> {
    (metadata.clip_id.is_some() || metadata.source.is_some()).then(|| StreamBatchKey {
        clip_id: metadata.clip_id.clone(),
        domain: metadata.domain.clone(),
        source: metadata.source.clone(),
    })
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
        TrainingBatchingMode::PackedStreams => None,
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

#[cfg(test)]
mod tests {
    use super::packed_stream_indices;

    #[test]
    fn packed_stream_indices_round_robin_more_streams_than_batch() {
        let groups = vec![
            vec![0, 1, 2],
            vec![10, 11, 12],
            vec![20, 21, 22],
            vec![30, 31, 32],
            vec![40, 41, 42],
        ];

        assert_eq!(packed_stream_indices(&groups, 0, 2), vec![0, 10]);
        assert_eq!(packed_stream_indices(&groups, 2, 2), vec![20, 30]);
        assert_eq!(packed_stream_indices(&groups, 4, 2), vec![40, 1]);
        assert_eq!(packed_stream_indices(&groups, 6, 2), vec![11, 21]);
        assert_eq!(packed_stream_indices(&groups, 8, 2), vec![31, 41]);
        assert_eq!(packed_stream_indices(&groups, 10, 2), vec![2, 12]);
    }
}
