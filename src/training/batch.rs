use super::config::TrainingBatchingMode;
use crate::{
    JepaDataset, JepaDatasetConfig, JepaSampleMetadata, JepaTensorBatch, VJepaConfig,
    dataset::{JepaCpuTensorBatch, jepa_tensor_batch_from_cpu, load_jepa_cpu_tensor_batch},
    dataset_from_config,
};
use anyhow::{Context, Result, ensure};
use burn::tensor::backend::Backend;
use std::sync::mpsc::{Receiver, SyncSender, channel, sync_channel};
use std::thread::{self, JoinHandle};

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
    let batch =
        load_training_cpu_batch_from_indices(dataset, dataset_config, model_config, indices)?;
    Ok(jepa_tensor_batch_from_cpu(batch, device))
}

pub(super) fn materialize_training_batch<B: Backend>(
    batch: JepaCpuTensorBatch,
    device: &B::Device,
) -> JepaTensorBatch<B> {
    jepa_tensor_batch_from_cpu(batch, device)
}

pub(super) fn load_training_cpu_batch_from_indices(
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    indices: &[usize],
) -> Result<JepaCpuTensorBatch> {
    ensure!(
        !indices.is_empty(),
        "training batch indices must be non-empty"
    );
    let mut student_values = Vec::new();
    let mut teacher_values = None::<Vec<f32>>;
    let mut metadata = Vec::with_capacity(indices.len());
    let mut shape = None::<[usize; 5]>;
    for &index in indices {
        let sample = dataset.sample(index)?;
        let batch = load_jepa_cpu_tensor_batch(&sample, dataset_config, model_config)?;
        let sample_shape = batch.shape;
        ensure!(
            sample_shape[0] == 1,
            "dataset sample loader must produce single-sample CPU batches"
        );
        if let Some(shape) = &shape {
            ensure!(
                shape[1..] == sample_shape[1..],
                "all training batch samples must have matching C/T/H/W shapes"
            );
        } else {
            shape = Some(sample_shape);
        }
        if let Some(sample_teacher) = batch.teacher_values {
            if teacher_values.is_none() {
                teacher_values = Some(student_values.clone());
            }
            teacher_values
                .as_mut()
                .expect("teacher values initialized")
                .extend(sample_teacher);
        } else if let Some(teacher_values) = teacher_values.as_mut() {
            teacher_values.extend_from_slice(&batch.student_values);
        }
        student_values.extend(batch.student_values);
        metadata.extend(batch.metadata);
    }
    let mut shape = shape.context("training batch must contain at least one sample")?;
    shape[0] = indices.len();
    Ok(JepaCpuTensorBatch {
        student_values,
        teacher_values,
        shape,
        metadata,
    })
}

pub(super) fn cpu_batch_from_planner(
    planner: &TrainingBatchPlanner,
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    start_index: usize,
    batch_size: usize,
) -> Result<JepaCpuTensorBatch> {
    let indices = planner.indices(dataset, start_index, batch_size)?;
    load_training_cpu_batch_from_indices(dataset, dataset_config, model_config, &indices)
}

pub(super) struct TrainingBatchPrefetcher {
    request_tx: SyncSender<Option<usize>>,
    response_rx: Receiver<Result<JepaCpuTensorBatch>>,
    handle: Option<JoinHandle<()>>,
}

impl TrainingBatchPrefetcher {
    pub(super) fn new(
        dataset_config: JepaDatasetConfig,
        model_config: VJepaConfig,
        batching: TrainingBatchingMode,
        batch_size: usize,
    ) -> Result<Self> {
        let (request_tx, request_rx) = sync_channel::<Option<usize>>(1);
        let (response_tx, response_rx) = channel::<Result<JepaCpuTensorBatch>>();
        let handle = thread::Builder::new()
            .name("burn-jepa-batch-prefetch".to_string())
            .spawn(move || {
                let init = dataset_from_config(&dataset_config, true).and_then(|dataset| {
                    let planner = TrainingBatchPlanner::new(dataset.as_ref(), batching)?;
                    Ok((dataset, planner))
                });
                let (dataset, planner) = match init {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = response_tx.send(Err(error));
                        return;
                    }
                };
                while let Ok(Some(start_index)) = request_rx.recv() {
                    let batch = cpu_batch_from_planner(
                        &planner,
                        dataset.as_ref(),
                        &dataset_config,
                        &model_config,
                        start_index,
                        batch_size,
                    );
                    if response_tx.send(batch).is_err() {
                        break;
                    }
                }
            })
            .context("spawn JEPA training batch prefetch thread")?;
        Ok(Self {
            request_tx,
            response_rx,
            handle: Some(handle),
        })
    }

    pub(super) fn request(&self, start_index: usize) -> Result<()> {
        self.request_tx
            .send(Some(start_index))
            .context("request prefetched JEPA training batch")
    }

    pub(super) fn recv(&self) -> Result<JepaCpuTensorBatch> {
        self.response_rx
            .recv()
            .context("receive prefetched JEPA training batch")?
    }
}

impl Drop for TrainingBatchPrefetcher {
    fn drop(&mut self) {
        let _ = self.request_tx.send(None);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
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
