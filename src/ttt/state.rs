use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct TttLayerState<B: Backend> {
    pub fast_weight: Option<Tensor<B, 3>>,
    pub fast_weight_banks: Option<Tensor<B, 4>>,
}

impl<B: Backend> TttLayerState<B> {
    pub fn empty() -> Self {
        Self {
            fast_weight: None,
            fast_weight_banks: None,
        }
    }

    pub fn detach(&mut self) {
        if let Some(weight) = self.fast_weight.take() {
            self.fast_weight = Some(weight.detach());
        }
        if let Some(weight) = self.fast_weight_banks.take() {
            self.fast_weight_banks = Some(weight.detach());
        }
    }

    pub fn decay(&mut self, factor: f64) {
        if let Some(weight) = self.fast_weight.take() {
            self.fast_weight = Some(weight.mul_scalar(factor));
        }
        if let Some(weight) = self.fast_weight_banks.take() {
            self.fast_weight_banks = Some(weight.mul_scalar(factor));
        }
    }
}

#[derive(Clone, Debug)]
pub struct TttState<B: Backend> {
    pub layers: Vec<TttLayerState<B>>,
}

impl<B: Backend> TttState<B> {
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: (0..layer_count).map(|_| TttLayerState::empty()).collect(),
        }
    }

    pub fn detach(&mut self) {
        for layer in &mut self.layers {
            layer.detach();
        }
    }

    pub fn decay(&mut self, factor: f64) {
        for layer in &mut self.layers {
            layer.decay(factor);
        }
    }

    pub fn has_fast_weights(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| layer.fast_weight.is_some() || layer.fast_weight_banks.is_some())
    }

    pub fn select_rows(&self, rows: &[usize]) -> Self {
        Self {
            layers: self
                .layers
                .iter()
                .map(|layer| TttLayerState {
                    fast_weight: layer
                        .fast_weight
                        .as_ref()
                        .map(|weight| select_rows3(weight.clone(), rows)),
                    fast_weight_banks: layer
                        .fast_weight_banks
                        .as_ref()
                        .map(|weight| select_rows4(weight.clone(), rows)),
                })
                .collect(),
        }
    }

    pub fn unpack_rows(&self, rows: usize) -> Vec<Self> {
        (0..rows)
            .map(|row| Self {
                layers: self
                    .layers
                    .iter()
                    .map(|layer| TttLayerState {
                        fast_weight: layer
                            .fast_weight
                            .as_ref()
                            .map(|weight| weight.clone().slice_dim(0, row..row + 1)),
                        fast_weight_banks: layer
                            .fast_weight_banks
                            .as_ref()
                            .map(|weight| weight.clone().slice_dim(0, row..row + 1)),
                    })
                    .collect(),
            })
            .collect()
    }

    pub fn pack_rows(row_states: &[Self]) -> Self {
        let layer_count = row_states
            .iter()
            .map(|state| state.layers.len())
            .max()
            .unwrap_or(0);
        let layers = (0..layer_count)
            .map(|layer_index| {
                let fast_template = row_states.iter().find_map(|state| {
                    state
                        .layers
                        .get(layer_index)
                        .and_then(|layer| layer.fast_weight.as_ref())
                });
                let fast_weight = fast_template.map(|template| {
                    let [_, rows, cols] = template.shape().dims::<3>();
                    Tensor::cat(
                        row_states
                            .iter()
                            .map(|state| {
                                state
                                    .layers
                                    .get(layer_index)
                                    .and_then(|layer| layer.fast_weight.as_ref())
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        Tensor::<B, 3>::zeros([1, rows, cols], &template.device())
                                    })
                            })
                            .collect(),
                        0,
                    )
                });
                let bank_template = row_states.iter().find_map(|state| {
                    state
                        .layers
                        .get(layer_index)
                        .and_then(|layer| layer.fast_weight_banks.as_ref())
                });
                let fast_weight_banks = bank_template.map(|template| {
                    let [_, banks, rows, cols] = template.shape().dims::<4>();
                    Tensor::cat(
                        row_states
                            .iter()
                            .map(|state| {
                                state
                                    .layers
                                    .get(layer_index)
                                    .and_then(|layer| layer.fast_weight_banks.as_ref())
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        Tensor::<B, 4>::zeros(
                                            [1, banks, rows, cols],
                                            &template.device(),
                                        )
                                    })
                            })
                            .collect(),
                        0,
                    )
                });
                TttLayerState {
                    fast_weight,
                    fast_weight_banks,
                }
            })
            .collect();
        Self { layers }
    }
}

fn select_rows3<B: Backend>(tensor: Tensor<B, 3>, rows: &[usize]) -> Tensor<B, 3> {
    Tensor::cat(
        rows.iter()
            .map(|&row| tensor.clone().slice_dim(0, row..row + 1))
            .collect(),
        0,
    )
}

fn select_rows4<B: Backend>(tensor: Tensor<B, 4>, rows: &[usize]) -> Tensor<B, 4> {
    Tensor::cat(
        rows.iter()
            .map(|&row| tensor.clone().slice_dim(0, row..row + 1))
            .collect(),
        0,
    )
}
