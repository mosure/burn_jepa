use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct TttLayerState<B: Backend> {
    pub fast_weight: Option<Tensor<B, 3>>,
}

impl<B: Backend> TttLayerState<B> {
    pub fn empty() -> Self {
        Self { fast_weight: None }
    }

    pub fn detach(&mut self) {
        if let Some(weight) = self.fast_weight.take() {
            self.fast_weight = Some(weight.detach());
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
}
