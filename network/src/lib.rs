mod data_structures;
mod helpers;
pub use data_structures::activations::{Activation, Activations};
pub use data_structures::batch::Batch;
pub use data_structures::neural_net::NeuralNet;

pub trait BatchProvider {
    fn provide_batch(&mut self) -> Option<Batch>;
    fn reset(&mut self);
}

pub trait Evaluable {
    fn create_inference_batch(&self) -> Batch;
    fn set_prediction(&mut self, batch: Batch);
}
