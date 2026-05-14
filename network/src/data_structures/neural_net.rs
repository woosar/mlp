use crate::data_structures::activations::Activations;
use crate::data_structures::batch::Batch;

use crate::data_structures::net_propagator::NetPropagator;
use crate::data_structures::net_properties::NetProperties;
use crate::{BatchProvider, Evaluable};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

const LEARNING_RATE: f32 = 0.01;

#[derive(Debug)]
pub struct NeuralNet {
    activation: Activations,
    properties: Arc<NetProperties>,
    parameters: Arc<RwLock<Arc<[f32]>>>,
    // transformer: Arc<dyn Transformer>,
}

impl NeuralNet {
    pub fn new(
        layout: Vec<usize>,
        activation: Activations,
        parameters: Option<Vec<f32>>,
        is_classifier: bool,
        // transformer: Arc<dyn Transformer>,
    ) -> Self {
        let parameters = if let Some(external_parameters) = parameters {
            Arc::<[f32]>::from(external_parameters)
        } else {
            Self::initialize_randomized_parameters(&layout)
        };
        let properties = NetProperties::new(layout, is_classifier);

        Self {
            activation,
            properties: Arc::new(properties),
            parameters: Arc::new(RwLock::new(parameters)),
            // transformer,
        }
    }

    fn create_propagator(&self, current_parameters: Arc<[f32]>) -> NetPropagator {
        NetPropagator::new(self.activation, current_parameters, self.properties.clone())
    }

    fn update_parameters(&self, gradient: Vec<f32>) {
        let current_params = Arc::clone(&*self.parameters.read().unwrap());
        let new_arc: Arc<[f32]> = current_params
            .iter()
            .zip(gradient.iter())
            .map(|(w, g)| w - (0.01 * g))
            .collect();
        let mut writer = self.parameters.write().unwrap();
        *writer = new_arc;
    }

    pub fn evaluate_data<A: Evaluable>(&self, dataset: &mut A) {
        let mut batch = dataset.create_inference_batch();
        self.make_inference_on_batch(&mut batch);
        dataset.set_prediction(batch)
    }

    pub fn make_inference_on_batch(&self, batch: &mut Batch) {
        let current_params = self.parameters.read().unwrap().clone();
        let mut propagator = self.create_propagator(current_params.clone());

        for sample_number in 0..batch.number_of_samples() {
            let input = batch.input(sample_number);
            let output = propagator.forward_pass(input);

            batch.set_output(sample_number, output)
        }
    }

    pub fn train_on_epoch<T: BatchProvider>(&self, data: &mut T) {
        while let Some(batch) = data.provide_batch() {
            self.backprop(&batch)
        }
    }

    pub fn train<T: BatchProvider>(&self, data: &mut T, epochs: usize) {
        let mut counter = 0;
        while counter < epochs {
            // println!("{counter}");
            if counter % 1000 == 0 {
                println!(
                    "epoch: {counter}/{epochs}, {:.2}",
                    100.0 * (counter as f32) / (epochs as f32)
                )
            }

            data.reset();
            self.train_on_epoch(data);
            counter += 1;
        }
    }

    pub fn backprop(&self, batch: &Batch) {
        let current_params = self.parameters.read().unwrap().clone();
        let num_params = self.properties.number_of_parameters();

        let chunk_size = (batch.number_of_samples() / rayon::current_num_threads()).max(1);

        let mut total_gradient = (0..batch.number_of_samples())
            .collect::<Vec<usize>>()
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut propagator = self.create_propagator(current_params.clone());
                let mut chunk_gradient = vec![0.0; num_params];

                for &i in chunk {
                    let input = batch.input(i); // transform here observation wise, because everything else assumes too much
                    let target = batch.output_sample(i);
                    propagator.backprop(input, target, &mut chunk_gradient);
                }

                chunk_gradient
            })
            .reduce(
                || vec![0.0; num_params],
                |mut a, b| {
                    for (i, val) in b.iter().enumerate() {
                        a[i] += val;
                    }
                    a
                },
            );

        let number = batch.number_of_samples() as f32;
        total_gradient.iter_mut().for_each(|elem| *elem /= number);
        self.update_parameters(total_gradient);
    }

    fn initialize_randomized_parameters(layout: &[usize]) -> Arc<[f32]> {
        let mut rng = rand::rng(); // This is the modern entry point
        let mut params = Vec::new();

        for i in 0..layout.len() - 1 {
            let n_in = layout[i] as f32;
            let n_out = layout[i + 1];

            let std_dev = (2.0 / n_in).sqrt();
            let dist = Normal::new(0.0, std_dev).expect("Invalid distribution parameters");

            let source_count = layout[i] + 1;
            let target_count = n_out;

            for _ in 0..(source_count * target_count) {
                params.push(dist.sample(&mut rng));
            }
        }

        Arc::from(params)
    }
}

impl Clone for NeuralNet {
    fn clone(&self) -> Self {
        Self {
            activation: self.activation.clone(),
            properties: Arc::clone(&self.properties),
            parameters: Arc::clone(&self.parameters),
        }
    }
}
