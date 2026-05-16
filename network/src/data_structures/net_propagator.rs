use crate::data_structures::activations::Activations;
use crate::data_structures::net_properties::NetProperties;
use crate::data_structures::signals::Signals;
use crate::data_structures::swap_buffer::SwapBuffer;
use std::sync::Arc;

#[derive(Debug)]
pub struct NetPropagator {
    swap_buffer: SwapBuffer,
    signals: Signals,
    activation: Activations,
    parameters: Arc<[f32]>,
    properties: Arc<NetProperties>,
}

impl NetPropagator {
    pub fn new(
        activation: Activations,
        parameters: Arc<[f32]>,
        properties: Arc<NetProperties>,
    ) -> Self {
        let swap_buffer = SwapBuffer::new(properties.capacity());
        let signals = Signals::new(properties.layout());
        Self {
            swap_buffer,
            signals,
            activation,
            parameters,
            properties,
        }
    }

    pub fn backprop(
        &mut self,
        input: &[f32],
        output: &[f32],
        local_gradient: &mut Vec<f32>,
        local_loss: &mut f32,
    ) {
        // forward pass to fill signals
        let new_output = self.forward_pass(input);

        // generate the first error signal
        // we assume MSE for regression and CE for classification
        // this yields the same first error signal
        let delta_l = output
            .iter()
            .zip(new_output)
            .map(|(y, z)| z - y)
            .collect::<Vec<_>>();

        // Regression -> Exact MSE Loss
        // Classification -> Brier Score (MSE on probabilities)
        // todo: ce loss later, maybe in the delta_l iterator parallely
        *local_loss += delta_l.iter().map(|d| d * d).sum::<f32>();

        {
            let (input_buffer, _) = self.swap_buffer.get_pair();
            input_buffer[0..delta_l.len()].copy_from_slice(&delta_l);
        }

        for (transition_idx, transition_offset_idx) in self
            .properties
            .transition_offsets()
            .iter()
            .enumerate()
            .rev()
        {
            // outer backprop loop
            let (input_buffer, output_buffer) = self.swap_buffer.get_pair();
            let source_activations = self.signals.activations(transition_idx);
            let source_neuron_count = self.properties.layout()[transition_idx] + 1; // including ghost
            let target_neuron_count = self.properties.layout()[transition_idx + 1];

            for i in 0..source_neuron_count {
                let parameter_offset = transition_offset_idx + i;
                let mut acc = 0.0;
                for j in 0..target_neuron_count {
                    let index = parameter_offset + j * source_neuron_count;
                    acc += self.parameters[index] * input_buffer[j];
                    local_gradient[index] += input_buffer[j] * source_activations[i];
                }
                output_buffer[i] = acc * (self.activation.derivative)(source_activations[i])
            }
            self.swap_buffer.swap();
        }
    }

    pub fn forward_pass(&mut self, input: &[f32]) -> &[f32] {
        let num_transitions = self.properties.transition_offsets().len();

        {
            let (input_buffer, _) = self.swap_buffer.get_pair();
            input_buffer[0..input.len()].copy_from_slice(input);
        }

        for transition_idx in 0..num_transitions {
            let transition_offset = self.properties.transition_offsets()[transition_idx];
            let n_source_layer = self.properties.layout()[transition_idx];
            let n_target_layer = self.properties.layout()[transition_idx + 1];

            let (input_buffer, output_buffer) = self.swap_buffer.get_pair();

            input_buffer[n_source_layer] = 1.0;
            self.signals.set_activations(transition_idx, input_buffer);

            let is_last_layer = transition_idx == num_transitions - 1;

            for i in 0..n_target_layer {
                let start_idx = transition_offset + i * (n_source_layer + 1);

                let mut acc: f32 = 0.0;

                for (j, &input_val) in input_buffer.iter().take(n_source_layer + 1).enumerate() {
                    acc += input_val * self.parameters[start_idx + j];
                }

                output_buffer[i] = if is_last_layer {
                    acc
                } else {
                    (self.activation.activation)(acc)
                };
            }
            // apply softmax for classifier
            if is_last_layer && self.properties.is_classifier() {
                let max_val = output_buffer
                    .iter()
                    .take(n_target_layer)
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let mut sum = 0.0;
                for i in 0..n_target_layer {
                    output_buffer[i] = (output_buffer[i] - max_val).exp();
                    sum += output_buffer[i];
                }

                for i in 0..n_target_layer {
                    output_buffer[i] /= sum;
                }
            }

            self.swap_buffer.swap();
        }

        let num_layers = num_transitions;
        let (final_output, _) = self.swap_buffer.get_pair();
        self.signals.set_activations(num_layers, final_output);

        self.signals.activations(num_layers)
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::activations::{Activation, Activations};
    use crate::data_structures::net_propagator::NetPropagator;
    use crate::data_structures::net_properties::NetProperties;
    use insta::assert_debug_snapshot;
    use std::sync::Arc;

    fn create_propagator() -> NetPropagator {
        let layout = vec![2, 3, 2];
        let props = NetProperties::new(layout, false);
        let parameters: Arc<[f32]> = vec![1.0; props.number_of_parameters()].into();
        let propagator = NetPropagator::new(
            Activations::new(Activation::ReLu),
            parameters,
            Arc::new(props),
        );
        propagator
    }
    #[test]
    fn test_forward_pass() {
        let mut propagator = create_propagator();
        let input = vec![1.0, 0.0];
        let result = propagator.forward_pass(&input);
        assert_debug_snapshot!(result)
    }

    #[test]
    fn test_backwards_pass() {
        let mut propagator = create_propagator();
        let input = vec![1.0, 0.0];
        let output = vec![1.0, 0.0];
        let mut asd = vec![0.0; 17];
        let mut x = 0.0;
        let result = propagator.backprop(&input, &output, &mut asd, &mut x);

        assert_debug_snapshot!(asd)
    }
}
