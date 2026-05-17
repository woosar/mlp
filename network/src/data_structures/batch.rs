use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Training;
#[derive(Debug, Clone, Copy)]
pub struct Inference;

#[derive(Debug, Clone)]
pub struct Batch<State = Training> {
    number_of_samples: usize,
    input_nodes: usize,
    output_nodes: usize,
    split_index: usize,
    data: Vec<f32>,
    _state: PhantomData<State>,
}

impl<State> Batch<State> {
    pub fn new(
        number_of_samples: usize,
        input_nodes: usize,
        output_nodes: usize,
        data: &[f32],
    ) -> Self {
        let split_index = number_of_samples * input_nodes;
        Self {
            number_of_samples,
            input_nodes,
            output_nodes,
            split_index,
            data: Vec::from(data),
            _state: PhantomData,
        }
    }

    pub fn number_of_samples(&self) -> usize {
        self.number_of_samples
    }

    pub fn input(&self, sample_number: usize) -> &[f32] {
        let start_index = sample_number * self.input_nodes; // beware of off-by-one errors!
        let end_index = (sample_number + 1) * self.input_nodes;
        &self.data[start_index..end_index]
    }

    pub fn output(&self) -> Vec<f32> {
        Vec::from(&self.data[self.split_index..])
    }

    fn output_index_calculation(&self, sample_number: usize) -> (usize, usize) {
        let start_index = self.split_index + sample_number * self.output_nodes;
        let end_index = self.split_index + (sample_number + 1) * self.output_nodes;
        (start_index, end_index)
    }
}

impl Batch<Training> {
    pub fn from_data(input_data: &[f32], output_data: &[f32], layout: &[usize]) -> Batch<Training> {
        if input_data.len() != output_data.len() {
            panic!("nope")
        }

        let asd = [input_data, output_data].concat();

        Batch::new(
            input_data.len(),
            layout[0],
            layout[layout.len() - 1],
            &asd,
        )
    }

    pub fn to_inference_batch(&self) -> Batch<Inference> {
        let mut new_data = self.data.clone();
        for i in self.split_index..new_data.len() {
            new_data[i] = 0.0;
        }
        Batch::<Inference> {
            number_of_samples: self.number_of_samples,
            input_nodes: self.input_nodes,
            output_nodes: self.output_nodes,
            split_index: self.split_index,
            data: new_data,
            _state: PhantomData,
        }
    }

    pub fn output_sample(&self, sample_number: usize) -> &[f32] {
        let (start_index, end_index) = self.output_index_calculation(sample_number);
        &self.data[start_index..end_index]
    }
}

impl Batch<Inference> {
    pub fn set_output(&mut self, sample_number: usize, output: &[f32]) {
        let (start_index, end_index) = self.output_index_calculation(sample_number);
        assert_eq!(
            output.len(),
            end_index - start_index,
            "output length must match number of output nodes"
        );
        self.data[start_index..end_index].copy_from_slice(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::batch::{Batch, Training};
    use insta::assert_debug_snapshot;

    fn create_test_batch() -> Batch<Training> {
        let data = vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 11.0, 11.0, 11.0, 22.0, 22.0, 22.0, 33.0, 33.0, 33.0,
        ];
        Batch::new(3, 2, 3, &data)
    }

    #[test]
    fn test_creation() {
        let batch = create_test_batch();
        assert_debug_snapshot!(batch)
    }

    #[test]
    fn test_slicing() {
        let batch = create_test_batch();
        let second_sample_input = batch.input(1);
        let second_sample_output = batch.output_sample(1);
        let result = vec![
            Vec::from(second_sample_input),
            Vec::from(second_sample_output),
        ];
        assert_debug_snapshot!(result)
    }
}
