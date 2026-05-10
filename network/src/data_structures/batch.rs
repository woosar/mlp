#[derive(Debug, Clone)]
pub struct Batch {
    number_of_samples: usize,
    input_nodes: usize,
    output_nodes: usize,
    split_index: usize,
    data: Vec<f32>,
    training: bool,
}

impl Batch {
    pub fn from_data(input_data: &[f32], output_data: &[f32], layout: &[usize]) -> Batch {
        if input_data.len() != output_data.len() {
            panic!("nope")
        }

        let asd = [input_data, output_data].concat();

        Batch::new(
            input_data.len(),
            layout[0],
            layout[layout.len() - 1],
            &asd,
            true,
        )
    }

    pub fn new(
        number_of_samples: usize,
        input_nodes: usize,
        output_nodes: usize,
        data: &[f32],
        training: bool,
    ) -> Self {
        let split_index = number_of_samples * input_nodes;
        Self {
            number_of_samples,
            input_nodes,
            output_nodes,
            split_index,
            data: Vec::from(data),
            training,
        }
    }

    pub fn number_of_samples(&self) -> usize {
        self.number_of_samples
    }

    pub fn to_inference_batch(&self) -> Option<Batch> {
        if !self.training {
            return None;
        }
        let mut new = self.clone();
        new.training = false;
        for i in 0..self.split_index {
            new.data[i + self.split_index] = 0.0
        }
        Some(new)
    }

    pub fn set_output(&mut self, sample_number: usize, output: &[f32]) {
        if self.training {
            return;
        }
        let (start_index, end_index) = self.output_index_calculation(sample_number);
        assert_eq!(
            output.len(),
            end_index - start_index,
            "output length must match number of output nodes"
        );
        self.data[start_index..end_index].copy_from_slice(output)
    }

    pub fn input(&self, sample_number: usize) -> &[f32] {
        let start_index = sample_number * self.input_nodes; // beware of off-by-one errors!
        let end_index = (sample_number + 1) * self.input_nodes;
        &self.data[start_index..end_index]
    }

    pub fn output(&self) -> Vec<f32> {
        Vec::from(&self.data[self.split_index..])
    }

    pub fn output_sample(&self, sample_number: usize) -> &[f32] {
        if !self.training {
            return &self.data[0..0];
        }
        let (start_index, end_index) = self.output_index_calculation(sample_number);
        &self.data[start_index..end_index]
    }

    fn output_index_calculation(&self, sample_number: usize) -> (usize, usize) {
        let start_index = self.split_index + sample_number * self.output_nodes;
        let end_index = self.split_index + (sample_number + 1) * self.output_nodes;
        (start_index, end_index)
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::batch::Batch;
    use insta::assert_debug_snapshot;

    fn create_test_batch() -> Batch {
        let data = vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 11.0, 11.0, 11.0, 22.0, 22.0, 22.0, 33.0, 33.0, 33.0,
        ];
        Batch::new(3, 2, 3, &data, true)
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
