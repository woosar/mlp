use crate::assert_dimension_validity;
use crate::owned_data::OwnedData;
use network::{Batch, BatchProvider};
use rand::distr::{Distribution, Uniform};
use std::fmt::{Debug, Formatter};

#[derive(Debug)]
pub struct OwnedDataset {
    data: OwnedData,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    batch_mode: BatchMode,
    drawn_observations: Vec<usize>,
    total_observations: usize,
    number_of_full_batches: usize,
}

impl OwnedDataset {
    pub fn input(&self) -> &[f32] {
        &self.data.input()
    }
    pub fn output(&self) -> &[f32] {
        &self.data.output()
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    pub fn create_inference_batch(&self) -> Batch {
        let len = self.input().len();
        let zeros = vec![0.0f32; self.output().len()];
        let new = [self.input(), &zeros].concat();
        // let new = [self.input.clone(), self.output.clone()].concat();
        Batch::new(len, self.input_dim, self.output_dim, &new, false)
    }
    pub fn new(
        data: OwnedData,
        batch_size: usize,
        input_dim: usize,
        output_dim: usize,
        batch_mode: BatchMode,
    ) -> Self {
        assert_dimension_validity(
            data.output().len(),
            data.input().len(),
            output_dim,
            input_dim,
        );

        let total_observations = data.input().len() / input_dim;
        let drawn_observations = match batch_mode {
            // for sequential draws, we misuse this as a simple cursor!
            BatchMode::Sequential => vec![0],
            BatchMode::Random => Vec::new(),
        };
        let number_of_full_batches = total_observations / batch_size;
        Self {
            data,
            input_dim,
            output_dim,
            batch_size,
            batch_mode,
            drawn_observations,
            total_observations,
            number_of_full_batches,
        }
    }

    pub fn next_random_batch(&mut self) -> Option<Batch> {
        if self.drawn_observations.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let current_pool_len = self.drawn_observations.len();
        let count = current_pool_len.min(self.batch_size);

        let mut batch_indices = Vec::with_capacity(count);

        for _ in 0..count {
            let remaining = self.drawn_observations.len();
            if remaining == 0 {
                break;
            }

            let dist = Uniform::new(0, remaining).unwrap();
            let pool_idx = dist.sample(&mut rng);

            let observation_idx = self.drawn_observations.swap_remove(pool_idx);
            batch_indices.push(observation_idx);
        }

        let mut data = Vec::with_capacity(count * (self.input_dim + self.output_dim));

        for &obs_idx in &batch_indices {
            let in_start = obs_idx * self.input_dim;
            let in_end = in_start + self.input_dim;
            let out_start = obs_idx * self.output_dim;
            let out_end = out_start + self.output_dim;

            data.extend_from_slice(&self.data.input_mut()[in_start..in_end]);
            data.extend_from_slice(&self.data.output_mut()[out_start..out_end]);
        }

        Some(Batch::new(
            count,
            self.input_dim,
            self.output_dim,
            &data,
            true,
        ))
    }
    pub fn next_sequential_batch(&mut self) -> Option<Batch> {
        // in sequential mode, we use the observation index vector as a simple cursor
        let cursor = self.drawn_observations[0];

        let next_batch_observation_count = self
            .total_observations
            .saturating_sub(cursor * self.batch_size)
            .min(self.batch_size);

        if next_batch_observation_count == 0 {
            return None;
        }

        let input_start = cursor * self.batch_size * self.input_dim;
        let input_end = input_start + next_batch_observation_count * self.input_dim;
        let output_start = cursor * self.batch_size * self.output_dim;
        let output_end = output_start + next_batch_observation_count * self.output_dim;
        let (input, output) = self.data.borrow_mut();
        let data = [
            &input[input_start..input_end],
            &output[output_start..output_end],
        ]
        .concat();

        self.drawn_observations[0] += 1;

        Some(Batch::new(
            next_batch_observation_count,
            self.input_dim,
            self.output_dim,
            &data,
            true,
        ))
    }
}

pub enum BatchMode {
    Sequential,
    Random,
}

impl Default for BatchMode {
    fn default() -> Self {
        BatchMode::Sequential
    }
}

impl Debug for BatchMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug_string = match self {
            BatchMode::Sequential => "Sequential",
            BatchMode::Random => "Random",
        };
        write!(f, "{}", debug_string)
    }
}

impl BatchProvider for OwnedDataset {
    fn provide_batch(&mut self) -> Option<Batch> {
        match self.batch_mode {
            BatchMode::Sequential => self.next_sequential_batch(),
            BatchMode::Random => self.next_random_batch(),
        }
    }

    fn reset(&mut self) {
        self.drawn_observations.clear();
        match self.batch_mode {
            BatchMode::Sequential => self.drawn_observations.push(0),
            BatchMode::Random => self.drawn_observations.extend(0..self.total_observations),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::owned_data::OwnedData;
    use crate::owned_dataset::{BatchMode, OwnedDataset};
    use insta::assert_debug_snapshot;
    use network::BatchProvider;

    fn generate_siso_dataset() -> OwnedDataset {
        let (inp, out) = (0..10)
            .map(|idx| ((idx + 1) as f32, ((idx + 1) * 10) as f32))
            .collect::<(Vec<f32>, Vec<f32>)>();

        OwnedDataset::new(OwnedData::new(inp, out), 3, 1, 1, BatchMode::Sequential)
    }

    fn generate_mimo_dataset(number_of_observations: usize) -> OwnedDataset {
        let inp = (0..number_of_observations)
            .map(|elem| vec![elem as f32, (2 * elem) as f32])
            .flatten()
            .collect::<Vec<f32>>();

        let out = (0..number_of_observations)
            .map(|elem| vec![(10 * elem) as f32, (20 * elem) as f32, (30 * elem) as f32])
            .flatten()
            .collect::<Vec<f32>>();

        OwnedDataset::new(OwnedData::new(inp, out), 3, 2, 3, BatchMode::Sequential)
    }

    #[test]
    fn test_siso_dataset() {
        let mut dataset = generate_siso_dataset();
        let mut lala = Vec::new();
        while let Some(batch) = dataset.provide_batch() {
            lala.push(batch)
        }

        assert_debug_snapshot!(lala)
    }

    #[test]
    fn test_mimo_dataset() {
        let mut dataset = generate_mimo_dataset(10);
        let mut lala = Vec::new();
        while let Some(batch) = dataset.provide_batch() {
            println!("{:#?}", batch);
            lala.push(batch)
        }

        assert_debug_snapshot!(lala)
    }
}


