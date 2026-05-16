use crate::{OwnedDataset, assert_dimension_validity, de_interleave};
use network::{Batch, Evaluable};
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct OwnedEvaluation {
    input: Vec<f32>,
    output: Vec<f32>,
    pub prediction: Option<Vec<f32>>,
    error: Option<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
}
#[derive(Serialize)]
pub struct RawData {
    pub input: Vec<Vec<f32>>,
    pub output: Vec<Vec<f32>>,
    pub prediction: Vec<Vec<f32>>,
}

impl OwnedEvaluation {
    pub fn to_raw_data(self) -> RawData {
        let input = de_interleave(self.input, self.input_dim);
        let output = de_interleave(self.output, self.output_dim);

        let prediction = self
            .prediction
            .map(|pred| de_interleave(pred, self.output_dim))
            .unwrap_or_default();

        RawData {
            input,
            output,
            prediction,
        }
    }
    pub fn new(input: Vec<f32>, output: Vec<f32>, input_dim: usize, output_dim: usize) -> Self {
        assert_dimension_validity(output.len(), input.len(), output_dim, input_dim);
        Self {
            input,
            output,
            prediction: None,
            input_dim,
            output_dim,
            error: None,
        }
    }
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;

        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, self)?;

        Ok(())
    }
}

impl From<&OwnedDataset> for OwnedEvaluation {
    fn from(value: &OwnedDataset) -> Self {
        Self {
            input: value.input().to_vec(),
            output: value.output().to_vec(),
            prediction: None,
            input_dim: value.input_dim(),
            output_dim: value.output_dim(),
            error: None,
        }
    }
}

impl Evaluable for OwnedEvaluation {
    fn create_inference_batch(&self) -> Batch {
        let mut data = vec![0.0; self.input.len() + self.output.len()];

        data[0..self.input.len()].copy_from_slice(&self.input);
        Batch::new(
            self.input.len() / self.input_dim,
            self.input_dim,
            self.output_dim,
            &data,
            false,
        )
    }

    fn set_prediction(&mut self, batch: Batch) {
        let error = self
            .output
            .iter()
            .zip(batch.output())
            .map(|(a, b)| b - a)
            .collect::<Vec<_>>();
        self.prediction = Some(batch.output());
        self.error = Some(error);
    }
}
