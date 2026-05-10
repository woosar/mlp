mod owned_dataset;
mod owned_evaluation;

pub use owned_dataset::{BatchMode, OwnedDataset};
pub use owned_evaluation::OwnedEvaluation;

pub fn assert_dimension_validity(
    output_len: usize,
    input_len: usize,
    output_dim: usize,
    input_dim: usize,
) {
    assert!(
        input_dim != 0
            && output_dim != 0
            && input_len % input_dim == 0
            && output_len % output_dim == 0
            && output_len / output_dim == input_len / input_dim,
        "invalid input"
    )
}
