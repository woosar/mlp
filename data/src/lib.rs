mod owned_data;
mod owned_dataset;
mod owned_evaluation;

pub use owned_data::OwnedData;
pub use owned_dataset::{BatchMode, OwnedDataset};
pub use owned_evaluation::{OwnedEvaluation, RawData};

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

pub fn de_interleave(data: Vec<f32>, dim: usize) -> Vec<Vec<f32>> {
    let mut result = vec![Vec::new(); dim];
    for (idx, val) in data.into_iter().enumerate() {
        result[idx % dim].push(val);
    }

    result
}

#[cfg(test)]
mod test {
    use crate::de_interleave;
    use insta::assert_debug_snapshot;

    #[test]
    fn test_unflatten() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let result = de_interleave(data, 2);
        assert_debug_snapshot!(result)
    }
}
