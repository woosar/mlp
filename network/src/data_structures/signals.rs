use crate::helpers::cumulative_sum;

#[derive(Clone, Debug)]
pub struct Signals {
    appended_layout: Vec<usize>,
    indices: Vec<usize>,
    activations: Vec<f32>, // will be calculated during the forward pass
}

impl Signals {
    pub fn new(layout: &[usize]) -> Self {
        let mut appended_layout: Vec<usize> = layout.iter().map(|&l| l + 1).collect();
        let last = appended_layout.len() - 1;
        appended_layout[last] -= 1;

        let total: usize = appended_layout.iter().sum();

        let mut indices = cumulative_sum(&appended_layout, 0usize);
        indices.pop();

        Self {
            appended_layout,
            indices,
            activations: vec![0.0; total],
        }
    }

    fn layer_info(&self, layer: usize) -> (usize, usize) {
        let node_count = self.appended_layout[layer];
        let start_index = self.indices[layer];
        (node_count, start_index)
    }

    pub fn set_activations(&mut self, layer: usize, activations: &[f32]) {
        let (node_count, start_index) = self.layer_info(layer);
        self.activations[start_index..start_index + node_count]
            .copy_from_slice(&activations[0..node_count])
    }

    pub fn activations(&self, layer: usize) -> &[f32] {
        let (node_count, start_index) = self.layer_info(layer);
        &self.activations[start_index..start_index + node_count]
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::signals::Signals;
    use insta::assert_debug_snapshot;

    #[test]
    fn test_signals() {
        let layout = [2, 3, 2];
        let signals = Signals::new(&layout);
        assert_debug_snapshot!(signals)
    }
}
