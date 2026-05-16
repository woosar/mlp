use crate::helpers::cumulative_sum;

#[derive(Debug, Clone)]
pub struct NetProperties {
    layout: Vec<usize>,
    transition_parameter_count: Vec<usize>,
    transition_offsets: Vec<usize>,
    number_of_parameters: usize,
    capacity: usize,
    max_layer: usize,
    is_classifier: bool,
}

impl NetProperties {
    pub fn new(layout: Vec<usize>, is_classifier: bool) -> Self {
        let capacity = layout
            .iter()
            .max()
            .expect("next time, provide a valid layout")
            + 1; // +1 for the bias 
        let transition_parameter_count: Vec<usize> = layout
            .windows(2)
            .map(|win| {
                let (in_nodes, out_nodes) = (win[0], win[1]);
                (in_nodes + 1) * out_nodes
            })
            .collect();

        let mut transition_offsets = cumulative_sum(&transition_parameter_count, 0);
        let number_of_parameters = transition_offsets.pop().expect("this cannot fail");
        let max_layer = transition_offsets.len();
        Self {
            layout,
            transition_parameter_count,
            transition_offsets,
            number_of_parameters,
            capacity,
            max_layer,
            is_classifier,
        }
    }

    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    pub fn transition_offsets(&self) -> &[usize] {
        &self.transition_offsets
    }

    pub fn number_of_parameters(&self) -> usize {
        self.number_of_parameters
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn layout(&self) -> &[usize] {
        &self.layout
    }

    pub fn is_classifier(&self) -> bool {
        self.is_classifier
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::net_properties::NetProperties;
    use insta::assert_debug_snapshot;

    #[test]
    fn test_properties() {
        let layout = vec![2, 3, 2];
        let properties = NetProperties::new(layout, true);
        assert_debug_snapshot!(properties)
    }
}
