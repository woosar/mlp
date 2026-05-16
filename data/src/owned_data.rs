#[derive(Debug)]
pub struct OwnedData {
    input: Vec<f32>,
    output: Vec<f32>,
}

impl OwnedData {
    pub fn input(&self) -> &[f32] {
        &self.input
    }
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub fn output_mut(&mut self) -> &mut [f32] {
        &mut self.output
    }

    pub fn input_mut(&mut self) -> &mut [f32] {
        &mut self.input
    }

    pub fn borrow_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.input, &mut self.output)
    }

    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self { input, output }
    }

    pub fn from_input_slice(input: &[f32]) -> Self {
        let input = Vec::from(input);
        let len = input.len();
        Self {
            input,
            output: vec![0.0; len],
        }
    }

    pub fn from_csv(content: &str, input_dim: usize, output_dim: usize) -> Self {
        let mut input = Vec::new();
        let mut output = Vec::new();

        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let values: Vec<f32> = line
                .split(',')
                .map(|v| v.trim().parse::<f32>().unwrap())
                .collect();

            assert_eq!(
                values.len(),
                input_dim + output_dim,
                "Expected {} columns per row",
                input_dim + output_dim
            );

            for i in 0..input_dim {
                input.push(values[i]);
            }
            for i in 0..output_dim {
                output.push(values[input_dim + i]);
            }
        }

        Self::new(input, output)
    }
}
