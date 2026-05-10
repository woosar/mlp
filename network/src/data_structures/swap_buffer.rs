#[derive(Debug)]
pub struct SwapBuffer {
    buffer_1: Vec<f32>,
    buffer_2: Vec<f32>,
}

impl SwapBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer_1: vec![0.0; capacity],
            buffer_2: vec![0.0; capacity],
        }
    }

    pub fn get_pair(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        (&mut self.buffer_1, &mut self.buffer_2)
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.buffer_1, &mut self.buffer_2);
    }
}
