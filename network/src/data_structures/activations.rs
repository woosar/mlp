use crate::data_structures::activations::activation_functions::{
    relu_activation, relu_derivative, sigmoid_activation, sigmoid_derivative,
};
#[derive(Clone, Copy, Debug)]
pub struct Activations {
    pub activation: fn(f32) -> f32,
    // the derivative is in terms of the activation
    pub derivative: fn(f32) -> f32,
}

pub enum Activation {
    ReLu,
    Sigmoid,
    Identity,
}

impl Activations {
    pub fn new(activation: Activation) -> Self {
        match activation {
            Activation::ReLu => Self {
                activation: relu_activation,
                derivative: relu_derivative,
            },

            Activation::Sigmoid => Self {
                activation: sigmoid_activation,
                derivative: sigmoid_derivative,
            },
            Activation::Identity => Self {
                activation: |input| input,
                derivative: |_| 1.0,
            },
        }
    }
}

mod activation_functions {
    pub fn relu_activation(input: f32) -> f32 {
        input.max(0.0)
    }
    pub fn relu_derivative(activation: f32) -> f32 {
        if activation > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn sigmoid_activation(input: f32) -> f32 {
        1.0 / (1.0 + (-input).exp())
    }

    pub fn sigmoid_derivative(input: f32) -> f32 {
        input * (1.0 - input)
    }
}
