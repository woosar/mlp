use data::{BatchMode, OwnedDataset, OwnedEvaluation};
use network::{Activation, Activations, NeuralNet};

pub fn single_main() {
    let layout = vec![1, 16,16, 1];
    let (inp, out) = (0..100)
        .map(|elem| {
            let x = (elem as f32 / 49.5) - 1.0;
            (x, x * x)
        })
        .collect::<(Vec<f32>, Vec<f32>)>();

    let mut dataset = OwnedDataset::new(
        inp,
        out,
        32,
        layout[0],
        layout[layout.len() - 1],
        BatchMode::Sequential,
    );

    let net = NeuralNet::new(layout, Activations::new(Activation::Sigmoid), None);

    net.train(&mut dataset);
    let mut evaluable = OwnedEvaluation::from(&dataset);
    net.evaluate_data(&mut evaluable);

    evaluable.save_to_json("hehe.json").unwrap()
}
