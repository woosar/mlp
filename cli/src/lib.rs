use data::{BatchMode, OwnedData, OwnedDataset, OwnedEvaluation};
use network::{Activation, Activations, NeuralNet};
use std::path::Path;

pub fn single_main() {
    let layout = [2, 24, 24, 24, 2];
    let training = Path::new(r"D:\Apps\Phyton\PythonProject10\data.txt");
    let evaluation = Path::new(r"D:\Apps\Phyton\PythonProject10\data2.txt");

    let mut training_dataset = OwnedDataset::new(
        OwnedData::from_csv(training.to_str().unwrap(), 2, 2),
        256,
        layout[0],
        layout[layout.len() - 1],
        BatchMode::Sequential,
    );

    let net = NeuralNet::new(
        layout,
        Activations::new(Activation::ReLu),
        None,
        true,
    );

    net.train(&mut training_dataset, 10000, |_, _| {});
    
    let evaluation_dataset = OwnedDataset::new(
        OwnedData::from_csv(evaluation.to_str().unwrap(), 2, 2),
        32,
        layout[0],
        layout[layout.len() - 1],
        BatchMode::Random,
    );
    let mut evaluable_2 = OwnedEvaluation::from(&training_dataset);

    net.evaluate_data(&mut evaluable_2);
    evaluable_2.save_to_json("training_export.json").unwrap();
    let mut evaluable = OwnedEvaluation::from(&evaluation_dataset);

    net.evaluate_data(&mut evaluable);

    evaluable.save_to_json("evaluation_export.json").unwrap()
}
