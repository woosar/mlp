use data::{BatchMode, OwnedData, OwnedDataset, OwnedEvaluation, RawData};
use network::{Activation, Activations, NeuralNet};
use std::sync::OnceLock;
use tauri::Emitter;

static NEURAL: OnceLock<NeuralNet<5>> = OnceLock::new();

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

fn get_net<'a>() -> &'a NeuralNet<5> {
    let layout = [2, 24, 24, 24, 2]; // todo global layout
    &NEURAL.get_or_init(|| {
        NeuralNet::new(
            layout,
            Activations::new(Activation::ReLu),
            None,
            true,
        )
    })
}

#[tauri::command]
fn train(app_handle: tauri::AppHandle) {
    println!("Starting training");

    std::thread::spawn(move || {
        // todo bake it in better
        let net_worker = get_net();
        let training = include_str!("../files/training.txt");
        let layout = [2, 24, 24, 24, 2];

        let mut training_dataset = OwnedDataset::new(
            OwnedData::from_csv(training, 2, 2),
            256,
            layout[0],
            layout[layout.len() - 1],
            BatchMode::Sequential,
        );
        let data_raw = training_dataset.clone().to_raw_data(); // todo wip
        let _ = app_handle.emit("training-data", data_raw);

        net_worker.train(&mut training_dataset, 10000, |epoch, loss| {
            if epoch % 10 == 0 {
                let _ = app_handle.emit("epoch-completed", epoch);
                let _ = app_handle.emit("loss", loss);
            }
        });

        println!("Training complete!");
    });
}
#[tauri::command]
fn calculate(input: Vec<f32>) -> RawData {
    let layout = [2, 24, 24, 24, 2];
    let data = OwnedData::from_input_slice(&input); // todo: redo ownership here
    let net = get_net();

    let evaluation_dataset = OwnedDataset::new(
        data,
        32,
        layout[0],
        layout[layout.len() - 1],
        BatchMode::Random,
    );

    let mut evaluation = OwnedEvaluation::from(&evaluation_dataset);
    net.evaluate_data(&mut evaluation);
    evaluation.to_raw_data()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, calculate, train])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
