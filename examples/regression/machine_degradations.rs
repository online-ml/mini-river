use light_river::mondrian_forest::mondrian_forest::MondrianForestRegressor;

use light_river::common::{RegTarget, Regressor};
use light_river::datasets::machine_degradation::MachineDegradation;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use num::ToPrimitive;

use std::fs::File;
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["H.e", "UD.t.i", "H.i", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    let sample = transactions.into_iter().next();
    let observation = sample.unwrap().unwrap().get_observation();
    let mut out: Vec<String> = observation.iter().map(|(k, _)| k.clone()).collect();
    out.sort();
    out
}

fn get_dataset_size(transactions: IterCsv<f32, File>) -> usize {
    let mut length = 0;
    for _ in transactions {
        length += 1;
    }
    length
}

fn main() {
    let n_trees: usize = 10;

    let transactions_f = MachineDegradation::load_data();
    let features = get_features(transactions_f);

    println!("Features: {:?}", features);

    let mut mf: MondrianForestRegressor<f32> =
        MondrianForestRegressor::new(n_trees, features.len());
    let mut err_total = 0.0;

    let transactions_l = MachineDegradation::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    let now = Instant::now();

    let transactions = MachineDegradation::load_data();
    for (idx, transaction) in transactions.enumerate() {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let x = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        let y = data.get_y().unwrap();
        let y = data.to_regression_target("pCut::Motor_Torque").unwrap();

        // println!("=M=1 idx={idx}, x={x}, y={y}");

        // Skip first sample since tree has still no node
        if idx != 0 {
            let pred = mf.predict_one(&x, &y);
            let err = (pred - y).powi(2);
            err_total += err;
            // println!("idx={idx}, x={x}, y={y}, pred: {pred}, err: {err}");
        }

        mf.learn_one(&x, &y);
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());

    println!(
        "MSE: {} / {} = {}",
        err_total,
        dataset_size - 1,
        err_total / (dataset_size - 1).to_f32().unwrap()
    );

    let forest_size = mf.get_forest_size();
    println!("Forest tree sizes: {:?}", forest_size);
}
