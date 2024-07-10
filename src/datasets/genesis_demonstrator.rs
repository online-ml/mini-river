use crate::datasets::utils;
use crate::stream::data_stream::Target;
use crate::stream::iter_csv::IterCsv;
use std::{fs::File, path::Path};

/// Genesis demonstrator data for machine learning
///
/// References
/// ----------
/// [^1]: [Genesis demonstrator data for machine learning](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
pub struct GenesisDemostrator;
impl GenesisDemostrator {
    pub fn load_data() -> IterCsv<f32, File> {
        // TODO: come up a way to follow the 302 redirect
        // let url = "https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning/download/vi2a2jUtKvt6eaRtcgfh%2Fversions%2FqvQVrKjulHBBJj5YC5fv%2Ffiles%2FGenesis_StateMachineLabel.csv?datasetVersionNumber=1";
        let file_name = "Genesis_StateMachineLabel.csv";

        if !Path::new(file_name).exists() {
            panic!("Dataset not downloaded. Download it in file '{file_name}'");
        }

        let file = File::open(file_name).unwrap();
        let y_cols = Some(Target::Name("Label".to_string()));
        IterCsv::<f32, File>::new(file, y_cols).unwrap()
    }
}
