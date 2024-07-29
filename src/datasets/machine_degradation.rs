use crate::datasets::utils;
use crate::stream::data_stream::Target;
use crate::stream::iter_csv::IterCsv;
use std::{fs::File, path::Path};

/// One Year Industrial Component Degradation
///
/// References
/// ----------
/// [^1]: [One Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
pub struct MachineDegradation;
impl MachineDegradation {
    pub fn load_data() -> IterCsv<f32, File> {
        // TODO: come up a way to follow the 302 redirect
        // let url = "https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation/download/fA53OHmuZ0enYASBqytj%2Fversions%2FvXObUJmxGJQSUSC2Wyc7%2Ffiles%2F01-04T184148_000_mode1.csv?datasetVersionNumber=1";
        let file_name = "one-year-industrial-component-degradation.csv";

        if !Path::new(file_name).exists() {
            panic!("Dataset not downloaded. Download it in file '{file_name}'");
        }

        let file = File::open(file_name).unwrap();
        let y_cols = Some(Target::Name("pCut::Motor_Torque".to_string()));
        IterCsv::<f32, File>::new(file, y_cols).unwrap()
    }
}
