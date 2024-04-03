use std::path::Path;
use std::fs::File;
use std::io::Error;

use uuid;
use serde::ser::{ Serialize, Serializer, SerializeStruct };
use ndarray::Array2;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct TSPSolution {
    pub tot_cost: f32,
    pub path: Vec<u64>,
    pub bssf: bool,
    pub calculation_time: f32,
    pub algorithm_name: String,
}

#[derive(Deserialize, Clone)]
pub struct TSPProblem {
    pub num_cities: u64,
    pub city_connections_w_costs: Array2<f32>,
}

#[derive(Deserialize, Clone)]
pub struct TSPPackage {
    pub uuid: String,
    pub problem_data: TSPProblem,
    pub solutions: Vec<TSPSolution>,
}

impl Serialize for TSPSolution {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("TSPSolution", 5)?;
        state.serialize_field("tot_cost", &self.tot_cost)?;
        state.serialize_field("path", &self.path)?;
        state.serialize_field("bssf", &self.bssf)?;
        state.serialize_field("calculation_time", &self.calculation_time)?;
        state.serialize_field("algorithm_name", &self.algorithm_name)?;
        state.end()
    }
}

impl Serialize for TSPProblem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("TSPProblem", 2)?;
        state.serialize_field("num_cities", &self.num_cities)?;
        state.serialize_field("city_connections_w_costs", &self.city_connections_w_costs)?;
        state.end()
    }
}

impl Serialize for TSPPackage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("TSPPackage", 3)?;
        state.serialize_field("uuid", &self.uuid)?;
        state.serialize_field("problem_data", &self.problem_data)?;
        state.serialize_field("solutions", &self.solutions)?;
        state.end()
    }
}

fn create_json_sub_dir_name(num_cities: u64) -> String {
    return format!("tsp_problems_of_{}_cities", num_cities);
}

fn create_json_file_name(tsp_prob_uuid: String) -> String {
    return format!("{}.json", tsp_prob_uuid);
}

impl TSPPackage {
    pub fn new(problem_data: TSPProblem) -> TSPPackage {
        let ts = uuid::Timestamp::now(uuid::NoContext);
        TSPPackage {
            uuid: uuid::Uuid::new_v7(ts).to_string(),
            problem_data,
            solutions: vec![],
        }
    }

    pub fn store_as_json(&self, primary_data_path: &Path) -> Result<(), Error> {
        // Create or verify the existence of the subdirectory
        let sub_directory_path = primary_data_path.join(
            create_json_sub_dir_name(self.problem_data.num_cities)
        );
        match sub_directory_path.exists() {
            true => {}
            false => {
                match std::fs::create_dir(sub_directory_path.clone()) {
                    Ok(_) => {}
                    Err(e) => {
                        match e.kind() {
                            // If the directory already exists, we can ignore the error
                            std::io::ErrorKind::AlreadyExists => {}
                            _ => {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }

        // Store the data as a json file
        let json_file_path = sub_directory_path.join(create_json_file_name(self.uuid.clone()));
        let json_file = File::create(json_file_path).expect(
            "Creating JSON file failed. Because of the uuid there should never be a duplicate file name."
        );
        serde_json::to_writer(json_file, self)?;

        return Ok(());
    }
}

pub fn get_num_existing_tsp_problems_by_size(
    data_path: &Path,
    problem_size: u64
) -> Result<u64, Error> {
    let subdirectory_path = data_path.join(create_json_sub_dir_name(problem_size));
    match subdirectory_path.exists() {
        false => {
            return Ok(0);
        }
        true => {}
    }

    // Count the number of json files in the subdirectory
    let mut num_json_files = 0;
    for entry in std::fs::read_dir(subdirectory_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            num_json_files += 1;
        }
    }

    return Ok(num_json_files);
}
