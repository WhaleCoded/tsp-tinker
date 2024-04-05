use std::path::{ Path, PathBuf };
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
    pub optimal: bool,
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

pub enum TSPAlgorithm {
    BranchNBound,
    LinKernighan,
    Pseudorandom,
    NaiveHeuristic,
}

impl TSPAlgorithm {
    pub fn to_string(&self) -> String {
        match self {
            TSPAlgorithm::BranchNBound => {
                return "Branch and Bound".to_string();
            }
            TSPAlgorithm::LinKernighan => {
                return "Lin-Kernighan".to_string();
            }
            TSPAlgorithm::Pseudorandom => {
                return "Pseudorandom".to_string();
            }
            TSPAlgorithm::NaiveHeuristic => {
                return "Naive Heuristic".to_string();
            }
        }
    }
}

impl Serialize for TSPSolution {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("TSPSolution", 5)?;
        state.serialize_field("tot_cost", &self.tot_cost)?;
        state.serialize_field("path", &self.path)?;
        state.serialize_field("optimal", &self.optimal)?;
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

pub fn get_subdirectories_of_tsp_problems(data_path: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut sub_dirs: Vec<PathBuf> = vec![];

    for entry in std::fs::read_dir(data_path)? {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // Check if the directory matches the naming convention for TSP problems
            let dir_name = path
                .file_name()
                .expect("Directories should always have a name")
                .to_str()
                .expect("Directory names should always be valid UTF-8");

            if dir_name.starts_with("tsp_problems_of_") && dir_name.ends_with("_cities") {
                sub_dirs.push(path);
            }
        }
    }

    //sort the subdirectories by the number of cities smallest to largest
    sub_dirs.sort_by(|a, b| {
        let a_num_cities = a
            .file_name()
            .expect("Directories should always have a name")
            .to_str()
            .expect("Directory names should always be valid UTF-8")
            .split("_")
            .collect::<Vec<&str>>()[3]
            .parse::<u64>()
            .expect("Directory names should always be valid u64");
        let b_num_cities = b
            .file_name()
            .expect("Directories should always have a name")
            .to_str()
            .expect("Directory names should always be valid UTF-8")
            .split("_")
            .collect::<Vec<&str>>()[3]
            .parse::<u64>()
            .expect("Directory names should always be valid u64");

        a_num_cities.cmp(&b_num_cities)
    });

    return Ok(sub_dirs);
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

    pub fn from_json(json_file_path: &Path) -> Result<TSPPackage, Error> {
        let json_file = File::open(json_file_path)?;
        let tsp_package: TSPPackage = serde_json::from_reader(json_file)?;

        return Ok(tsp_package);
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

    pub fn has_been_solved_by_algorithm(&self, algorithm: &TSPAlgorithm) -> bool {
        for solution in &self.solutions {
            if solution.algorithm_name == algorithm.to_string() {
                return true;
            }
        }

        return false;
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

pub fn get_num_existing_tsp_problems_by_sub_dir(sub_dir_path: &Path) -> Result<u64, Error> {
    match sub_dir_path.exists() {
        false => {
            return Ok(0);
        }
        true => {}
    }

    // Count the number of json files in the subdirectory
    let mut num_json_files = 0;
    for entry in std::fs::read_dir(sub_dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            num_json_files += 1;
        }
    }

    return Ok(num_json_files);
}

pub fn get_tsp_problem_file_paths_by_sub_dir(sub_dir_path: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut json_file_paths: Vec<PathBuf> = vec![];

    for entry in std::fs::read_dir(sub_dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            json_file_paths.push(path);
        }
    }

    return Ok(json_file_paths);
}
