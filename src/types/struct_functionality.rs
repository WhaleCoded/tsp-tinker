use std::{ fs::File, io::Error, path::Path };

use super::{ TSPAlgorithm, TSPProblem, TSPSolution, TSPPackage };

pub fn create_json_sub_dir_name(num_cities: u64) -> String {
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

    pub fn remove_solution_by_algorithm(&mut self, algorithm: &TSPAlgorithm) {
        self.solutions.retain(|solution| solution.algorithm_name != algorithm.to_string());
    }

    pub fn get_solution_by_algorithm(&self, algorithm: &TSPAlgorithm) -> Option<&TSPSolution> {
        for solution in &self.solutions {
            if solution.algorithm_name == algorithm.to_string() {
                return Some(solution);
            }
        }

        return None;
    }
}
