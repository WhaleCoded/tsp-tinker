use std::{ fmt::Display, fs::File, hash::{ Hash, Hasher }, io::Error, path::Path };

use super::{ TSPAlgorithm, TSPProblem, TSPSolution, TSPPackage, UndirectedEdge };

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

impl PartialEq for UndirectedEdge {
    fn eq(&self, other: &Self) -> bool {
        // Sort the cities in the edge so that the comparison is order independent
        let mut ordered_cities = vec![self.city_a, self.city_b];
        ordered_cities.sort();

        let mut other_ordered_cities = vec![other.city_a, other.city_b];
        other_ordered_cities.sort();

        return ordered_cities == other_ordered_cities;
    }
}

impl Eq for UndirectedEdge {}
impl Hash for UndirectedEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut ordered_cities = vec![self.city_a, self.city_b];
        ordered_cities.sort();
        ordered_cities.hash(state);
    }
}

impl Display for UndirectedEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.city_a, self.city_b)
    }
}

impl UndirectedEdge {
    pub fn new(city_a: u64, city_b: u64) -> UndirectedEdge {
        return UndirectedEdge {
            city_a,
            city_b,
        };
    }
}
