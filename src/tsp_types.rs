use std::fmt;
use std::path::{ Path, PathBuf };
use std::fs::File;
use std::io::Error;

use uuid;
use serde::ser::{ Serialize, Serializer, SerializeStruct };
use ndarray::{ s, Array1, Array2 };

#[derive(serde::Deserialize, Clone, serde::Serialize)]
pub struct TSPSolution {
    pub tot_cost: f32,
    pub path: Vec<u64>,
    pub optimal: bool,
    pub calculation_time: f32,
    pub algorithm_name: String,
}

#[derive(serde::Deserialize, Clone, serde::Serialize)]
pub struct UndirectedEdge {
    pub city_a: u64,
    pub city_b: u64,
    pub cost: f32,
}

#[derive(Clone)]
pub struct TSPProblem {
    pub num_cities: u64,
    pub undirected_edges: bool,
    pub city_connections_w_costs: Array2<f32>,
}

#[derive(serde::Deserialize, Clone, serde::Serialize)]
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

impl Serialize for TSPProblem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("TSPProblem", 2)?;
        state.serialize_field("num_cities", &self.num_cities)?;
        state.serialize_field("undirected", &self.undirected_edges)?;

        // If the edges are undirected, we only need to store the upper triangle of the matrix
        // To save space, we will store the upper triangle as a 1D array
        if self.undirected_edges {
            let mut upper_triangle_edges: Array1<f32> = Array1::zeros(
                ((self.num_cities * (self.num_cities + 1)) / 2) as usize
            );
            let mut triangle_index = 0;
            for i in 0..self.num_cities as usize {
                // Transfer over in slices
                let slice = self.city_connections_w_costs.slice(s![i, i..]);
                upper_triangle_edges
                    .slice_mut(s![triangle_index..triangle_index + slice.len()])
                    .assign(&slice);
                triangle_index += slice.len();
            }

            state.serialize_field("upper_triangle_edges", &upper_triangle_edges.to_vec())?;
        } else {
            state.serialize_field("city_connections_w_costs", &self.city_connections_w_costs)?;
        }

        state.end()
    }
}

// Implementing Deserialize for TSPProblem
impl<'de> serde::Deserialize<'de> for TSPProblem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {
        enum Field {
            NumCities,
            UndirectedEdges,
            UpperTriangleEdges,
            CityConnectionsWCosts,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: serde::Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str(
                            "`num_cities`, `undirected`, `upper_triangle_edges`, or `city_connections_w_costs`"
                        )
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E> where E: serde::de::Error {
                        match value {
                            "num_cities" => Ok(Field::NumCities),
                            "undirected" => Ok(Field::UndirectedEdges),
                            "upper_triangle_edges" => Ok(Field::UpperTriangleEdges),
                            "city_connections_w_costs" => Ok(Field::CityConnectionsWCosts),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TSPProblemVisitor;

        impl<'de> serde::de::Visitor<'de> for TSPProblemVisitor {
            type Value = TSPProblem;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TSPProblem")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TSPProblem, V::Error>
                where V: serde::de::MapAccess<'de>
            {
                let mut num_cities = None;
                let mut undirected_edges = None;
                let mut upper_triangle_edges = None::<Vec<f32>>;
                let mut city_connections_w_costs = None::<Array2<f32>>;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::NumCities => {
                            if num_cities.is_some() {
                                return Err(serde::de::Error::duplicate_field("num_cities"));
                            }
                            num_cities = Some(map.next_value()?);
                        }
                        Field::UndirectedEdges => {
                            if undirected_edges.is_some() {
                                return Err(serde::de::Error::duplicate_field("undirected"));
                            }
                            undirected_edges = Some(map.next_value()?);
                        }
                        Field::UpperTriangleEdges => {
                            if upper_triangle_edges.is_some() {
                                return Err(
                                    serde::de::Error::duplicate_field("upper_triangle_edges")
                                );
                            }
                            upper_triangle_edges = Some(map.next_value()?);
                        }
                        Field::CityConnectionsWCosts => {
                            if city_connections_w_costs.is_some() {
                                return Err(
                                    serde::de::Error::duplicate_field("city_connections_w_costs")
                                );
                            }
                            city_connections_w_costs = Some(map.next_value()?);
                        }
                    }
                }

                let num_cities: u64 = num_cities.ok_or_else(||
                    serde::de::Error::missing_field("num_cities")
                )?;
                let undirected_edges: bool = undirected_edges.ok_or_else(||
                    serde::de::Error::missing_field("undirected")
                )?;

                // Construct the city_connections_w_costs Array2<f32> depending on whether we have undirected edges
                let city_connections_w_costs = if let Some(edges) = upper_triangle_edges {
                    let mut connections = Array2::<f32>::zeros((
                        num_cities as usize,
                        num_cities as usize,
                    ));
                    let mut edge_iter = edges.iter();
                    for i in 0..num_cities as usize {
                        for j in i..num_cities as usize {
                            if let Some(&cost) = edge_iter.next() {
                                connections[[i, j]] = cost;
                                connections[[j, i]] = cost; // Mirror for undirected graph
                            }
                        }
                    }
                    connections
                } else if let Some(connections) = city_connections_w_costs {
                    connections
                } else {
                    return Err(serde::de::Error::missing_field("city_connections_w_costs"));
                };

                Ok(TSPProblem {
                    num_cities,
                    undirected_edges,
                    city_connections_w_costs,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "num_cities",
            "undirected",
            "upper_triangle_edges",
            "city_connections_w_costs",
        ];
        deserializer.deserialize_struct("TSPProblem", FIELDS, TSPProblemVisitor)
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
