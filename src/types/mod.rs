use ndarray::Array2;

mod helpers;
mod serialization;
mod struct_functionality;

pub use helpers::{
    convert_tour_into_undirected_edges, convert_undirected_edges_into_tour,
    convert_undirected_matrix_to_edges, get_num_existing_tsp_problems_by_size,
    get_num_existing_tsp_problems_by_sub_dir, get_subdirectories_of_tsp_problems,
    get_tsp_problem_file_paths_by_sub_dir,
};

#[derive(serde::Deserialize, Clone, serde::Serialize)]
pub struct TSPSolution {
    pub tot_cost: f32,
    pub path: Vec<u64>,
    pub optimal: bool,
    pub calculation_time: f32,
    pub algorithm_name: String,
}

#[derive(Debug, serde::Deserialize, Copy, Clone, serde::Serialize)]
pub struct UndirectedEdge {
    pub city_a: u64,
    pub city_b: u64,
}

#[derive(Clone)]
pub struct TSPProblem {
    pub num_cities: u64,
    pub undirected_edges: bool,
    pub city_connections_w_costs: Array2<f32>,
    pub generation_method: TSPGenerationMethod,
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

#[derive(serde::Deserialize, Clone, serde::Serialize)]
pub enum TSPGenerationMethod {
    Euclidean,
    RandomUndirected,
    RandomDirected,
}

impl TSPGenerationMethod {
    pub fn to_string(&self) -> String {
        match self {
            TSPGenerationMethod::Euclidean => {
                return "Euclidean".to_string();
            }
            TSPGenerationMethod::RandomUndirected => {
                return "Random Undirected".to_string();
            }
            TSPGenerationMethod::RandomDirected => {
                return "Random Directed".to_string();
            }
        }
    }
}
