use ndarray::Array2;

mod serialization;
mod struct_functionality;
mod helpers;

pub use helpers::{
    get_num_existing_tsp_problems_by_size,
    get_subdirectories_of_tsp_problems,
    get_num_existing_tsp_problems_by_sub_dir,
    get_tsp_problem_file_paths_by_sub_dir,
    convert_undirected_matrix_to_edges,
    convert_undirected_edges_into_tour,
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
