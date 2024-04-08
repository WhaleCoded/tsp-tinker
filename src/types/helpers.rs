use ndarray::Array1;
use std::{
    collections::HashSet,
    io::Error,
    path::{Path, PathBuf},
};

use crate::types::struct_functionality::create_json_sub_dir_name;

use super::UndirectedEdge;

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

pub fn get_num_existing_tsp_problems_by_size(
    data_path: &Path,
    problem_size: u64,
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

pub fn convert_undirected_matrix_to_edges(num_cities: &u64) -> Vec<UndirectedEdge> {
    let mut edges: Vec<UndirectedEdge> = vec![];

    for i in 0..*num_cities {
        for j in i + 1..*num_cities {
            edges.push(UndirectedEdge {
                city_a: i as u64,
                city_b: j as u64,
            });
        }
    }

    return edges;
}

fn group_undirected_edges_into_array(
    num_cities: u64,
    undirected_edges: &Vec<UndirectedEdge>,
) -> Array1<Vec<UndirectedEdge>> {
    let mut grouped_edges: Array1<Vec<UndirectedEdge>> =
        Array1::from(vec![vec![]; num_cities as usize]);

    for edge in undirected_edges {
        grouped_edges[edge.city_a as usize].push(edge.clone());
        grouped_edges[edge.city_b as usize].push(edge.clone());
    }

    return grouped_edges;
}

pub fn convert_undirected_edges_into_tour(
    num_cities: u64,
    undirected_edges: &Vec<UndirectedEdge>,
) -> Vec<u64> {
    let mut tour = vec![];

    let mut grouped_edges = group_undirected_edges_into_array(num_cities, undirected_edges);

    // There must be 2 edges for each city to be a valid tour
    for city_edges in grouped_edges.iter() {
        assert_eq!(city_edges.len(), 2);
    }

    let mut curr_city: u64 = 0;
    while tour.len() < (num_cities as usize) {
        // Get edge that hasn't been visited for the current city
        let next_edge = grouped_edges[curr_city as usize].pop().expect(&format!(
            "There should be exactly 2 edges for each city. Current tour is {}--->{:?}",
            curr_city, tour
        ));
        // println!("Current city: {}", curr_city);
        // println!("Next edge: {}", next_edge);

        // Add the other city of the edge to the tour
        let next_city = match next_edge.city_a {
            _ if curr_city == next_edge.city_a => next_edge.city_b,
            _ => next_edge.city_a,
        };
        // println!("Next city: {}", next_city);

        // Remove next_ctiy's edge to curr_city
        grouped_edges[next_city as usize].retain(|edge| edge != &next_edge);

        tour.push(next_city);
        curr_city = next_city;

        // println!("Tour length: {}/{}", tour.len(), num_cities);
    }

    assert!(tour.len() == (num_cities as usize));

    return tour;
}

pub fn convert_tour_into_undirected_edges(tour: &Vec<u64>) -> HashSet<UndirectedEdge> {
    let mut t_prime_edges: HashSet<UndirectedEdge> = HashSet::new();

    // Convert T to edges
    for (i, node) in tour.iter().enumerate() {
        if i == 0 {
            t_prime_edges.insert(UndirectedEdge::new(0, *node));
        } else {
            t_prime_edges.insert(UndirectedEdge::new(tour[i - 1], *node));
        }
    }

    return t_prime_edges;
}
