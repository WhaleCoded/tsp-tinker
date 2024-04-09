use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::Instant,
};

use ndarray::Array2;
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};

use crate::types::{TSPAlgorithm, TSPProblem, TSPSolution};

use super::naive_heuristic::generate_naive_heuristic_solution;

const DEPTH_BEFORE_BOPUNDING: u64 = 4;
const MAX_THREADS: usize = 30;

fn sorted_push(vec: &mut Vec<(usize, usize, f32)>, edge: (usize, usize, f32)) {
    // Default goes at the end
    let mut insert_index = vec.len();
    if vec.len() == 0 {
        vec.push(edge);
        return;
    }

    let new_cost = edge.2;
    for (i, (_, _, curr_cost)) in vec.iter().enumerate() {
        if *curr_cost >= new_cost {
            insert_index = i;
            break;
        }
    }
    vec.insert(insert_index, edge);
}

fn find_parent(parent_and_rank: &mut Vec<(usize, u64)>, node: usize) -> usize {
    if parent_and_rank[node as usize].0 != node {
        parent_and_rank[node as usize].0 =
            find_parent(parent_and_rank, parent_and_rank[node as usize].0);
    }
    return parent_and_rank[node as usize].0;
}

fn union_sets(parent_and_rank: &mut Vec<(usize, u64)>, node_a: usize, node_b: usize) {
    let a_parent = find_parent(parent_and_rank, node_a);
    let b_parent = find_parent(parent_and_rank, node_b);
    if parent_and_rank[a_parent as usize].1 > parent_and_rank[b_parent as usize].1 {
        parent_and_rank[b_parent as usize].0 = a_parent;
    } else if parent_and_rank[a_parent as usize].1 < parent_and_rank[b_parent as usize].1 {
        parent_and_rank[a_parent as usize].0 = b_parent;
    } else {
        parent_and_rank[b_parent as usize].0 = a_parent;
        parent_and_rank[a_parent as usize].1 += 1;
    }
}

fn is_cycle(parent_and_rank: &mut Vec<(usize, u64)>, start_node: usize, end_node: usize) -> bool {
    let start_parent = find_parent(parent_and_rank, start_node);
    let end_parent = find_parent(parent_and_rank, end_node);
    return start_parent == end_parent;
}

fn calc_minimum_spanning_tree(
    nodes_in_tree: &HashSet<u64>,
    city_connections_w_costs: &Array2<f32>,
) -> f32 {
    // We need to convert the sparse city indexes to continuous indexes
    let mut city_index_to_tree_index: HashMap<u64, usize> = HashMap::new();
    for (i, city) in nodes_in_tree.iter().enumerate() {
        city_index_to_tree_index.insert(*city, i);
    }

    let mut tree_edges: Vec<(usize, usize, f32)> = vec![];
    let nodes_as_vec: Vec<u64> = nodes_in_tree.iter().cloned().collect();
    // Gather the edges for our tree
    // To avoid O(elog(e)) sorting, we keep tree_edges sorted as we go which is only O(e)
    for node in nodes_in_tree {
        let node_tree_index = city_index_to_tree_index[node];
        for end_node in &nodes_as_vec {
            let end_node_tree_index = city_index_to_tree_index[end_node];
            if *node != *end_node as u64 {
                sorted_push(
                    &mut tree_edges,
                    (
                        node_tree_index,
                        end_node_tree_index,
                        city_connections_w_costs[[*node as usize, *end_node as usize]],
                    ),
                );
            }
        }
    }

    // Using kruskals algorithm to find the minimum spanning tree
    let mut mst_cost = 0.0;
    let mut mst_edges: Vec<(usize, usize)> = vec![];
    let mut parent_and_rank = nodes_in_tree
        .iter()
        .map(|x| (city_index_to_tree_index[x], 0))
        .collect::<Vec<(usize, u64)>>();
    for (start_node, end_node, cost) in tree_edges {
        // Check for cycle
        if is_cycle(&mut parent_and_rank, start_node, end_node) {
            continue;
        }

        // Add edge to mst
        mst_cost += cost;
        mst_edges.push((start_node, end_node));
        union_sets(&mut parent_and_rank, start_node, end_node);

        if mst_edges.len() == nodes_in_tree.len() - 1 {
            // minimum spanning tree is complete
            // no new edges will be added
            break;
        }
    }

    return mst_cost;
}

fn calc_lower_bound(
    fixed_connections: &Vec<(u64, u64)>,
    city_connections_w_costs: &Array2<f32>,
) -> f32 {
    let problem_size = city_connections_w_costs.shape()[0] as u64;
    let mut remaining_cities: HashSet<u64> = HashSet::from_iter(0..problem_size);

    // Cost up to now
    let mut current_cost = 0.0;
    for (city_a, city_b) in fixed_connections {
        current_cost += city_connections_w_costs[[*city_a as usize, *city_b as usize]];
        remaining_cities.remove(&city_a);
        remaining_cities.remove(&city_b);
    }

    // Cost of remaining minimum spanning tree
    let mst_cost = calc_minimum_spanning_tree(&remaining_cities, &city_connections_w_costs);

    let lower_bound = current_cost + mst_cost;
    return lower_bound;
}

fn branch_from_root(
    starting_upper_bound: f32,
    city_connections_w_costs: &Array2<f32>,
) -> (f32, Vec<u64>) {
    let fixed_edges: Vec<(u64, u64)> = vec![];
    let available_cities: HashSet<u64> =
        HashSet::from_iter(1..city_connections_w_costs.shape()[0] as u64);

    let best_upper_bound = starting_upper_bound;
    let best_tour = vec![];
    let upper_bound_arc = Arc::new(RwLock::new((best_upper_bound, best_tour)));

    let num_threads = match MAX_THREADS {
        _ if available_cities.len() > MAX_THREADS => MAX_THREADS,
        _ => available_cities.len(),
    };
    let pool = match ThreadPoolBuilder::new().num_threads(num_threads).build() {
        Ok(pool) => pool,
        Err(e) => {
            panic!("Error creating thread pool: {}", e);
        }
    };

    pool.install(|| {
        available_cities.par_iter().for_each(|available_city| {
            let child_upper_bound = upper_bound_arc.clone();
            let mut child_fixed_edges = fixed_edges.clone();
            child_fixed_edges.push((0, *available_city));
            let mut child_available_cities = available_cities.clone();
            child_available_cities.remove(&available_city);
            branch_from_child(
                &child_upper_bound,
                city_connections_w_costs,
                child_fixed_edges,
                child_available_cities,
                1,
            );
        });
    });

    let (final_best_cost, final_best_tour) = upper_bound_arc
        .read()
        .expect("We should always be able to read the upper bound")
        .clone();
    return (final_best_cost, final_best_tour);
}

fn branch_from_child(
    starting_upper_bound: &Arc<RwLock<(f32, Vec<u64>)>>,
    city_connections_w_costs: &Array2<f32>,
    mut fixed_edges: Vec<(u64, u64)>,
    available_cities: HashSet<u64>,
    depth: u64,
) {
    let curr_city = fixed_edges
        .last()
        .expect("there should always be a curr_city in the child")
        .1;

    if available_cities.len() == 0 {
        // We have reached the end of the tree
        fixed_edges.push((curr_city, 0));

        let mut tour_cost = 0.0;
        for (city_a, city_b) in &fixed_edges {
            tour_cost += city_connections_w_costs[[*city_a as usize, *city_b as usize]];
        }

        let tour: Vec<u64> = fixed_edges.iter().map(|x| x.0).collect();
        assert_eq!(tour.len(), city_connections_w_costs.shape()[0]);

        // While this read check is redundant, it prevents most of the leaf nodes from acquiring the write lock
        let current_best_cost;
        {
            // Read the current best cost
            let bssf_tuple = starting_upper_bound
                .read()
                .expect("We should always be able to read the upper bound");
            current_best_cost = bssf_tuple.0;
        }

        if tour_cost < current_best_cost {
            // Update the best tour
            let mut upper_bound = starting_upper_bound
                .write()
                .expect("We should always be able to write the upper bound");
            if tour_cost < upper_bound.0 {
                *upper_bound = (tour_cost, tour);
            }
        }
    }

    for available_city in &available_cities {
        if depth >= DEPTH_BEFORE_BOPUNDING {
            let lower_bound = calc_lower_bound(&fixed_edges, &city_connections_w_costs);
            let best_upper_bound: f32;
            {
                let best_bound_tuple = starting_upper_bound
                    .read()
                    .expect("We should always be able to read the upper bound");
                best_upper_bound = best_bound_tuple.0;
            }
            if lower_bound > best_upper_bound {
                // Prune this branch
                continue;
            }
        }

        let mut child_fixed_edges = fixed_edges.clone();
        child_fixed_edges.push((curr_city, *available_city));
        let mut child_available_cities = available_cities.clone();
        child_available_cities.remove(&available_city);
        branch_from_child(
            starting_upper_bound,
            city_connections_w_costs,
            child_fixed_edges,
            child_available_cities,
            depth + 1,
        );
    }

    return;
}

pub fn calc_branch_n_bound(tsp_problem: &TSPProblem, timeout: &Option<u32>) -> Option<TSPSolution> {
    if !tsp_problem.undirected_edges {
        // This current implementation has only been tested on undirected graphs
        return None;
    }

    // Get start time
    let start_time = Instant::now();

    // Initialize the best tour and cost
    let starting_upper_bound = generate_naive_heuristic_solution(tsp_problem);

    // Branch and bound
    let (new_cost, new_tour) = branch_from_root(
        starting_upper_bound.tot_cost,
        &tsp_problem.city_connections_w_costs,
    );

    // Initialize the priority queue
    // Get end time
    let end_time = start_time.elapsed().as_secs_f32();

    // Return the best solution
    return Some(TSPSolution {
        tot_cost: new_cost,
        path: new_tour,
        optimal: true,
        calculation_time: end_time,
        algorithm_name: TSPAlgorithm::BranchNBound.to_string(),
    });
}
