use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::Instant,
};

use ndarray::Array2;
use rand::prelude::IteratorRandom;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};

use super::utils::calculate_cost_of_tour;
use crate::types::{
    convert_tour_into_undirected_edges, convert_undirected_edges_into_tour, TSPAlgorithm,
    TSPProblem, TSPSolution, UndirectedEdge,
};

const MAX_Y_OPTIONS: usize = 5;
const NUM_SOLVES_BEFORE_REDUCTION: u32 = 30;

pub fn generate_pseudorandom_solution(tsp_problem: &TSPProblem) -> TSPSolution {
    // Get start time
    let start_time = Instant::now();

    let mut cost = 0.0;
    let mut tour = vec![];

    let mut curr_city = 0;
    let mut available_cities = (1..tsp_problem.num_cities).collect::<Vec<u64>>();
    while tour.len() < (tsp_problem.num_cities as usize) {
        if tour.len() == (tsp_problem.num_cities as usize) - 1 {
            tour.push(0);
            cost += tsp_problem.city_connections_w_costs[[curr_city as usize, 0]];
        } else {
            let next_city_index = (0..available_cities.len())
                .choose(&mut rand::thread_rng())
                .unwrap();
            let next_city = available_cities.swap_remove(next_city_index);
            tour.push(next_city);
            cost += tsp_problem.city_connections_w_costs[[curr_city as usize, next_city as usize]];
            curr_city = next_city;
        }
    }

    // Get end time
    let end_time = Instant::now();

    // Create the TSPSolution
    let tsp_solution = TSPSolution {
        algorithm_name: TSPAlgorithm::Pseudorandom.to_string(),
        path: tour,
        tot_cost: cost,
        optimal: false,
        calculation_time: end_time.duration_since(start_time).as_secs_f32(),
    };

    return tsp_solution;
}

fn get_edges_for_node(node: u64, tsp_tour: &Vec<u64>) -> Vec<UndirectedEdge> {
    if node == 0 {
        return vec![
            UndirectedEdge::new(tsp_tour[tsp_tour.len() - 2], 0),
            UndirectedEdge::new(0, tsp_tour[0]),
        ];
    }

    let mut in_edge = UndirectedEdge::new(0, 0);
    let mut out_edge = UndirectedEdge::new(0, 0);
    for (i, city) in tsp_tour.iter().enumerate() {
        if *city == node {
            match i {
                0 => {
                    in_edge = UndirectedEdge::new(tsp_tour[tsp_tour.len() - 1], node);
                    out_edge = UndirectedEdge::new(node, tsp_tour[i + 1]);
                }
                _ => {
                    if i == tsp_tour.len() - 1 {
                        in_edge = UndirectedEdge::new(tsp_tour[i - 1], node);
                        out_edge = UndirectedEdge::new(node, tsp_tour[0]);
                    } else {
                        in_edge = UndirectedEdge::new(tsp_tour[i - 1], node);
                        out_edge = UndirectedEdge::new(node, tsp_tour[i + 1]);
                    }
                }
            }

            break;
        }
    }

    return vec![in_edge, out_edge];
}

fn get_viable_y_edges_ordered_by_best_value(
    t_nodes: &Vec<u64>,
    x_connections: &Vec<UndirectedEdge>,
    available_nodes: &HashSet<u64>,
    connection_and_cost_matrix: &Array2<f32>,
    broken_connections: &HashSet<UndirectedEdge>,
    joined_edges: &HashSet<UndirectedEdge>,
    curr_t_prime_edges: &Vec<UndirectedEdge>,
) -> Vec<(UndirectedEdge, u64, f32, f32)> {
    let mut y_edges: Vec<(UndirectedEdge, u64, f32, f32)> = vec![];

    let t_2i = t_nodes[t_nodes.len() - 1];
    let last_xi = x_connections[x_connections.len() - 1];
    let x_cost = connection_and_cost_matrix[[last_xi.city_a as usize, last_xi.city_b as usize]];

    //println!("Available nodes: {:?}", available_nodes);
    //println!("T nodes: {:?}", t_nodes);
    for node in available_nodes.iter() {
        let y_edge_candidate = UndirectedEdge::new(t_2i, *node);
        //println!("Checking y edge candidate: {}", y_edge_candidate);
        if !broken_connections.contains(&y_edge_candidate) {
            let y_cost = connection_and_cost_matrix[[t_2i as usize, *node as usize]];
            //println!("Cost of y edge candidate: {}", y_cost);
            let improvement = x_cost - y_cost;
            //println!("Improvement: {}", improvement);

            // Gain can temporarily be negative
            y_edges.push((y_edge_candidate, *node, improvement, y_cost));
        }
    }

    // Sort by improvement
    y_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    y_edges.truncate(MAX_Y_OPTIONS);

    // We now have the nearest neighbors now calculate |xi+1| - |yi| and sort by that value
    let mut y_edges_to_remove = vec![];
    for (i, (_, y_node, _, y_cost)) in y_edges.iter_mut().enumerate() {
        let t1 = t_nodes[0];
        let t2i = t_nodes[t_nodes.len() - 1];
        let t2i_plus_1 = *y_node;
        let xi_plus_one = choose_x_deterministic_edge(
            t2i_plus_1,
            t2i,
            t1,
            curr_t_prime_edges,
            joined_edges,
            true,
        );

        match xi_plus_one {
            Some((next_xi_edge, _)) => {
                let xi_plus_one_cost = connection_and_cost_matrix
                    [[next_xi_edge.city_a as usize, next_xi_edge.city_b as usize]];

                //println!("Cost of yi: {}", y_cost);
                //println!("Cost of xi+1: {}", xi_plus_one_cost);
                *y_cost = xi_plus_one_cost - *y_cost;
                //println!("Cost of xi+1 - yi: {}", y_cost);
            }
            None => {
                // This y edge is not viable remove it
                y_edges_to_remove.push(i);
            }
        }
    }
    for i in y_edges_to_remove.iter().rev() {
        y_edges.remove(*i);
    }
    y_edges.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    return y_edges;
}

fn choose_y_edge(
    x_connections: &Vec<UndirectedEdge>,
    connection_and_cost_matrix: &Array2<f32>,
    t_nodes: &Vec<u64>,
    available_nodes: &HashSet<u64>,
    broken_connections: &HashSet<UndirectedEdge>,
) -> Option<(UndirectedEdge, u64, f32)> {
    let mut y_edge = None;

    let t_2i = t_nodes[t_nodes.len() - 1];
    let last_xi = x_connections[x_connections.len() - 1];
    let x_cost = connection_and_cost_matrix[[last_xi.city_a as usize, last_xi.city_b as usize]];
    let mut best_improvement = 0.0;
    let mut t2i_plus_1 = None;

    //println!("Available nodes: {:?}", available_nodes);
    //println!("T nodes: {:?}", t_nodes);
    for node in available_nodes.iter() {
        let y_edge_candidate = UndirectedEdge::new(t_2i, *node);
        //println!("Checking y edge candidate: {}", y_edge_candidate);
        if !broken_connections.contains(&y_edge_candidate) {
            let y_cost = connection_and_cost_matrix[[t_2i as usize, *node as usize]];
            //println!("Cost of y edge candidate: {}", y_cost);
            let improvement = x_cost - y_cost;
            //println!("Improvement: {}", improvement);
            if improvement > best_improvement {
                best_improvement = improvement;
                y_edge = Some(y_edge_candidate);
                t2i_plus_1 = Some(*node);
            }
        }
    }

    match y_edge {
        Some(edge) => {
            let t2i_plus_1 = t2i_plus_1.expect("If an edge is selected t2i+1 should be set.");
            //println!("Selected yi: {}", edge);
            //println!("This implies t2i + 1: {}", t2i_plus_1);
            return Some((edge, t2i_plus_1, best_improvement));
        }
        None => {
            //println!("No profitable y edge found.");
        }
    }

    return None;
}

fn choose_x_deterministic_edge(
    t_2i_plus_1: u64,
    t_2i: u64,
    t1: u64,
    curr_t_prime_edges: &Vec<UndirectedEdge>,
    joined_edges: &HashSet<UndirectedEdge>,
    testing_y_options: bool,
) -> Option<(UndirectedEdge, u64)> {
    //println!(
    //     "Choosing a deterministic xi with t1: {}, t2i: {}, t2i+1: {}",
    //     t1, t_2i, t_2i_plus_1
    // );
    let mut consideration_edges = curr_t_prime_edges.clone();
    let y_index = consideration_edges
        .iter()
        .position(|&x| x == UndirectedEdge::new(t_2i, t_2i_plus_1));
    if !testing_y_options {
        consideration_edges.remove(y_index.expect("y edge not found in consideration edges."));
    }

    //println!(
    //     "Consideration edges: [{}]",
    //     consideration_edges
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    // Traverse the edges until we find which edge is on the opposite side of t1
    let mut ordered_edges = HashMap::new();
    for edge in consideration_edges.iter() {
        let a_edge_vec = ordered_edges.entry(edge.city_a).or_insert(vec![]);
        a_edge_vec.push(edge);
        let b_edge_vec = ordered_edges.entry(edge.city_b).or_insert(vec![]);
        b_edge_vec.push(edge);
    }

    assert_eq!(ordered_edges[&t1].len(), 1);
    let mut curr_edge = ordered_edges[&t1][0];
    let mut curr_node = t1;
    let mut xi = None;
    let mut new_t_node = 0;
    while curr_node != t_2i_plus_1 {
        //println!("Current node: {}, Current edge: {}", curr_node, curr_edge);
        let next_node = match curr_node {
            _ if curr_node == curr_edge.city_a => curr_edge.city_b,
            _ => curr_edge.city_a,
        };

        let possible_nodes = &ordered_edges[&next_node];
        //println!(
        //     "Possible nodes for {}: {}",
        //     next_node,
        //     possible_nodes
        //         .iter()
        //         .map(|p| p.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );
        assert_eq!(possible_nodes.len(), 2);
        let next_edge = match possible_nodes[0] {
            // if this is the next edge curr_node won't be present
            _ if curr_node == possible_nodes[0].city_a || curr_node == possible_nodes[0].city_b => {
                possible_nodes[1]
            }
            _ => possible_nodes[0],
        };

        curr_node = next_node;
        curr_edge = next_edge;

        if curr_node == t_2i_plus_1 {
            // Return the edge that is opposite of t1
            new_t_node = match next_edge {
                _ if next_edge.city_a == curr_node => next_edge.city_b,
                _ => next_edge.city_a,
            };
            xi = Some(next_edge);
            break;
        }
    }
    assert_eq!(xi.is_some(), true);
    //println!("Selected xi: {}", xi.unwrap());

    if !joined_edges.contains(&xi.unwrap()) {
        // curr node is the next t2i
        return Some((*xi.unwrap(), new_t_node));
    }

    //println!("xi already in joined edges.");
    return None;
}

fn step_5(best_improvement: f32, t_prime_edges: &Vec<UndirectedEdge>) -> Option<Vec<u64>> {
    //println!("Step 5");
    // if best_improvement is positive then apply the changes to form T`
    if best_improvement > 0.0 {
        //println!(
        //     "Positive gain found {}. Constructing T'...",
        //     best_improvement
        // );

        let num_cities = t_prime_edges.len() as u64;
        //println!("Number of cities: {}", num_cities);
        //println!(
        //     "T' edges: [{}]",
        //     t_prime_edges
        //         .iter()
        //         .map(|p| p.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );
        let t_prime = convert_undirected_edges_into_tour(num_cities, &t_prime_edges);
        //println!(
        //     "Constructed T': [{}]",
        //     t_prime
        //         .iter()
        //         .map(|p| p.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );

        return Some(t_prime);
    }

    return None;
}

fn loop_should_be_closed(
    best_improvement: f32,
    tot_curr_gain: f32,
    x_edge: &UndirectedEdge,
    t_nodes: &Vec<u64>,
    city_connections_w_costs: &Array2<f32>,
) -> Option<f32> {
    // check gain of closing the loop between t-2i and t1 compared to xi
    //println!("Checking gain of closing the loop between t-2i and t1 compared to xi...");
    let x_cost = city_connections_w_costs[[x_edge.city_a as usize, x_edge.city_b as usize]];
    let close_loop_cost =
        city_connections_w_costs[[t_nodes[0] as usize, t_nodes[t_nodes.len() - 1] as usize]];
    let close_loop_gain = x_cost - close_loop_cost;
    let gi_star = tot_curr_gain + close_loop_gain;
    //println!("Gain of closing the loop: {}", close_loop_gain);
    //println!("Total gain: {}", gi_star);
    //println!("Best improvement so far: {}", best_improvement);

    if gi_star > best_improvement {
        //println!("New best improvement: {}", gi_star);
        return Some(gi_star);
    }

    return None;
}

fn step_4_change_loop(
    best_improvement: f32,
    best_t_prime_edges: &Vec<UndirectedEdge>,
    tot_gain: f32,
    pre_selected_t_nodes: &Vec<u64>,
    pre_selected_x_connections: &Vec<UndirectedEdge>,
    pre_selected_y_connections: &Vec<UndirectedEdge>,
    pre_selected_available_nodes: &HashSet<u64>,
    pre_selected_joined_connections: &HashSet<UndirectedEdge>,
    pre_selected_broken_connections: &HashSet<UndirectedEdge>,
    city_connections_w_costs: &Array2<f32>,
    pre_selected_curr_t_prime_edges: &Vec<UndirectedEdge>,
    reduction_edges: &Option<HashSet<UndirectedEdge>>,
) -> Option<Vec<u64>> {
    let mut best_improvement = best_improvement;
    let mut best_t_prime_edges = best_t_prime_edges.clone();
    let mut tot_gain = tot_gain;
    let mut t_nodes = pre_selected_t_nodes.clone();
    let mut x_connections = pre_selected_x_connections.clone();
    let mut y_connections = pre_selected_y_connections.clone();
    let mut available_nodes = pre_selected_available_nodes.clone();
    let mut broken_connections = pre_selected_broken_connections.clone();
    let mut joined_connections = pre_selected_joined_connections.clone();
    let mut curr_t_prime_edges = pre_selected_curr_t_prime_edges.clone();

    loop {
        if x_connections.len() == 4 {
            // i is now == 4 so the reduction rule can be applied
            if let Some(edges_to_prohibit) = reduction_edges {
                for edge in edges_to_prohibit.iter() {
                    joined_connections.insert(*edge);
                }
            }
        }

        if
        // implies t-2i
        let Some((x_edge, t2i)) = choose_x_deterministic_edge(
            t_nodes[t_nodes.len() - 1],
            t_nodes[t_nodes.len() - 2],
            t_nodes[0],
            &curr_t_prime_edges,
            &joined_connections,
            false,
        ) {
            x_connections.push(x_edge);
            broken_connections.insert(x_edge);
            t_nodes.push(t2i);
            available_nodes.remove(&t2i);
            let x_position = curr_t_prime_edges
                .iter()
                .position(|&p| p == x_edge)
                .expect("x edge not found in current edges");
            curr_t_prime_edges.remove(x_position);

            // check gain of closing the loop between t-2i and t1 compared to xi
            let close_loop_gain = loop_should_be_closed(
                best_improvement,
                tot_gain,
                &x_edge,
                &t_nodes,
                city_connections_w_costs,
            );
            match close_loop_gain {
                Some(gi_star) => {
                    best_improvement = gi_star;
                    let closure_edge = UndirectedEdge::new(t_nodes[0], t_nodes[t_nodes.len() - 1]);
                    best_t_prime_edges = curr_t_prime_edges.clone();
                    best_t_prime_edges.push(closure_edge);
                }
                None => {
                    // Stopping criteria mentioned in step 5
                    //println!("Stopping criteria met. Moving to step 5...");
                    break;
                }
            }

            // choose new y-i implies t-2i+1
            if let Some((y_edge, t2i_plus_1, gain)) = choose_y_edge(
                &x_connections,
                city_connections_w_costs,
                &t_nodes,
                &available_nodes,
                &broken_connections,
            ) {
                y_connections.push(y_edge);
                joined_connections.insert(y_edge);
                t_nodes.push(t2i_plus_1);
                available_nodes.remove(&t2i_plus_1);
                tot_gain += gain;
                curr_t_prime_edges.push(y_edge);

                // verify gain is positive else stop
            } else {
                // Go to Step 5
                //println!("No profitable y edge found. Moving to step 5...");
                break;
            }
        } else {
            // 4-(e) xi+1 could not be broken remove the last yi and ti
            //println!("xi+1 could not be broken. Removing the last yi and ti...");
            let last_y_edge = y_connections.pop().unwrap();
            //println!("Removed yi: {}", last_y_edge);
            joined_connections.remove(&last_y_edge);
            let y_position = curr_t_prime_edges
                .iter()
                .position(|&p| p == last_y_edge)
                .expect("y edge not found in current edges");
            curr_t_prime_edges.remove(y_position);
            let last_t_node = t_nodes.pop().unwrap();
            //println!("Removed ti: {}", last_t_node);
            available_nodes.insert(last_t_node);
            //println!(
            //     "y-edges: [{}]",
            //     y_connections
            //         .iter()
            //         .map(|p| p.to_string())
            //         .collect::<Vec<_>>()
            //         .join(", ")
            // );
            //println!(
            //     "t-nodes: [{}]",
            //     t_nodes
            //         .iter()
            //         .map(|p| p.to_string())
            //         .collect::<Vec<_>>()
            //         .join(", ")
            // );
            //println!(
            //     "Current T` edges: [{}]",
            //     curr_t_prime_edges
            //         .iter()
            //         .map(|p| p.to_string())
            //         .collect::<Vec<_>>()
            //         .join(", ")
            // );
            break;
        }
    }

    return step_5(best_improvement, &best_t_prime_edges);
}

fn step_4_with_back_tracking(
    tsp_problem: &TSPProblem,
    pre_selected_t_nodes: &Vec<u64>,
    pre_selected_x_connections: &Vec<UndirectedEdge>,
    pre_selected_y_connections: &Vec<UndirectedEdge>,
    pre_selected_available_nodes: &HashSet<u64>,
    pre_selected_broken_connections: &HashSet<UndirectedEdge>,
    pre_selected_joined_connections: &HashSet<UndirectedEdge>,
    pre_selected_tot_gain: &f32,
    reduction_edges: &Option<HashSet<UndirectedEdge>>,
    pre_selected_curr_t_prime_edges: &Vec<UndirectedEdge>,
) -> Option<Vec<u64>> {
    let mut best_improvement = 0.0;
    let mut best_t_prime_edges = Vec::new();
    let mut tot_gain = *pre_selected_tot_gain;
    let mut t_nodes = pre_selected_t_nodes.clone();
    let mut x_connections = pre_selected_x_connections.clone();
    let mut y_connections = pre_selected_y_connections.clone();
    let mut available_nodes = pre_selected_available_nodes.clone();
    let mut broken_connections = pre_selected_broken_connections.clone();
    let mut joined_connections = pre_selected_joined_connections.clone();
    let mut curr_t_prime_edges = pre_selected_curr_t_prime_edges.clone();

    // Step 4
    //println!("Step 4");

    // Choose x2 and y2 separately because they can be backtracked
    if let Some((x_edge, t2i)) = choose_x_deterministic_edge(
        t_nodes[t_nodes.len() - 1],
        t_nodes[t_nodes.len() - 2],
        t_nodes[0],
        &curr_t_prime_edges,
        &joined_connections,
        false,
    ) {
        x_connections.push(x_edge);
        broken_connections.insert(x_edge);
        t_nodes.push(t2i);
        available_nodes.remove(&t2i);
        let x_position = curr_t_prime_edges
            .iter()
            .position(|&p| p == x_edge)
            .expect("x edge not found in current edges");
        curr_t_prime_edges.remove(x_position);

        // check gain of closing the loop between t-2i and t1 compared to xi
        let close_loop_gain = loop_should_be_closed(
            best_improvement,
            tot_gain,
            &x_edge,
            &t_nodes,
            &tsp_problem.city_connections_w_costs,
        );
        match close_loop_gain {
            Some(gi_star) => {
                best_improvement = gi_star;
                let closure_edge = UndirectedEdge::new(t_nodes[0], t_nodes[t_nodes.len() - 1]);
                best_t_prime_edges = curr_t_prime_edges.clone();
                best_t_prime_edges.push(closure_edge);
            }
            None => {
                // Stopping criteria mentioned in step 5
                //println!("Stopping criteria met. Moving to step 5...");
                return step_5(best_improvement, &best_t_prime_edges);
            }
        }

        // Go through the y2 viable options in order of value
        let possible_y2_options = get_viable_y_edges_ordered_by_best_value(
            &t_nodes,
            &x_connections,
            &available_nodes,
            &tsp_problem.city_connections_w_costs,
            &broken_connections,
            &joined_connections,
            &curr_t_prime_edges,
        );
        //println!(
        //     "Possible y2 options: [{}]",
        //     possible_y2_options
        //         .iter()
        //         .map(|p| format!("{} : {}", p.2.to_string(), p.0.to_string()))
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );

        if possible_y2_options.is_empty() {
            // Go to Step 5
            //println!("No profitable y2 edge found. Moving to step 5...");
            return step_5(best_improvement, &best_t_prime_edges);
        }

        for (y_edge, t2i_plus_1, gain, _) in possible_y2_options {
            // select y-2 implies t-5
            y_connections.push(y_edge);
            joined_connections.insert(y_edge);
            t_nodes.push(t2i_plus_1);
            available_nodes.remove(&t2i_plus_1);
            tot_gain += gain;
            curr_t_prime_edges.push(y_edge);

            // We already know we had positive gain so keep trying to improve
            let t_prime = step_4_change_loop(
                best_improvement,
                &best_t_prime_edges,
                tot_gain,
                &t_nodes,
                &x_connections,
                &y_connections,
                &available_nodes,
                &joined_connections,
                &broken_connections,
                &tsp_problem.city_connections_w_costs,
                &curr_t_prime_edges,
                reduction_edges,
            );

            if t_prime.is_some() {
                return t_prime;
            }

            // Backtrack and undo changes
            //println!("Backtracking to try new y2 edge...");
            let t5 = t_nodes
                .pop()
                .expect("We pushed t5 we should be able to pop.");
            available_nodes.insert(t5);
            let y2 = y_connections
                .pop()
                .expect("We pushed y2 we should be able to pop.");
            joined_connections.remove(&y2);
            tot_gain -= gain;
            let y_position = curr_t_prime_edges
                .iter()
                .position(|&p| p == y2)
                .expect("y edge not found in current edges");
            curr_t_prime_edges.remove(y_position);
        }
    } else {
        // x2 could not be broken
        //println!("x2 could not be broken. No changes can be made backtracking...");
        return None;
    }

    return None;
}

fn step_3_with_back_tracking(
    tsp_problem: &TSPProblem,
    pre_selected_t_nodes: &Vec<u64>,
    pre_selected_x_connections: &Vec<UndirectedEdge>,
    pre_selected_available_nodes: &HashSet<u64>,
    pre_selected_broken_connections: &HashSet<UndirectedEdge>,
    reduction_edges: &Option<HashSet<UndirectedEdge>>,
    pre_selected_curr_t_prime_edges: &Vec<UndirectedEdge>,
) -> Option<Vec<u64>> {
    let mut t_nodes = pre_selected_t_nodes.clone();
    let mut y_connections: Vec<UndirectedEdge> = vec![];
    let mut available_nodes = pre_selected_available_nodes.clone();
    let mut joined_connections: HashSet<UndirectedEdge> = HashSet::new();
    let mut curr_t_prime_edges = pre_selected_curr_t_prime_edges.clone();

    // Step 3
    //println!("Step 3");
    //println!(
    //     "Current t prime edges: [{}]",
    //     curr_t_prime_edges
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );
    let mut tot_gain = 0.0;

    let possible_y1_options = get_viable_y_edges_ordered_by_best_value(
        &t_nodes,
        &pre_selected_x_connections,
        &available_nodes,
        &tsp_problem.city_connections_w_costs,
        &pre_selected_broken_connections,
        &joined_connections,
        &curr_t_prime_edges,
    );
    //println!(
    //     "Possible y1 options: [{}]",
    //     possible_y1_options
    //         .iter()
    //         .map(|p| format!("{} : {}", p.2.to_string(), p.0.to_string()))
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    for (y_edge, t_3, gain, _) in possible_y1_options {
        // select y-1 implies t-3
        //println!("Selected y1: {}", y_edge);
        //println!("This implies t3: {}", t_3);
        y_connections.push(y_edge);
        t_nodes.push(t_3);
        available_nodes.remove(&t_3);
        joined_connections.insert(y_edge);
        tot_gain += gain;
        curr_t_prime_edges.push(y_edge);

        // Move to next step
        let t_prime = step_4_with_back_tracking(
            tsp_problem,
            &t_nodes,
            &pre_selected_x_connections,
            &y_connections,
            &available_nodes,
            &pre_selected_broken_connections,
            &joined_connections,
            &tot_gain,
            reduction_edges,
            &curr_t_prime_edges,
        );

        if t_prime.is_some() {
            return t_prime;
        }

        // Backtrack and undo changes
        //println!("Backtracking to try new y1 edge...");
        let t3 = t_nodes
            .pop()
            .expect("We pushed t3 we should be able to pop.");
        available_nodes.insert(t3);
        let y1 = y_connections
            .pop()
            .expect("We pushed y1 we should be able to pop.");
        joined_connections.remove(&y1);
        tot_gain -= gain;
        let y_position = curr_t_prime_edges
            .iter()
            .position(|&p| p == y1)
            .expect("y edge not found in current edges");
        curr_t_prime_edges.remove(y_position);
    }

    return None;
}

fn step_2_with_back_tracking(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
    reduction_edges: &Option<HashSet<UndirectedEdge>>,
) -> Option<Vec<u64>> {
    // Step 2
    //println!("Step 2");
    let t_node_choices = (0..tsp_problem.num_cities).collect::<Vec<u64>>();
    let mut available_nodes: HashSet<u64> = HashSet::from_iter(t_node_choices.iter().cloned());
    let mut x_connections: Vec<UndirectedEdge> = vec![];
    let mut broken_connections: HashSet<UndirectedEdge> = HashSet::new();
    let mut t_nodes: Vec<u64> = vec![];
    let mut curr_t_prime_edges: Vec<UndirectedEdge> =
        convert_tour_into_undirected_edges(starting_path)
            .into_iter()
            .collect();
    //println!(
    //     "Original Edges {}: [{}]",
    //     curr_t_prime_edges.len(),
    //     curr_t_prime_edges
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    // get t-1
    let mut t_prime: Option<Vec<u64>> = None;
    for t_node in t_node_choices.iter() {
        //println!("Selected t-1: {}", t_node);
        t_nodes.push(*t_node);
        available_nodes.remove(&t_node);

        // select x-1 which implies t-2
        let possible_x_edges = get_edges_for_node(*t_node, starting_path);
        for x_edge in possible_x_edges {
            //println!("Testing {} as x1...", x_edge);
            x_connections.push(x_edge);
            let x_position = curr_t_prime_edges
                .iter()
                .position(|&p| p == x_edge)
                .expect("x edge not found in current edges");
            curr_t_prime_edges.remove(x_position);

            // This implies t2 push it to t_nodes
            let t2: u64 = match t_node {
                _ if *t_node == x_edge.city_a => x_edge.city_b,
                _ => x_edge.city_a,
            };
            //println!("This implies t2: {}", t2);
            broken_connections.insert(x_edge);
            t_nodes.push(t2);
            available_nodes.remove(&t2);
            //println!("Available nodes: {:?}", available_nodes);
            //println!("T nodes: {:?}", t_nodes);

            // Step 3 and beyond
            t_prime = step_3_with_back_tracking(
                tsp_problem,
                &t_nodes,
                &x_connections,
                &available_nodes,
                &broken_connections,
                reduction_edges,
                &curr_t_prime_edges,
            );

            if t_prime.is_some() {
                break;
            }

            // Undo our changes
            //println!("Backtracking to try new x1 edge...");
            let t2 = t_nodes
                .pop()
                .expect("We pushed t2 we should be able to pop.");
            available_nodes.insert(t2);
            x_connections.pop();
            broken_connections.remove(&x_edge);
            curr_t_prime_edges.push(x_edge);
        }

        if t_prime.is_some() {
            break;
        }

        // Undo our changes
        let t1 = t_nodes.pop().expect("We pushed t1 therefore we can pop 2.");
        available_nodes.insert(t1);
    }

    return t_prime;
}

fn run_steps_1_through_6(
    tsp_problem: &TSPProblem,
    reduction_edges: &Option<HashSet<UndirectedEdge>>,
    checkout_time_avoider: Arc<RwLock<HashSet<Vec<u64>>>>,
) -> (Vec<u64>, f32) {
    // Step 1
    let starting_solution = generate_pseudorandom_solution(tsp_problem);
    // let starting_solution = TSPSolution {
    //     algorithm_name: TSPAlgorithm::LinKernighan.to_string(),
    //     path: vec![2, 4, 1, 3, 0],
    //     tot_cost: 5.276084,
    //     optimal: false,
    //     calculation_time: 0.0,
    // };
    let mut curr_best_tour = starting_solution.path.clone();
    let mut curr_best_cost = starting_solution.tot_cost;
    //println!("Step 1: {:?} : {}", curr_best_tour, curr_best_cost);
    {
        if checkout_time_avoider
            .read()
            .expect("Could not acquire read lock on checkout time avoider.")
            .contains(&curr_best_tour)
        {
            //println!("Avoiding duplicate tour...");
            return (curr_best_tour, curr_best_cost);
        }
    }
    {
        checkout_time_avoider
            .write()
            .expect("Could not acquire write lock on checkout time avoider")
            .insert(curr_best_tour.clone());
    }

    // XXXXX backtrack to all possible y2 connections (TOP 5 Best y1 options)
    // choose alternative x2 and use special logic to convert t` into a valid tour
    // XXXXX try all y1 options starting at smallest to largest (TOP 5 Best y1 options)
    // XXXXX try the alternative x1
    // XXXXX try a different t1

    // if at any time during backtracking we find a positive gain we apply the changes and return
    // we only stop computing once (starting from step 1 with T`) we exhaust all backtracking options without gain

    // According to the paper we can limit backtracking to the first two choices for y2 and y1 and still maintain the majority of effectiveness
    // Keep a list of T` tours calculated if at any time we run into a previously discovered tour we stop and return

    // Reduction: run the algorithm a few times and pay attention to the common connections of the resulting local optima
    // pose a rule that these common connections cannot be broken (add them to joined_connections?)
    // if after a we trials and there are only 2 local optima we can return and assume no further improvement is likely
    // (The official paper mentions only enforcing the reduction on i > 4)

    // Lookahead: instead of choosing the smallest yi look at the 5 smallest yi and consider them relative to the xi connection that will be broken
    // for those 5 yi explore the yi with the greatest value of |xi+1| - |yi|

    // Gain is |xi| - |yi|
    // Check gain of closing loop before constructing y3...
    // y2... can be negative as long as the tot gain is still positive
    // keep the best tour along with G*
    // The reduction rule should be applied after x4 is chosen not t4

    while {
        //println!(
        //     "Going into step 2 with tour {}",
        //     curr_best_tour
        //         .iter()
        //         .map(|p| p.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );
        let t_prime_opt = step_2_with_back_tracking(&curr_best_tour, tsp_problem, reduction_edges);

        match t_prime_opt {
            Some(t_prime) => {
                //println!(
                //     "We found an improved T` tour: [{}]",
                //     t_prime
                //         .iter()
                //         .map(|p| p.to_string())
                //         .collect::<Vec<_>>()
                //         .join(", ")
                // );
                // Calc cost of tour
                let t_prime_cost =
                    calculate_cost_of_tour(&t_prime, &tsp_problem.city_connections_w_costs);
                //println!(
                //     "Cost of T`: {} which gives us a gain of {}",
                //     t_prime_cost,
                //     curr_best_cost - t_prime_cost
                // );
                // assert!(
                //     t_prime_cost <= curr_best_cost,
                //     "T` should never be worse than T"
                // );
                curr_best_cost = t_prime_cost;

                curr_best_tour = t_prime;

                let t_prime_already_registered;
                {
                    t_prime_already_registered = checkout_time_avoider
                        .read()
                        .expect("Could not get read lock for checkout time avoider.")
                        .contains(&curr_best_tour);
                }

                if t_prime_already_registered {
                    //println!("Avoiding duplicate tour...");
                    return (curr_best_tour, curr_best_cost);
                } else {
                    {
                        checkout_time_avoider
                            .write()
                            .expect("Could not get write lock for checkout time avoider")
                            .insert(curr_best_tour.clone());
                    }
                }
                true
            }
            None => {
                // Backtracking failed to find a better tour
                //println!(
                //     "No improvement found after backtracking. Returning best solution so far..."
                // );
                false
            }
        }
    } {}

    // Calc cost of final tour
    let tot_cost = calculate_cost_of_tour(&curr_best_tour, &tsp_problem.city_connections_w_costs);

    return (curr_best_tour, tot_cost);
}

pub fn calc_lin_kernighan_heuristic(
    tsp_problem: &TSPProblem,
    stopping_metric: u32,
) -> Option<TSPSolution> {
    if !tsp_problem.undirected_edges {
        return None;
    }

    // Get start time
    let start_time = Instant::now();

    let mut best_tour = vec![];
    let mut best_cost = f32::INFINITY;
    let mut local_optima = HashSet::new();
    let checkout_time_avoider: HashSet<Vec<u64>> = HashSet::new();
    let checkout_avoider_lock = Arc::new(RwLock::new(checkout_time_avoider));

    let pool = match ThreadPoolBuilder::new()
        .num_threads((NUM_SOLVES_BEFORE_REDUCTION * 2).try_into().unwrap())
        .build()
    {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("Error creating thread pool: {}", e);
            return None;
        }
    };

    // Explore run_steps_1_through_6 N times
    // if there are 4 or more unique local optima invoke the reduction rule and keep going another N times
    // else return the best local optima so far (it is likely there will be no significant improvement)
    //println!(
    //     "Running Lin-Kernighan heuristic on problem size {}...",
    //     tsp_problem.num_cities
    // );
    let initial_solve_results: Vec<(Vec<u64>, f32)> = pool.install(|| {
        (0..NUM_SOLVES_BEFORE_REDUCTION)
            .into_par_iter()
            .map(|_| run_steps_1_through_6(tsp_problem, &None, checkout_avoider_lock.clone()))
            .collect()
    });

    assert_eq!(
        initial_solve_results.len(),
        NUM_SOLVES_BEFORE_REDUCTION as usize
    );
    for (tour, cost) in initial_solve_results {
        if cost < best_cost {
            best_tour = tour.clone();
            best_cost = cost;
        }

        local_optima.insert(tour);
    }
    // println!("Best Cost After Initial Solve: {}", best_cost);
    // println!(
    //     "Best Current Tour After Initial Solve: [{}]",
    //     best_tour
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );
    // println!("Unique Local Optima: {}", local_optima.len());
    // for _ in 0..NUM_SOLVES_BEFORE_REDUCTION {
    //     let (tour, cost) = run_steps_1_through_6(tsp_problem, &None, &checkout_avoider_lock);
    //     if cost < best_cost {
    //         best_tour = tour.clone();
    //         best_cost = cost;
    //     }

    //     local_optima.insert(tour);
    // }
    //println!("Best Cost After Initial Solve: {}", best_cost);
    //println!(
    //     "Best Current Tour After Initial Solve: [{}]",
    //     best_tour
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    //println!(
    //     "Unique Local Optima: {}. Needed {} local optima in order to run reduction.",
    //     local_optima.len(),
    //     (NUM_SOLVES_BEFORE_REDUCTION as usize) / 2
    // );
    if local_optima.len() >= 4 {
        //println!("Running reduction rule...");
        //println!("Unique Local Optima: {:?}", local_optima);
        // Invoke the reduction rule
        // Create a list of edges used in most of the local optima
        let mut common_edges: HashMap<UndirectedEdge, u32> = HashMap::new();
        for tour in local_optima.iter() {
            let mut previous_city = 0;
            for city in tour {
                let edge = UndirectedEdge::new(previous_city, *city);
                let count = common_edges.entry(edge).or_insert(0);
                *count += 1;
                previous_city = *city;
            }
        }
        //println!("Common Edges: {:?}", common_edges);

        let reduction_cutoff = (local_optima.len() as f32 * 0.6) as u32;
        let reduction_edges: Option<HashSet<UndirectedEdge>> = Some(
            common_edges
                .iter()
                .filter(|(_, count)| **count >= reduction_cutoff)
                .map(|(edge, _)| edge.clone())
                .collect(),
        );
        //println!("Reduction Edges: {:?}", reduction_edges.clone().unwrap());

        // for _ in 0..(NUM_SOLVES_BEFORE_REDUCTION * 2) {
        //     let (tour, cost) =
        //         run_steps_1_through_6(tsp_problem, &reduction_edges, &checkout_avoider_lock);
        //     if cost < best_cost {
        //         best_tour = tour.clone();
        //         best_cost = cost;
        //     }

        //     local_optima.insert(tour);
        // }
        let reduction_solve_results: Vec<(Vec<u64>, f32)> = pool.install(|| {
            (0..NUM_SOLVES_BEFORE_REDUCTION * 2)
                .into_par_iter()
                .map(|_| {
                    run_steps_1_through_6(
                        tsp_problem,
                        &reduction_edges,
                        checkout_avoider_lock.clone(),
                    )
                })
                .collect()
        });
        for (tour, cost) in reduction_solve_results {
            if cost < best_cost {
                best_tour = tour.clone();
                best_cost = cost;
            }

            local_optima.insert(tour);
        }

        // println!("Best Cost After Reduction: {}", best_cost);
        // println!(
        //     "Best Current Tour After REduction: [{}]",
        //     best_tour
        //         .iter()
        //         .map(|p| p.to_string())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );
        // println!("Unique Local Optima: {}", local_optima.len());
    } else {
        // There were not many unique local optima so we can assume no further improvement is likely
    }

    //println!("Final Cost: {}", best_cost);
    //println!(
    //     "Final Tour: [{}]",
    //     best_tour
    //         .iter()
    //         .map(|p| p.to_string())
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    // Get end time
    let end_time = Instant::now();

    // Create the TSPSolution
    let tsp_solution = TSPSolution {
        algorithm_name: TSPAlgorithm::LinKernighan.to_string(),
        path: best_tour,
        tot_cost: best_cost,
        optimal: false,
        calculation_time: end_time.duration_since(start_time).as_secs_f32(),
    };

    return Some(tsp_solution);
}
