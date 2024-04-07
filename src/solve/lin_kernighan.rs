use core::panic;
use std::{collections::HashSet, time::Instant};

use ndarray::Array2;
use rand::prelude::IteratorRandom;

use super::utils::calculate_cost_of_tour;
use crate::types::{
    convert_undirected_edges_into_tour, TSPAlgorithm, TSPProblem, TSPSolution, UndirectedEdge,
};

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

    println!("Available nodes: {:?}", available_nodes);
    println!("T nodes: {:?}", t_nodes);
    for node in available_nodes.iter() {
        let y_edge_candidate = UndirectedEdge::new(t_2i, *node);
        println!("Checking y edge candidate: {}", y_edge_candidate);
        if !broken_connections.contains(&y_edge_candidate) {
            let y_cost = connection_and_cost_matrix[[t_2i as usize, *node as usize]];
            println!("Cost of y edge candidate: {}", y_cost);
            let improvement = x_cost - y_cost;
            println!("Improvement: {}", improvement);
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
            println!("Selected yi: {}", edge);
            println!("This implies t2i + 1: {}", t2i_plus_1);
            return Some((edge, t2i_plus_1, best_improvement));
        }
        None => {
            println!("No profitable y edge found.");
        }
    }

    return None;
}

fn x_edge_is_in_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &Vec<u64>,
    x_edges: &Vec<UndirectedEdge>,
) -> bool {
    let last_xi = x_edges[x_edges.len() - 1];
    let t_2i = t_nodes[t_nodes.len() - 2];
    let t_2i_minus_1 = t_nodes[t_nodes.len() - 3];

    assert_eq!(UndirectedEdge::new(t_2i, t_2i_minus_1), last_xi);
    println!("Checking if ({}, {}) is an in-edge...", t_2i_minus_1, t_2i);

    if t_2i_minus_1 == 0 && t_2i == tsp_tour[0] {
        // This is the in-edge for 0
        return true;
    }
    for node in tsp_tour.iter() {
        if *node == t_2i_minus_1 {
            println!("Found t2i-1 first.");
            return true;
        } else if *node == t_2i {
            return false;
        }
    }

    panic!("t2i-1 and t2i not found in tsp_tour.");
}

fn choose_x_deterministic_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &Vec<u64>,
    available_nodes: &HashSet<u64>,
    joined_edges: &HashSet<UndirectedEdge>,
    x_edges: &Vec<UndirectedEdge>,
) -> Option<(UndirectedEdge, u64)> {
    let t_2i_minus_1 = t_nodes[t_nodes.len() - 1];
    let mut t_2i = None;
    let is_in_edge = x_edge_is_in_edge(tsp_tour, t_nodes, x_edges);
    println!("Is xi in edge: {}", is_in_edge);

    println!("Available nodes: {:?}", available_nodes);
    println!("T nodes: {:?}", t_nodes);
    for (i, node) in tsp_tour.iter().enumerate() {
        if *node == t_2i_minus_1 {
            let mut prospective_t2i = tsp_tour[0];
            if !is_in_edge {
                if i != tsp_tour.len() - 1 {
                    prospective_t2i = tsp_tour[i + 1];
                }
            } else {
                prospective_t2i = tsp_tour[tsp_tour.len() - 1];
                if i != 0 {
                    prospective_t2i = tsp_tour[i - 1];
                }
            }

            if available_nodes.contains(&prospective_t2i) {
                t_2i = Some(prospective_t2i);
                println!("Selected t2i: {}", prospective_t2i);
                break;
            }

            println!("t2i not in available nodes. {}", prospective_t2i);
        }
    }

    match t_2i {
        Some(node) => {
            let selected_x_edge = UndirectedEdge::new(node, t_2i_minus_1);
            if !joined_edges.contains(&selected_x_edge) {
                println!("Selected xi: {}", selected_x_edge);
                return Some((selected_x_edge, node));
            }
            println!("xi already in joined edges.");
        }
        None => {}
    }

    println!("No valid xi found.");
    return None;
}

fn construct_t_prime(
    t_nodes: &Vec<u64>,
    x_connections: &Vec<UndirectedEdge>,
    y_connections: &Vec<UndirectedEdge>,
    starting_tour: &Vec<u64>,
) -> Vec<u64> {
    let mut t_prime_edges: HashSet<UndirectedEdge> = HashSet::new();

    // Convert T to edges
    for (i, node) in starting_tour.iter().enumerate() {
        if i == 0 {
            t_prime_edges.insert(UndirectedEdge::new(0, *node));
        } else {
            t_prime_edges.insert(UndirectedEdge::new(starting_tour[i - 1], *node));
        }
    }
    println!(
        "Original Edges: [{}]",
        t_prime_edges
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Remove x-edges
    println!(
        "T` can't include x-edges: [{}]",
        x_connections
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    for x_edge in x_connections.iter() {
        let removed_x_edge = t_prime_edges.remove(x_edge);
        if !removed_x_edge {
            panic!(
                "x_edge {} not found in T: [{}]",
                x_edge,
                t_prime_edges
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    // Add y-edges
    println!(
        "T` will include y-edges: [{}] and closure edge {:?}",
        y_connections
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        (t_nodes[t_nodes.len() - 1], t_nodes[0])
    );
    for (i, y_edge) in y_connections.iter().enumerate() {
        let add_result = t_prime_edges.insert(y_edge.clone());
        println!("Added y_edge: {} with result {}", y_edge, add_result);
        if !add_result {
            panic!(
                "y_edge {} already in T: [{}]",
                y_edge,
                t_prime_edges
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        if i == y_connections.len() - 1 {
            // add t2i-t1 connection
            let add_result =
                t_prime_edges.insert(UndirectedEdge::new(t_nodes[t_nodes.len() - 1], t_nodes[0]));
            println!(
                "Added closure edge: {:?} with result {}",
                (t_nodes[t_nodes.len() - 1], t_nodes[0]),
                add_result
            );
            if !add_result {
                panic!(
                    "closure edge {:?} already in T: [{}]",
                    (t_nodes[t_nodes.len() - 1], t_nodes[0]),
                    t_prime_edges
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }
    println!(
        "Constructed T' edges: [{}]",
        t_prime_edges
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Convert edges back to T'
    let t_prime_edge_vec = t_prime_edges.into_iter().collect::<Vec<UndirectedEdge>>();
    let t_prime = convert_undirected_edges_into_tour(starting_tour.len() as u64, &t_prime_edge_vec);

    return t_prime;
}

fn step_4_with_back_tracking(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
    pre_selected_t_nodes: &Vec<u64>,
    pre_selected_x_connections: &Vec<UndirectedEdge>,
    pre_selected_y_connections: &Vec<UndirectedEdge>,
    pre_selected_available_nodes: &HashSet<u64>,
    pre_selected_broken_connections: &HashSet<UndirectedEdge>,
    pre_selected_joined_connections: &HashSet<UndirectedEdge>,
    pre_selected_tot_gain: &f32,
) -> Option<Vec<u64>> {
    let mut best_improvement = 0.0;
    let mut tot_gain = *pre_selected_tot_gain;
    let mut t_nodes = pre_selected_t_nodes.clone();
    let mut x_connections = pre_selected_x_connections.clone();
    let mut y_connections = pre_selected_y_connections.clone();
    let mut available_nodes = pre_selected_available_nodes.clone();
    let mut broken_connections = pre_selected_broken_connections.clone();
    let mut joined_connections = pre_selected_joined_connections.clone();

    // Step 4
    println!("Step 4");

    // choose new x-i
    loop {
        if
        // implies t-2i
        let Some((x_edge, t2i)) = choose_x_deterministic_edge(
            &starting_path,
            &t_nodes,
            &available_nodes,
            &joined_connections,
            &x_connections,
        ) {
            x_connections.push(x_edge);
            broken_connections.insert(x_edge);
            t_nodes.push(t2i);
            available_nodes.remove(&t2i);

            // check gain of closing the loop between t-2i and t1 compared to xi
            println!("Checking gain of closing the loop between t-2i and t1 compared to xi...");
            let x_cost = tsp_problem.city_connections_w_costs
                [[x_edge.city_a as usize, x_edge.city_b as usize]];
            let close_loop_cost = tsp_problem.city_connections_w_costs
                [[t_nodes[0] as usize, t_nodes[t_nodes.len() - 1] as usize]];
            let close_loop_gain = x_cost - close_loop_cost;
            let gi_star = tot_gain + close_loop_gain;
            println!("Gain of closing the loop: {}", close_loop_gain);
            println!("Total gain: {}", gi_star);
            println!("Best improvement so far: {}", best_improvement);
            if gi_star > best_improvement {
                best_improvement = gi_star;
                println!("New best improvement: {}", best_improvement);
            } else {
                // Stopping criteria mentioned in step 5
                println!("Stopping criteria met. Moving to step 5...");
                break;
            }

            // choose new y-i implies t-2i+1
            if let Some((y_edge, t2i_plus_1, gain)) = choose_y_edge(
                &x_connections,
                &tsp_problem.city_connections_w_costs,
                &t_nodes,
                &available_nodes,
                &broken_connections,
            ) {
                y_connections.push(y_edge);
                joined_connections.insert(y_edge);
                t_nodes.push(t2i_plus_1);
                available_nodes.remove(&t2i_plus_1);
                tot_gain += gain;

                // verify gain is positive else stop
            } else {
                // Go to Step 5
                println!("No profitable y edge found. Moving to step 5...");
                break;
            }
        } else {
            // 4-(e) xi+1 could not be broken remove the last yi and ti
            println!("xi+1 could not be broken. Removing the last yi and ti...");
            let last_y_edge = y_connections.pop().unwrap();
            println!("Removed yi: {}", last_y_edge);
            joined_connections.remove(&last_y_edge);
            let last_t_node = t_nodes.pop().unwrap();
            println!("Removed ti: {}", last_t_node);
            available_nodes.insert(last_t_node);
            println!(
                "y-edges: [{}]",
                y_connections
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "t-nodes: [{}]",
                t_nodes
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            break;
        }
    }

    // Step 5
    println!("Step 5");
    // if best_improvement is positive then apply the changes to form T`
    if best_improvement > 0.0 {
        println!("Positive gain found. Constructing T'...");
        let t_prime = construct_t_prime(&t_nodes, &x_connections, &y_connections, &starting_path);
        println!(
            "Constructed T': [{}]",
            t_prime
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        return Some(t_prime);
    }

    println!("No positive gain found. Backtracking...");
    return None;
}

fn step_3_with_back_tracking(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
    pre_selected_t_nodes: &Vec<u64>,
    pre_selected_x_connections: &Vec<UndirectedEdge>,
    pre_selected_available_nodes: &HashSet<u64>,
    pre_selected_broken_connections: &HashSet<UndirectedEdge>,
) -> Option<Vec<u64>> {
    let mut t_nodes = pre_selected_t_nodes.clone();
    let mut y_connections: Vec<UndirectedEdge> = vec![];
    let mut available_nodes = pre_selected_available_nodes.clone();
    let mut joined_connections: HashSet<UndirectedEdge> = HashSet::new();

    // Step 3
    println!("Step 3");
    let mut tot_gain = 0.0;

    // select y-1 implies t-3
    if let Some((y_edge, t_3, gain)) = choose_y_edge(
        &pre_selected_x_connections,
        &tsp_problem.city_connections_w_costs,
        &t_nodes,
        &available_nodes,
        &pre_selected_broken_connections,
    ) {
        y_connections.push(y_edge);
        t_nodes.push(t_3);
        available_nodes.remove(&t_3);
        joined_connections.insert(y_edge);
        tot_gain += gain;
    } else {
        // Go to Step 6(d)
        return None;
    }

    let t_prime = step_4_with_back_tracking(
        starting_path,
        tsp_problem,
        &t_nodes,
        &pre_selected_x_connections,
        &y_connections,
        &available_nodes,
        &pre_selected_broken_connections,
        &joined_connections,
        &tot_gain,
    );

    return t_prime;
}

fn step_2_with_back_tracking(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
) -> Option<Vec<u64>> {
    // Step 2
    println!("Step 2");
    let t_node_choices = (0..tsp_problem.num_cities).collect::<Vec<u64>>();
    let mut available_nodes: HashSet<u64> = HashSet::from_iter(t_node_choices.iter().cloned());
    let mut x_connections: Vec<UndirectedEdge> = vec![];
    let mut broken_connections: HashSet<UndirectedEdge> = HashSet::new();
    let mut t_nodes: Vec<u64> = vec![];

    // get t-1
    let mut t_prime: Option<Vec<u64>> = None;
    for t_node in t_node_choices.iter() {
        println!("Selected t-1: {}", t_node);
        t_nodes.push(*t_node);
        available_nodes.remove(&t_node);

        // select x-1 which implies t-2
        let possible_x_edges = get_edges_for_node(*t_node, starting_path);
        for x_edge in possible_x_edges {
            assert_eq!(t_nodes.len(), 1, "We should only have t-1 at this point.");
            assert_eq!(
                x_connections.len(),
                0,
                "We should not have any x_edges yet."
            );
            assert_eq!(
                broken_connections.len(),
                0,
                "We should not have any broken connections yet."
            );
            assert_eq!(
                available_nodes.len(),
                (tsp_problem.num_cities as usize) - 1,
                "All nodes should be available except t1."
            );

            println!("Testing {} as x1...", x_edge);
            x_connections.push(x_edge);

            // This implies t2 push it to t_nodes
            let t2: u64 = match t_node {
                _ if *t_node == x_edge.city_a => x_edge.city_b,
                _ => x_edge.city_a,
            };
            println!("This implies t2: {}", t2);
            broken_connections.insert(x_edge);
            t_nodes.push(t2);
            available_nodes.remove(&t2);
            println!("Available nodes: {:?}", available_nodes);
            println!("T nodes: {:?}", t_nodes);

            // Step 3 and beyond
            t_prime = step_3_with_back_tracking(
                starting_path,
                tsp_problem,
                &t_nodes,
                &x_connections,
                &available_nodes,
                &broken_connections,
            );

            if t_prime.is_some() {
                break;
            }

            // Undo our changes
            println!("Backtracking to try new x1 edge...");
            let t2 = t_nodes
                .pop()
                .expect("We pushed t2 we should be able to pop.");
            available_nodes.insert(t2);
            x_connections.pop();
            broken_connections.remove(&x_edge);
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

fn run_steps_1_through_6(tsp_problem: &TSPProblem) -> (Vec<u64>, f32) {
    // Step 1
    // let starting_solution = generate_pseudorandom_solution(tsp_problem);
    let starting_solution = TSPSolution {
        algorithm_name: TSPAlgorithm::Pseudorandom.to_string(),
        path: vec![3, 1, 4, 2, 0],
        tot_cost: 8.58086,
        optimal: false,
        calculation_time: 0.0,
    };
    let mut curr_best_tour = starting_solution.path.clone();
    let mut curr_best_cost = starting_solution.tot_cost;
    println!("Step 1: {:?} : {}", curr_best_tour, curr_best_cost);
    // backtrack to all possible y2 connections (TOP 5 Best y1 options)
    // choose alternative x2 and use special logic to convert t` into a valid tour
    // try all y1 options starting at smallest to largest (TOP 5 Best y1 options)
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

    while {
        let t_prime_opt = step_2_with_back_tracking(&curr_best_tour, tsp_problem);

        match t_prime_opt {
            Some(t_prime) => {
                println!(
                    "We found an improved T` tour: [{}]",
                    t_prime
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                // Calc cost of tour
                let t_prime_cost =
                    calculate_cost_of_tour(&t_prime, &tsp_problem.city_connections_w_costs);
                println!(
                    "Cost of T`: {} which gives us a gain of {}",
                    t_prime_cost,
                    curr_best_cost - t_prime_cost
                );
                curr_best_cost = t_prime_cost;

                curr_best_tour = t_prime;
                true
            }
            None => {
                // Backtracking failed to find a better tour
                println!(
                    "No improvement found after backtracking. Returning best solution so far..."
                );
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

    // Explore run_steps_1_through_6 N times
    // if there are 4 or more unique local optima invoke the reduction rule and keep going another N times
    // else return the best local optima so far (it is likely there will be no significant improvement)
    println!(
        "Running Lin-Kernighan heuristic on problem size {}...",
        tsp_problem.num_cities
    );
    for _ in 0..1 {
        let (tour, cost) = run_steps_1_through_6(tsp_problem);
        if cost < best_cost {
            best_tour = tour;
            best_cost = cost;
        }
    }
    println!("Final Cost: {}", best_cost);
    println!(
        "Final Tour: [{}]",
        best_tour
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

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
