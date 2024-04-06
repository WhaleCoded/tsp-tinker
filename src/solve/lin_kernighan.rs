use std::{ collections::HashSet, time::Instant };

use ndarray::Array2;
use rand::prelude::IteratorRandom;

use crate::types::{
    convert_undirected_edges_into_tour,
    convert_undirected_matrix_to_edges,
    TSPAlgorithm,
    TSPProblem,
    TSPSolution,
    UndirectedEdge,
};
use super::utils::calculate_cost_of_tour;

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

fn choose_new_t_node(available_nodes: &mut Vec<u64>) -> u64 {
    let node_index = (0..available_nodes.len()).choose(&mut rand::thread_rng()).unwrap();

    let node = available_nodes.swap_remove(node_index);

    return node;
}

fn get_edges_for_node(node: u64, tsp_tour: &Vec<u64>) -> (UndirectedEdge, UndirectedEdge) {
    if node == 0 {
        return (
            UndirectedEdge::new(tsp_tour[tsp_tour.len() - 2], 0),
            UndirectedEdge::new(0, tsp_tour[0]),
        );
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

    return (in_edge, out_edge);
}

fn choose_x_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &mut Vec<u64>,
    joined_edges: &HashSet<UndirectedEdge>
) -> Option<UndirectedEdge> {
    let t_node = t_nodes[t_nodes.len() - 1];
    let (in_edge, out_edge) = get_edges_for_node(t_node, tsp_tour);

    let mut x_edge = Some(in_edge);
    match t_node {
        _ if t_node == in_edge.city_a => {
            t_nodes.push(in_edge.city_b);
        }
        _ => {
            t_nodes.push(in_edge.city_a);
        }
    }
    // match (joined_edges.contains(&in_edge), joined_edges.contains(&out_edge)) {
    //     (true, true) => {
    //         return None;
    //     }
    //     (false, false) => {
    //         match rand::random::<bool>() {
    //             // x-i implies t-2i
    //             true => {
    //                 x_edge = Some(in_edge);
    //                 t_nodes.push(in_edge.0);
    //             }
    //             false => {
    //                 x_edge = Some(out_edge);
    //                 t_nodes.push(out_edge.1);
    //             }
    //         }
    //     }
    //     (true, false) => {
    //         x_edge = Some(out_edge);
    //         t_nodes.push(out_edge.1);
    //     }
    //     (false, true) => {
    //         x_edge = Some(in_edge);
    //         t_nodes.push(in_edge.0);
    //     }
    // }

    println!("Trying to determine x1 from possible edges: {} and {}...", in_edge, out_edge);
    println!("Selected x1: {}", x_edge.unwrap());
    println!("This implies t2: {}", t_nodes[t_nodes.len() - 1]);

    return x_edge;
}

fn choose_y_edge(
    x_connections: &Vec<UndirectedEdge>,
    connection_and_cost_matrix: &Array2<f32>,
    t_nodes: &mut Vec<u64>,
    available_nodes: &mut Vec<u64>,
    broken_connections: &HashSet<UndirectedEdge>
) -> Option<(UndirectedEdge, f32)> {
    let mut y_edge = None;

    let t_2i = t_nodes[t_nodes.len() - 1];
    let last_xi = x_connections[x_connections.len() - 1];
    let x_cost = connection_and_cost_matrix[[last_xi.city_a as usize, last_xi.city_b as usize]];
    let mut best_improvement = 0.0;
    let mut t2i_plus_1 = None;

    println!("Available nodes: {:?}", available_nodes);
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
            println!("Selected yi: {}", edge);
            available_nodes.retain(|&x| x != t2i_plus_1.unwrap());
            t_nodes.push(t2i_plus_1.unwrap());
            println!("This implies ti: {}", t_nodes[t_nodes.len() - 1]);
            return Some((edge, best_improvement));
        }
        None => {
            println!("No profitable y edge found.");
        }
    }

    return None;
}

fn choose_x_deterministic_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &mut Vec<u64>,
    available_nodes: &mut Vec<u64>,
    joined_edges: &HashSet<UndirectedEdge>
) -> Option<UndirectedEdge> {
    let t_2i_minus_1 = t_nodes[t_nodes.len() - 1];
    let mut t_2i = None;

    println!("Available nodes: {:?}", available_nodes);
    for (i, node) in tsp_tour.iter().enumerate() {
        if *node == t_2i_minus_1 {
            let mut prospective_t2i = tsp_tour[0];
            if i != tsp_tour.len() - 1 {
                prospective_t2i = tsp_tour[i + 1];
            }

            if available_nodes.contains(&prospective_t2i) {
                t_2i = Some(prospective_t2i);
                available_nodes.retain(|&x| x != prospective_t2i);
                println!("Selected t2i: {}", prospective_t2i);
                t_nodes.push(prospective_t2i);
                break;
            }

            println!("t2i not in available nodes. {}", prospective_t2i);
        }
    }

    match t_2i {
        Some(node) => {
            // flipped this to test deterministic xi
            let selected_x_edge = UndirectedEdge::new(node, t_2i_minus_1);
            if !joined_edges.contains(&selected_x_edge) {
                println!("Selected xi: {}", selected_x_edge);
                return Some(selected_x_edge);
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
    starting_tour: &Vec<u64>
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
            let add_result = t_prime_edges.insert(
                UndirectedEdge::new(t_nodes[t_nodes.len() - 1], t_nodes[0])
            );
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

fn run_steps_2_through_5(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
    all_undirected_edges: &Vec<UndirectedEdge>,
    t_nodes: &mut Vec<u64>,
    x_connections: &mut Vec<UndirectedEdge>,
    y_connections: &mut Vec<UndirectedEdge>,
    available_nodes: &mut Vec<u64>,
    broken_connections: &mut HashSet<UndirectedEdge>,
    joined_connections: &mut HashSet<UndirectedEdge>
) -> Option<Vec<u64>> {
    // Step 2
    println!("Step 2");
    let mut best_improvement = 0.0;
    let mut tot_gain = 0.0;
    // get t-1
    let t_node = choose_new_t_node(available_nodes);
    println!("Selected t-1: {}", t_node);
    t_nodes.push(t_node);
    // select x-1 which implies t-2
    let x_edge = choose_x_edge(&starting_path, t_nodes, &joined_connections).expect(
        "first x_edge should always be valid."
    );
    x_connections.push(x_edge);
    broken_connections.insert(x_edge);
    available_nodes.retain(|&x| x != t_nodes[t_nodes.len() - 1]);

    // Step 3
    // select y-1 implies t-3
    println!("Step 3");
    if
        let Some((y_edge, gain)) = choose_y_edge(
            &x_connections,
            &tsp_problem.city_connections_w_costs,
            t_nodes,
            available_nodes,
            &broken_connections
        )
    {
        y_connections.push(y_edge);
        joined_connections.insert(y_edge);
        tot_gain += gain;
    } else {
        // Go to Step 6(d)
        return None;
    }

    // Step 4
    println!("Step 4");
    // choose new x-i
    loop {
        if
            // implies t-2i
            let Some(x_edge) = choose_x_deterministic_edge(
                &starting_path,
                t_nodes,
                available_nodes,
                &joined_connections
            )
        {
            x_connections.push(x_edge);
            broken_connections.insert(x_edge);

            // check gain of closing the loop between t-2i and t1 compared to xi
            println!("Checking gain of closing the loop between t-2i and t1 compared to xi...");
            let x_cost =
                tsp_problem.city_connections_w_costs
                    [[x_edge.city_a as usize, x_edge.city_b as usize]];
            let close_loop_cost =
                tsp_problem.city_connections_w_costs
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
            if
                let Some((y_edge, gain)) = choose_y_edge(
                    &x_connections,
                    &tsp_problem.city_connections_w_costs,
                    t_nodes,
                    available_nodes,
                    &broken_connections
                )
            {
                y_connections.push(y_edge);
                joined_connections.insert(y_edge);
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
            available_nodes.push(last_t_node);
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

fn run_steps_1_through_6(
    tsp_problem: &TSPProblem,
    undirected_edges: &Vec<UndirectedEdge>
) -> (Vec<u64>, f32) {
    // Step 1
    let starting_solution = generate_pseudorandom_solution(tsp_problem);
    let mut curr_best_tour = starting_solution.path.clone();
    let mut curr_best_cost = starting_solution.tot_cost;
    println!("Step 1: {:?} : {}", curr_best_tour, curr_best_cost);

    let mut t_nodes: Vec<u64> = vec![];
    let mut x_connections: Vec<UndirectedEdge> = vec![];
    let mut y_connections: Vec<UndirectedEdge> = vec![];
    let mut broken_connections: HashSet<UndirectedEdge> = HashSet::new();
    let mut joined_connections: HashSet<UndirectedEdge> = HashSet::new();
    let mut available_nodes: Vec<u64> = (0..tsp_problem.num_cities).collect();

    // backtrack to all possible y2 connections
    // choose alternative x2 and use special logic to convert t` into a valid tour
    // try all y1 options starting at smallest to largest
    // try the alternative x1
    // try a different t1

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

    while (
        {
            let t_prime_opt = run_steps_2_through_5(
                &curr_best_tour,
                tsp_problem,
                undirected_edges,
                &mut t_nodes,
                &mut x_connections,
                &mut y_connections,
                &mut available_nodes,
                &mut broken_connections,
                &mut joined_connections
            );

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
                    let t_prime_cost = calculate_cost_of_tour(
                        &t_prime,
                        &tsp_problem.city_connections_w_costs
                    );
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
                    // back track technically step 6
                    println!("No improvement found. Backtracking...");
                    false
                }
            }
        }
    ) {}

    // Calc cost of final tour
    let tot_cost = calculate_cost_of_tour(&curr_best_tour, &tsp_problem.city_connections_w_costs);

    return (curr_best_tour, tot_cost);
}

pub fn calc_lin_kernighan_heuristic(
    tsp_problem: &TSPProblem,
    stopping_metric: u32
) -> Option<TSPSolution> {
    if !tsp_problem.undirected_edges {
        return None;
    }

    // Get start time
    let start_time = Instant::now();

    let mut best_tour = vec![];
    let mut best_cost = f32::INFINITY;

    let undirected_edges = convert_undirected_matrix_to_edges(&tsp_problem.num_cities);

    // Explore run_steps_1_through_6 N times
    // if there are 4 or more unique local optima invoke the reduction rule and keep going another N times
    // else return the best local optima so far (it is likely there will be no significant improvement)
    println!("Running Lin-Kernighan heuristic on problem size {}...", tsp_problem.num_cities);
    for _ in 0..1 {
        let (tour, cost) = run_steps_1_through_6(tsp_problem, &undirected_edges);
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
