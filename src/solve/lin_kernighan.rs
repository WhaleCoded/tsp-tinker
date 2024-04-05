use std::{ collections::HashSet, time::Instant };

use ndarray::Array2;
use rand::prelude::IteratorRandom;

use crate::tsp_types::{ TSPProblem, TSPSolution, TSPAlgorithm };

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

fn get_edges_for_node(node: u64, tsp_tour: &Vec<u64>) -> ((u64, u64), (u64, u64)) {
    if node == 0 {
        return ((tsp_tour[tsp_tour.len() - 1], 0), (0, tsp_tour[0]));
    }

    let mut in_edge = (0, 0);
    let mut out_edge = (0, 0);
    for (i, city) in tsp_tour.iter().enumerate() {
        if *city == node {
            match i {
                0 => {
                    in_edge = (tsp_tour[tsp_tour.len() - 1], node);
                    out_edge = (node, tsp_tour[i + 1]);
                }
                _ => {
                    if i == tsp_tour.len() - 1 {
                        in_edge = (tsp_tour[i - 1], node);
                        out_edge = (node, tsp_tour[0]);
                    } else {
                        in_edge = (tsp_tour[i - 1], node);
                        out_edge = (node, tsp_tour[i + 1]);
                    }
                }
            }
        }
    }

    return (in_edge, out_edge);
}

fn choose_x_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &mut Vec<u64>,
    joined_edges: &HashSet<(u64, u64)>
) -> Option<(u64, u64)> {
    let t_node = t_nodes[t_nodes.len() - 1];
    let (in_edge, out_edge) = get_edges_for_node(t_node, tsp_tour);

    let mut x_edge = None;
    match (joined_edges.contains(&in_edge), joined_edges.contains(&out_edge)) {
        (true, true) => {
            return None;
        }
        (false, false) => {
            match rand::random::<bool>() {
                // x-i implies t-2i
                true => {
                    x_edge = Some(in_edge);
                    t_nodes.push(in_edge.0);
                }
                false => {
                    x_edge = Some(out_edge);
                    t_nodes.push(out_edge.1);
                }
            }
        }
        (true, false) => {
            x_edge = Some(out_edge);
            t_nodes.push(out_edge.1);
        }
        (false, true) => {
            x_edge = Some(in_edge);
            t_nodes.push(in_edge.0);
        }
    }

    return x_edge;
}

fn choose_y_edge(
    x_connections: &Vec<(u64, u64)>,
    connection_and_cost_matrix: &Array2<f32>,
    t_nodes: &mut Vec<u64>,
    available_nodes: &mut Vec<u64>,
    broken_connections: &HashSet<(u64, u64)>
) -> Option<((u64, u64), f32)> {
    let mut y_edge = None;

    let t_2i = t_nodes[t_nodes.len() - 1];
    let x_cost =
        connection_and_cost_matrix[[x_connections[0].0 as usize, x_connections[0].1 as usize]];
    let mut best_improvement = 0.0;

    for node in available_nodes.iter() {
        let y_edge_candidate = (t_2i, *node);
        if !broken_connections.contains(&y_edge_candidate) {
            let y_cost = connection_and_cost_matrix[[t_2i as usize, *node as usize]];
            let improvement = x_cost - y_cost;
            if improvement > best_improvement {
                best_improvement = improvement;
                y_edge = Some(y_edge_candidate);
            }
        }
    }

    match y_edge {
        Some(edge) => {
            available_nodes.retain(|&x| x != edge.1);
            t_nodes.push(edge.1);
            return Some((edge, best_improvement));
        }
        None => {}
    }

    return None;
}

fn choose_x_deterministic_edge(
    tsp_tour: &Vec<u64>,
    t_nodes: &mut Vec<u64>,
    available_nodes: &mut Vec<u64>,
    joined_edges: &HashSet<(u64, u64)>
) -> Option<(u64, u64)> {
    let t_2i_minus_1 = t_nodes[t_nodes.len() - 1];
    let mut t_2i = None;

    for (i, node) in tsp_tour.iter().enumerate() {
        if *node == t_2i_minus_1 {
            let mut prospective_t2i = tsp_tour[tsp_tour.len() - 1];
            if i != 0 {
                prospective_t2i = tsp_tour[i - 1];
            }

            if available_nodes.contains(&prospective_t2i) {
                t_2i = Some(prospective_t2i);
                available_nodes.retain(|&x| x != prospective_t2i);
                t_nodes.push(prospective_t2i);
                break;
            }
        }
    }

    match t_2i {
        Some(node) => {
            if !joined_edges.contains(&(node, t_2i_minus_1)) {
                return Some((node, t_2i_minus_1));
            }
        }
        None => {}
    }

    return None;
}

fn construct_t_prime(
    t_nodes: &Vec<u64>,
    x_connections: &Vec<(u64, u64)>,
    y_connections: &Vec<(u64, u64)>,
    starting_tour: &Vec<u64>
) -> Vec<u64> {
    let mut t_prime_edges: HashSet<(u64, u64)> = HashSet::new();

    // Convert T to edges
    for (i, node) in starting_tour.iter().enumerate() {
        if i == 0 {
            t_prime_edges.insert((0, *node));
        } else {
            t_prime_edges.insert((starting_tour[i - 1], *node));
        }
    }

    // Remove x-edges
    for x_edge in x_connections.iter() {
        let removed_x_edge = t_prime_edges.remove(x_edge);
        if !removed_x_edge {
            panic!("x_edge {:?} not found in T: {:?}", x_edge, t_prime_edges);
        }
    }

    // Add y-edges
    for (i, y_edge) in y_connections.iter().enumerate() {
        let add_result = t_prime_edges.insert(y_edge.clone());
        if !add_result {
            panic!("y_edge {:?} already in T: {:?}", y_edge, t_prime_edges);
        }

        if i == y_connections.len() - 1 {
            // add t2i-t1 connection
            let add_result = t_prime_edges.insert((t_nodes[t_nodes.len() - 1], t_nodes[0]));
            if !add_result {
                panic!("y_edge {:?} already in T: {:?}", y_edge, t_prime_edges);
            }
        }
    }

    // Convert edges back to T'
    let mut t_prime: Vec<u64> = vec![];
    let mut curr_node = 0;
    while t_prime.len() < starting_tour.len() {
        let next_node = t_prime_edges
            .iter()
            .find(|(from, _)| *from == curr_node)
            .unwrap().1;
        t_prime.push(next_node);
        curr_node = next_node;
    }

    return t_prime;
}

fn run_steps_2_through_5(
    starting_path: &Vec<u64>,
    tsp_problem: &TSPProblem,
    t_nodes: &mut Vec<u64>,
    x_connections: &mut Vec<(u64, u64)>,
    y_connections: &mut Vec<(u64, u64)>,
    available_nodes: &mut Vec<u64>,
    broken_connections: &mut HashSet<(u64, u64)>,
    joined_connections: &mut HashSet<(u64, u64)>
) -> Option<Vec<u64>> {
    // Step 2
    let mut best_improvement = 0.0;
    let mut tot_gain = 0.0;
    // get t-1
    let t_node = choose_new_t_node(available_nodes);
    t_nodes.push(t_node);
    // select x-1 which implies t-2
    let x_edge = choose_x_edge(&starting_path, t_nodes, &joined_connections).expect(
        "first x_edge should always be valid."
    );
    x_connections.push(x_edge);
    broken_connections.insert(x_edge);

    // Step 3
    // select y-1 implies t-3
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
            let x_cost =
                tsp_problem.city_connections_w_costs[[x_edge.0 as usize, x_edge.1 as usize]];
            let close_loop_cost =
                tsp_problem.city_connections_w_costs
                    [[t_nodes[0] as usize, t_nodes[t_nodes.len() - 1] as usize]];
            let close_loop_gain = x_cost - close_loop_cost;
            let gi_star = tot_gain + close_loop_gain;
            if gi_star > best_improvement {
                best_improvement = gi_star;
            } else {
                // Stopping criteria mentioned in step 5
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
                break;
            }
        } else {
            // 4-(e) xi+1 could not be broken remove the last yi and ti
            let last_y_edge = y_connections.pop().unwrap();
            joined_connections.remove(&last_y_edge);
            let last_t_node = t_nodes.pop().unwrap();
            available_nodes.push(last_t_node);
            break;
        }
    }

    // Step 5
    // if best_improvement is positive then apply the changes to form T`
    if best_improvement > 0.0 {
        let t_prime = construct_t_prime(&t_nodes, &x_connections, &y_connections, &starting_path);

        return Some(t_prime);
    }

    return None;
}

fn run_steps_1_through_6(tsp_problem: &TSPProblem) -> (Vec<u64>, f32) {
    // Step 1
    let starting_solution = generate_pseudorandom_solution(tsp_problem);
    let mut curr_best_tour = starting_solution.path.clone();

    let mut t_nodes: Vec<u64> = vec![];
    let mut x_connections: Vec<(u64, u64)> = vec![];
    let mut y_connections: Vec<(u64, u64)> = vec![];
    let mut broken_connections: HashSet<(u64, u64)> = HashSet::new();
    let mut joined_connections: HashSet<(u64, u64)> = HashSet::new();
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
                &mut t_nodes,
                &mut x_connections,
                &mut y_connections,
                &mut available_nodes,
                &mut broken_connections,
                &mut joined_connections
            );

            match t_prime_opt {
                Some(t_prime) => {
                    curr_best_tour = t_prime;
                    true
                }
                None => {
                    // back track technically step 6
                    false
                }
            }
        }
    ) {}

    // Calc cost of final tour
    let mut tot_cost = 0.0;
    for (i, city) in curr_best_tour.iter().enumerate() {
        if i == 0 {
            tot_cost += tsp_problem.city_connections_w_costs[[0, *city as usize]];
        } else {
            tot_cost +=
                tsp_problem.city_connections_w_costs
                    [[curr_best_tour[i - 1] as usize, *city as usize]];
        }
    }

    return (curr_best_tour, tot_cost);
}

pub fn calc_lin_kernighan_heuristic(tsp_problem: &TSPProblem, stopping_metric: u32) -> TSPSolution {
    // Get start time
    let start_time = Instant::now();

    let mut best_tour = vec![];
    let mut best_cost = f32::INFINITY;

    // Explore run_steps_1_through_6 N times
    // if there are 4 or more unique local optima invoke the reduction rule and keep going another N times
    // else return the best local optima so far (it is likely there will be no significant improvement)
    for _ in 0..stopping_metric {
        let (tour, cost) = run_steps_1_through_6(tsp_problem);
        if cost < best_cost {
            best_tour = tour;
            best_cost = cost;
        }
    }

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

    return tsp_solution;
}
