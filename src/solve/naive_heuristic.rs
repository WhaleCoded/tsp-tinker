use std::time::Instant;

use crate::types::{ TSPProblem, TSPSolution, TSPAlgorithm };

fn calc_naive_heuristic_for_single_city(
    priority_city: u64,
    tsp_problem: &TSPProblem
) -> (Vec<u64>, f32) {
    // The naive heuristic will choose the smallest connection available starting with the priority city
    // With each selection of the smallest connection, we limit future more optimal tour possibilities
    // This is a greedy algorithm that will not always find the optimal solution. We can however increase our chances
    // of finding a more efficient route by running the algorithm multiple times with different priority cities
    // The route then has to be reordered to start with the city 0
    let mut tour = vec![];
    let mut cost = 0.0;

    // Choose the smallest connection starting at the first city
    let mut current_city: u64 = priority_city;
    while tour.len() < (tsp_problem.num_cities as usize) {
        // get the row of the current city
        let row = &tsp_problem.city_connections_w_costs.row(current_city as usize);

        if tour.len() == (tsp_problem.num_cities as usize) - 1 {
            // add the first city to the end of the tour
            tour.push(priority_city);
            cost += row[priority_city as usize];
        } else {
            // find the smallest connection other than the current city
            let mut smallest_cost = f32::INFINITY;
            let mut smallest_city = 0;
            for (city, &cost) in row.iter().enumerate() {
                if
                    cost < smallest_cost &&
                    !tour.contains(&(city as u64)) &&
                    city != (current_city as usize) &&
                    city != (priority_city as usize)
                {
                    smallest_cost = cost;
                    smallest_city = city as u64;
                }
            }

            // add the smallest city to the tour
            tour.push(smallest_city);
            cost += smallest_cost;
            current_city = smallest_city;
        }
    }

    // Reorder the tour to start with the first city
    if priority_city != 0 {
        let mut ordered_tour = vec![];
        let mut start_index = 0;
        for (i, &city) in tour.iter().enumerate() {
            if city == 0 {
                start_index = i;
                break;
            }
        }

        // The starting city is assumed and thus only appears at the end of the tour signifying a return to the start
        for i in start_index + 1..tour.len() {
            ordered_tour.push(tour[i]);
        }
        for i in 0..start_index {
            ordered_tour.push(tour[i]);
        }
        ordered_tour.push(0);

        return (ordered_tour, cost);
    }

    return (tour, cost);
}

pub fn generate_naive_heuristic_solution(tsp_problem: &TSPProblem) -> TSPSolution {
    // Get start time
    let start_time = Instant::now();

    // Run the naive heuristic for each city
    let mut best_tour = vec![];
    let mut best_cost = f32::INFINITY;

    for city in 0..tsp_problem.num_cities {
        let (tour, cost) = calc_naive_heuristic_for_single_city(city, tsp_problem);
        if cost < best_cost {
            best_tour = tour;
            best_cost = cost;
        }
    }

    // Get end time
    let end_time = Instant::now();
    let calculation_time = end_time.duration_since(start_time).as_secs_f32();

    return TSPSolution {
        algorithm_name: TSPAlgorithm::NaiveHeuristic.to_string(),
        path: best_tour,
        tot_cost: best_cost,
        optimal: false,
        calculation_time: calculation_time,
    };
}
