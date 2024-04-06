mod lin_kernighan;
mod naive_heuristic;
mod utils;

use std::path::Path;
use std::io::Error;

use indicatif::{ MultiProgress, ProgressBar, ProgressStyle };

use crate::types::{
    get_subdirectories_of_tsp_problems,
    get_num_existing_tsp_problems_by_sub_dir,
    get_tsp_problem_file_paths_by_sub_dir,
    TSPPackage,
    TSPAlgorithm,
};

pub fn solve_tsp(
    data_path: &Path,
    selected_algorithms_and_timeout: Vec<(TSPAlgorithm, Option<u32>)>,
    force_resolve: bool
) -> Result<(), Error> {
    // Use the data path to find the TSP problems to solve
    // setup a progress to track our overall progress
    let sub_dir_paths = get_subdirectories_of_tsp_problems(data_path)?;
    let mut tot_problems_to_solve = 0;
    for sub_dir_path in &sub_dir_paths {
        tot_problems_to_solve += get_num_existing_tsp_problems_by_sub_dir(sub_dir_path)?;
    }

    let multi_pb = MultiProgress::new();
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}"
    ).unwrap();
    let overall_progress_pb = multi_pb.add(ProgressBar::new(tot_problems_to_solve));
    overall_progress_pb.set_style(sty.clone());
    overall_progress_pb.set_message("Total TSP Problems Solved");

    // Iterate over the subdirectories and solve the TSP problems
    for sub_dir_path in sub_dir_paths {
        overall_progress_pb.set_message(
            format!("Solving: {:?}", sub_dir_path.file_name().expect("No directory name found."))
        );
        // Grab the tsp problem file paths for this subdirectory
        let tsp_problem_file_paths = get_tsp_problem_file_paths_by_sub_dir(&sub_dir_path)?;
        for tsp_problem_file_path in tsp_problem_file_paths {
            let mut tsp_packaged_prob = match TSPPackage::from_json(&tsp_problem_file_path) {
                Ok(tsp_packaged_prob) => tsp_packaged_prob,
                Err(e) => {
                    eprintln!(
                        "Error reading TSP problem from file ({:?}): {} Skipping...",
                        tsp_problem_file_path,
                        e
                    );
                    overall_progress_pb.inc(1);
                    continue;
                }
            };

            // Solve the TSP problem
            for (algorithm, timeout) in &selected_algorithms_and_timeout {
                // Check if the problem has already been solved
                if tsp_packaged_prob.has_been_solved_by_algorithm(algorithm) {
                    if force_resolve {
                        // remove the previous solution
                        tsp_packaged_prob.remove_solution_by_algorithm(algorithm);
                    } else {
                        // skip this algorithm
                        continue;
                    }
                }

                match algorithm {
                    TSPAlgorithm::BranchNBound => {
                        // Solve the TSP problem using the Branch and Bound algorithm
                        // with the provided timeout
                    }
                    TSPAlgorithm::LinKernighan => {
                        // Solve the TSP problem using the Lin-Kernighan algorithm
                        // with the provided timeout
                        // will return None if the graph is directed
                        if
                            let Some(solution) = lin_kernighan::calc_lin_kernighan_heuristic(
                                &tsp_packaged_prob.problem_data,
                                10
                            )
                        {
                            tsp_packaged_prob.solutions.push(solution);
                        }
                    }
                    TSPAlgorithm::Pseudorandom => {
                        // Solve the TSP problem using the Pseudorandom algorithm
                        // with the provided timeout
                        let solution = lin_kernighan::generate_pseudorandom_solution(
                            &tsp_packaged_prob.problem_data
                        );
                        tsp_packaged_prob.solutions.push(solution);
                    }
                    TSPAlgorithm::NaiveHeuristic => {
                        // Solve the TSP problem using the Naive Heuristic algorithm
                        // with the provided timeout
                        let solution = naive_heuristic::generate_naive_heuristic_solution(
                            &tsp_packaged_prob.problem_data
                        );
                        tsp_packaged_prob.solutions.push(solution);
                    }
                }
            }

            // Save the updated TSP problem
            match tsp_packaged_prob.store_as_json(data_path) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "Error saving TSP problem to file ({:?}): {} Skipping...",
                        tsp_problem_file_path,
                        e
                    );
                }
            }

            // Update the overall progress bar
            overall_progress_pb.inc(1);
        }
    }

    return Ok(());
}
