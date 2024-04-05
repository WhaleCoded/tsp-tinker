mod euclidean;

use std::path::Path;
use std::io::Error;

use indicatif::{ ProgressBar, ProgressStyle };
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use euclidean::{ generate_city_coordinates, generate_euclidean_distance_matrix };
use crate::tsp_types::{ TSPProblem, TSPPackage, get_num_existing_tsp_problems_by_size };

const THREAD_POOL_SIZE: usize = 30;

fn generate_single_problem(problem_size: u64, data_path: &Path) -> Result<(), Error> {
    let city_coordinates = generate_city_coordinates(problem_size, 4);
    let distance_matrix = generate_euclidean_distance_matrix(&city_coordinates);

    let tsp_problem = TSPProblem {
        num_cities: problem_size,
        city_connections_w_costs: distance_matrix,
    };
    let tsp_package = TSPPackage::new(tsp_problem);

    tsp_package.store_as_json(data_path)?;

    return Ok(());
}

pub fn generate_tsp_problems(
    data_path: &Path,
    num_problems_per_size: u64,
    starting_problem_size: u64,
    ending_problem_size: Option<u64>
) -> Result<(), Error> {
    let mut problem_sizes = vec![starting_problem_size];

    match ending_problem_size {
        Some(ending_problem_size) => {
            for i in starting_problem_size + 1..=ending_problem_size {
                problem_sizes.push(i);
            }
        }
        None => {}
    }

    // Setup progress bar
    let pb = ProgressBar::new((problem_sizes.len() as u64) * num_problems_per_size);

    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}"
    ).unwrap();
    pb.set_style(sty);

    // Generate TSP problems using the euclidean distance method
    let pool = match ThreadPoolBuilder::new().num_threads(THREAD_POOL_SIZE).build() {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("Error creating thread pool: {}", e);
            return Err(Error::new(std::io::ErrorKind::Other, "Error creating thread pool"));
        }
    };
    for problem_size in problem_sizes {
        pb.set_message(format!("Generating TSP problems for problem size: {}", problem_size));

        // Calculate how many TSP problems we still need to generate
        let current_num_problems_generated = get_num_existing_tsp_problems_by_size(
            data_path,
            problem_size
        )?;

        if current_num_problems_generated >= num_problems_per_size {
            pb.inc(num_problems_per_size);
            continue;
        } else {
            pb.inc(current_num_problems_generated);
        }

        // Concurrently generate the remaining TSP problems
        let num_left_to_generate = num_problems_per_size - current_num_problems_generated;
        pool.install(|| {
            (0..num_left_to_generate).into_par_iter().for_each(|_| {
                match generate_single_problem(problem_size, data_path) {
                    Ok(_) => pb.inc(1),
                    Err(e) => eprintln!("Error generating TSP problem: {}", e),
                }
            });
        });
    }

    return Ok(());
}
