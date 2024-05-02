mod euclidean;
mod random;

use std::io::Error;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::types::{
    get_num_existing_tsp_problems_by_size, TSPGenerationMethod, TSPPackage, TSPProblem,
};
use euclidean::{generate_city_coordinates, generate_euclidean_distance_matrix};
use random::generate_random_cost_matrix;

const THREAD_POOL_SIZE: usize = 30;

fn generate_single_problem(
    problem_size: u64,
    data_path: &Path,
    gen_method: &TSPGenerationMethod,
    kill_sig_sent: Arc<AtomicBool>,
) -> Result<(), Error> {
    if kill_sig_sent.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(Error::new(
            std::io::ErrorKind::Other,
            "Received kill signal. Exiting...",
        ));
    }

    let (cost_matrix, undirected) = match gen_method {
        TSPGenerationMethod::Euclidean => {
            let city_coordinates = generate_city_coordinates(problem_size, 4);
            let distance_matrix = generate_euclidean_distance_matrix(&city_coordinates);
            (distance_matrix, true)
        }
        TSPGenerationMethod::RandomUndirected => {
            let cost_matrix = generate_random_cost_matrix(problem_size, true);
            (cost_matrix, true)
        }
        TSPGenerationMethod::RandomDirected => {
            let cost_matrix = generate_random_cost_matrix(problem_size, false);
            (cost_matrix, false)
        }
    };

    let tsp_problem = TSPProblem {
        num_cities: problem_size,
        city_connections_w_costs: cost_matrix,
        undirected_edges: undirected,
    };
    let tsp_package = TSPPackage::new(tsp_problem);

    tsp_package.store_as_json(data_path)?;

    return Ok(());
}

pub fn generate_tsp_problems(
    data_path: &Path,
    generation_method: TSPGenerationMethod,
    num_problems_per_size: u64,
    starting_problem_size: u64,
    ending_problem_size: Option<u64>,
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
        "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap();
    pb.set_style(sty);

    // Generate TSP problems using the euclidean distance method
    let pool = match ThreadPoolBuilder::new()
        .num_threads(THREAD_POOL_SIZE)
        .build()
    {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("Error creating thread pool: {}", e);
            return Err(Error::new(
                std::io::ErrorKind::Other,
                "Error creating thread pool",
            ));
        }
    };

    // Setup Ctrl-C handler to avoid corrupted files
    let kill_sig_sent = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let kill_sig_sent = kill_sig_sent.clone();
        move || {
            println!("Received kill signal. Gracefully shutting down...");
            kill_sig_sent.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    })
    .expect("Error setting Ctrl-C handler");

    for problem_size in problem_sizes {
        pb.set_message(format!(
            "Generating TSP problems for problem size: {}",
            problem_size
        ));

        // Calculate how many TSP problems we still need to generate
        let current_num_problems_generated =
            get_num_existing_tsp_problems_by_size(data_path, problem_size)?;

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
                match generate_single_problem(
                    problem_size,
                    data_path,
                    &generation_method,
                    kill_sig_sent.clone(),
                ) {
                    Ok(_) => pb.inc(1),
                    Err(e) => match e.kind() {
                        std::io::ErrorKind::Other => {
                            // Received kill signal
                        }
                        _ => {
                            eprintln!("Error generating TSP problem: {}", e);
                        }
                    },
                }
            });
        });

        if kill_sig_sent.load(std::sync::atomic::Ordering::Relaxed) {
            // Kill signal sent and graceful shutdown has already occurred
            println!("Graceful shutdown complete.");
            std::process::exit(0);
        }
    }

    // The ctrlc_handler is only needed for generation, so we reset it here
    ctrlc::set_handler(|| {
        println!("Received kill signal. Exiting...");
        std::process::exit(0);
    })
    .expect("Error re-setting Ctrl-C handler");

    return Ok(());
}
