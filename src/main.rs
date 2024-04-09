mod generate;
mod solve;
mod types;

use clap::{Arg, ArgAction, Command};
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

fn main() {
    // Data Path
    let data_path_arg = Arg::new("data-path")
        .short('d')
        .action(ArgAction::Set)
        .required(true)
        .value_parser(PathBuf::from_str)
        .help(
            "Path to the data folder. If you are generating data, this is where the data will be stored. If you are solving data, this is where the data is located."
        );
    // Generate TSP problems range of city size number of problems per size
    let generate_arg = Arg::new("generate")
        .short('g')
        .action(ArgAction::Set)
        .required(false)
        .num_args(2..=3)
        .value_delimiter(' ')
        .help(
            "Generate TSP problems. The first argument is the number of problems to generate for each problem size. The second argument is the problem size (number of cities) to generate. If a third argument is provided, we will generate TSP problems for a range of sizes where the second argument is starting size and the third argument is the final size."
        );
    // Solve TSP problems located in the data path specify algorithm
    let lin_kernighan_arg = Arg::new("lin-kernighan")
        .short('l')
        .action(ArgAction::SetTrue)
        .required(false)
        .help("Solve TSP problems using the Lin-Kernighan algorithm.");
    let b_n_b_arg = Arg::new("branch-n-bound")
        .short('b')
        .action(ArgAction::SetTrue)
        .required(false)
        .help("Solve TSP problems using the Branch and Bound algorithm.");
    // Override any data at the path specified
    let force_arg = Arg::new("force")
        .short('f')
        .action(ArgAction::SetTrue)
        .required(false)
        .help(
            "Force the re-solve of TSP problems even if they were already solved by the specified algorithm."
        );

    // Take all the arguments and create a command line interface
    let args = Command::new("tsp")
        .arg(data_path_arg)
        .arg(generate_arg)
        .arg(lin_kernighan_arg)
        .arg(b_n_b_arg)
        .arg(force_arg);

    // Parse the command line arguments
    let matches = args.get_matches();
    let force: bool = *matches.get_one("force").unwrap();

    // Get the data path
    let data_path = Path::new(matches.get_one::<PathBuf>("data-path").unwrap());
    if !data_path.exists() {
        println!("Data path does not exist. Creating data path.");
        if let Err(_) = fs::create_dir_all(data_path) {
            println!(
                "Failed to create the data folder {}. Please make sure your path is formatted correctly.",
                data_path.display()
            );
        }
    }

    // Check if we are generating data
    if let Some(gen_value_refs) = matches.get_many::<String>("generate") {
        let mut generate_value: Vec<u64> = vec![];

        for gen_value in gen_value_refs {
            match gen_value.parse::<u64>() {
                Ok(value) => generate_value.push(value),
                Err(_) => {
                    println!(
                        "Invalid argument for generate. Please provide a positive whole number."
                    );
                    return;
                }
            }
        }

        if generate_value.len() > 3 {
            println!(
                "Too many arguments for generate. Please provide 2 or 3 arguments. See help for more info."
            );
            return;
        } else if generate_value.len() < 2 {
            println!(
                "Not enough arguments for generate. Please provide 2 or 3 arguments. See help for more info."
            );
            return;
        }

        let num_problems_per_size = generate_value[0];
        let start_size = generate_value[1];
        let end_size = generate_value.get(2);

        let generation_result = match end_size {
            Some(end_size) => {
                if start_size > *end_size {
                    println!("Starting size must be less than or equal to the ending size.");
                    return;
                }

                println!(
                    "Generating {} TSP problems for each problem size: {}-{}",
                    num_problems_per_size, start_size, end_size
                );

                generate::generate_tsp_problems(
                    data_path,
                    num_problems_per_size,
                    start_size,
                    Some(*end_size),
                )
            }
            None => {
                println!(
                    "Generating {} TSP problems for the problem size {}",
                    num_problems_per_size, start_size
                );

                generate::generate_tsp_problems(data_path, num_problems_per_size, start_size, None)
            }
        };

        match generation_result {
            Ok(_) => println!("TSP problems generated successfully."),
            Err(e) => println!(
                "Failed to generate TSP problems because of an IO error: {}",
                e
            ),
        }
    }

    // Check if we are solving data
    match solve::solve_tsp(
        data_path,
        vec![
            (types::TSPAlgorithm::NaiveHeuristic, None),
            (types::TSPAlgorithm::LinKernighan, None),
            (types::TSPAlgorithm::BranchNBound, None),
        ],
        force,
    ) {
        Ok(_) => println!("TSP problems solved successfully."),
        Err(e) => println!("Failed to solve TSP problems because of an IO error: {}", e),
    }
}
