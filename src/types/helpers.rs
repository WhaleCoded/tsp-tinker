use std::{ io::Error, path::{ Path, PathBuf } };
use crate::types::struct_functionality::create_json_sub_dir_name;

pub fn get_subdirectories_of_tsp_problems(data_path: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut sub_dirs: Vec<PathBuf> = vec![];

    for entry in std::fs::read_dir(data_path)? {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // Check if the directory matches the naming convention for TSP problems
            let dir_name = path
                .file_name()
                .expect("Directories should always have a name")
                .to_str()
                .expect("Directory names should always be valid UTF-8");

            if dir_name.starts_with("tsp_problems_of_") && dir_name.ends_with("_cities") {
                sub_dirs.push(path);
            }
        }
    }

    //sort the subdirectories by the number of cities smallest to largest
    sub_dirs.sort_by(|a, b| {
        let a_num_cities = a
            .file_name()
            .expect("Directories should always have a name")
            .to_str()
            .expect("Directory names should always be valid UTF-8")
            .split("_")
            .collect::<Vec<&str>>()[3]
            .parse::<u64>()
            .expect("Directory names should always be valid u64");
        let b_num_cities = b
            .file_name()
            .expect("Directories should always have a name")
            .to_str()
            .expect("Directory names should always be valid UTF-8")
            .split("_")
            .collect::<Vec<&str>>()[3]
            .parse::<u64>()
            .expect("Directory names should always be valid u64");

        a_num_cities.cmp(&b_num_cities)
    });

    return Ok(sub_dirs);
}

pub fn get_num_existing_tsp_problems_by_size(
    data_path: &Path,
    problem_size: u64
) -> Result<u64, Error> {
    let subdirectory_path = data_path.join(create_json_sub_dir_name(problem_size));
    match subdirectory_path.exists() {
        false => {
            return Ok(0);
        }
        true => {}
    }

    // Count the number of json files in the subdirectory
    let mut num_json_files = 0;
    for entry in std::fs::read_dir(subdirectory_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            num_json_files += 1;
        }
    }

    return Ok(num_json_files);
}

pub fn get_num_existing_tsp_problems_by_sub_dir(sub_dir_path: &Path) -> Result<u64, Error> {
    match sub_dir_path.exists() {
        false => {
            return Ok(0);
        }
        true => {}
    }

    // Count the number of json files in the subdirectory
    let mut num_json_files = 0;
    for entry in std::fs::read_dir(sub_dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            num_json_files += 1;
        }
    }

    return Ok(num_json_files);
}

pub fn get_tsp_problem_file_paths_by_sub_dir(sub_dir_path: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut json_file_paths: Vec<PathBuf> = vec![];

    for entry in std::fs::read_dir(sub_dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "json" {
            json_file_paths.push(path);
        }
    }

    return Ok(json_file_paths);
}
