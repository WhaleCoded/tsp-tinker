use std::vec;

use ndarray::{Array1, Array2};
use rand::{self, Rng};

pub fn generate_city_coordinates(num_cities: u64, dimensionality: u64) -> Vec<Array1<f32>> {
    let mut city_coordinates: Vec<Array1<f32>> = vec![];
    let mut num_generator = rand::thread_rng();

    for _ in 0..num_cities {
        let mut city_coordinate: Array1<f32> = Array1::zeros(dimensionality as usize);

        for i in 0..dimensionality {
            city_coordinate[i as usize] = num_generator.gen_range(-1.0..1.0);
        }

        city_coordinates.push(city_coordinate);
    }

    return city_coordinates;
}

pub fn generate_euclidean_distance_matrix(city_coordinates: &Vec<Array1<f32>>) -> Array2<f32> {
    let num_cities = city_coordinates.len();
    let mut distance_matrix = Array2::zeros((num_cities, num_cities));

    for i in 0..num_cities {
        for j in i + 1..num_cities {
            let distance = (city_coordinates
                .get(i)
                .expect("A city index was out of range. This should never happen.")
                - city_coordinates
                    .get(j)
                    .expect("A city index was out of range. This should never happen."))
            .mapv(|x| x.powi(2))
            .sum()
            .sqrt();
            distance_matrix[[i, j]] = distance;
            distance_matrix[[j, i]] = distance;
        }
    }

    return distance_matrix;
}
