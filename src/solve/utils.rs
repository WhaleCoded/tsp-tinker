use ndarray::Array2;

pub fn calculate_cost_of_tour(tour: &Vec<u64>, city_connections_w_costs: &Array2<f32>) -> f32 {
    let mut tot_cost = 0.0;
    for (i, city) in tour.iter().enumerate() {
        if i == 0 {
            tot_cost += city_connections_w_costs[[0, *city as usize]];
        } else {
            tot_cost += city_connections_w_costs[[tour[i - 1] as usize, *city as usize]];
        }
    }

    return tot_cost;
}
