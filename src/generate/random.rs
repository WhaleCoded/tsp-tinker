use ndarray::Array2;
use rand::Rng;

pub fn generate_random_cost_matrix(num_cities: u64, undirected_edges: bool) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let mut cost_matrix = Array2::<f32>::zeros((num_cities as usize, num_cities as usize));
    for i in 0..num_cities {
        let bottom_range = match undirected_edges {
            true => i + 1,
            false => 0,
        };

        cost_matrix[[i as usize, i as usize]] = 0.0;

        for j in bottom_range..num_cities {
            let cost: f32 = rng.gen_range(0.0..1.0);
            cost_matrix[[i as usize, j as usize]] = cost;

            if undirected_edges {
                cost_matrix[[j as usize, i as usize]] = cost;
            }
        }
    }
    cost_matrix
}
