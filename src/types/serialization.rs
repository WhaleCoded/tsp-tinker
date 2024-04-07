use std::fmt;

use ndarray::{s, Array1, Array2};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use super::TSPProblem;

impl Serialize for TSPProblem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TSPProblem", 2)?;
        state.serialize_field("num_cities", &self.num_cities)?;
        state.serialize_field("undirected", &self.undirected_edges)?;

        // If the edges are undirected, we only need to store the upper triangle of the matrix
        // To save space, we will store the upper triangle as a 1D array
        if self.undirected_edges {
            let mut upper_triangle_edges: Array1<f32> =
                Array1::zeros(((self.num_cities * (self.num_cities + 1)) / 2) as usize);
            let mut triangle_index = 0;
            for i in 0..self.num_cities as usize {
                // Transfer over in slices
                let slice = self.city_connections_w_costs.slice(s![i, i..]);
                upper_triangle_edges
                    .slice_mut(s![triangle_index..triangle_index + slice.len()])
                    .assign(&slice);
                triangle_index += slice.len();
            }

            state.serialize_field("upper_triangle_edges", &upper_triangle_edges.to_vec())?;
        } else {
            state.serialize_field("city_connections_w_costs", &self.city_connections_w_costs)?;
        }

        state.end()
    }
}

// Implementing Deserialize for TSPProblem
impl<'de> serde::Deserialize<'de> for TSPProblem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum Field {
            NumCities,
            UndirectedEdges,
            UpperTriangleEdges,
            CityConnectionsWCosts,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str(
                            "`num_cities`, `undirected`, `upper_triangle_edges`, or `city_connections_w_costs`"
                        )
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "num_cities" => Ok(Field::NumCities),
                            "undirected" => Ok(Field::UndirectedEdges),
                            "upper_triangle_edges" => Ok(Field::UpperTriangleEdges),
                            "city_connections_w_costs" => Ok(Field::CityConnectionsWCosts),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TSPProblemVisitor;

        impl<'de> serde::de::Visitor<'de> for TSPProblemVisitor {
            type Value = TSPProblem;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TSPProblem")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TSPProblem, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut num_cities = None;
                let mut undirected_edges = None;
                let mut upper_triangle_edges = None::<Vec<f32>>;
                let mut city_connections_w_costs = None::<Array2<f32>>;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::NumCities => {
                            if num_cities.is_some() {
                                return Err(serde::de::Error::duplicate_field("num_cities"));
                            }
                            num_cities = Some(map.next_value()?);
                        }
                        Field::UndirectedEdges => {
                            if undirected_edges.is_some() {
                                return Err(serde::de::Error::duplicate_field("undirected"));
                            }
                            undirected_edges = Some(map.next_value()?);
                        }
                        Field::UpperTriangleEdges => {
                            if upper_triangle_edges.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "upper_triangle_edges",
                                ));
                            }
                            upper_triangle_edges = Some(map.next_value()?);
                        }
                        Field::CityConnectionsWCosts => {
                            if city_connections_w_costs.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "city_connections_w_costs",
                                ));
                            }
                            city_connections_w_costs = Some(map.next_value()?);
                        }
                    }
                }

                let num_cities: u64 =
                    num_cities.ok_or_else(|| serde::de::Error::missing_field("num_cities"))?;
                let undirected_edges: bool = undirected_edges
                    .ok_or_else(|| serde::de::Error::missing_field("undirected"))?;

                // Construct the city_connections_w_costs Array2<f32> depending on whether we have undirected edges
                let city_connections_w_costs = if let Some(edges) = upper_triangle_edges {
                    let mut connections =
                        Array2::<f32>::zeros((num_cities as usize, num_cities as usize));
                    let mut edge_iter = edges.iter();
                    for i in 0..num_cities as usize {
                        for j in i..num_cities as usize {
                            if let Some(&cost) = edge_iter.next() {
                                connections[[i, j]] = cost;
                                connections[[j, i]] = cost; // Mirror for undirected graph
                            }
                        }
                    }
                    connections
                } else if let Some(connections) = city_connections_w_costs {
                    connections
                } else {
                    return Err(serde::de::Error::missing_field("city_connections_w_costs"));
                };

                Ok(TSPProblem {
                    num_cities,
                    undirected_edges,
                    city_connections_w_costs,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "num_cities",
            "undirected",
            "upper_triangle_edges",
            "city_connections_w_costs",
        ];
        deserializer.deserialize_struct("TSPProblem", FIELDS, TSPProblemVisitor)
    }
}
