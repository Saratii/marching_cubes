use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    math::Vec3,
    render::mesh::{Indices, Mesh, PrimitiveTopology},
};

use crate::{
    terrain_generation::{CHUNK_SIZE, VOXEL_SIZE, VOXELS_PER_DIM},
    triangle_table::TRIANGLE_TABLE,
};

const ISO_LEVEL: f32 = 0.0;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeId {
    x: usize,
    y: usize,
    z: usize,
    direction: u8,
}

struct VertexCache {
    edge_to_vertex: HashMap<EdgeId, u32>,
    vertices: Vec<Vec3>,
}

impl VertexCache {
    fn new() -> Self {
        Self {
            edge_to_vertex: HashMap::new(),
            vertices: Vec::new(),
        }
    }
    fn get_or_create_vertex(&mut self, edge_id: EdgeId, position: Vec3) -> u32 {
        if let Some(&vertex_index) = self.edge_to_vertex.get(&edge_id) {
            vertex_index
        } else {
            let vertex_index = self.vertices.len() as u32;
            self.vertices.push(position);
            self.edge_to_vertex.insert(edge_id, vertex_index);
            vertex_index
        }
    }
}

pub fn march_cubes(densities: &Vec<f32>) -> Mesh {
    let mut vertex_cache = VertexCache::new();
    let mut indices = Vec::new();
    for x in 0..VOXELS_PER_DIM - 1 {
        for y in 0..VOXELS_PER_DIM - 1 {
            for z in 0..VOXELS_PER_DIM - 1 {
                process_cube_with_cache(x, y, z, &mut vertex_cache, &mut indices, densities);
            }
        }
    }
    build_mesh_from_cache_and_indices(vertex_cache, indices, densities)
}

fn calculate_cube_index(values: &[f32; 8]) -> u8 {
    let mut cube_index = 0;
    for i in 0..8 {
        if values[i] > ISO_LEVEL {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_cube_vertices(x: usize, y: usize, z: usize) -> [Vec3; 8] {
    let chunk_coord = (0, 0, 0);
    let start_pos = Vec3::new(
        (chunk_coord.0 as f32 - 0.5) * CHUNK_SIZE,
        (chunk_coord.1 as f32 - 0.5) * CHUNK_SIZE,
        (chunk_coord.2 as f32 - 0.5) * CHUNK_SIZE,
    );
    let base_x = start_pos.x + x as f32 * VOXEL_SIZE;
    let base_y = start_pos.y + y as f32 * VOXEL_SIZE;
    let base_z = start_pos.z + z as f32 * VOXEL_SIZE;
    [
        Vec3::new(base_x, base_y, base_z),
        Vec3::new(base_x + VOXEL_SIZE, base_y, base_z),
        Vec3::new(base_x + VOXEL_SIZE, base_y + VOXEL_SIZE, base_z),
        Vec3::new(base_x, base_y + VOXEL_SIZE, base_z),
        Vec3::new(base_x, base_y, base_z + VOXEL_SIZE),
        Vec3::new(base_x + VOXEL_SIZE, base_y, base_z + VOXEL_SIZE),
        Vec3::new(
            base_x + VOXEL_SIZE,
            base_y + VOXEL_SIZE,
            base_z + VOXEL_SIZE,
        ),
        Vec3::new(base_x, base_y + VOXEL_SIZE, base_z + VOXEL_SIZE),
    ]
}

fn process_cube_with_cache(
    x: usize,
    y: usize,
    z: usize,
    vertex_cache: &mut VertexCache,
    indices: &mut Vec<u32>,
    densities: &[f32],
) {
    let cube_vertices = get_cube_vertices(x, y, z);
    let cube_values = sample_cube_values_from_densities(x, y, z, densities);
    let cube_index = calculate_cube_index(&cube_values);
    if cube_index == 0 || cube_index == 255 {
        return;
    }
    let triangles = triangulate_cube_with_cache(
        cube_index,
        &cube_vertices,
        &cube_values,
        x,
        y,
        z,
        vertex_cache,
    );
    for triangle in triangles {
        indices.extend(triangle);
    }
}

fn get_edge_table_for_cube(cube_index: u8) -> Vec<i32> {
    TRIANGLE_TABLE[cube_index as usize].to_vec()
}

fn triangulate_cube_with_cache(
    cube_index: u8,
    vertices: &[Vec3; 8],
    values: &[f32; 8],
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    vertex_cache: &mut VertexCache,
) -> Vec<[u32; 3]> {
    let edge_table = get_edge_table_for_cube(cube_index);
    let mut result = Vec::new();
    let mut i = 0;
    while i < edge_table.len() && edge_table[i] != -1 {
        if i + 2 < edge_table.len() && edge_table[i + 1] != -1 && edge_table[i + 2] != -1 {
            let v1 = get_or_create_edge_vertex(
                edge_table[i] as usize,
                vertices,
                values,
                cube_x,
                cube_y,
                cube_z,
                vertex_cache,
            );
            let v2 = get_or_create_edge_vertex(
                edge_table[i + 1] as usize,
                vertices,
                values,
                cube_x,
                cube_y,
                cube_z,
                vertex_cache,
            );
            let v3 = get_or_create_edge_vertex(
                edge_table[i + 2] as usize,
                vertices,
                values,
                cube_x,
                cube_y,
                cube_z,
                vertex_cache,
            );
            result.push([v1, v2, v3]);
            i += 3;
        } else {
            break;
        }
    }
    result
}

fn get_or_create_edge_vertex(
    edge_index: usize,
    vertices: &[Vec3; 8],
    values: &[f32; 8],
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    vertex_cache: &mut VertexCache,
) -> u32 {
    let edge_id = get_canonical_edge_id(edge_index, cube_x, cube_y, cube_z);
    let position = interpolate_edge(edge_index, vertices, values);
    vertex_cache.get_or_create_vertex(edge_id, position)
}

fn get_canonical_edge_id(edge_index: usize, cube_x: usize, cube_y: usize, cube_z: usize) -> EdgeId {
    match edge_index {
        0 => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z,
            direction: 0,
        },
        1 => EdgeId {
            x: cube_x + 1,
            y: cube_y,
            z: cube_z,
            direction: 1,
        },
        2 => EdgeId {
            x: cube_x,
            y: cube_y + 1,
            z: cube_z,
            direction: 0,
        },
        3 => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z,
            direction: 1,
        },
        4 => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z + 1,
            direction: 0,
        },
        5 => EdgeId {
            x: cube_x + 1,
            y: cube_y,
            z: cube_z + 1,
            direction: 1,
        },
        6 => EdgeId {
            x: cube_x,
            y: cube_y + 1,
            z: cube_z + 1,
            direction: 0,
        },
        7 => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z + 1,
            direction: 1,
        },
        8 => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z,
            direction: 2,
        },
        9 => EdgeId {
            x: cube_x + 1,
            y: cube_y,
            z: cube_z,
            direction: 2,
        },
        10 => EdgeId {
            x: cube_x + 1,
            y: cube_y + 1,
            z: cube_z,
            direction: 2,
        },
        11 => EdgeId {
            x: cube_x,
            y: cube_y + 1,
            z: cube_z,
            direction: 2,
        },
        _ => EdgeId {
            x: cube_x,
            y: cube_y,
            z: cube_z,
            direction: 0,
        },
    }
}

fn interpolate_edge(edge_index: usize, vertices: &[Vec3; 8], values: &[f32; 8]) -> Vec3 {
    let (v1_idx, v2_idx) = get_edge_vertex_indices(edge_index);
    let v1 = vertices[v1_idx];
    let v2 = vertices[v2_idx];
    let val1 = values[v1_idx];
    let val2 = values[v2_idx];
    let t = if (val2 - val1).abs() < 0.0001 {
        0.5
    } else {
        ((ISO_LEVEL - val1) / (val2 - val1)).clamp(0.0, 1.0)
    };
    v1 + t * (v2 - v1)
}

fn sample_cube_values_from_densities(x: usize, y: usize, z: usize, densities: &[f32]) -> [f32; 8] {
    let get_density = |dx: usize, dy: usize, dz: usize| -> f32 {
        let idx = (dz * VOXELS_PER_DIM * VOXELS_PER_DIM + dy * VOXELS_PER_DIM + dx)
            .min(densities.len() - 1);
        densities[idx]
    };
    [
        get_density(x, y, z),
        get_density(x + 1, y, z),
        get_density(x + 1, y + 1, z),
        get_density(x, y + 1, z),
        get_density(x, y, z + 1),
        get_density(x + 1, y, z + 1),
        get_density(x + 1, y + 1, z + 1),
        get_density(x, y + 1, z + 1),
    ]
}

fn calculate_vertex_normal(point: Vec3, densities: &[f32]) -> Vec3 {
    let epsilon = 0.001;
    let grad_x = sample_density_at_point(point + Vec3::new(epsilon, 0.0, 0.0), densities)
        - sample_density_at_point(point - Vec3::new(epsilon, 0.0, 0.0), densities);
    let grad_y = sample_density_at_point(point + Vec3::new(0.0, epsilon, 0.0), densities)
        - sample_density_at_point(point - Vec3::new(0.0, epsilon, 0.0), densities);
    let grad_z = sample_density_at_point(point + Vec3::new(0.0, 0.0, epsilon), densities)
        - sample_density_at_point(point - Vec3::new(0.0, 0.0, epsilon), densities);
    Vec3::new(-grad_x, -grad_y, -grad_z).normalize_or_zero()
}

fn sample_density_at_point(point: Vec3, densities: &[f32]) -> f32 {
    let voxel_x =
        ((point.x + HALF_CHUNK) / VOXEL_SIZE).clamp(0.0, (VOXELS_PER_DIM - 1) as f32) as usize;
    let voxel_y =
        ((point.y + HALF_CHUNK) / VOXEL_SIZE).clamp(0.0, (VOXELS_PER_DIM - 1) as f32) as usize;
    let voxel_z =
        ((point.z + HALF_CHUNK) / VOXEL_SIZE).clamp(0.0, (VOXELS_PER_DIM - 1) as f32) as usize;
    let idx = (voxel_z * VOXELS_PER_DIM * VOXELS_PER_DIM + voxel_y * VOXELS_PER_DIM + voxel_x)
        .min(densities.len() - 1);
    densities[idx]
}

fn build_mesh_from_cache_and_indices(
    vertex_cache: VertexCache,
    indices: Vec<u32>,
    densities: &[f32],
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    if vertex_cache.vertices.is_empty() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        mesh.insert_indices(Indices::U32(Vec::new()));
    } else {
        let positions: Vec<[f32; 3]> = vertex_cache
            .vertices
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect();
        let normals: Vec<[f32; 3]> = vertex_cache
            .vertices
            .iter()
            .map(|v| calculate_vertex_normal(*v, densities).into())
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_indices(Indices::U32(indices));
    }
    mesh
}

fn get_edge_vertex_indices(edge_index: usize) -> (usize, usize) {
    match edge_index {
        0 => (0, 1),
        1 => (1, 2),
        2 => (2, 3),
        3 => (3, 0),
        4 => (4, 5),
        5 => (5, 6),
        6 => (6, 7),
        7 => (7, 4),
        8 => (0, 4),
        9 => (1, 5),
        10 => (2, 6),
        11 => (3, 7),
        _ => (0, 0),
    }
}

pub fn add_triangle_colors(mut mesh: Mesh) -> Mesh {
    if let Some(positions_attr) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        if let Some(indices) = mesh.indices() {
            let positions: Vec<[f32; 3]> = positions_attr.as_float3().unwrap().to_vec();
            let mut colors = vec![[1.0, 1.0, 1.0, 1.0]; positions.len()];
            match indices {
                Indices::U32(idx) => {
                    for (tri_index, chunk) in idx.chunks(3).enumerate() {
                        if chunk.len() == 3 {
                            let t = (tri_index as f32 * 0.618034) % 1.0;
                            let color = (
                                (t * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                                ((t + 0.33) * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                                ((t + 0.66) * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                            );
                            for &vertex_idx in chunk {
                                colors[vertex_idx as usize] = [color.0, color.1, color.2, 1.0];
                            }
                        }
                    }
                }
                Indices::U16(idx) => {
                    for (tri_index, chunk) in idx.chunks(3).enumerate() {
                        if chunk.len() == 3 {
                            let t = (tri_index as f32 * 0.618034) % 1.0;
                            let color = (
                                (t * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                                ((t + 0.33) * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                                ((t + 0.66) * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5,
                            );
                            for &vertex_idx in chunk {
                                colors[vertex_idx as usize] = [color.0, color.1, color.2, 1.0];
                            }
                        }
                    }
                }
            }
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
    }
    mesh
}
