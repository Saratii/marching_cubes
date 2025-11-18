use std::collections::HashMap;

use bevy::prelude::Vec3;

use crate::marching_cubes::tables::{EDGE_VERTICES, TRIANGLE_TABLE};

pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
}

impl MeshBuffers {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: Vec::new(),
        }
    }
}

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
    uvs: Vec<[f32; 2]>,
}

impl VertexCache {
    fn new() -> Self {
        Self {
            edge_to_vertex: HashMap::new(),
            vertices: Vec::new(),
            uvs: Vec::new(),
        }
    }

    fn get_or_create_vertex(&mut self, edge_id: EdgeId, position: Vec3, material: u8) -> u32 {
        if let Some(&vertex_index) = self.edge_to_vertex.get(&edge_id) {
            vertex_index
        } else {
            let vertex_index = self.vertices.len() as u32;
            let uv = encode_material_to_uv(material);
            self.vertices.push(position);
            self.uvs.push(uv);
            self.edge_to_vertex.insert(edge_id, vertex_index);
            vertex_index
        }
    }
}

fn voxel_data_from_index(
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    corner: usize,
    materials: &[u8],
    samples_per_chunk_dim: usize,
) -> u8 {
    let (dx, dy, dz) = match corner {
        0 => (0, 0, 0),
        1 => (1, 0, 0),
        2 => (1, 1, 0),
        3 => (0, 1, 0),
        4 => (0, 0, 1),
        5 => (1, 0, 1),
        6 => (1, 1, 1),
        7 => (0, 1, 1),
        _ => (0, 0, 0),
    };
    let x = cube_x + dx;
    let y = cube_y + dy;
    let z = cube_z + dz;
    let idx = z * samples_per_chunk_dim * samples_per_chunk_dim + y * samples_per_chunk_dim + x;
    let material = materials[idx];
    material
}

pub fn mc_mesh_generation(
    mesh_buffers: &mut MeshBuffers,
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    half_extent: f32,
) {
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = (half_extent * 2.0) / (samples_per_chunk_dim - 1) as f32;
    let mut vertex_cache = VertexCache::new();
    let mut indices = Vec::new();
    for x in 0..cubes_per_chunk_dim {
        for y in 0..cubes_per_chunk_dim {
            for z in 0..cubes_per_chunk_dim {
                process_cube_with_cache(
                    x,
                    y,
                    z,
                    &mut vertex_cache,
                    &mut indices,
                    densities,
                    materials,
                    samples_per_chunk_dim,
                    half_extent,
                    voxel_size,
                );
            }
        }
    }
    build_mesh_buffers_from_cache_and_indices(
        mesh_buffers,
        vertex_cache,
        indices,
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    );
}

fn calculate_cube_index(values: &[f32; 8]) -> u8 {
    let mut cube_index = 0;
    for i in 0..8 {
        if values[i] > 0.0 {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_cube_vertices(x: usize, y: usize, z: usize, half_extent: f32, voxel_size: f32) -> [Vec3; 8] {
    let start_pos = Vec3::splat(-half_extent);
    let base_x = start_pos.x + x as f32 * voxel_size;
    let base_y = start_pos.y + y as f32 * voxel_size;
    let base_z = start_pos.z + z as f32 * voxel_size;
    [
        Vec3::new(base_x, base_y, base_z),
        Vec3::new(base_x + voxel_size, base_y, base_z),
        Vec3::new(base_x + voxel_size, base_y + voxel_size, base_z),
        Vec3::new(base_x, base_y + voxel_size, base_z),
        Vec3::new(base_x, base_y, base_z + voxel_size),
        Vec3::new(base_x + voxel_size, base_y, base_z + voxel_size),
        Vec3::new(
            base_x + voxel_size,
            base_y + voxel_size,
            base_z + voxel_size,
        ),
        Vec3::new(base_x, base_y + voxel_size, base_z + voxel_size),
    ]
}

fn process_cube_with_cache(
    x: usize,
    y: usize,
    z: usize,
    vertex_cache: &mut VertexCache,
    indices: &mut Vec<u32>,
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
) {
    let cube_vertices = get_cube_vertices(x, y, z, half_extent, voxel_size);
    let cube_values = sample_cube_values_from_sdf(x, y, z, densities, samples_per_chunk_dim);
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
        materials,
        samples_per_chunk_dim,
    );
    for triangle in triangles {
        indices.extend_from_slice(&triangle);
    }
}

fn get_edge_table_for_cube(cube_index: u8) -> &'static [i32] {
    &TRIANGLE_TABLE[cube_index as usize]
}

fn triangulate_cube_with_cache(
    cube_index: u8,
    vertices: &[Vec3; 8],
    values: &[f32; 8],
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    vertex_cache: &mut VertexCache,
    materials: &[u8],
    samples_per_chunk_dim: usize,
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
                materials,
                samples_per_chunk_dim,
            );
            let v2 = get_or_create_edge_vertex(
                edge_table[i + 1] as usize,
                vertices,
                values,
                cube_x,
                cube_y,
                cube_z,
                vertex_cache,
                materials,
                samples_per_chunk_dim,
            );
            let v3 = get_or_create_edge_vertex(
                edge_table[i + 2] as usize,
                vertices,
                values,
                cube_x,
                cube_y,
                cube_z,
                vertex_cache,
                materials,
                samples_per_chunk_dim,
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
    materials: &[u8],
    samples_per_chunk_dim: usize,
) -> u32 {
    let edge_id = get_canonical_edge_id(edge_index, cube_x, cube_y, cube_z);
    let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
    let position = interpolate_edge(edge_index, vertices, values);
    let material1 = voxel_data_from_index(
        cube_x,
        cube_y,
        cube_z,
        v1_idx,
        materials,
        samples_per_chunk_dim,
    );
    let material2 = voxel_data_from_index(
        cube_x,
        cube_y,
        cube_z,
        v2_idx,
        materials,
        samples_per_chunk_dim,
    );

    let material = if material1 == 2 || material2 == 2 {
        2
    } else if material1 != 0 {
        material1
    } else {
        material2
    };

    vertex_cache.get_or_create_vertex(edge_id, position, material)
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
    let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
    let v1 = vertices[v1_idx];
    let v2 = vertices[v2_idx];
    let val1 = values[v1_idx];
    let val2 = values[v2_idx];
    let t = if (val2 - val1).abs() < 0.0001 {
        0.5
    } else {
        ((0.0 - val1) / (val2 - val1)).clamp(0.0, 1.0)
    };
    v1 + t * (v2 - v1)
}

fn sample_cube_values_from_sdf(
    x: usize,
    y: usize,
    z: usize,
    densities: &[i16],
    samples_per_chunk_dim: usize,
) -> [f32; 8] {
    let get_sdf = |x: usize, y: usize, z: usize| -> f32 {
        let idx = z * samples_per_chunk_dim * samples_per_chunk_dim + y * samples_per_chunk_dim + x;
        densities[idx] as f32
    };
    [
        get_sdf(x, y, z),
        get_sdf(x + 1, y, z),
        get_sdf(x + 1, y + 1, z),
        get_sdf(x, y + 1, z),
        get_sdf(x, y, z + 1),
        get_sdf(x + 1, y, z + 1),
        get_sdf(x + 1, y + 1, z + 1),
        get_sdf(x, y + 1, z + 1),
    ]
}

fn calculate_vertex_normal(
    point: Vec3,
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
) -> Vec3 {
    let epsilon = voxel_size;
    let grad_x = sample_sdf_at_point_with_interpolation(
        point + Vec3::new(epsilon, 0.0, 0.0),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    ) - sample_sdf_at_point_with_interpolation(
        point - Vec3::new(epsilon, 0.0, 0.0),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    );
    let grad_y = sample_sdf_at_point_with_interpolation(
        point + Vec3::new(0.0, epsilon, 0.0),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    ) - sample_sdf_at_point_with_interpolation(
        point - Vec3::new(0.0, epsilon, 0.0),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    );
    let grad_z = sample_sdf_at_point_with_interpolation(
        point + Vec3::new(0.0, 0.0, epsilon),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    ) - sample_sdf_at_point_with_interpolation(
        point - Vec3::new(0.0, 0.0, epsilon),
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
    );
    Vec3::new(grad_x, grad_y, grad_z).normalize_or_zero()
}

fn sample_sdf_at_point_with_interpolation(
    point: Vec3,
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
) -> f32 {
    let voxel_x = (point.x + half_extent) / voxel_size;
    let voxel_y = (point.y + half_extent) / voxel_size;
    let voxel_z = (point.z + half_extent) / voxel_size;
    let voxel_x = voxel_x.clamp(0.0, (samples_per_chunk_dim - 1) as f32);
    let voxel_y = voxel_y.clamp(0.0, (samples_per_chunk_dim - 1) as f32);
    let voxel_z = voxel_z.clamp(0.0, (samples_per_chunk_dim - 1) as f32);
    let x0 = voxel_x.floor() as usize;
    let y0 = voxel_y.floor() as usize;
    let z0 = voxel_z.floor() as usize;
    let x1 = (x0 + 1).min(samples_per_chunk_dim - 1);
    let y1 = (y0 + 1).min(samples_per_chunk_dim - 1);
    let z1 = (z0 + 1).min(samples_per_chunk_dim - 1);
    let fx = voxel_x - x0 as f32;
    let fy = voxel_y - y0 as f32;
    let fz = voxel_z - z0 as f32;
    let get_sdf = |x: usize, y: usize, z: usize| -> f32 {
        let idx = z * samples_per_chunk_dim * samples_per_chunk_dim + y * samples_per_chunk_dim + x;
        densities[idx] as f32
    };
    let c000 = get_sdf(x0, y0, z0);
    let c100 = get_sdf(x1, y0, z0);
    let c010 = get_sdf(x0, y1, z0);
    let c110 = get_sdf(x1, y1, z0);
    let c001 = get_sdf(x0, y0, z1);
    let c101 = get_sdf(x1, y0, z1);
    let c011 = get_sdf(x0, y1, z1);
    let c111 = get_sdf(x1, y1, z1);
    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;
    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

fn build_mesh_buffers_from_cache_and_indices(
    mesh_buffers: &mut MeshBuffers,
    vertex_cache: VertexCache,
    indices: Vec<u32>,
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
) {
    if vertex_cache.vertices.is_empty() {
        return;
    }
    let positions: Vec<[f32; 3]> = vertex_cache
        .vertices
        .iter()
        .map(|v| [v.x, v.y, v.z])
        .collect();
    let normals: Vec<[f32; 3]> = vertex_cache
        .vertices
        .iter()
        .map(|v| {
            calculate_vertex_normal(
                *v,
                densities,
                samples_per_chunk_dim,
                half_extent,
                voxel_size,
            )
            .into()
        })
        .collect();
    mesh_buffers.positions = positions;
    mesh_buffers.normals = normals;
    mesh_buffers.indices = indices;
    mesh_buffers.uvs = vertex_cache.uvs;
}

fn encode_material_to_uv(material: u8) -> [f32; 2] {
    [material as f32, 0.0]
}