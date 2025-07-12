use bevy::{
    math::Vec3,
    render::mesh::{Indices, Mesh, PrimitiveTopology},
    utils::default,
};
use rand::Rng;

use crate::{
    terrain_generation::{CHUNK_SIZE, TerrainChunk, VOXEL_SIZE, VOXELS_PER_DIM},
    triangle_table::TRIANGLE_TABLE,
};

pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;

const EDGE_TO_CORNERS: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

const CORNER_OFFSETS: [Vec3; 8] = [
    Vec3::ZERO,
    Vec3::new(VOXEL_SIZE, 0.0, 0.0),
    Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
    Vec3::new(0.0, VOXEL_SIZE, 0.0),
    Vec3::new(0.0, 0.0, VOXEL_SIZE),
    Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
    Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
    Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
];

pub fn march_cubes_for_chunk_into_mesh(terrain: &TerrainChunk) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    let mut rng = rand::rng();
    for z in 0..VOXELS_PER_DIM - 1 {
        let zc = z as f32 * VOXEL_SIZE - HALF_CHUNK;
        for y in 0..VOXELS_PER_DIM - 1 {
            let yc = y as f32 * VOXEL_SIZE - HALF_CHUNK;
            for x in 0..VOXELS_PER_DIM - 1 {
                let pos = Vec3::new(x as f32 * VOXEL_SIZE - HALF_CHUNK, yc, zc);
                let (cube_index, corner_densities) = get_cube_index_and_densities(terrain, x, y, z);
                if cube_index == 0 || cube_index == 255 {
                    continue;
                }
                let triangles = TRIANGLE_TABLE[cube_index as usize];
                for triangle in triangles.chunks_exact(3) {
                    let color = [
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        1.0,
                    ];
                    if triangle[0] == -1 {
                        break;
                    }
                    let edge_indices = [
                        triangle[0] as usize,
                        triangle[1] as usize,
                        triangle[2] as usize,
                    ];
                    let triangle_vertices = [
                        get_edge_vertex_density_cached(
                            &corner_densities,
                            pos,
                            edge_indices[0],
                            terrain.iso_level,
                        ),
                        get_edge_vertex_density_cached(
                            &corner_densities,
                            pos,
                            edge_indices[1],
                            terrain.iso_level,
                        ),
                        get_edge_vertex_density_cached(
                            &corner_densities,
                            pos,
                            edge_indices[2],
                            terrain.iso_level,
                        ),
                    ];

                    let base_vertex_index = vertices.len() as u32;
                    for vertex in triangle_vertices {
                        vertices.push([vertex.x, vertex.y, vertex.z]);
                        colors.push(color);
                    }
                    indices.push(base_vertex_index);
                    indices.push(base_vertex_index + 1);
                    indices.push(base_vertex_index + 2);
                }
            }
        }
    }
    if vertices.is_empty() {
        let mut empty_mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new());
        return empty_mesh;
    }
    let mut normals = vec![[0.0; 3]; vertices.len()];
    indices.chunks_exact(3).for_each(|triangle| {
        let [v0, v1, v2] = [
            Vec3::from(vertices[triangle[0] as usize]),
            Vec3::from(vertices[triangle[1] as usize]),
            Vec3::from(vertices[triangle[2] as usize]),
        ];
        let normal = (v1 - v0).cross(v2 - v0);
        if normal.length_squared() > 1e-6 {
            let normal = normal.normalize();
            triangle.iter().for_each(|&idx| {
                let n = &mut normals[idx as usize];
                n[0] += normal.x;
                n[1] += normal.y;
                n[2] += normal.z;
            });
        }
    });
    for normal in &mut normals {
        let len_squared = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len_squared > 0.0 {
            let len = len_squared.sqrt();
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }
    let mut uvs = Vec::with_capacity(vertices.len());
    for vertex in &vertices {
        uvs.push([
            (vertex[0] + HALF_CHUNK) / CHUNK_SIZE,
            (vertex[2] + HALF_CHUNK) / CHUNK_SIZE,
        ]);
    }
    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn get_cube_index_and_densities(
    terrain: &TerrainChunk,
    x: usize,
    y: usize,
    z: usize,
) -> (u8, [f32; 8]) {
    let corner_coords = [
        (x, y, z),
        (x + 1, y, z),
        (x + 1, y + 1, z),
        (x, y + 1, z),
        (x, y, z + 1),
        (x + 1, y, z + 1),
        (x + 1, y + 1, z + 1),
        (x, y + 1, z + 1),
    ];
    let mut cube_index = 0u8;
    let mut densities = [0.0; 8];
    for (i, (cx, cy, cz)) in corner_coords.iter().enumerate() {
        let density = terrain.get_density(*cx as i32, *cy as i32, *cz as i32);
        densities[i] = density;
        if density <= terrain.iso_level {
            cube_index |= 1 << i;
        }
    }
    (cube_index, densities)
}

fn get_edge_vertex_density_cached(
    corner_densities: &[f32; 8],
    pos: Vec3,
    edge_idx: usize,
    iso_level: f32,
) -> Vec3 {
    let (c1, c2) = EDGE_TO_CORNERS[edge_idx];
    let p1 = pos + CORNER_OFFSETS[c1];
    let p2 = pos + CORNER_OFFSETS[c2];
    let density1 = corner_densities[c1];
    let density2 = corner_densities[c2];
    let t = if (density2 - density1).abs() < 0.00001 {
        0.5
    } else {
        ((iso_level - density1) / (density2 - density1)).clamp(0.0, 1.0)
    };
    p1.lerp(p2, t)
}
