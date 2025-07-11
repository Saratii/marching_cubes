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

pub fn march_cubes_for_chunk_into_mesh(terrain: &TerrainChunk) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    let mut rng = rand::rng();
    for x in 0..VOXELS_PER_DIM - 1 {
        for y in 0..VOXELS_PER_DIM - 1 {
            for z in 0..VOXELS_PER_DIM - 1 {
                let half_chunk = CHUNK_SIZE / 2.0;
                let pos = Vec3::new(
                    x as f32 * VOXEL_SIZE - half_chunk,
                    y as f32 * VOXEL_SIZE - half_chunk,
                    z as f32 * VOXEL_SIZE - half_chunk,
                );
                let cube_index = get_cube_index_density(terrain, x, y, z);
                let triangles = TRIANGLE_TABLE[cube_index as usize];
                let mut i = 0;
                while triangles[i] != -1 {
                    let triangle_color = [
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        1.0,
                    ];
                    let edge_indices = [
                        triangles[i] as usize,
                        triangles[i + 1] as usize,
                        triangles[i + 2] as usize,
                    ];
                    let mut triangle_vertices = Vec::new();
                    for &edge_idx in edge_indices.iter() {
                        let edge_vertex = get_edge_vertex_density(terrain, pos, edge_idx);
                        triangle_vertices.push(edge_vertex);
                    }
                    let base_vertex_index = vertices.len() as u32;
                    for vertex in triangle_vertices {
                        vertices.push([vertex.x, vertex.y, vertex.z]);
                        colors.push(triangle_color);
                    }
                    indices.extend_from_slice(&[
                        base_vertex_index,
                        base_vertex_index + 1,
                        base_vertex_index + 2,
                    ]);
                    i += 3;
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
    for triangle in indices.chunks(3) {
        let v0 = Vec3::from(vertices[triangle[0] as usize]);
        let v1 = Vec3::from(vertices[triangle[1] as usize]);
        let v2 = Vec3::from(vertices[triangle[2] as usize]);
        let cross = (v1 - v0).cross(v2 - v0);
        if cross.length_squared() > 0.000001 {
            let normal = cross.normalize();
            for &idx in triangle {
                let n = &mut normals[idx as usize];
                n[0] += normal.x;
                n[1] += normal.y;
                n[2] += normal.z;
            }
        }
    }
    for normal in &mut normals {
        let len_squared = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len_squared > 0.0 {
            let len = len_squared.sqrt();
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }
    let uvs: Vec<[f32; 2]> = vertices
        .iter()
        .map(|v| {
            [
                (v[0] + CHUNK_SIZE / 2.0) / CHUNK_SIZE,
                (v[2] + CHUNK_SIZE / 2.0) / CHUNK_SIZE,
            ]
        })
        .collect();
    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn get_cube_index_density(terrain: &TerrainChunk, x: usize, y: usize, z: usize) -> u8 {
    if z == 63 || y == 63 || x == 63 {
        println!("Cube index at ({}, {}, {}) is 0", x, y, z);
    }
    let mut cube_index = 0u8;
    let corners = [
        (x, y, z),
        (x + 1, y, z),
        (x + 1, y + 1, z),
        (x, y + 1, z),
        (x, y, z + 1),
        (x + 1, y, z + 1),
        (x + 1, y + 1, z + 1),
        (x, y + 1, z + 1),
    ];
    for (i, (cx, cy, cz)) in corners.iter().enumerate() {
        if terrain.get_density(*cx as i32, *cy as i32, *cz as i32) <= terrain.iso_level {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_edge_vertex_density(terrain: &TerrainChunk, pos: Vec3, edge_idx: usize) -> Vec3 {
    let edge_vertices = [
        (Vec3::ZERO, Vec3::new(VOXEL_SIZE, 0.0, 0.0)),
        (
            Vec3::new(VOXEL_SIZE, 0.0, 0.0),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
            Vec3::new(0.0, VOXEL_SIZE, 0.0),
        ),
        (Vec3::new(0.0, VOXEL_SIZE, 0.0), Vec3::ZERO),
        (
            Vec3::new(0.0, 0.0, VOXEL_SIZE),
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
            Vec3::new(0.0, 0.0, VOXEL_SIZE),
        ),
        (Vec3::ZERO, Vec3::new(0.0, 0.0, VOXEL_SIZE)),
        (
            Vec3::new(VOXEL_SIZE, 0.0, 0.0),
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(0.0, VOXEL_SIZE, 0.0),
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
        ),
    ];
    let (v1, v2) = edge_vertices[edge_idx];
    let p1 = pos + v1;
    let p2 = pos + v2;
    let voxel_coords1 = get_voxel_coords_from_local_pos(p1);
    let voxel_coords2 = get_voxel_coords_from_local_pos(p2);
    let density1 = terrain.get_density(voxel_coords1.0, voxel_coords1.1, voxel_coords1.2);
    let density2 = terrain.get_density(voxel_coords2.0, voxel_coords2.1, voxel_coords2.2);
    let iso_level = terrain.iso_level;
    let t = if (density2 - density1).abs() < 0.00001 {
        0.5
    } else {
        ((iso_level - density1) / (density2 - density1)).clamp(0.0, 1.0)
    };
    p1.lerp(p2, t)
}

fn get_voxel_coords_from_local_pos(local_pos: Vec3) -> (i32, i32, i32) {
    let half_chunk = CHUNK_SIZE / 2.0;
    let x = ((local_pos.x + half_chunk) / VOXEL_SIZE).floor() as i32;
    let y = ((local_pos.y + half_chunk) / VOXEL_SIZE).floor() as i32;
    let z = ((local_pos.z + half_chunk) / VOXEL_SIZE).floor() as i32;
    (
        x.min(VOXELS_PER_DIM as i32 - 1),
        y.min(VOXELS_PER_DIM as i32 - 1),
        z.min(VOXELS_PER_DIM as i32 - 1),
    )
}
