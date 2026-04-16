use std::collections::hash_map::Entry;

use bevy::math::Vec3;
use rustc_hash::{FxBuildHasher, FxHashMap as HashMap};

use crate::{
    marching_cubes::tables::{CORNER_OFFSETS, EDGE_ID_OFFSETS, EDGE_VERTICES, TRIANGLE_TABLE},
    terrain::chunk_generator::MaterialCode,
};

type EdgeKey = u64;

#[inline(always)]
fn make_edge_key(x: u16, y: u16, z: u16, dir: u8) -> u64 {
    (x as u64) | ((y as u64) << 16) | ((z as u64) << 32) | ((dir as u64) << 48)
}

pub fn mc_mesh_generation(
    densities: &[i16],
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    half_extent: f32,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>, Vec<u32>) {
    let mut edge_to_vertex: HashMap<EdgeKey, u32> = HashMap::with_hasher(FxBuildHasher::default());
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = (half_extent * 2.0) / (samples_per_chunk_dim - 1) as f32;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut material_ids = Vec::new();
    let mut indices = Vec::new();
    let stride = samples_per_chunk_dim * samples_per_chunk_dim;
    for z_idx in 0..cubes_per_chunk_dim {
        let cube_z_world_pos = -half_extent + z_idx as f32 * voxel_size;
        let z_base = z_idx * stride;
        for y_idx in 0..cubes_per_chunk_dim {
            let cube_y_world_pos = -half_extent + y_idx as f32 * voxel_size;
            let yz_base = z_base + y_idx * samples_per_chunk_dim;
            for x_idx in 0..cubes_per_chunk_dim {
                let cube_x_world_pos = -half_extent + x_idx as f32 * voxel_size;
                let cube_world_pos =
                    Vec3::new(cube_x_world_pos, cube_y_world_pos, cube_z_world_pos);
                let voxel_idx = yz_base + x_idx;
                let cube_corner_densities = sample_cube_corner_densities(
                    densities,
                    samples_per_chunk_dim,
                    stride,
                    voxel_idx,
                );
                let sdf_sign_mask = compute_sdf_sign_mask(&cube_corner_densities);
                if sdf_sign_mask == 0 || sdf_sign_mask == 255 {
                    continue;
                }
                let edge_table = &TRIANGLE_TABLE[sdf_sign_mask as usize];
                triangulate_cube_with_cache(
                    x_idx,
                    y_idx,
                    z_idx,
                    cube_world_pos,
                    voxel_idx,
                    &mut vertices,
                    &mut normals,
                    &mut material_ids,
                    &mut indices,
                    densities,
                    materials,
                    samples_per_chunk_dim,
                    voxel_size,
                    stride,
                    &mut edge_to_vertex,
                    &cube_corner_densities,
                    edge_table,
                );
            }
        }
    }
    for normal in normals.iter_mut() {
        let len = normal.length();
        if len > 0.0001 {
            *normal /= len;
        }
    }
    (vertices, normals, material_ids, indices)
}

#[inline(always)]
fn compute_sdf_sign_mask(values: &[f32; 8]) -> u8 {
    let mut mask = 0u8;
    for i in 0..8 {
        mask |= (((values[i].to_bits() >> 31) ^ 1) as u8) << i;
    }
    mask
}

fn triangulate_cube_with_cache(
    x_idx: usize,
    y_idx: usize,
    z_idx: usize,
    cube_world_pos: Vec3,
    voxel_idx: usize,
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    material_ids: &mut Vec<u32>,
    indices: &mut Vec<u32>,
    densities: &[i16],
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    voxel_size: f32,
    stride: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
    cube_corner_densities: &[f32; 8],
    edge_table: &[i32; 16],
) {
    let mut i = 0;
    while edge_table[i] != -1 {
        let edge_index = edge_table[i] as usize;
        let (dx, dy, dz, dir) = EDGE_ID_OFFSETS[edge_index];
        let edge_id = make_edge_key(x_idx as u16 + dx, y_idx as u16 + dy, z_idx as u16 + dz, dir);
        let v1 = get_or_create_edge_vertex(
            edge_index,
            cube_corner_densities,
            cube_world_pos,
            voxel_size,
            vertices,
            normals,
            material_ids,
            materials,
            samples_per_chunk_dim,
            stride,
            voxel_idx,
            edge_to_vertex,
            edge_id,
        );
        let edge_index = edge_table[i + 1] as usize;
        let (dx, dy, dz, dir) = EDGE_ID_OFFSETS[edge_index];
        let edge_id = make_edge_key(x_idx as u16 + dx, y_idx as u16 + dy, z_idx as u16 + dz, dir);
        let v2 = get_or_create_edge_vertex(
            edge_index,
            cube_corner_densities,
            cube_world_pos,
            voxel_size,
            vertices,
            normals,
            material_ids,
            materials,
            samples_per_chunk_dim,
            stride,
            voxel_idx,
            edge_to_vertex,
            edge_id,
        );
        let edge_index = edge_table[i + 2] as usize;
        let (dx, dy, dz, dir) = EDGE_ID_OFFSETS[edge_index];
        let edge_id = make_edge_key(x_idx as u16 + dx, y_idx as u16 + dy, z_idx as u16 + dz, dir);
        let v3 = get_or_create_edge_vertex(
            edge_index,
            cube_corner_densities,
            cube_world_pos,
            voxel_size,
            vertices,
            normals,
            material_ids,
            materials,
            samples_per_chunk_dim,
            stride,
            voxel_idx,
            edge_to_vertex,
            edge_id,
        );
        let i1 = v1 as usize;
        let i2 = v2 as usize;
        let i3 = v3 as usize;
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        let p3 = vertices[i3];
        let n = (p2 - p1).cross(p3 - p1);
        normals[i1] += n;
        normals[i2] += n;
        normals[i3] += n;
        let m1 = material_ids[i1];
        let m2 = material_ids[i2];
        let m3 = material_ids[i3];
        if m1 == m2 && m2 == m3 {
            indices.push(v1);
            indices.push(v2);
            indices.push(v3);
        } else {
            split_mixed_triangle(
                v1,
                v2,
                v3,
                m1,
                m2,
                m3,
                vertices,
                normals,
                material_ids,
                indices,
                materials,
                densities,
            );
        }
        i += 3;
    }
}

#[inline(always)]
fn get_or_create_edge_vertex(
    edge_index: usize,
    cube_corner_densities: &[f32; 8],
    cube_world_pos: Vec3,
    voxel_size: f32,
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    material_ids: &mut Vec<u32>,
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    stride: usize,
    voxel_idx: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
    edge_id: u64,
) -> u32 {
    match edge_to_vertex.entry(edge_id) {
        Entry::Occupied(e) => *e.get(),
        Entry::Vacant(e) => {
            let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
            let d1 = CORNER_OFFSETS[v1_idx];
            let d2 = CORNER_OFFSETS[v2_idx];
            let v1 = d1 * voxel_size + cube_world_pos;
            let v2 = d2 * voxel_size + cube_world_pos;
            let position =
                interpolate_edge_from_base(v1_idx, v2_idx, cube_corner_densities, v1, v2);
            let material1 = materials[voxel_idx
                + d1.z as usize * stride
                + d1.y as usize * samples_per_chunk_dim
                + d1.x as usize];
            let material2 = materials[voxel_idx
                + d2.z as usize * stride
                + d2.y as usize * samples_per_chunk_dim
                + d2.x as usize];
            let material = if material1 == MaterialCode::Grass || material2 == MaterialCode::Grass {
                MaterialCode::Grass
            } else if material1 != MaterialCode::Air {
                material1
            } else {
                material2
            };
            let idx = vertices.len() as u32;
            vertices.push(position);
            material_ids.push(material as u32);
            normals.push(Vec3::ZERO);
            e.insert(idx);
            idx
        }
    }
}

#[inline(always)]
fn interpolate_edge_from_base(
    v1_idx: usize,
    v2_idx: usize,
    cube_corner_densities: &[f32; 8],
    v1: Vec3,
    v2: Vec3,
) -> Vec3 {
    let val1 = cube_corner_densities[v1_idx];
    let val2 = cube_corner_densities[v2_idx];
    let denom = val2 - val1;
    let t = if denom.abs() < 0.0001 {
        0.5
    } else {
        (-val1 / denom).clamp(0.0, 1.0)
    };
    v1 + t * (v2 - v1)
}

#[inline(always)]
fn sample_cube_corner_densities(
    densities: &[i16],
    samples_per_chunk_dim: usize,
    stride: usize,
    voxel_idx: usize,
) -> [f32; 8] {
    [
        densities[voxel_idx] as f32,
        densities[voxel_idx + 1] as f32,
        densities[voxel_idx + samples_per_chunk_dim + 1] as f32,
        densities[voxel_idx + samples_per_chunk_dim] as f32,
        densities[voxel_idx + stride] as f32,
        densities[voxel_idx + stride + 1] as f32,
        densities[voxel_idx + stride + samples_per_chunk_dim + 1] as f32,
        densities[voxel_idx + stride + samples_per_chunk_dim] as f32,
    ]
}

fn interp(vertices: &[Vec3], normals: &[Vec3], a: usize, b: usize) -> (Vec3, Vec3) {
    let pos = (vertices[a] + vertices[b]) * 0.5;
    let raw_n = (normals[a] + normals[b]) * 0.5;
    let len = raw_n.length();
    let norm = if len > 0.0001 { raw_n / len } else { Vec3::Y };
    (pos, norm)
}

fn make_seam(
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    material_ids: &mut Vec<u32>,
    pos: Vec3,
    norm: Vec3,
    mat_near: u32,
    mat_far: u32,
) -> (u32, u32) {
    let near = vertices.len() as u32;
    vertices.push(pos);
    normals.push(norm);
    material_ids.push(mat_near);
    let far = vertices.len() as u32;
    vertices.push(pos);
    normals.push(norm);
    material_ids.push(mat_far);
    (near, far)
}

fn split_mixed_triangle(
    v1: u32,
    v2: u32,
    v3: u32,
    m1: u32,
    m2: u32,
    m3: u32,
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    material_ids: &mut Vec<u32>,
    indices: &mut Vec<u32>,
    materials: &[MaterialCode],
    densities: &[i16],
) {
    if m1 == m2 && m1 != m3 {
        let (p13, n13) = interp(vertices, normals, v1 as usize, v3 as usize);
        let (p23, n23) = interp(vertices, normals, v2 as usize, v3 as usize);
        let (s13_top, s13_bot) = make_seam(vertices, normals, material_ids, p13, n13, m1, m3);
        let (s23_top, s23_bot) = make_seam(vertices, normals, material_ids, p23, n23, m1, m3);
        indices.push(v1);
        indices.push(v2);
        indices.push(s23_top);
        indices.push(v1);
        indices.push(s23_top);
        indices.push(s13_top);
        indices.push(s13_bot);
        indices.push(s23_bot);
        indices.push(v3);
    } else if m1 == m3 && m1 != m2 {
        let (p12, n12) = interp(vertices, normals, v1 as usize, v2 as usize);
        let (p23, n23) = interp(vertices, normals, v2 as usize, v3 as usize);
        let (s12_top, s12_bot) = make_seam(vertices, normals, material_ids, p12, n12, m1, m2);
        let (s23_top, s23_bot) = make_seam(vertices, normals, material_ids, p23, n23, m1, m2);
        indices.push(v1);
        indices.push(s12_top);
        indices.push(s23_top);
        indices.push(v1);
        indices.push(s23_top);
        indices.push(v3);
        indices.push(s12_bot);
        indices.push(v2);
        indices.push(s23_bot);
    } else if m2 == m3 && m2 != m1 {
        let (p12, n12) = interp(vertices, normals, v1 as usize, v2 as usize);
        let (p13, n13) = interp(vertices, normals, v1 as usize, v3 as usize);
        let (s12_top, s12_bot) = make_seam(vertices, normals, material_ids, p12, n12, m1, m2);
        let (s13_top, s13_bot) = make_seam(vertices, normals, material_ids, p13, n13, m1, m3);
        indices.push(v1);
        indices.push(s12_top);
        indices.push(s13_top);
        indices.push(s12_bot);
        indices.push(v2);
        indices.push(v3);
        indices.push(s12_bot);
        indices.push(v3);
        indices.push(s13_bot);
    } else {
        indices.push(v1);
        indices.push(v2);
        indices.push(v3);
    }
}
