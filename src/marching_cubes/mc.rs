use std::collections::hash_map::Entry;

use rustc_hash::{FxBuildHasher, FxHashMap as HashMap};

use crate::marching_cubes::tables::{
    CORNER_OFFSETS, EDGE_ID_OFFSETS, EDGE_VERTICES, TRIANGLE_TABLE,
};

type EdgeKey = u64;

#[inline(always)]
fn make_edge_key(x: u16, y: u16, z: u16, dir: u8) -> u64 {
    (x as u64) | ((y as u64) << 16) | ((z as u64) << 32) | ((dir as u64) << 48)
}

pub fn mc_mesh_generation(
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    half_extent: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
    let mut edge_to_vertex: HashMap<EdgeKey, u32> = HashMap::with_hasher(FxBuildHasher::default());
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = (half_extent * 2.0) / (samples_per_chunk_dim - 1) as f32;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let stride = samples_per_chunk_dim * samples_per_chunk_dim;
    for z in 0..cubes_per_chunk_dim {
        let base_z = -half_extent + z as f32 * voxel_size;
        let z_base = z * stride;
        for y in 0..cubes_per_chunk_dim {
            let base_y = -half_extent + y as f32 * voxel_size;
            let yz_base = z_base + y * samples_per_chunk_dim;
            for x in 0..cubes_per_chunk_dim {
                let base_x = -half_extent + x as f32 * voxel_size;
                let base = yz_base + x;
                process_cube_with_cache(
                    x,
                    y,
                    z,
                    base_x,
                    base_y,
                    base_z,
                    base,
                    &mut vertices,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                    densities,
                    materials,
                    samples_per_chunk_dim,
                    voxel_size,
                    stride,
                    &mut edge_to_vertex,
                );
            }
        }
    }
    (vertices, normals, uvs, indices)
}

#[inline(always)]
fn calculate_cube_index(values: &[f32; 8]) -> u8 {
    let mut mask = 0u8;
    for i in 0..8 {
        mask |= (((values[i].to_bits() >> 31) ^ 1) as u8) << i;
    }
    mask
}

fn process_cube_with_cache(
    x: usize,
    y: usize,
    z: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    base: usize,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    voxel_size: f32,
    stride: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
) {
    let cube_values = sample_cube_values_from_sdf(densities, samples_per_chunk_dim, stride, base);
    let cube_index = calculate_cube_index(&cube_values);
    if cube_index == 0 || cube_index == 255 {
        return;
    }
    triangulate_cube_with_cache(
        cube_index,
        &cube_values,
        x,
        y,
        z,
        base_x,
        base_y,
        base_z,
        voxel_size,
        vertices,
        normals,
        uvs,
        materials,
        samples_per_chunk_dim,
        indices,
        stride,
        base,
        edge_to_vertex,
    );
}

fn triangulate_cube_with_cache(
    cube_index: u8,
    values: &[f32; 8],
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    materials: &[u8],
    samples_per_chunk_dim: usize,
    indices: &mut Vec<u32>,
    stride: usize,
    base: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
) {
    let edge_table = &TRIANGLE_TABLE[cube_index as usize];
    let mut i = 0;
    while edge_table[i] != -1 {
        let v1 = get_or_create_edge_vertex(
            edge_table[i] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertices,
            normals,
            uvs,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
            edge_to_vertex,
        );
        let v2 = get_or_create_edge_vertex(
            edge_table[i + 1] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertices,
            normals,
            uvs,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
            edge_to_vertex,
        );
        let v3 = get_or_create_edge_vertex(
            edge_table[i + 2] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertices,
            normals,
            uvs,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
            edge_to_vertex,
        );
        let p1 = vertices[v1 as usize];
        let p2 = vertices[v2 as usize];
        let p3 = vertices[v3 as usize];
        let ux = p2[0] - p1[0];
        let uy = p2[1] - p1[1];
        let uz = p2[2] - p1[2];
        let vx = p3[0] - p1[0];
        let vy = p3[1] - p1[1];
        let vz = p3[2] - p1[2];
        let nx = uy * vz - uz * vy;
        let ny = uz * vx - ux * vz;
        let nz = ux * vy - uy * vx;
        let i1 = v1 as usize;
        let i2 = v2 as usize;
        let i3 = v3 as usize;
        let n = &mut normals[i1];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        let n = &mut normals[i2];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        let n = &mut normals[i3];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        indices.push(v1);
        indices.push(v2);
        indices.push(v3);
        i += 3;
    }
}

#[inline(always)]
fn get_or_create_edge_vertex(
    edge_index: usize,
    values: &[f32; 8],
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    materials: &[u8],
    samples_per_chunk_dim: usize,
    stride: usize,
    base: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
) -> u32 {
    let (dx, dy, dz, dir) = EDGE_ID_OFFSETS[edge_index];
    let edge_id = make_edge_key(
        cube_x as u16 + dx,
        cube_y as u16 + dy,
        cube_z as u16 + dz,
        dir,
    );
    match edge_to_vertex.entry(edge_id) {
        Entry::Occupied(e) => *e.get(),
        Entry::Vacant(e) => {
            let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
            let position = interpolate_edge_from_base(
                v1_idx, v2_idx, base_x, base_y, base_z, voxel_size, values,
            );
            let (dx1, dy1, dz1) = CORNER_OFFSETS[v1_idx];
            let (dx2, dy2, dz2) = CORNER_OFFSETS[v2_idx];
            let material1 = materials[base + dz1 * stride + dy1 * samples_per_chunk_dim + dx1];
            let material2 = materials[base + dz2 * stride + dy2 * samples_per_chunk_dim + dx2];
            let material = if material1 == 2 || material2 == 2 {
                2
            } else if material1 != 0 {
                material1
            } else {
                material2
            };
            let idx = vertices.len() as u32;
            vertices.push(position);
            uvs.push([material as f32, 0.0]);
            normals.push([0.0, 0.0, 0.0]);
            e.insert(idx);
            idx
        }
    }
}

#[inline(always)]
fn interpolate_edge_from_base(
    v1_idx: usize,
    v2_idx: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    values: &[f32; 8],
) -> [f32; 3] {
    let (dx1, dy1, dz1) = CORNER_OFFSETS[v1_idx];
    let (dx2, dy2, dz2) = CORNER_OFFSETS[v2_idx];
    let v1x = base_x + dx1 as f32 * voxel_size;
    let v1y = base_y + dy1 as f32 * voxel_size;
    let v1z = base_z + dz1 as f32 * voxel_size;
    let v2x = base_x + dx2 as f32 * voxel_size;
    let v2y = base_y + dy2 as f32 * voxel_size;
    let v2z = base_z + dz2 as f32 * voxel_size;
    let val1 = values[v1_idx];
    let val2 = values[v2_idx];
    let denom = val2 - val1;
    let t = if denom.abs() < 0.0001 {
        0.5
    } else {
        (-val1 / denom).clamp(0.0, 1.0)
    };
    [
        v1x + t * (v2x - v1x),
        v1y + t * (v2y - v1y),
        v1z + t * (v2z - v1z),
    ]
}

#[inline(always)]
fn sample_cube_values_from_sdf(
    densities: &[i16],
    samples_per_chunk_dim: usize,
    stride: usize,
    base: usize,
) -> [f32; 8] {
    [
        densities[base] as f32,
        densities[base + 1] as f32,
        densities[base + samples_per_chunk_dim + 1] as f32,
        densities[base + samples_per_chunk_dim] as f32,
        densities[base + stride] as f32,
        densities[base + stride + 1] as f32,
        densities[base + stride + samples_per_chunk_dim + 1] as f32,
        densities[base + stride + samples_per_chunk_dim] as f32,
    ]
}
