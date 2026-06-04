use std::collections::hash_map::Entry;

use bevy::math::Vec3;
use rustc_hash::{FxBuildHasher, FxHashMap as HashMap};

use crate::{
    constants::{
        CHUNK_WORLD_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, SAMPLES_PER_CHUNK_DIM_PADDED,
    },
    marching_cubes::tables::{CORNER_OFFSETS, EDGE_ID_OFFSETS, EDGE_VERTICES, TRIANGLE_TABLE},
    terrain::chunk_generator::MaterialCode,
};

type EdgeKey = u64;

#[inline(always)]
fn make_edge_key(x: u16, y: u16, z: u16, dir: u8) -> u64 {
    (x as u64) | ((y as u64) << 16) | ((z as u64) << 32) | ((dir as u64) << 48)
}

#[inline(always)]
fn sample_full_res(
    densities_full_res: &[i16],
    x: usize,
    y: usize,
    z: usize,
    padded_dim: usize,
) -> f32 {
    let stride = padded_dim * padded_dim;
    densities_full_res[z * stride + y * padded_dim + x] as f32
}

fn sample_full_res_trilinear(densities_full_res: &[i16], local_pos: Vec3) -> f32 {
    let inv_voxel = (SAMPLES_PER_CHUNK_DIM - 1) as f32 / CHUNK_WORLD_SIZE;
    let fx = (local_pos.x + HALF_CHUNK) * inv_voxel + 1.0;
    let fy = (local_pos.y + HALF_CHUNK) * inv_voxel + 1.0;
    let fz = (local_pos.z + HALF_CHUNK) * inv_voxel + 1.0;
    let fx = fx.clamp(0.0, (SAMPLES_PER_CHUNK_DIM_PADDED - 1) as f32);
    let fy = fy.clamp(0.0, (SAMPLES_PER_CHUNK_DIM_PADDED - 1) as f32);
    let fz = fz.clamp(0.0, (SAMPLES_PER_CHUNK_DIM_PADDED - 1) as f32);
    let ix = fx as usize;
    let iy = fy as usize;
    let iz = fz as usize;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;
    let ix1 = (ix + 1).min(SAMPLES_PER_CHUNK_DIM_PADDED - 1);
    let iy1 = (iy + 1).min(SAMPLES_PER_CHUNK_DIM_PADDED - 1);
    let iz1 = (iz + 1).min(SAMPLES_PER_CHUNK_DIM_PADDED - 1);
    let d000 = sample_full_res(densities_full_res, ix, iy, iz, SAMPLES_PER_CHUNK_DIM_PADDED);
    let d100 = sample_full_res(
        densities_full_res,
        ix1,
        iy,
        iz,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d010 = sample_full_res(
        densities_full_res,
        ix,
        iy1,
        iz,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d110 = sample_full_res(
        densities_full_res,
        ix1,
        iy1,
        iz,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d001 = sample_full_res(
        densities_full_res,
        ix,
        iy,
        iz1,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d101 = sample_full_res(
        densities_full_res,
        ix1,
        iy,
        iz1,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d011 = sample_full_res(
        densities_full_res,
        ix,
        iy1,
        iz1,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let d111 = sample_full_res(
        densities_full_res,
        ix1,
        iy1,
        iz1,
        SAMPLES_PER_CHUNK_DIM_PADDED,
    );
    let c00 = d000 + tx * (d100 - d000);
    let c10 = d010 + tx * (d110 - d010);
    let c01 = d001 + tx * (d101 - d001);
    let c11 = d011 + tx * (d111 - d011);
    let c0 = c00 + ty * (c10 - c00);
    let c1 = c01 + ty * (c11 - c01);
    c0 + tz * (c1 - c0)
}

#[inline(always)]
fn compute_full_res_gradient(densities_full_res: &[i16], local_pos: Vec3) -> Vec3 {
    let h = CHUNK_WORLD_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32 * 0.5;
    let dx = sample_full_res_trilinear(densities_full_res, local_pos + Vec3::new(h, 0.0, 0.0))
        - sample_full_res_trilinear(densities_full_res, local_pos - Vec3::new(h, 0.0, 0.0));
    let dy = sample_full_res_trilinear(densities_full_res, local_pos + Vec3::new(0.0, h, 0.0))
        - sample_full_res_trilinear(densities_full_res, local_pos - Vec3::new(0.0, h, 0.0));
    let dz = sample_full_res_trilinear(densities_full_res, local_pos + Vec3::new(0.0, 0.0, h))
        - sample_full_res_trilinear(densities_full_res, local_pos - Vec3::new(0.0, 0.0, h));
    Vec3::new(dx, dy, dz)
}

pub fn mc_mesh_generation(
    densities: &[i16],
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    densities_padded: bool,
    densities_full_res: &[i16],
) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>, Vec<u32>) {
    let mut edge_to_vertex: HashMap<EdgeKey, u32> = HashMap::with_hasher(FxBuildHasher::default());
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = CHUNK_WORLD_SIZE / (samples_per_chunk_dim - 1) as f32;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut material_ids = Vec::new();
    let mut indices = Vec::new();
    let density_dim = if densities_padded {
        samples_per_chunk_dim + 2
    } else {
        samples_per_chunk_dim
    };
    let stride = density_dim * density_dim;
    let density_offset = if densities_padded {
        stride + density_dim + 1
    } else {
        0
    };
    let mat_stride = samples_per_chunk_dim * samples_per_chunk_dim;
    for z_idx in 0..cubes_per_chunk_dim {
        let cube_z_world_pos = -HALF_CHUNK + z_idx as f32 * voxel_size;
        let z_base = z_idx * stride;
        let mat_z_base = z_idx * mat_stride;
        for y_idx in 0..cubes_per_chunk_dim {
            let cube_y_world_pos = -HALF_CHUNK + y_idx as f32 * voxel_size;
            let yz_base = z_base + y_idx * density_dim;
            let mat_yz_base = mat_z_base + y_idx * samples_per_chunk_dim;
            for x_idx in 0..cubes_per_chunk_dim {
                let cube_x_world_pos = -HALF_CHUNK + x_idx as f32 * voxel_size;
                let cube_world_pos =
                    Vec3::new(cube_x_world_pos, cube_y_world_pos, cube_z_world_pos);
                let voxel_idx = density_offset + yz_base + x_idx;
                let mat_voxel_idx = mat_yz_base + x_idx;
                let cube_corner_densities =
                    sample_cube_corner_densities(densities, density_dim, stride, voxel_idx);
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
                    mat_voxel_idx,
                    &mut vertices,
                    &mut normals,
                    &mut material_ids,
                    &mut indices,
                    materials,
                    samples_per_chunk_dim,
                    voxel_size,
                    mat_stride,
                    &mut edge_to_vertex,
                    &cube_corner_densities,
                    edge_table,
                    densities_full_res,
                );
            }
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
    mat_voxel_idx: usize,
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    material_ids: &mut Vec<u32>,
    indices: &mut Vec<u32>,
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    voxel_size: f32,
    mat_stride: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
    cube_corner_densities: &[f32; 8],
    edge_table: &[i32; 16],
    densities_full_res: &[i16],
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
            mat_stride,
            mat_voxel_idx,
            edge_to_vertex,
            edge_id,
            densities_full_res,
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
            mat_stride,
            mat_voxel_idx,
            edge_to_vertex,
            edge_id,
            densities_full_res,
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
            mat_stride,
            mat_voxel_idx,
            edge_to_vertex,
            edge_id,
            densities_full_res,
        );
        let m1 = material_ids[v1 as usize];
        let m2 = material_ids[v2 as usize];
        let m3 = material_ids[v3 as usize];
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
    mat_stride: usize,
    mat_voxel_idx: usize,
    edge_to_vertex: &mut HashMap<EdgeKey, u32>,
    edge_id: u64,
    densities_full_res: &[i16],
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
            let material1 = materials[mat_voxel_idx
                + d1.z as usize * mat_stride
                + d1.y as usize * samples_per_chunk_dim
                + d1.x as usize];
            let material2 = materials[mat_voxel_idx
                + d2.z as usize * mat_stride
                + d2.y as usize * samples_per_chunk_dim
                + d2.x as usize];
            let material = if material1 == MaterialCode::Grass || material2 == MaterialCode::Grass {
                MaterialCode::Grass
            } else if material1 == MaterialCode::Sand || material2 == MaterialCode::Sand {
                MaterialCode::Sand
            } else if material1 != MaterialCode::Air {
                material1
            } else {
                material2
            };
            let gradient = compute_full_res_gradient(densities_full_res, position);
            let normal = if gradient.length_squared() > 0.0001 {
                gradient.normalize()
            } else {
                Vec3::Y
            };
            let idx = vertices.len() as u32;
            vertices.push(position);
            normals.push(normal);
            material_ids.push(material as u32);
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
