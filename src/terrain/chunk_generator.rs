use bevy::prelude::*;
use core::arch::x86_64::*;
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};

use crate::{
    constants::{CHUNK_WORLD_SIZE, HALF_CHUNK, NOISE_AMPLITUDE, NOISE_FREQUENCY, NOISE_SEED},
    terrain::terrain::Uniformity,
};

const SCALE: f32 = 32767.0 / 10.0; // Map [-10, 10] to [-32767, 32767]
const SCALE_INV: f32 = 1.0 / SCALE;

#[repr(u8)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum MaterialCode {
    Air = 0,
    Dirt = 1,
    Grass = 2,
    Sand = 3,
}

pub fn get_fbm() -> GeneratorWrapper<SafeNode> {
    let mountains = opensimplex2().ridged(0.5, 0.5, 5, 2.0);
    (mountains).build()
}

pub fn generate_chunk_into_buffers(
    fbm: &GeneratorWrapper<SafeNode>,
    chunk_start: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
    heightmap_buffer: &mut [f32],
    dhdx_buffer: &mut [f32],
    dhdz_buffer: &mut [f32],
    samples_per_chunk_dim: usize,
) -> Uniformity {
    let noise_samples = generate_noise_height_samples(chunk_start.x, chunk_start.z, fbm);
    generate_terrain_heights(heightmap_buffer, samples_per_chunk_dim, &noise_samples);
    compute_heightmap_gradients(
        dhdx_buffer,
        dhdz_buffer,
        samples_per_chunk_dim,
        &noise_samples,
    );
    let is_uniform = fill_voxel_densities(
        density_buffer,
        material_buffer,
        &chunk_start,
        heightmap_buffer,
        samples_per_chunk_dim,
        dhdx_buffer,
        dhdz_buffer,
    );
    let uniformity = if !is_uniform {
        Uniformity::NonUniform
    } else {
        if material_buffer[0] == MaterialCode::Dirt {
            Uniformity::Dirt
        } else if material_buffer[0] == MaterialCode::Air {
            Uniformity::Air
        } else {
            println!("materials[0]: {:?}", material_buffer[0]);
            panic!("Generated uniform chunk with unknown material type!");
        }
    };
    uniformity
}

pub fn generate_noise_height_samples(
    chunk_start_x: f32, //assumed to be even and integer
    chunk_start_z: f32, //assumed to be even and integer
    fbm: &GeneratorWrapper<SafeNode>,
) -> [f32; 25] {
    let mut noise_grid = [0.0; 25];
    let x_start = ((chunk_start_x - HALF_CHUNK) / HALF_CHUNK) as i32;
    let z_start = ((chunk_start_z - HALF_CHUNK) / HALF_CHUNK) as i32;
    fbm.gen_uniform_grid_2d(
        &mut noise_grid,
        x_start,
        z_start,
        5,
        5,
        NOISE_FREQUENCY * HALF_CHUNK,
        NOISE_SEED,
    );
    for v in &mut noise_grid {
        *v *= NOISE_AMPLITUDE;
    }
    noise_grid
}

pub fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_WORLD_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_WORLD_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_WORLD_SIZE - HALF_CHUNK,
    )
}

//it may be better to store a byte signifying if a chunk contains a surface when saving to disk
pub fn chunk_contains_surface(density_buffer: &[i16]) -> bool {
    let mut has_positive = false;
    let mut has_negative = false;
    for &density in density_buffer {
        if density > 0 {
            has_positive = true;
        } else if density < 0 {
            has_negative = true;
        }
        if has_positive && has_negative {
            return true;
        }
    }
    false
}

//Sample the fbm noise at a higher resolution and then bilinearly interpolate to get smooth terrain heights
pub fn generate_terrain_heights(
    heightmap_buffer: &mut [f32],
    samples_per_chunk_dim: usize,
    noise_samples: &[f32],
) {
    let inv_samples = 1.0 / (samples_per_chunk_dim - 1) as f32;
    let mut roller = 0;
    for z in 0..samples_per_chunk_dim {
        let t_z = z as f32 * inv_samples;
        let grid_z = 1.0 + t_z * 2.0;
        let grid_z_idx = grid_z as usize;
        let gz0 = grid_z_idx.saturating_sub(1).min(4);
        let gz1 = grid_z_idx.min(4);
        let gz2 = (grid_z_idx + 1).min(4);
        let gz3 = (grid_z_idx + 2).min(4);
        let local_t_z = grid_z - grid_z_idx as f32;
        let b0 = gz0 * 5;
        let b1 = gz1 * 5;
        let b2 = gz2 * 5;
        let b3 = gz3 * 5;
        for x in 0..samples_per_chunk_dim {
            let t_x = x as f32 * inv_samples;
            let grid_x = 1.0 + t_x * 2.0;
            let grid_x_idx = grid_x as usize;
            let gx0 = grid_x_idx.saturating_sub(1).min(4);
            let gx1 = grid_x_idx.min(4);
            let gx2 = (grid_x_idx + 1).min(4);
            let gx3 = (grid_x_idx + 2).min(4);
            let local_t_x = grid_x - grid_x_idx as f32;
            let g = [
                noise_samples[b0 + gx0],
                noise_samples[b1 + gx0],
                noise_samples[b2 + gx0],
                noise_samples[b3 + gx0],
                noise_samples[b0 + gx1],
                noise_samples[b1 + gx1],
                noise_samples[b2 + gx1],
                noise_samples[b3 + gx1],
                noise_samples[b0 + gx2],
                noise_samples[b1 + gx2],
                noise_samples[b2 + gx2],
                noise_samples[b3 + gx2],
                noise_samples[b0 + gx3],
                noise_samples[b1 + gx3],
                noise_samples[b2 + gx3],
                noise_samples[b3 + gx3],
            ];
            let height = bicubic_4x4_simd(&g, local_t_x, local_t_z);
            heightmap_buffer[roller] = height;
            roller += 1;
        }
    }
}

//optimized by first iterating over the boundary voxels to quickly determine uniformity
//assumes that the surface passes through one of the chunk sides
//will break if a "cave" is fully enclosed within the chunk
pub fn fill_voxel_densities(
    densities: &mut [i16],
    materials: &mut [MaterialCode],
    chunk_start: &Vec3,
    terrain_heights: &[f32],
    samples_per_chunk_dim: usize,
    dhdx: &[f32],
    dhdz: &[f32],
) -> bool {
    let voxel_size = CHUNK_WORLD_SIZE / (samples_per_chunk_dim - 1) as f32;
    let heightmap_grid_size: usize = samples_per_chunk_dim * samples_per_chunk_dim;
    let solid_threshold = quantize_f32_to_i16(-1.0);
    let mut is_uniform = true;
    let mut init_distance = 0;
    let mut init_material = MaterialCode::Air;
    let mut has_init = false;
    for z in [0, samples_per_chunk_dim - 1] {
        let height_base = z * samples_per_chunk_dim;
        let z_base = z * heightmap_grid_size;
        for y in 0..samples_per_chunk_dim {
            let world_y = chunk_start.y + y as f32 * voxel_size;
            let below_sea = world_y < 0.0;
            let base_rolling_voxel_index = z_base + y * samples_per_chunk_dim;
            for x in 0..samples_per_chunk_dim {
                let rolling_voxel_index = base_rolling_voxel_index + x;
                let terrain_height = terrain_heights[height_base + x];
                let distance_to_surface = {
                    let vertical_dist = world_y - terrain_height;
                    let gidx = z * samples_per_chunk_dim + x;
                    let dhdx = dhdx[gidx];
                    let dhdz = dhdz[gidx];
                    (vertical_dist / (1.0 + dhdx * dhdx + dhdz * dhdz).sqrt()).clamp(-10.0, 10.0)
                };
                let quantized_distance_to_surface = quantize_f32_to_i16(distance_to_surface);
                densities[rolling_voxel_index] = quantized_distance_to_surface;
                let mat = if quantized_distance_to_surface >= 0 {
                    MaterialCode::Air
                } else if quantized_distance_to_surface < solid_threshold {
                    MaterialCode::Dirt
                } else if below_sea {
                    MaterialCode::Sand
                } else {
                    MaterialCode::Grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = quantized_distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != quantized_distance_to_surface || init_material != mat
                    {
                        is_uniform = false;
                    }
                }
            }
        }
    }
    for z in 1..samples_per_chunk_dim - 1 {
        let height_base = z * samples_per_chunk_dim;
        let z_base = z * heightmap_grid_size;
        for y in [0, samples_per_chunk_dim - 1] {
            let world_y = chunk_start.y + y as f32 * voxel_size;
            let below_sea = world_y < 0.0;
            let base_rolling_voxel_index = z_base + y * samples_per_chunk_dim;
            for x in 0..samples_per_chunk_dim {
                let rolling_voxel_index = base_rolling_voxel_index + x;
                let terrain_height = terrain_heights[height_base + x];
                let distance_to_surface = {
                    let vertical_dist = world_y - terrain_height;
                    let gidx = z * samples_per_chunk_dim + x;
                    let dhdx = dhdx[gidx];
                    let dhdz = dhdz[gidx];
                    (vertical_dist / (1.0 + dhdx * dhdx + dhdz * dhdz).sqrt()).clamp(-10.0, 10.0)
                };
                let quantized_distance_to_surface = quantize_f32_to_i16(distance_to_surface);
                densities[rolling_voxel_index] = quantized_distance_to_surface;
                let mat = if quantized_distance_to_surface >= 0 {
                    MaterialCode::Air
                } else if quantized_distance_to_surface < solid_threshold {
                    MaterialCode::Dirt
                } else if below_sea {
                    MaterialCode::Sand
                } else {
                    MaterialCode::Grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = quantized_distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != quantized_distance_to_surface || init_material != mat
                    {
                        is_uniform = false;
                    }
                }
            }
        }
    }
    for z in 1..samples_per_chunk_dim - 1 {
        let height_base = z * samples_per_chunk_dim;
        let z_base = z * heightmap_grid_size;
        for y in 1..samples_per_chunk_dim - 1 {
            let world_y = chunk_start.y + y as f32 * voxel_size;
            let below_sea = world_y < 0.0;
            let base_rolling_voxel_index = z_base + y * samples_per_chunk_dim;
            for x in [0, samples_per_chunk_dim - 1] {
                let rolling_voxel_index = base_rolling_voxel_index + x;
                let terrain_height = terrain_heights[height_base + x];
                let distance_to_surface = {
                    let vertical_dist = world_y - terrain_height;
                    let gidx = z * samples_per_chunk_dim + x;
                    let dhdx = dhdx[gidx];
                    let dhdz = dhdz[gidx];
                    (vertical_dist / (1.0 + dhdx * dhdx + dhdz * dhdz).sqrt()).clamp(-10.0, 10.0)
                };
                let quantized_distance_to_surface = quantize_f32_to_i16(distance_to_surface);
                densities[rolling_voxel_index] = quantized_distance_to_surface;
                let mat = if quantized_distance_to_surface >= 0 {
                    MaterialCode::Air
                } else if quantized_distance_to_surface < solid_threshold {
                    MaterialCode::Dirt
                } else if below_sea {
                    MaterialCode::Sand
                } else {
                    MaterialCode::Grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = quantized_distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != quantized_distance_to_surface || init_material != mat
                    {
                        is_uniform = false;
                    }
                }
            }
        }
    }
    if is_uniform {
        return is_uniform;
    } else {
        for z in 1..samples_per_chunk_dim - 1 {
            let height_base = z * samples_per_chunk_dim;
            let z_base = z * heightmap_grid_size;
            for y in 1..samples_per_chunk_dim - 1 {
                let world_y = chunk_start.y + y as f32 * voxel_size;
                let below_sea = world_y < 0.0;
                let base_rolling_voxel_index = z_base + y * samples_per_chunk_dim;
                for x in 1..samples_per_chunk_dim - 1 {
                    let rolling_voxel_index = base_rolling_voxel_index + x;
                    let terrain_height = terrain_heights[height_base + x];
                    let distance_to_surface = {
                        let vertical_dist = world_y - terrain_height;
                        let gidx = z * samples_per_chunk_dim + x;
                        let dhdx = dhdx[gidx];
                        let dhdz = dhdz[gidx];
                        (vertical_dist / (1.0 + dhdx * dhdx + dhdz * dhdz).sqrt())
                            .clamp(-10.0, 10.0)
                    };
                    let quantized_distance_to_surface = quantize_f32_to_i16(distance_to_surface);
                    densities[rolling_voxel_index] = quantized_distance_to_surface;
                    let mat = if quantized_distance_to_surface >= 0 {
                        MaterialCode::Air
                    } else if quantized_distance_to_surface < solid_threshold {
                        MaterialCode::Dirt
                    } else if below_sea {
                        MaterialCode::Sand
                    } else {
                        MaterialCode::Grass
                    };
                    materials[rolling_voxel_index] = mat;
                    if is_uniform {
                        if !has_init {
                            init_distance = quantized_distance_to_surface;
                            init_material = mat;
                            has_init = true;
                        } else if init_distance != quantized_distance_to_surface
                            || init_material != mat
                        {
                            is_uniform = false;
                        }
                    }
                }
            }
        }
    }
    is_uniform
}

#[inline(always)]
pub fn quantize_f32_to_i16(value: f32) -> i16 {
    (value * SCALE).round() as i16
}

#[inline(always)]
pub fn dequantize_i16_to_f32(q: i16) -> f32 {
    q as f32 * SCALE_INV
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Density = trilinear SDF sample at the output grid point
// Material = scan the corresponding input block;
// prefer grass/sand if any exist near the surface,
// otherwise pick the closest-to-surface solid.
pub fn downscale(
    densities_in: &[i16],
    materials_in: &[MaterialCode],
    in_dim: usize,
    densities_out: &mut [i16],
    materials_out: &mut [MaterialCode],
    out_dim: usize,
) {
    let in_max = in_dim - 1;
    let out_max = out_dim - 1;
    let stride = in_max / out_max;
    for target_z in 0..out_dim {
        let high_sample_z = (target_z as f32 / out_max as f32) * in_max as f32;
        let full_res_target_z = target_z * stride;
        let full_res_end_z = (full_res_target_z + stride).min(in_dim - 1);
        let z_base = target_z * out_dim;
        for target_y in 0..out_dim {
            let high_sample_y = (target_y as f32 / out_max as f32) * in_max as f32;
            let full_res_target_y = target_y * stride;
            let full_res_end_y = (full_res_target_y + stride).min(in_dim - 1);
            let zy_base = (z_base + target_y) * out_dim;
            for target_x in 0..out_dim {
                let full_res_target_x = target_x * stride;
                let high_sample_x = (target_x as f32 / out_max as f32) * in_max as f32;
                let new_density = sample_trilinear_density(
                    densities_in,
                    in_dim,
                    high_sample_x,
                    high_sample_y,
                    high_sample_z,
                );
                let out_i = zy_base + target_x;
                densities_out[out_i] = quantize_f32_to_i16(new_density);
                let full_res_end_x = (full_res_target_x + stride).min(in_dim - 1);
                let new_material = pick_surface_material_block_prefer_biomes(
                    densities_in,
                    materials_in,
                    in_dim,
                    full_res_target_x,
                    full_res_target_y,
                    full_res_target_z,
                    full_res_end_x,
                    full_res_end_y,
                    full_res_end_z,
                );
                materials_out[out_i] = new_material;
            }
        }
    }
}

fn pick_surface_material_block_prefer_biomes(
    densities: &[i16],
    materials: &[MaterialCode],
    in_dim: usize,
    full_res_start_x: usize,
    full_res_start_y: usize,
    full_res_start_z: usize,
    full_res_end_x: usize,
    full_res_end_y: usize,
    full_res_end_z: usize,
) -> MaterialCode {
    let mut best_m = MaterialCode::Air;
    let mut best_abs = i16::MAX;
    for full_res_z in full_res_start_z..=full_res_end_z {
        for full_res_y in full_res_start_y..=full_res_end_y {
            let base = (full_res_z * in_dim + full_res_y) * in_dim;
            for full_res_x in full_res_start_x..=full_res_end_x {
                let index = base + full_res_x;
                let density = densities[index];
                let material = materials[index];
                if material == MaterialCode::Air || density >= 0 {
                    continue;
                }
                let abs = density.abs();
                if abs < best_abs {
                    best_abs = abs;
                    best_m = material;
                }
            }
        }
    }
    if best_m != MaterialCode::Air {
        return best_m;
    }
    let mut best_any_m = MaterialCode::Air;
    let mut best_any_abs = i16::MAX;
    let mut best_any_solid = false;
    for full_res_z in full_res_start_z..=full_res_end_z {
        for full_res_y in full_res_start_y..=full_res_end_y {
            let base = (full_res_z * in_dim + full_res_y) * in_dim;
            for full_res_x in full_res_start_x..=full_res_end_x {
                let index = base + full_res_x;
                let density = densities[index];
                let material = materials[index];
                if material == MaterialCode::Air {
                    continue;
                }
                let abs = density.abs();
                let solid = density < 0;
                if abs < best_any_abs || (abs == best_any_abs && solid && !best_any_solid) {
                    best_any_abs = abs;
                    best_any_solid = solid;
                    best_any_m = material;
                }
            }
        }
    }
    best_any_m
}

fn sample_trilinear_density(
    d: &[i16],
    dim: usize,
    high_sample_x: f32,
    high_sample_y: f32,
    high_sample_z: f32,
) -> f32 {
    let x0 = high_sample_x.floor() as isize;
    let y0 = high_sample_y.floor() as isize;
    let z0 = high_sample_z.floor() as isize;
    let x1 = (x0 + 1).min((dim - 1) as isize);
    let y1 = (y0 + 1).min((dim - 1) as isize);
    let z1 = (z0 + 1).min((dim - 1) as isize);
    let tx = high_sample_x - x0 as f32;
    let ty = high_sample_y - y0 as f32;
    let tz = high_sample_z - z0 as f32;
    let idx = |x: isize, y: isize, z: isize| -> usize {
        (z as usize * dim + y as usize) * dim + x as usize
    };
    let d000 = dequantize_i16_to_f32(d[idx(x0, y0, z0)]);
    let d100 = dequantize_i16_to_f32(d[idx(x1, y0, z0)]);
    let d010 = dequantize_i16_to_f32(d[idx(x0, y1, z0)]);
    let d110 = dequantize_i16_to_f32(d[idx(x1, y1, z0)]);
    let d001 = dequantize_i16_to_f32(d[idx(x0, y0, z1)]);
    let d101 = dequantize_i16_to_f32(d[idx(x1, y0, z1)]);
    let d011 = dequantize_i16_to_f32(d[idx(x0, y1, z1)]);
    let d111 = dequantize_i16_to_f32(d[idx(x1, y1, z1)]);
    let c00 = lerp(d000, d100, tx);
    let c10 = lerp(d010, d110, tx);
    let c01 = lerp(d001, d101, tx);
    let c11 = lerp(d011, d111, tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    lerp(c0, c1, tz)
}

pub fn compute_heightmap_gradients(
    dhdx: &mut [f32],
    dhdz: &mut [f32],
    dim: usize,
    noise_height_samples: &[f32],
) {
    const INV_CHUNK_SIZE: f32 = 1.0 / CHUNK_WORLD_SIZE;
    let mut gx = [0.0; 25];
    let mut gz = [0.0; 25];
    for zz in 0..5 {
        let zz_5 = zz * 5;
        for xx in 0..5 {
            let xm = if xx > 0 { xx - 1 } else { 0 };
            let xp = if xx < 4 { xx + 1 } else { 4 };
            let zm = if zz > 0 { zz - 1 } else { 0 };
            let zp = if zz < 4 { zz + 1 } else { 4 };
            gx[zz_5 + xx] = (noise_height_samples[zz * 5 + xp] - noise_height_samples[zz_5 + xm])
                * INV_CHUNK_SIZE;
            gz[zz_5 + xx] = (noise_height_samples[zp * 5 + xx] - noise_height_samples[zm * 5 + xx])
                * INV_CHUNK_SIZE;
        }
    }
    let mut vg = [0.0; 16];
    let mut zg = [0.0; 16];
    let inv_samples = 1.0 / (dim - 1) as f32;
    for z in 0..dim {
        let t_z = z as f32 * inv_samples;
        let grid_z = 1.0 + t_z * 2.0;
        let iz = grid_z as usize;
        let gz0 = iz.saturating_sub(1).min(4);
        let gz1 = iz.min(4);
        let gz2 = (iz + 1).min(4);
        let gz3 = (iz + 2).min(4);
        let tz = grid_z - iz as f32;
        let b0 = gz0 * 5;
        let b1 = gz1 * 5;
        let b2 = gz2 * 5;
        let b3 = gz3 * 5;
        let base_z = z * dim;
        for x in 0..dim {
            let t_x = x as f32 * inv_samples;
            let grid_x = 1.0 + t_x * 2.0;
            let ix = grid_x as usize;
            let gx0i = ix.saturating_sub(1).min(4);
            let gx1i = ix.min(4);
            let gx2i = (ix + 1).min(4);
            let gx3i = (ix + 2).min(4);
            let tx = grid_x - ix as f32;
            let out = base_z + x;
            let i0 = b0 + gx0i;
            let i1 = b0 + gx1i;
            let i2 = b0 + gx2i;
            let i3 = b0 + gx3i;
            let i4 = b1 + gx0i;
            let i5 = b1 + gx1i;
            let i6 = b1 + gx2i;
            let i7 = b1 + gx3i;
            let i8 = b2 + gx0i;
            let i9 = b2 + gx1i;
            let i10 = b2 + gx2i;
            let i11 = b2 + gx3i;
            let i12 = b3 + gx0i;
            let i13 = b3 + gx1i;
            let i14 = b3 + gx2i;
            let i15 = b3 + gx3i;
            vg[0] = gx[i0];
            vg[1] = gx[i4];
            vg[2] = gx[i8];
            vg[3] = gx[i12];
            vg[4] = gx[i1];
            vg[5] = gx[i5];
            vg[6] = gx[i9];
            vg[7] = gx[i13];
            vg[8] = gx[i2];
            vg[9] = gx[i6];
            vg[10] = gx[i10];
            vg[11] = gx[i14];
            vg[12] = gx[i3];
            vg[13] = gx[i7];
            vg[14] = gx[i11];
            vg[15] = gx[i15];
            zg[0] = gz[i0];
            zg[1] = gz[i4];
            zg[2] = gz[i8];
            zg[3] = gz[i12];
            zg[4] = gz[i1];
            zg[5] = gz[i5];
            zg[6] = gz[i9];
            zg[7] = gz[i13];
            zg[8] = gz[i2];
            zg[9] = gz[i6];
            zg[10] = gz[i10];
            zg[11] = gz[i14];
            zg[12] = gz[i3];
            zg[13] = gz[i7];
            zg[14] = gz[i11];
            zg[15] = gz[i15];
            dhdx[out] = bicubic_4x4_simd(&vg, tx, tz);
            dhdz[out] = bicubic_4x4_simd(&zg, tx, tz);
        }
    }
}

#[inline(always)]
fn bicubic_4x4_simd(g: &[f32; 16], t_x: f32, t_z: f32) -> f32 {
    unsafe {
        let c0 = _mm_loadu_ps(g.as_ptr());
        let c1 = _mm_loadu_ps(g.as_ptr().add(4));
        let c2 = _mm_loadu_ps(g.as_ptr().add(8));
        let c3 = _mm_loadu_ps(g.as_ptr().add(12));
        let three = _mm_set1_ps(3.0);
        let two = _mm_set1_ps(2.0);
        let four = _mm_set1_ps(4.0);
        let five = _mm_set1_ps(5.0);
        let half = _mm_set1_ps(0.5);
        let txv = _mm_set1_ps(t_x);
        let a = _mm_add_ps(
            _mm_sub_ps(_mm_mul_ps(three, c1), _mm_mul_ps(three, c2)),
            _mm_sub_ps(c3, c0),
        );
        let b = _mm_sub_ps(
            _mm_add_ps(_mm_mul_ps(two, c0), _mm_mul_ps(four, c2)),
            _mm_add_ps(_mm_mul_ps(five, c1), c3),
        );
        let c = _mm_sub_ps(c2, c0);
        let d = _mm_mul_ps(two, c1);
        let y = _mm_fmadd_ps(txv, a, b);
        let y = _mm_fmadd_ps(txv, y, c);
        let y = _mm_fmadd_ps(txv, y, d);
        let cols = _mm_mul_ps(half, y);
        let t2 = t_z * t_z;
        let t3 = t2 * t_z;
        let w0 = 0.5 * (-t3 + 2.0 * t2 - t_z);
        let w1 = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0);
        let w2 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t_z);
        let w3 = 0.5 * (t3 - t2);
        let wz = _mm_set_ps(w3, w2, w1, w0);
        let sum = _mm_dp_ps(cols, wz, 0xF1);
        _mm_cvtss_f32(sum)
    }
}
