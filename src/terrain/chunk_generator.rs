use bevy::prelude::*;
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};

use crate::terrain::terrain::{CHUNK_SIZE, HALF_CHUNK};
pub const NOISE_SEED: i32 = 111; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.0005; // Frequency of the noise
pub const NOISE_AMPLITUDE: f32 = 300.0; // Amplitude of the noise

pub fn get_fbm() -> GeneratorWrapper<SafeNode> {
    let mountains = opensimplex2().ridged(0.5, 0.5, 5, 2.0);
    (mountains).build()
}

pub fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_SIZE - HALF_CHUNK,
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
    chunk_start_x: f32, //assumed to be even and integer
    chunk_start_z: f32, //assumed to be even and integer
    fbm: &GeneratorWrapper<SafeNode>,
    heightmap_buffer: &mut [f32],
    samples_per_chunk_dim: usize,
) {
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
            let height = bicubic_interp16(
                noise_grid[b0 + gx0],
                noise_grid[b0 + gx1],
                noise_grid[b0 + gx2],
                noise_grid[b0 + gx3],
                noise_grid[b1 + gx0],
                noise_grid[b1 + gx1],
                noise_grid[b1 + gx2],
                noise_grid[b1 + gx3],
                noise_grid[b2 + gx0],
                noise_grid[b2 + gx1],
                noise_grid[b2 + gx2],
                noise_grid[b2 + gx3],
                noise_grid[b3 + gx0],
                noise_grid[b3 + gx1],
                noise_grid[b3 + gx2],
                noise_grid[b3 + gx3],
                local_t_x,
                local_t_z,
            );
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
    materials: &mut [u8],
    chunk_start: &Vec3,
    terrain_heights: &[f32],
    samples_per_chunk_dim: usize,
) -> bool {
    let voxel_size = CHUNK_SIZE / (samples_per_chunk_dim - 1) as f32;
    let heightmap_grid_size: usize = samples_per_chunk_dim * samples_per_chunk_dim;
    let solid_threshold = quantize_f32_to_i16(-1.0);
    let mut is_uniform = true;
    let mut init_distance = 0;
    let mut init_material = 0;
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
                let distance_to_surface =
                    quantize_f32_to_i16((world_y - terrain_height).clamp(-10.0, 10.0));
                densities[rolling_voxel_index] = distance_to_surface;
                let mat = if distance_to_surface >= 0 {
                    0 //air
                } else if distance_to_surface < solid_threshold {
                    1 //dirt
                } else if below_sea {
                    3 //sand
                } else {
                    2 //grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != distance_to_surface || init_material != mat {
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
                let distance_to_surface =
                    quantize_f32_to_i16((world_y - terrain_height).clamp(-10.0, 10.0));
                densities[rolling_voxel_index] = distance_to_surface;
                let mat = if distance_to_surface >= 0 {
                    0 //air
                } else if distance_to_surface < solid_threshold {
                    1 //dirt
                } else if below_sea {
                    3 //sand
                } else {
                    2 //grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != distance_to_surface || init_material != mat {
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
                let distance_to_surface =
                    quantize_f32_to_i16((world_y - terrain_height).clamp(-10.0, 10.0));
                densities[rolling_voxel_index] = distance_to_surface;
                let mat = if distance_to_surface >= 0 {
                    0 //air
                } else if distance_to_surface < solid_threshold {
                    1 //dirt
                } else if below_sea {
                    3 //sand
                } else {
                    2 //grass
                };
                materials[rolling_voxel_index] = mat;
                if is_uniform {
                    if !has_init {
                        init_distance = distance_to_surface;
                        init_material = mat;
                        has_init = true;
                    } else if init_distance != distance_to_surface || init_material != mat {
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
                    let distance_to_surface =
                        quantize_f32_to_i16((world_y - terrain_height).clamp(-10.0, 10.0));
                    densities[rolling_voxel_index] = distance_to_surface;
                    let mat = if distance_to_surface >= 0 {
                        0 //air
                    } else if distance_to_surface < solid_threshold {
                        1 //dirt
                    } else if below_sea {
                        3 //sand
                    } else {
                        2 //grass
                    };
                    materials[rolling_voxel_index] = mat;
                    if is_uniform {
                        if !has_init {
                            init_distance = distance_to_surface;
                            init_material = mat;
                            has_init = true;
                        } else if init_distance != distance_to_surface || init_material != mat {
                            is_uniform = false;
                        }
                    }
                }
            }
        }
    }
    is_uniform
}

const SCALE: f32 = 32767.0 / 10.0; // Map [-10, 10] to [-32767, 32767]

#[inline(always)]
pub fn quantize_f32_to_i16(value: f32) -> i16 {
    (value * SCALE).round() as i16
}

#[inline(always)]
pub fn dequantize_i16_to_f32(q: i16) -> f32 {
    q as f32 / SCALE
}

#[inline(always)]
fn cubic_interp(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[inline(always)]
fn bicubic_interp16(
    g00: f32,
    g01: f32,
    g02: f32,
    g03: f32,
    g10: f32,
    g11: f32,
    g12: f32,
    g13: f32,
    g20: f32,
    g21: f32,
    g22: f32,
    g23: f32,
    g30: f32,
    g31: f32,
    g32: f32,
    g33: f32,
    t_x: f32,
    t_z: f32,
) -> f32 {
    let col0 = cubic_interp(g00, g01, g02, g03, t_x);
    let col1 = cubic_interp(g10, g11, g12, g13, t_x);
    let col2 = cubic_interp(g20, g21, g22, g23, t_x);
    let col3 = cubic_interp(g30, g31, g32, g33, t_x);
    cubic_interp(col0, col1, col2, col3, t_z)
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
    materials_in: &[u8],
    in_dim: usize,
    densities_out: &mut [i16],
    materials_out: &mut [u8],
    out_dim: usize,
) {
    let in_max = (in_dim - 1) as f32;
    let out_max = (out_dim - 1) as f32;
    for oz in 0..out_dim {
        let fz = (oz as f32 / out_max) * in_max;
        for oy in 0..out_dim {
            let fy = (oy as f32 / out_max) * in_max;
            for ox in 0..out_dim {
                let fx = (ox as f32 / out_max) * in_max;
                let d = sample_trilinear_density(densities_in, in_dim, fx, fy, fz);
                let out_i = (oz * out_dim + oy) * out_dim + ox;
                densities_out[out_i] = quantize_f32_to_i16(d);
                let m = pick_surface_material_block_prefer_biomes(
                    densities_in,
                    materials_in,
                    in_dim,
                    ox,
                    oy,
                    oz,
                    out_dim,
                );
                materials_out[out_i] = m;
            }
        }
    }
}

fn pick_surface_material_block_prefer_biomes(
    densities: &[i16],
    materials: &[u8],
    in_dim: usize,
    ox: usize,
    oy: usize,
    oz: usize,
    out_dim: usize,
) -> u8 {
    let rf = (in_dim - 1) / (out_dim - 1);
    let x0 = ox * rf;
    let y0 = oy * rf;
    let z0 = oz * rf;
    let x1 = (x0 + rf).min(in_dim - 1);
    let y1 = (y0 + rf).min(in_dim - 1);
    let z1 = (z0 + rf).min(in_dim - 1);
    let mut best_m: u8 = 0;
    let mut best_abs: i32 = i32::MAX;
    for z in z0..=z1 {
        for y in y0..=y1 {
            let base = (z * in_dim + y) * in_dim;
            for x in x0..=x1 {
                let i = base + x;
                let d = densities[i];
                let m = materials[i];
                if m == 0 || d >= 0 {
                    continue;
                }
                let abs = (d as i32).abs();
                if abs < best_abs {
                    best_abs = abs;
                    best_m = m;
                }
            }
        }
    }
    if best_m != 0 {
        return best_m;
    }
    let mut best_any_m: u8 = 0;
    let mut best_any_abs: i32 = i32::MAX;
    let mut best_any_solid = false;
    for z in z0..=z1 {
        for y in y0..=y1 {
            let base = (z * in_dim + y) * in_dim;
            for x in x0..=x1 {
                let i = base + x;
                let d = densities[i];
                let m = materials[i];
                if m == 0 {
                    continue;
                }
                let abs = (d as i32).abs();
                let solid = d < 0;
                if abs < best_any_abs || (abs == best_any_abs && solid && !best_any_solid) {
                    best_any_abs = abs;
                    best_any_solid = solid;
                    best_any_m = m;
                }
            }
        }
    }
    best_any_m
}

fn sample_trilinear_density(d: &[i16], dim: usize, fx: f32, fy: f32, fz: f32) -> f32 {
    let x0 = fx.floor() as isize;
    let y0 = fy.floor() as isize;
    let z0 = fz.floor() as isize;
    let x1 = (x0 + 1).min((dim - 1) as isize);
    let y1 = (y0 + 1).min((dim - 1) as isize);
    let z1 = (z0 + 1).min((dim - 1) as isize);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let tz = fz - z0 as f32;
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
