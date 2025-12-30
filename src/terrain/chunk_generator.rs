use bevy::math::Vec3;
use fastnoise2::{SafeNode, generator::GeneratorWrapper};

use crate::terrain::terrain::{
    CHUNK_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk, VOXEL_SIZE,
};

pub const NOISE_SEED: i32 = 100; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.02; // Frequency of the noise
pub const NOISE_AMPLITUDE: f32 = 120.0; // Amplitude of the noise
pub const HEIGHT_SCALE: f32 = 50.0; // Scale factor for heightmap
const HEIGHT_MAP_GRID_SIZE: usize = SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
const NOISE_SAMPLES_PER_SIDE: usize = 9;
const NOISE_SAMPLES_M1: usize = NOISE_SAMPLES_PER_SIDE - 1;
const SAMPLES_M1: usize = SAMPLES_PER_CHUNK_DIM - 1;

pub fn generate_densities(
    fbm: &GeneratorWrapper<SafeNode>,
    first_sample_reuse: f32,
    chunk_start: Vec3,
) -> (Box<[i16]>, Box<[u8]>, bool) {
    let mut densities = vec![0; SAMPLES_PER_CHUNK];
    let mut materials = vec![0; SAMPLES_PER_CHUNK];
    let terrain_heights = generate_terrain_heights(&chunk_start, fbm, first_sample_reuse);
    let is_uniform = fill_voxel_densities(
        &mut densities,
        &mut materials,
        &chunk_start,
        &terrain_heights,
    );
    (
        densities.try_into().unwrap(),
        materials.try_into().unwrap(),
        is_uniform,
    )
}

pub fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_SIZE - HALF_CHUNK,
    )
}

pub fn heights_contains_surface(chunk_start: &Vec3, terrain_heights: &[f32]) -> bool {
    let chunk_bottom = chunk_start.y;
    let chunk_top = chunk_start.y + CHUNK_SIZE;
    let mut min_height = f32::INFINITY;
    let mut max_height = f32::NEG_INFINITY;
    for &height in terrain_heights {
        min_height = min_height.min(height);
        max_height = max_height.max(height);
    }
    max_height >= chunk_bottom && min_height < chunk_top
}

//it may be better to store a byte signifying if a chunk contains a surface when saving to disk
pub fn chunk_contains_surface(chunk: &TerrainChunk) -> bool {
    let mut has_positive = false;
    let mut has_negative = false;
    for &density in &chunk.densities {
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

pub fn sample_fbm(fbm: &GeneratorWrapper<SafeNode>, x: f32, z: f32) -> f32 {
    // Base terrain - this creates the actual height variation
    let base = fbm.gen_single_2d(x * 0.0001, z * 0.001, NOISE_SEED);
    let detail = fbm.gen_single_2d(x * 0.003, z * 0.003, NOISE_SEED + 1);
    let fine = fbm.gen_single_2d(x * 0.008, z * 0.008, NOISE_SEED + 2);

    // Combine with decreasing weights
    let height = base * 100.0 + detail * 40.0 + fine * 15.0;

    // Continentalness for biome variation
    let cont = fbm.gen_single_2d(x * 0.0003, z * 0.0003, NOISE_SEED + 3) * 0.5 + 0.5;

    // Mountain modifier - makes some areas extra tall
    let mountain_mask = ((cont - 0.4) / 0.3).clamp(0.0, 1.0);
    let mountain_boost = mountain_mask * mountain_mask * 150.0;

    height + mountain_boost
}

//Sample the fbm noise at a higher resolution and then bilinearly interpolate to get smooth terrain heights
pub fn generate_terrain_heights(
    chunk_start: &Vec3,
    fbm: &GeneratorWrapper<SafeNode>,
    first_sample_reuse: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; HEIGHT_MAP_GRID_SIZE];
    let mut noise_samples = vec![0.0; NOISE_SAMPLES_PER_SIDE * NOISE_SAMPLES_PER_SIDE];
    noise_samples[0] = first_sample_reuse;
    for nz in 0..NOISE_SAMPLES_PER_SIDE {
        let tz = nz as f32 / NOISE_SAMPLES_M1 as f32;
        let wz = chunk_start.z + tz * CHUNK_SIZE;
        for nx in 0..NOISE_SAMPLES_PER_SIDE {
            if nz == 0 && nx == 0 {
                continue;
            }
            let tx = nx as f32 / NOISE_SAMPLES_M1 as f32;
            let wx = chunk_start.x + tx * CHUNK_SIZE;
            noise_samples[nz * NOISE_SAMPLES_PER_SIDE + nx] = sample_fbm(fbm, wx, wz);
        }
    }
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let tz = z as f32 / SAMPLES_M1 as f32;
        let fz = tz * NOISE_SAMPLES_M1 as f32;
        let nz0 = fz.floor() as usize;
        let nz1 = (nz0 + 1).min(NOISE_SAMPLES_M1);
        let tz_frac = fz - nz0 as f32;
        let omtz = 1.0 - tz_frac;
        let base = z * SAMPLES_PER_CHUNK_DIM;
        let nz0_base = nz0 * NOISE_SAMPLES_PER_SIDE;
        let nz1_base = nz1 * NOISE_SAMPLES_PER_SIDE;
        for x in 0..SAMPLES_PER_CHUNK_DIM {
            let tx = x as f32 / SAMPLES_M1 as f32;
            let fx = tx * NOISE_SAMPLES_M1 as f32;
            let nx0 = fx.floor() as usize;
            let nx1 = (nx0 + 1).min(NOISE_SAMPLES_M1);
            let tx_frac = fx - nx0 as f32;
            let omtx = 1.0 - tx_frac;
            let s00 = noise_samples[nz0_base + nx0];
            let s10 = noise_samples[nz0_base + nx1];
            let s01 = noise_samples[nz1_base + nx0];
            let s11 = noise_samples[nz1_base + nx1];
            let s0 = s00 * omtx + s10 * tx_frac;
            let s1 = s01 * omtx + s11 * tx_frac;
            out[base + x] = s0 * omtz + s1 * tz_frac;
        }
    }
    out
}

fn fill_voxel_densities(
    densities: &mut [i16],
    materials: &mut [u8],
    chunk_start: &Vec3,
    terrain_heights: &[f32],
) -> bool {
    let solid_threshold = quantize_f32_to_i16(-1.0);
    let mut is_uniform = true;
    let mut init_distance = 0i16;
    let mut init_material = 0u8;
    let mut has_init = false;
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let height_base = z * SAMPLES_PER_CHUNK_DIM;
        let z_base = z * HEIGHT_MAP_GRID_SIZE;
        for y in 0..SAMPLES_PER_CHUNK_DIM {
            let world_y = chunk_start.y + y as f32 * VOXEL_SIZE;
            let below_sea = world_y < 0.0;
            let mut rolling_voxel_index = z_base + y * SAMPLES_PER_CHUNK_DIM;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
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
                rolling_voxel_index += 1;
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
