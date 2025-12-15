use bevy::{ecs::message::Message, math::Vec3};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};

use crate::terrain::terrain::{
    CHUNK_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk, VOXEL_SIZE,
};

pub const NOISE_SEED: i32 = 100; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.02; // Frequency of the noise
pub const NOISE_AMPLITUDE: f32 = 120.0; // Amplitude of the noise
pub const HEIGHT_SCALE: f32 = 50.0; // Scale factor for heightmap

#[derive(Message)]
pub struct GenerateChunkEvent {
    pub chunk_coords: Vec<(i16, i16, i16)>,
}

#[derive(Message)]
pub struct LoadChunksEvent {
    pub chunk_coords: Vec<(i16, i16, i16)>,
}

pub fn generate_densities(
    chunk_coord: &(i16, i16, i16),
    fbm: &GeneratorWrapper<SafeNode>,
) -> (Box<[i16]>, Box<[u8]>, bool) {
    let mut densities = vec![0; SAMPLES_PER_CHUNK];
    let mut materials = vec![0; SAMPLES_PER_CHUNK];
    let chunk_start = calculate_chunk_start(chunk_coord);
    let terrain_heights = generate_terrain_heights(&chunk_start, fbm);
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

fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
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

fn generate_terrain_heights(chunk_start: &Vec3, fbm: &GeneratorWrapper<SafeNode>) -> Vec<f32> {
    let mut out = vec![0.0; SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM];
    let stride = 4;
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let wz = chunk_start.z + z as f32 * VOXEL_SIZE;
        let base = z * SAMPLES_PER_CHUNK_DIM;
        for x in 0..SAMPLES_PER_CHUNK_DIM {
            if x % stride == 0
                || z % stride == 0
                || x == 0
                || x == SAMPLES_PER_CHUNK_DIM - 1
                || z == 0
                || z == SAMPLES_PER_CHUNK_DIM - 1
            {
                let wx = chunk_start.x + x as f32 * VOXEL_SIZE;
                out[base + x] = sample_fbm(fbm, wx, wz);
            }
        }
    }
    for z in 1..SAMPLES_PER_CHUNK_DIM - 1 {
        let base = z * SAMPLES_PER_CHUNK_DIM;
        for x in 1..SAMPLES_PER_CHUNK_DIM - 1 {
            if x % stride != 0 && z % stride != 0 {
                let sx = (x / stride) * stride;
                let sz = (z / stride) * stride;
                let sx1 = (sx + stride).min(SAMPLES_PER_CHUNK_DIM - 1);
                let sz1 = (sz + stride).min(SAMPLES_PER_CHUNK_DIM - 1);
                let tx = (x - sx) as f32 / stride as f32;
                let tz = (z - sz) as f32 / stride as f32;
                let s00 = out[sz * SAMPLES_PER_CHUNK_DIM + sx];
                let s10 = out[sz * SAMPLES_PER_CHUNK_DIM + sx1];
                let s01 = out[sz1 * SAMPLES_PER_CHUNK_DIM + sx];
                let s11 = out[sz1 * SAMPLES_PER_CHUNK_DIM + sx1];
                let s0 = s00 * (1.0 - tx) + s10 * tx;
                let s1 = s01 * (1.0 - tx) + s11 * tx;
                out[base + x] = s0 * (1.0 - tz) + s1 * tz;
            }
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
    let mut is_uniform = true;
    let mut initial: Option<(i16, u8)> = None;
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let height_base = z * SAMPLES_PER_CHUNK_DIM;
        for y in 0..SAMPLES_PER_CHUNK_DIM {
            let world_y = chunk_start.y + y as f32 * VOXEL_SIZE;
            let index_base =
                z * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM + y * SAMPLES_PER_CHUNK_DIM;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let terrain_height = terrain_heights[height_base + x];
                let voxel_index = index_base + x;
                let distance_to_surface =
                    quantize_f32_to_i16((world_y - terrain_height).clamp(-10.0, 10.0));
                densities[voxel_index] = distance_to_surface;
                if distance_to_surface < 0 {
                    if distance_to_surface < quantize_f32_to_i16(-1.0) {
                        materials[voxel_index] = 1; //dirt
                    } else {
                        let global_y = chunk_start.y + y as f32 * VOXEL_SIZE;
                        if global_y < 0.0 {
                            //sea level
                            materials[voxel_index] = 3; //sand
                        } else {
                            materials[voxel_index] = 2; //grass
                        }
                    }
                } else {
                    materials[voxel_index] = 0; //air
                }
                if let Some((init_distance, init_mat)) = initial {
                    if init_distance != distance_to_surface || init_mat != materials[voxel_index] {
                        {
                            is_uniform = false;
                        }
                    }
                } else {
                    initial = Some((distance_to_surface, materials[voxel_index]));
                }
            }
        }
    }
    is_uniform
}

#[inline]
pub fn quantize_f32_to_i16(value: f32) -> i16 {
    let scale = 32767.0 / 10.0; // Map [-10, 10] to [-32767, 32767]
    (value * scale).round() as i16
}

#[inline]
pub fn dequantize_i16_to_f32(q: i16) -> f32 {
    let scale = 32767.0 / 10.0;
    q as f32 / scale
}
