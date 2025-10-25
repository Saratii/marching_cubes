use bevy::{ecs::event::Event, math::Vec3};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};

use crate::terrain::terrain::{
    CHUNK_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk, VOXEL_SIZE,
};

pub const NOISE_SEED: u32 = 100; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.02; // Frequency of the noise

#[derive(Event)]
pub struct GenerateChunkEvent {
    pub chunk_coords: Vec<(i16, i16, i16)>,
}

#[derive(Event)]
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
    let contains_surface = heights_contains_surface(&chunk_start, &terrain_heights);
    fill_voxel_densities(
        &mut densities,
        &mut materials,
        &chunk_start,
        &terrain_heights,
    );
    (
        densities.try_into().unwrap(),
        materials.try_into().unwrap(),
        contains_surface,
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

fn generate_terrain_heights(chunk_start: &Vec3, fbm: &GeneratorWrapper<SafeNode>) -> Vec<f32> {
    let mut terrain_heights = vec![0.0f32; SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM];
    for z in 0..SAMPLES_PER_CHUNK_DIM {
        let world_z = chunk_start.z + z as f32 * VOXEL_SIZE;
        let height_base = z * SAMPLES_PER_CHUNK_DIM;
        for x in 0..SAMPLES_PER_CHUNK_DIM {
            let world_x = chunk_start.x + x as f32 * VOXEL_SIZE;
            terrain_heights[height_base + x] = fbm.gen_single_2d(
                world_x * NOISE_FREQUENCY,
                world_z * NOISE_FREQUENCY,
                NOISE_SEED as i32,
            );
        }
    }
    terrain_heights
}

fn fill_voxel_densities(
    densities: &mut [i16],
    materials: &mut [u8],
    chunk_start: &Vec3,
    terrain_heights: &[f32],
) {
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
                    if distance_to_surface < quantize_f32_to_i16(-VOXEL_SIZE * 2.0) {
                        materials[voxel_index] = 1;
                    } else {
                        materials[voxel_index] = 2;
                    }
                } else {
                    materials[voxel_index] = 0;
                }
            }
        }
    }
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
