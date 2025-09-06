use bevy::{ecs::event::Event, math::Vec3};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};

use crate::terrain::terrain::{
    CHUNK_SIZE, HALF_CHUNK, VOXEL_SIZE, VOXELS_PER_CHUNK, VOXELS_PER_CHUNK_DIM, VoxelData,
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
) -> Box<[VoxelData; VOXELS_PER_CHUNK]> {
    let mut densities = vec![
        VoxelData {
            sdf: 0.0,
            material: 255
        };
        VOXELS_PER_CHUNK
    ];
    let chunk_start = calculate_chunk_start(chunk_coord);
    let terrain_heights = generate_terrain_heights(&chunk_start, fbm);
    fill_voxel_densities(&mut densities, &chunk_start, &terrain_heights);
    densities.try_into().unwrap()
}

fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_SIZE - HALF_CHUNK,
    )
}

fn generate_terrain_heights(chunk_start: &Vec3, fbm: &GeneratorWrapper<SafeNode>) -> Vec<f32> {
    let mut terrain_heights = vec![0.0f32; VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM];
    for z in 0..VOXELS_PER_CHUNK_DIM {
        let world_z = chunk_start.z + z as f32 * VOXEL_SIZE;
        let height_base = z * VOXELS_PER_CHUNK_DIM;
        for x in 0..VOXELS_PER_CHUNK_DIM {
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

fn fill_voxel_densities(densities: &mut [VoxelData], chunk_start: &Vec3, terrain_heights: &[f32]) {
    for z in 0..VOXELS_PER_CHUNK_DIM {
        let height_base = z * VOXELS_PER_CHUNK_DIM;
        for y in 0..VOXELS_PER_CHUNK_DIM {
            let world_y = chunk_start.y + y as f32 * VOXEL_SIZE;
            let index_base =
                z * VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM + y * VOXELS_PER_CHUNK_DIM;
            for x in 0..VOXELS_PER_CHUNK_DIM {
                let terrain_height = terrain_heights[height_base + x];
                let voxel_index = index_base + x;
                let distance_to_surface = terrain_height - world_y;
                if distance_to_surface < 0.0 {
                    densities[voxel_index] = VoxelData {
                        sdf: distance_to_surface,
                        material: 0,
                    };
                } else if distance_to_surface < VOXEL_SIZE * 2.0 {
                    densities[voxel_index] = VoxelData {
                        sdf: distance_to_surface,
                        material: 2,
                    };
                } else {
                    densities[voxel_index] = VoxelData {
                        sdf: distance_to_surface,
                        material: 1,
                    };
                }
            }
        }
    }
}
