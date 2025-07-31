use bevy::{ecs::event::Event, math::Vec3};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};

use crate::terrain::terrain::{
    CHUNK_SIZE, Density, HALF_CHUNK, VOXEL_SIZE, VOXELS_PER_CHUNK, VOXELS_PER_DIM,
};

pub const NOISE_SEED: u32 = 100; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.02; // Frequency of the noise

#[derive(Event)]
pub struct GenerateChunkEvent {
    pub chunk_data: Vec<((i16, i16, i16), bool)>,
}

pub fn generate_densities(
    chunk_coord: &(i16, i16, i16),
    fbm: &GeneratorWrapper<SafeNode>,
    needs_noise: bool,
) -> Box<[Density; VOXELS_PER_CHUNK]> {
    if !needs_noise {
        return vec![
            Density {
                dirt: 255,
                grass: 0
            };
            VOXELS_PER_CHUNK
        ]
        .try_into()
        .unwrap();
    }
    let mut densities = vec![Density { dirt: 0, grass: 0 }; VOXELS_PER_CHUNK];
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
    let mut terrain_heights = vec![0.0f32; VOXELS_PER_DIM * VOXELS_PER_DIM];
    for z in 0..VOXELS_PER_DIM {
        let world_z = chunk_start.z + z as f32 * VOXEL_SIZE;
        let height_base = z * VOXELS_PER_DIM;
        for x in 0..VOXELS_PER_DIM {
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

fn fill_voxel_densities(densities: &mut [Density], chunk_start: &Vec3, terrain_heights: &[f32]) {
    for z in 0..VOXELS_PER_DIM {
        let height_base = z * VOXELS_PER_DIM;
        for y in 0..VOXELS_PER_DIM {
            let world_y = chunk_start.y + y as f32 * VOXEL_SIZE;
            let index_base = z * VOXELS_PER_DIM * VOXELS_PER_DIM + y * VOXELS_PER_DIM;
            for x in 0..VOXELS_PER_DIM {
                let terrain_height = terrain_heights[height_base + x];
                let voxel_index = index_base + x;
                let distance_to_surface = terrain_height - world_y;
                let smooth_density = calculate_smooth_density(distance_to_surface);
                if smooth_density > 0.0 {
                    let base_dirt = smooth_density * 255.0;
                    densities[voxel_index] =
                        calculate_material_densities(base_dirt, distance_to_surface);
                }
            }
        }
    }
}

fn calculate_smooth_density(distance_to_surface: f32) -> f32 {
    const TRANSITION_WIDTH: f32 = 1.0;
    const INV_DOUBLE_TRANSITION: f32 = 0.5;
    if distance_to_surface <= -TRANSITION_WIDTH {
        0.0
    } else if distance_to_surface >= TRANSITION_WIDTH {
        1.0
    } else {
        let smoothstep_param = (distance_to_surface + TRANSITION_WIDTH) * INV_DOUBLE_TRANSITION;
        smoothstep_param * smoothstep_param * (3.0 - 2.0 * smoothstep_param)
    }
}

fn calculate_material_densities(base_dirt: f32, distance_to_surface: f32) -> Density {
    const GRASS_FACTOR: f32 = 0.7;
    const DIRT_FACTOR: f32 = 0.1;
    const GRASS_DOMINANT: f32 = 0.9;
    const SURFACE_THRESHOLD: f32 = 0.5;
    const SUBSURFACE_THRESHOLD: f32 = -0.5;
    if distance_to_surface > SURFACE_THRESHOLD {
        Density {
            dirt: base_dirt as u8,
            grass: 0,
        }
    } else if distance_to_surface > SUBSURFACE_THRESHOLD {
        let grass_factor = -distance_to_surface + SURFACE_THRESHOLD;
        let grass_amount = grass_factor * GRASS_FACTOR;
        Density {
            dirt: (base_dirt * (1.0 - grass_amount)) as u8,
            grass: (base_dirt * grass_amount) as u8,
        }
    } else {
        Density {
            dirt: (base_dirt * DIRT_FACTOR) as u8,
            grass: (base_dirt * GRASS_DOMINANT) as u8,
        }
    }
}
