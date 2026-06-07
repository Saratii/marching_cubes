use crate::constants::{
    CHUNK_WORLD_SIZE, SAMPLES_PER_CHUNK_DIM, SAMPLES_PER_CHUNK_DIM_PADDED, VOXEL_WORLD_SIZE,
    WORLD_SEED,
};
use crate::terrain::chunk_generator::{MaterialCode, quantize_f32_to_i16};
use bevy::math::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

//in world units
const TREE_HEIGHT_MIN: f32 = 8.0;
const TREE_HEIGHT_MAX: f32 = 40.0;
const TREE_RADIUS_MIN: f32 = 0.4;
const TREE_RADIUS_MAX: f32 = 2.0;
const TREE_COUNT_MIN: usize = 0;
const TREE_COUNT_MAX: usize = 3;

struct Tree {
    root: Vec3,
    height: f32,
    radius: f32,
}

//carves a single tree cylinder into the density and material buffers for the current chunk
fn apply_tree_to_chunk(
    tree: &Tree,
    chunk_world_min: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
) {
    let half_h = tree.height * 0.5;
    let cy_center = tree.root.y + half_h;
    for z_pad in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
        let world_z = chunk_world_min.z + (z_pad as f32 - 1.0) * VOXEL_WORLD_SIZE;
        let dz = world_z - tree.root.z;
        for x_pad in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
            let world_x = chunk_world_min.x + (x_pad as f32 - 1.0) * VOXEL_WORLD_SIZE;
            let dx = world_x - tree.root.x;
            let radial_dist = (dx * dx + dz * dz).sqrt() - tree.radius;
            for y_pad in 0..SAMPLES_PER_CHUNK_DIM_PADDED {
                let world_y = chunk_world_min.y + (y_pad as f32 - 1.0) * VOXEL_WORLD_SIZE;
                let cy = world_y - cy_center;
                let dist = radial_dist.max(cy.abs() - half_h);
                let density_idx =
                    z_pad * SAMPLES_PER_CHUNK_DIM_PADDED * SAMPLES_PER_CHUNK_DIM_PADDED
                        + y_pad * SAMPLES_PER_CHUNK_DIM_PADDED
                        + x_pad;
                let new_density = quantize_f32_to_i16(dist);
                if new_density < density_buffer[density_idx] {
                    density_buffer[density_idx] = new_density;
                }
                let interior = x_pad >= 1
                    && x_pad <= SAMPLES_PER_CHUNK_DIM
                    && y_pad >= 1
                    && y_pad <= SAMPLES_PER_CHUNK_DIM
                    && z_pad >= 1
                    && z_pad <= SAMPLES_PER_CHUNK_DIM;
                if interior && dist < 0.0 {
                    let mat_idx = (z_pad - 1) * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM
                        + (y_pad - 1) * SAMPLES_PER_CHUNK_DIM
                        + (x_pad - 1);
                    material_buffer[mat_idx] = MaterialCode::Dirt;
                }
            }
        }
    }
}

//entry point for uniform air chunks — initializes buffers and applies tree only if it intersects
//returns true if the chunk was modified (is no longer uniform air)
pub fn place_trees_uniform_air(
    chunk_start: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
    heightmap_buffer: &[f32],
) -> bool {
    let trees = trees_for_chunk(chunk_start, heightmap_buffer);
    let chunk_world_max_y = chunk_start.y + CHUNK_WORLD_SIZE;
    let any_intersects = trees.iter().any(|tree| {
        let trunk_top_y = tree.root.y + tree.height;
        trunk_top_y >= chunk_start.y && tree.root.y <= chunk_world_max_y
    });
    if !any_intersects {
        return false;
    }
    let air_density = quantize_f32_to_i16(10.0);
    density_buffer.fill(air_density);
    material_buffer.fill(MaterialCode::Air);
    for tree in &trees {
        apply_tree_to_chunk(tree, chunk_start, density_buffer, material_buffer);
    }
    true
}

//entry point — generates a tree fully within the chunk and applies it to the chunk buffers
pub fn place_trees(
    chunk_start: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [MaterialCode],
    heightmap_buffer: &[f32],
) {
    for tree in trees_for_chunk(chunk_start, heightmap_buffer) {
        apply_tree_to_chunk(&tree, chunk_start, density_buffer, material_buffer);
    }
}

fn trees_for_chunk(chunk_start: Vec3, heightmap_buffer: &[f32]) -> Vec<Tree> {
    let cx = (chunk_start.x / CHUNK_WORLD_SIZE) as i64;
    let cz = (chunk_start.z / CHUNK_WORLD_SIZE) as i64;
    let chunk_seed = (WORLD_SEED as i64)
        .wrapping_add(cx.wrapping_mul(2654435761))
        .wrapping_add(cz.wrapping_mul(805459861));
    let mut rng = SmallRng::seed_from_u64(chunk_seed as u64);
    let count = rng.random_range(TREE_COUNT_MIN..=TREE_COUNT_MAX);
    let mut trees = Vec::with_capacity(count);
    for _ in 0..count {
        let local_x = rng.random_range(0.0..SAMPLES_PER_CHUNK_DIM as f32);
        let local_z = rng.random_range(0.0..SAMPLES_PER_CHUNK_DIM as f32);
        let heightmap_x = (local_x as usize + 1).min(SAMPLES_PER_CHUNK_DIM_PADDED - 1);
        let heightmap_z = (local_z as usize + 1).min(SAMPLES_PER_CHUNK_DIM_PADDED - 1);
        let surface_y = heightmap_buffer[heightmap_z * SAMPLES_PER_CHUNK_DIM_PADDED + heightmap_x];
        let size = rng.random_range(0.0f32..1.0);
        let height = TREE_HEIGHT_MIN + size * (TREE_HEIGHT_MAX - TREE_HEIGHT_MIN);
        let radius_jitter = rng.random_range(-0.15..0.15);
        let radius_t = (size + radius_jitter).clamp(0.0, 1.0);
        let radius = TREE_RADIUS_MIN + radius_t * (TREE_RADIUS_MAX - TREE_RADIUS_MIN);
        let lo = radius;
        let hi = CHUNK_WORLD_SIZE - radius;
        if lo >= hi {
            continue;
        }
        let root_x = chunk_start.x + (local_x * VOXEL_WORLD_SIZE).clamp(lo, hi);
        let root_z = chunk_start.z + (local_z * VOXEL_WORLD_SIZE).clamp(lo, hi);
        let overlaps = trees.iter().any(|t: &Tree| {
            let dx = t.root.x - root_x;
            let dz = t.root.z - root_z;
            (dx * dx + dz * dz).sqrt() < t.radius + radius
        });
        if overlaps {
            continue;
        }
        trees.push(Tree {
            root: Vec3::new(root_x, surface_y, root_z),
            height,
            radius,
        });
    }
    trees
}
