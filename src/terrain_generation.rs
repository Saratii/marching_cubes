use std::collections::HashMap;

use bevy::{
    asset::Assets,
    color::Color,
    ecs::{
        component::Component,
        entity::Entity,
        resource::Resource,
        system::{Commands, ResMut},
    },
    math::Vec3,
    pbr::{MeshMaterial3d, StandardMaterial},
    render::mesh::{Mesh, Mesh3d},
    transform::components::Transform,
    utils::default,
};

use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};

use crate::{
    conversions::chunk_coord_to_world_pos,
    marching_cubes::{HALF_CHUNK, add_triangle_colors, march_cubes},
};

pub const CHUNK_SIZE: f32 = 10.0; // World size in units (8×8×8 world units)
pub const VOXELS_PER_DIM: usize = 32; // Voxels per dimension per chunk (32×32×32 voxels)
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / (VOXELS_PER_DIM - 1) as f32;
const VOXELS_PER_CHUNK: usize =
    VOXELS_PER_DIM as usize * VOXELS_PER_DIM as usize * VOXELS_PER_DIM as usize; // Total voxels in a chunk
const NOISE_SEED: u32 = 100; // Seed for noise generation
const NOISE_FREQUENCY: f64 = 0.02; // Frequency of the noise

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

pub fn setup_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut chunk_map = ChunkMap::new();
    let entity = chunk_map.spawn_chunk(&mut commands, &mut meshes, &mut materials, (0, 0, 0), &fbm);
    chunk_map.0.insert((0, 0, 0), entity);
    commands.insert_resource(chunk_map);
    commands.insert_resource(NoiseFunction(fbm));
}

#[derive(Resource)]
pub struct ChunkMap(pub HashMap<(i32, i32, i32), (Entity, TerrainChunk)>);

impl ChunkMap {
    fn new() -> Self {
        Self { 0: HashMap::new() }
    }

    pub fn get_chunk_coord_from_world_pos(world_pos: Vec3) -> (i32, i32, i32) {
        let chunk_x = (world_pos.x / CHUNK_SIZE).round() as i32;
        let chunk_y = (world_pos.y / CHUNK_SIZE).round() as i32;
        let chunk_z = (world_pos.z / CHUNK_SIZE).round() as i32;
        (chunk_x, chunk_y, chunk_z)
    }

    pub fn get_chunk_center_from_coord(chunk_coord: (i32, i32, i32)) -> Vec3 {
        Vec3::new(
            chunk_coord.0 as f32 * CHUNK_SIZE,
            chunk_coord.1 as f32 * CHUNK_SIZE,
            chunk_coord.2 as f32 * CHUNK_SIZE,
        )
    }

    pub fn spawn_chunk(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        chunk_coord: (i32, i32, i32),
        fbm: &GeneratorWrapper<SafeNode>,
    ) -> (Entity, TerrainChunk) {
        let chunk_center = Self::get_chunk_center_from_coord(chunk_coord);
        let terrain_chunk = TerrainChunk::new(chunk_coord, fbm);
        let chunk_mesh = add_triangle_colors(march_cubes(&terrain_chunk.densities));
        let entity = commands
            .spawn((
                Mesh3d(meshes.add(chunk_mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    cull_mode: None,
                    ..default()
                })),
                Transform::from_translation(chunk_center),
                ChunkTag,
            ))
            .id();
        (entity, terrain_chunk)
    }
}

pub fn generate_densities(
    chunk_coord: &(i32, i32, i32),
    fbm: &GeneratorWrapper<SafeNode>,
) -> Vec<f32> {
    let mut densities = vec![0.0; VOXELS_PER_CHUNK];
    let chunk_center = Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE,
        chunk_coord.1 as f32 * CHUNK_SIZE,
        chunk_coord.2 as f32 * CHUNK_SIZE,
    );
    for z in 0..VOXELS_PER_DIM {
        for y in 0..VOXELS_PER_DIM {
            for x in 0..VOXELS_PER_DIM {
                let world_x = chunk_center.x - HALF_CHUNK + x as f32 * VOXEL_SIZE;
                let world_y = chunk_center.y - HALF_CHUNK + y as f32 * VOXEL_SIZE;
                let world_z = chunk_center.z - HALF_CHUNK + z as f32 * VOXEL_SIZE;
                let terrain_height = fbm.gen_single_2d(
                    world_x * NOISE_FREQUENCY as f32,
                    world_z * NOISE_FREQUENCY as f32,
                    NOISE_SEED as i32,
                );

                let i = z * VOXELS_PER_DIM * VOXELS_PER_DIM + y * VOXELS_PER_DIM + x;
                densities[i] = terrain_height - world_y;
            }
        }
    }
    densities
}

#[derive(Component)]
pub struct TerrainChunk {
    pub densities: Vec<f32>,
    pub chunk_coord: (i32, i32, i32),
    pub world_position: Vec3,
}

#[derive(Component)]
pub struct ChunkTag;

impl TerrainChunk {
    pub fn new(chunk_coord: (i32, i32, i32), fbm: &GeneratorWrapper<SafeNode>) -> Self {
        Self {
            densities: generate_densities(&chunk_coord, fbm),
            chunk_coord,
            world_position: chunk_coord_to_world_pos(chunk_coord),
        }
    }

    pub fn set_density(&mut self, x: i32, y: i32, z: i32, density: f32) {
        let index = self.get_voxel_index(x, y, z);
        self.densities[index] = density;
    }

    fn get_voxel_index(&self, x: i32, y: i32, z: i32) -> usize {
        (z * VOXELS_PER_DIM as i32 * VOXELS_PER_DIM as i32 + y * VOXELS_PER_DIM as i32 + x) as usize
    }

    pub fn get_density(&self, x: i32, y: i32, z: i32) -> f32 {
        let index = self.get_voxel_index(x, y, z);
        self.densities[index]
    }

    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_density(x, y, z) > 0.
    }

    pub fn dig_sphere(&mut self, center: Vec3, radius: f32, strength: f32) {
        let center_voxel = Vec3::new(
            (center.x + HALF_CHUNK) / VOXEL_SIZE,
            (center.y + HALF_CHUNK) / VOXEL_SIZE,
            (center.z + HALF_CHUNK) / VOXEL_SIZE,
        );
        let voxel_radius = radius / VOXEL_SIZE;
        let min_x = ((center_voxel.x - voxel_radius).floor() as i32).max(0);
        let max_x = ((center_voxel.x + voxel_radius).ceil() as i32).min(VOXELS_PER_DIM as i32 - 1);
        let min_y = ((center_voxel.y - voxel_radius).floor() as i32).max(0);
        let max_y = ((center_voxel.y + voxel_radius).ceil() as i32).min(VOXELS_PER_DIM as i32 - 1);
        let min_z = ((center_voxel.z - voxel_radius).floor() as i32).max(0);
        let max_z = ((center_voxel.z + voxel_radius).ceil() as i32).min(VOXELS_PER_DIM as i32 - 1);
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    let voxel_pos = Vec3::new(x as f32, y as f32, z as f32);
                    let distance = voxel_pos.distance(center_voxel);
                    if distance <= voxel_radius {
                        let falloff = 1.0 - (distance / voxel_radius).clamp(0.0, 1.0);
                        let dig_amount = strength * falloff;
                        let current_density = self.get_density(x, y, z);
                        let new_density = current_density - dig_amount;
                        self.set_density(x, y as i32, z as i32, new_density);
                    }
                }
            }
        }
    }
}
