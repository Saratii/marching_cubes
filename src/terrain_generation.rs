use bevy::{
    asset::{Assets, Handle},
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
use bevy_rapier3d::prelude::*;
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use std::collections::HashMap;

use crate::{
    conversions::chunk_coord_to_world_pos,
    marching_cubes::{HALF_CHUNK, march_cubes},
};

pub const CHUNK_SIZE: f32 = 10.0; // World size in meters
pub const VOXELS_PER_DIM: usize = 64; // Voxels per dimension per chunk (32×32×32 voxels)
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / (VOXELS_PER_DIM - 1) as f32;
pub const VOXELS_PER_CHUNK: usize =
    VOXELS_PER_DIM as usize * VOXELS_PER_DIM as usize * VOXELS_PER_DIM as usize; // Total voxels in a chunk
pub const NOISE_SEED: u32 = 100; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.02; // Frequency of the noise

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

#[derive(Resource)]
pub struct StandardTerrainMaterialHandle(pub Handle<StandardMaterial>);

#[derive(Resource)]
pub struct ChunkMap(pub HashMap<(i16, i16, i16), (Entity, TerrainChunk)>);

#[derive(Clone, Copy, Debug)]
pub struct Density {
    pub dirt: u8,
    pub grass: u8,
}

impl Density {
    pub fn sum(&self) -> f32 {
        self.dirt as f32 / 255. + self.grass as f32 / 255.
    }
}

#[derive(Component)]
pub struct TerrainChunk {
    pub densities: Box<[Density; VOXELS_PER_CHUNK]>,
    pub world_position: Vec3,
}

pub fn setup_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let standard_terrain_material_handle = materials.add(StandardMaterial { ..default() });
    let mut chunk_map = ChunkMap::new();
    let entity = chunk_map.spawn_chunk(
        &mut commands,
        &mut meshes,
        (0, 0, 0),
        &fbm,
        &standard_terrain_material_handle,
        true,
    );
    chunk_map.0.insert((0, 0, 0), entity);
    commands.insert_resource(chunk_map);
    commands.insert_resource(NoiseFunction(fbm));
    commands.insert_resource(StandardTerrainMaterialHandle(
        standard_terrain_material_handle,
    ));
}

impl ChunkMap {
    fn new() -> Self {
        Self { 0: HashMap::new() }
    }

    pub fn get_chunk_coord_from_world_pos(world_pos: Vec3) -> (i16, i16, i16) {
        let chunk_x = (world_pos.x / CHUNK_SIZE).round() as i16;
        let chunk_y = (world_pos.y / CHUNK_SIZE).round() as i16;
        let chunk_z = (world_pos.z / CHUNK_SIZE).round() as i16;
        (chunk_x, chunk_y, chunk_z)
    }

    pub fn get_chunk_center_from_coord(chunk_coord: (i16, i16, i16)) -> Vec3 {
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
        chunk_coord: (i16, i16, i16),
        fbm: &GeneratorWrapper<SafeNode>,
        standard_terrain_material_handle: &Handle<StandardMaterial>,
        needs_noise: bool,
    ) -> (Entity, TerrainChunk) {
        let chunk_center = Self::get_chunk_center_from_coord(chunk_coord);
        let terrain_chunk = TerrainChunk::new(chunk_coord, fbm, needs_noise);
        let chunk_mesh = march_cubes(&terrain_chunk.densities);
        if chunk_mesh.count_vertices() == 0 {
            //this is not ideal
            let entity = commands
                .spawn((
                    Mesh3d(meshes.add(chunk_mesh)),
                    MeshMaterial3d(standard_terrain_material_handle.clone()),
                    Transform::from_translation(chunk_center),
                    ChunkTag,
                ))
                .id();
            return (entity, terrain_chunk);
        }
        let entity = commands
            .spawn((
                Collider::from_bevy_mesh(
                    &chunk_mesh,
                    &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                )
                .unwrap(),
                Mesh3d(meshes.add(chunk_mesh)),
                MeshMaterial3d(standard_terrain_material_handle.clone()),
                Transform::from_translation(chunk_center),
                ChunkTag,
            ))
            .id();
        (entity, terrain_chunk)
    }

    pub fn dig_sphere(&mut self, center: Vec3, radius: f32, strength: f32) -> Vec<(i16, i16, i16)> {
        let mut modified_chunks = Vec::new();
        let voxel_radius = radius / VOXEL_SIZE;
        let min_world = center - Vec3::splat(radius);
        let max_world = center + Vec3::splat(radius);
        let min_chunk_x = (min_world.x / CHUNK_SIZE).floor() as i16;
        let max_chunk_x = (max_world.x / CHUNK_SIZE).ceil() as i16;
        let min_chunk_y = (min_world.y / CHUNK_SIZE).floor() as i16;
        let max_chunk_y = (max_world.y / CHUNK_SIZE).ceil() as i16;
        let min_chunk_z = (min_world.z / CHUNK_SIZE).floor() as i16;
        let max_chunk_z = (max_world.z / CHUNK_SIZE).ceil() as i16;
        for chunk_x in min_chunk_x..=max_chunk_x {
            for chunk_y in min_chunk_y..=max_chunk_y {
                for chunk_z in min_chunk_z..=max_chunk_z {
                    let chunk_coord = (chunk_x, chunk_y, chunk_z);
                    if !self.0.contains_key(&chunk_coord) {
                        continue;
                    }
                    let chunk_center = Self::get_chunk_center_from_coord(chunk_coord);
                    let chunk_modified = self.modify_chunk_voxels(
                        chunk_coord,
                        chunk_center,
                        center,
                        voxel_radius,
                        strength,
                    );
                    if chunk_modified && !modified_chunks.contains(&chunk_coord) {
                        modified_chunks.push(chunk_coord);
                    }
                }
            }
        }
        modified_chunks
    }

    fn modify_chunk_voxels(
        &mut self,
        chunk_coord: (i16, i16, i16),
        chunk_center: Vec3,
        dig_center: Vec3,
        voxel_radius: f32,
        strength: f32,
    ) -> bool {
        let mut chunk_modified = false;
        if let Some((_, chunk)) = self.0.get_mut(&chunk_coord) {
            for z in 0..VOXELS_PER_DIM {
                for y in 0..VOXELS_PER_DIM {
                    for x in 0..VOXELS_PER_DIM {
                        let world_x = chunk_center.x - HALF_CHUNK + x as f32 * VOXEL_SIZE;
                        let world_y = chunk_center.y - HALF_CHUNK + y as f32 * VOXEL_SIZE;
                        let world_z = chunk_center.z - HALF_CHUNK + z as f32 * VOXEL_SIZE;
                        let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                        let distance = voxel_world_pos.distance(dig_center);
                        if distance <= voxel_radius * VOXEL_SIZE {
                            let falloff =
                                1.0 - (distance / (voxel_radius * VOXEL_SIZE)).clamp(0.0, 1.0);
                            let dig_amount = strength * falloff;
                            let current_density =
                                chunk.get_mut_density(x as i32, y as i32, z as i32);
                            let density_sum = current_density.sum();
                            if density_sum > 0.0 {
                                let dirt_ratio = current_density.dirt as f32 / 255. / density_sum;
                                let grass_ratio = current_density.grass as f32 / 255. / density_sum;
                                current_density.dirt =
                                    ((current_density.dirt as f32 / 255. - dig_amount * dirt_ratio)
                                        .max(0.0)
                                        * 255.0) as u8;
                                current_density.grass = ((current_density.grass as f32 / 255.
                                    - dig_amount * grass_ratio)
                                    .max(0.0)
                                    * 255.0)
                                    as u8;
                                chunk_modified = true;
                            }
                        }
                    }
                }
            }
        }
        chunk_modified
    }
}

pub fn generate_densities(
    chunk_coord: &(i16, i16, i16),
    fbm: &GeneratorWrapper<SafeNode>,
    needs_noise: bool,
) -> Box<[Density; VOXELS_PER_CHUNK]> {
    if !needs_noise {
        return (vec![
            Density {
                dirt: 255,
                grass: 0,
            };
            VOXELS_PER_CHUNK
        ])
        .try_into()
        .unwrap();
    }
    let mut densities = vec![Density { dirt: 0, grass: 0 }; VOXELS_PER_CHUNK];
    let chunk_start = Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_SIZE - HALF_CHUNK,
    );
    for z in 0..VOXELS_PER_DIM {
        let world_z = chunk_start.z + z as f32 * VOXEL_SIZE;
        for x in 0..VOXELS_PER_DIM {
            let world_x = chunk_start.x + x as f32 * VOXEL_SIZE;
            let terrain_height = fbm.gen_single_2d(
                world_x * NOISE_FREQUENCY,
                world_z * NOISE_FREQUENCY,
                NOISE_SEED as i32,
            );
            for y in 0..VOXELS_PER_DIM {
                let world_y = chunk_start.y + y as f32 * VOXEL_SIZE;
                let i = z * VOXELS_PER_DIM * VOXELS_PER_DIM + y * VOXELS_PER_DIM + x;
                let distance_to_surface = terrain_height - world_y;
                let transition_width = 1.0;
                let smooth_density = if distance_to_surface <= -transition_width {
                    0.0
                } else if distance_to_surface >= transition_width {
                    1.0
                } else {
                    let t = (distance_to_surface + transition_width) / (2.0 * transition_width);
                    let t = t.clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                if smooth_density > 0.0 {
                    if distance_to_surface > 0.5 {
                        densities[i].dirt = (smooth_density * 255.0) as u8;
                        densities[i].grass = 0;
                    } else if distance_to_surface > -0.5 {
                        let grass_factor = (-distance_to_surface + 0.5).clamp(0.0, 1.0);
                        densities[i].dirt =
                            (smooth_density * (1.0 - grass_factor * 0.7) * 255.) as u8;
                        densities[i].grass = (smooth_density * grass_factor * 0.7 * 255.) as u8;
                    } else {
                        densities[i].dirt = (smooth_density * 0.1 * 255.) as u8;
                        densities[i].grass = (smooth_density * 0.9 * 255.) as u8;
                    }
                }
            }
        }
    }
    densities.try_into().unwrap()
}

#[derive(Component)]
pub struct ChunkTag;

impl TerrainChunk {
    pub fn new(
        chunk_coord: (i16, i16, i16),
        fbm: &GeneratorWrapper<SafeNode>,
        needs_noise: bool,
    ) -> Self {
        Self {
            densities: generate_densities(&chunk_coord, fbm, needs_noise),
            world_position: chunk_coord_to_world_pos(chunk_coord),
        }
    }

    pub fn set_density(&mut self, x: i32, y: i32, z: i32, density: Density) {
        let index = self.get_voxel_index(x, y, z);
        self.densities[index] = density;
    }

    fn get_voxel_index(&self, x: i32, y: i32, z: i32) -> usize {
        (z * VOXELS_PER_DIM as i32 * VOXELS_PER_DIM as i32 + y * VOXELS_PER_DIM as i32 + x) as usize
    }

    pub fn get_density(&self, x: i32, y: i32, z: i32) -> &Density {
        let index = self.get_voxel_index(x, y, z);
        &self.densities[index]
    }

    pub fn get_mut_density(&mut self, x: i32, y: i32, z: i32) -> &mut Density {
        let index = self.get_voxel_index(x, y, z);
        &mut self.densities[index]
    }

    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_density(x, y, z).sum() > 0.5
    }
}
