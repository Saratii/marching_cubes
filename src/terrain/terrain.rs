use std::collections::HashMap;

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
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};

use crate::{
    conversions::chunk_coord_to_world_pos, marching_cubes::march_cubes,
    terrain::chunk_generator::generate_densities,
};

pub const VOXELS_PER_CHUNK_DIM: usize = 16; // Number of voxel sample points
pub const VOXEL_SIZE: f32 = 0.125; // Size of each voxel in meters
pub const CUBES_PER_CHUNK_DIM: usize = VOXELS_PER_CHUNK_DIM - 1; // 63 cubes
pub const CHUNK_SIZE: f32 = CUBES_PER_CHUNK_DIM as f32 * VOXEL_SIZE; // 7.875 meters
pub const VOXELS_PER_CHUNK: usize =
    VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const CHUNK_CREATION_RADIUS: i16 = 20; //in world units
pub const CHUNK_GENERATION_CIRCULAR_RADIUS_SQUARED: f32 =
    (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE) * (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE);

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

#[derive(Resource)]
pub struct StandardTerrainMaterialHandle(pub Handle<StandardMaterial>);

#[derive(Clone, Copy, Debug)]
pub struct VoxelData {
    pub sdf: f32,
    pub material: u8,
}

#[derive(Component)]
pub struct TerrainChunk {
    pub densities: Box<[VoxelData; VOXELS_PER_CHUNK]>,
    pub world_position: Vec3,
}

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

    pub fn set_density(&mut self, x: u32, y: u32, z: u32, density: VoxelData) {
        let index = self.get_voxel_index(x, y, z);
        self.densities[index as usize] = density;
    }

    fn get_voxel_index(&self, x: u32, y: u32, z: u32) -> u32 {
        z * VOXELS_PER_CHUNK_DIM as u32 * VOXELS_PER_CHUNK_DIM as u32
            + y * VOXELS_PER_CHUNK_DIM as u32
            + x
    }

    pub fn get_density(&self, x: u32, y: u32, z: u32) -> &VoxelData {
        let index = self.get_voxel_index(x, y, z);
        &self.densities[index as usize]
    }

    pub fn get_mut_density(&mut self, x: u32, y: u32, z: u32) -> &mut VoxelData {
        let index = self.get_voxel_index(x, y, z);
        &mut self.densities[index as usize]
    }

    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_density(x, y, z).sdf > 0.0
    }
}

#[derive(Resource)]
pub struct ChunkMap(pub HashMap<(i16, i16, i16), (Entity, TerrainChunk)>);

impl ChunkMap {
    fn new() -> Self {
        Self { 0: HashMap::new() }
    }

    pub fn spawn_chunk(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        standard_terrain_material_handle: Handle<StandardMaterial>,
        terrain_chunk: TerrainChunk,
        mesh: Mesh,
        transform: Transform,
        collider: Option<Collider>,
    ) -> (Entity, TerrainChunk) {
        let bundle = (
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(standard_terrain_material_handle),
            ChunkTag,
            transform,
        );
        let entity = match collider {
            Some(collider) => commands.spawn((bundle, collider)).id(),
            None => commands.spawn(bundle).id(),
        };
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
                    let chunk_center = chunk_coord_to_world_pos(chunk_coord);
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
            for z in 0..VOXELS_PER_CHUNK_DIM {
                for y in 0..VOXELS_PER_CHUNK_DIM {
                    for x in 0..VOXELS_PER_CHUNK_DIM {
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
                                chunk.get_mut_density(x as u32, y as u32, z as u32);
                            let density_sum = current_density.sdf;
                            if density_sum > 0.0 {
                                current_density.sdf -= dig_amount;
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

pub fn setup_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let standard_terrain_material_handle = materials.add(StandardMaterial { ..default() });
    let mut chunk_map = ChunkMap::new();
    let terrain_chunk = TerrainChunk::new((0, 0, 0), &fbm, true);
    let mesh = march_cubes(&terrain_chunk.densities);
    let collider = Collider::from_bevy_mesh(
        &mesh,
        &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
    );
    let entity = chunk_map.spawn_chunk(
        &mut commands,
        &mut meshes,
        standard_terrain_material_handle.clone(),
        terrain_chunk,
        mesh,
        Transform::from_translation(Vec3::ZERO),
        collider,
    );
    chunk_map.0.insert((0, 0, 0), entity);
    commands.insert_resource(chunk_map);
    commands.insert_resource(NoiseFunction(fbm));
    commands.insert_resource(StandardTerrainMaterialHandle(
        standard_terrain_material_handle,
    ));
}
