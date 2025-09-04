use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use serde::{Deserialize, Serialize};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::chunk_loader::{
        ChunkDataFile, ChunkIndexFile, ChunkIndexMap, create_chunk_file_data, load_chunk_data,
    },
    marching_cubes::march_cubes,
    player::player::PLAYER_SPAWN,
    terrain::chunk_generator::generate_densities,
};

pub const VOXELS_PER_CHUNK_DIM: usize = 32; // Number of voxel sample points
pub const VOXEL_SIZE: f32 = 0.1; // Size of each voxel in meters
pub const CUBES_PER_CHUNK_DIM: usize = VOXELS_PER_CHUNK_DIM - 1; // 63 cubes
pub const CHUNK_SIZE: f32 = CUBES_PER_CHUNK_DIM as f32 * VOXEL_SIZE; // 7.875 meters
pub const VOXELS_PER_CHUNK: usize =
    VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM * VOXELS_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const CHUNK_CREATION_RADIUS: f32 = 50.0; //in world units
pub const CHUNK_CREATION_RADIUS_SQUARED: f32 = CHUNK_CREATION_RADIUS * CHUNK_CREATION_RADIUS;

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

#[derive(Resource)]
pub struct StandardTerrainMaterialHandle(pub Handle<StandardMaterial>);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VoxelData {
    pub sdf: f32,
    pub material: u8,
}

#[derive(Component, Serialize, Deserialize)]
pub struct TerrainChunk {
    pub densities: Box<[VoxelData]>,
}

impl TerrainChunk {
    pub fn new(chunk_coord: (i16, i16, i16), fbm: &GeneratorWrapper<SafeNode>) -> Self {
        Self {
            densities: generate_densities(&chunk_coord, fbm),
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

    pub fn dig_sphere(&mut self, center: Vec3, radius: f32, strength: f32) -> Vec<(i16, i16, i16)> {
        let mut modified_chunks = Vec::new();
        let min_world = center - Vec3::splat(radius);
        let max_world = center + Vec3::splat(radius);
        let min_chunk = world_pos_to_chunk_coord(&min_world);
        let max_chunk = world_pos_to_chunk_coord(&max_world);
        for chunk_x in min_chunk.0..=max_chunk.0 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                for chunk_z in min_chunk.2..=max_chunk.2 {
                    let chunk_coord = (chunk_x, chunk_y, chunk_z);
                    if !self.0.contains_key(&chunk_coord) {
                        continue;
                    }
                    let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
                    let chunk_modified = self.modify_chunk_voxels(
                        chunk_coord,
                        chunk_center,
                        center,
                        radius,
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
        radius: f32,
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
                        if distance <= radius {
                            let falloff = 1.0 - (distance / radius).clamp(0.0, 1.0);
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

pub fn spawn_chunk(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    standard_terrain_material_handle: Handle<StandardMaterial>,
    mesh: Mesh,
    transform: Transform,
    collider: Option<Collider>,
) -> Entity {
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
    entity
}

pub fn setup_map(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let standard_terrain_material_handle = materials.add(StandardMaterial { ..default() });
    commands.insert_resource(ChunkMap::new());
    commands.insert_resource(NoiseFunction(fbm));
    commands.insert_resource(StandardTerrainMaterialHandle(
        standard_terrain_material_handle,
    ));
}

pub fn spawn_initial_chunks(
    mut commands: Commands,
    mut chunk_index_map: ResMut<ChunkIndexMap>,
    mut chunk_map: ResMut<ChunkMap>,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_material: Res<StandardTerrainMaterialHandle>,
    fbm: Res<NoiseFunction>,
    mut index_file: ResMut<ChunkIndexFile>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
) {
    let player_chunk = world_pos_to_chunk_coord(&PLAYER_SPAWN);
    let min_chunk = (
        player_chunk.0 - CHUNK_CREATION_RADIUS as i16,
        player_chunk.1 - CHUNK_CREATION_RADIUS as i16,
        player_chunk.2 - CHUNK_CREATION_RADIUS as i16,
    );
    let max_chunk = (
        player_chunk.0 + CHUNK_CREATION_RADIUS as i16,
        player_chunk.1 + CHUNK_CREATION_RADIUS as i16,
        player_chunk.2 + CHUNK_CREATION_RADIUS as i16,
    );
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                if chunk_world_pos.distance_squared(PLAYER_SPAWN) < CHUNK_CREATION_RADIUS_SQUARED {
                    let terrain_chunk = if chunk_index_map.0.contains_key(&chunk_coord) {
                        load_chunk_data(&mut chunk_data_file.0, &chunk_index_map.0, chunk_coord)
                    } else {
                        let chunk = TerrainChunk::new(chunk_coord, &fbm.0);
                        create_chunk_file_data(
                            &chunk,
                            chunk_coord,
                            &mut chunk_index_map.0,
                            &mut chunk_data_file.0,
                            &mut index_file.0,
                        );
                        chunk
                    };
                    let mesh = march_cubes(&terrain_chunk.densities);
                    let transform =
                        Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord));
                    let collider = if mesh.count_vertices() > 0 {
                        Collider::from_bevy_mesh(
                            &mesh,
                            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                        )
                    } else {
                        None
                    };
                    let entity = spawn_chunk(
                        &mut commands,
                        &mut meshes,
                        standard_material.0.clone(),
                        mesh,
                        transform,
                        collider,
                    );
                    chunk_map.0.insert(chunk_coord, (entity, terrain_chunk));
                }
            }
        }
    }
}
