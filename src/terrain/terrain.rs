use std::{process::exit, sync::Arc};

use bevy::{
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
    render::render_resource::{AddressMode, SamplerDescriptor},
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use serde::{Deserialize, Serialize};

use crate::{
    conversions::{chunk_coord_to_world_pos, flatten_index, world_pos_to_chunk_coord},
    data_loader::file_loader::{DataBaseEnvHandle, DataBaseHandle, LoadedChunkKeys, deserialize_chunk_data, serialize_chunk_data,
    },
    marching_cubes::march_cubes,
    player::player::PLAYER_SPAWN,
    sparse_voxel_octree::ChunkSvo,
    terrain::chunk_generator::generate_densities,
};

pub const SDF_VALUES_PER_CHUNK_DIM: usize = 32; // Number of voxel sample points
pub const VOXEL_SIZE: f32 = 0.1; // Size of each voxel in meters
pub const CUBES_PER_CHUNK_DIM: usize = SDF_VALUES_PER_CHUNK_DIM - 1; // 63 cubes
pub const CHUNK_SIZE: f32 = CUBES_PER_CHUNK_DIM as f32 * VOXEL_SIZE; // 7.875 meters
pub const VOXELS_PER_CHUNK: usize =
    SDF_VALUES_PER_CHUNK_DIM * SDF_VALUES_PER_CHUNK_DIM * SDF_VALUES_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const Z1_RADIUS: f32 = 20.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z1_RADIUS_SQUARED: f32 = Z1_RADIUS * Z1_RADIUS;
pub const Z2_RADIUS: f32 = 90.0; //in world units. Distance where chunks are loaded but not physically simulated.
pub const Z2_RADIUS_SQUARED: f32 = Z2_RADIUS * Z2_RADIUS;

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub Arc<GeneratorWrapper<SafeNode>>);

#[derive(Resource)]
pub struct StandardTerrainMaterialHandle(pub Handle<StandardMaterial>);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VoxelData {
    pub sdf: f32,
    pub material: u8,
}

#[derive(Resource)]
pub struct TextureAtlasHandle(pub Handle<Image>);

#[derive(Component, Serialize, Deserialize, Debug)]
pub struct TerrainChunk {
    pub sdfs: Box<[VoxelData]>,
}

impl TerrainChunk {
    pub fn new(chunk_coord: (i16, i16, i16), fbm: &GeneratorWrapper<SafeNode>) -> Self {
        Self {
            sdfs: generate_densities(&chunk_coord, fbm),
        }
    }

    pub fn set_density(&mut self, x: u32, y: u32, z: u32, density: VoxelData) {
        let index = flatten_index(x, y, z, SDF_VALUES_PER_CHUNK_DIM);
        self.sdfs[index as usize] = density;
    }

    pub fn get_density(&self, x: u32, y: u32, z: u32) -> &VoxelData {
        let index = flatten_index(x, y, z, SDF_VALUES_PER_CHUNK_DIM);
        &self.sdfs[index as usize]
    }

    pub fn get_mut_density(&mut self, x: u32, y: u32, z: u32) -> &mut VoxelData {
        let index = flatten_index(x, y, z, SDF_VALUES_PER_CHUNK_DIM);
        &mut self.sdfs[index as usize]
    }

    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_density(x, y, z).sdf > 0.0
    }
}

pub fn setup_map(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let atlas_texture_handle: Handle<Image> = asset_server
        .load_with_settings::<Image, ImageLoaderSettings>("texture_atlas.png", |settings| {
            settings.sampler = ImageSampler::Descriptor(
                SamplerDescriptor {
                    address_mode_u: AddressMode::MirrorRepeat,
                    address_mode_v: AddressMode::MirrorRepeat,
                    address_mode_w: AddressMode::MirrorRepeat,
                    ..Default::default()
                }
                .into(),
            )
        });
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let standard_terrain_material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(atlas_texture_handle.clone()),
        ..default()
    });
    commands.insert_resource(ChunkSvo::new());
    commands.insert_resource(NoiseFunction(Arc::new(fbm)));
    commands.insert_resource(StandardTerrainMaterialHandle(
        standard_terrain_material_handle,
    ));
    commands.insert_resource(TextureAtlasHandle(atlas_texture_handle));
}

pub fn spawn_initial_chunks(
    mut commands: Commands,
    loaded_chunk_keys: Res<LoadedChunkKeys>,
    mut svo: ResMut<ChunkSvo>,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_material: Res<StandardTerrainMaterialHandle>,
    fbm: Res<NoiseFunction>,
    database_env: Res<DataBaseEnvHandle>,
    database: Res<DataBaseHandle>,
) {
    let player_chunk = world_pos_to_chunk_coord(&PLAYER_SPAWN);
    let min_chunk = (
        player_chunk.0 - Z1_RADIUS as i16,
        player_chunk.1 - Z1_RADIUS as i16,
        player_chunk.2 - Z1_RADIUS as i16,
    );
    let max_chunk = (
        player_chunk.0 + Z1_RADIUS as i16,
        player_chunk.1 + Z1_RADIUS as i16,
        player_chunk.2 + Z1_RADIUS as i16,
    );
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let mut locked_loaded_chunk_keys = loaded_chunk_keys.0.lock().unwrap();
                let chunk_data = if locked_loaded_chunk_keys.contains(&chunk_coord) {
                    drop(locked_loaded_chunk_keys);
                    let rtxn = database_env.0.read_txn().unwrap();
                    let bytes = database.0.get(&rtxn, &chunk_coord).unwrap().unwrap();
                    deserialize_chunk_data(&bytes)
                } else {
                    locked_loaded_chunk_keys.insert(chunk_coord);
                    drop(locked_loaded_chunk_keys);
                    let chunk = TerrainChunk::new(chunk_coord, &fbm.0);
                    let mut wtxn = database_env.0.write_txn().unwrap();
                    let bytes = serialize_chunk_data(&chunk);
                    database.0.put(&mut wtxn, &chunk_coord, &bytes).unwrap();
                    wtxn.commit().unwrap();
                    chunk
                };
                let mesh = march_cubes(
                    &chunk_data.sdfs,
                    CUBES_PER_CHUNK_DIM,
                    SDF_VALUES_PER_CHUNK_DIM,
                );
                let transform = Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord));
                let entity: Entity = if mesh.count_vertices() > 0 {
                    let collider = Collider::from_bevy_mesh(
                        &mesh,
                        &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                    )
                    .unwrap();
                    commands
                        .spawn((
                            Mesh3d(meshes.add(mesh)),
                            MeshMaterial3d(standard_material.0.clone()),
                            ChunkTag,
                            transform,
                            collider,
                        ))
                        .id()
                } else {
                    commands.spawn((ChunkTag, transform)).id()
                };
                svo.root.insert(chunk_coord, entity, chunk_data);
            }
        }
    }
}

//writes data to disk for a large amount of chunks without saving to memory
// 900GB written in 8 minutes HEHE
pub fn generate_large_map_utility(
    loaded_chunk_keys: Res<LoadedChunkKeys>,
    fbm: Res<NoiseFunction>,
    database_env: Res<DataBaseEnvHandle>,
    database: Res<DataBaseHandle>,
) {
    const CREATION_RADIUS: f32 = 100.0;
    const CREATION_RADIUS_SQUARED: f32 = CREATION_RADIUS * CREATION_RADIUS;
    let player_chunk = world_pos_to_chunk_coord(&PLAYER_SPAWN);
    let min_chunk = (
        player_chunk.0 - CREATION_RADIUS as i16,
        player_chunk.1 - CREATION_RADIUS as i16,
        player_chunk.2 - CREATION_RADIUS as i16,
    );
    let max_chunk = (
        player_chunk.0 + CREATION_RADIUS as i16,
        player_chunk.1 + CREATION_RADIUS as i16,
        player_chunk.2 + CREATION_RADIUS as i16,
    );
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                if chunk_world_pos.distance_squared(PLAYER_SPAWN) < CREATION_RADIUS_SQUARED {
                    let mut locked_index_map = loaded_chunk_keys.0.lock().unwrap();
                    if !locked_index_map.contains(&chunk_coord) {
                        let chunk = TerrainChunk::new(chunk_coord, &fbm.0);
                        locked_index_map.insert(chunk_coord);
                        drop(locked_index_map);
                        let mut wtxn = database_env.0.write_txn().unwrap();
                        let bytes = serialize_chunk_data(&chunk);
                        database.0.put(&mut wtxn, &chunk_coord, &bytes).unwrap();
                    } else {
                        drop(locked_index_map);
                    }
                }
            }
        }
    }
    exit(0);
}
