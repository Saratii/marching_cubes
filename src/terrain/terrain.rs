use std::{process::exit, sync::Arc};

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_resource::{AddressMode, SamplerDescriptor},
    },
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use isomesh::marching_cubes::mc::{MeshBuffers, mc_mesh_generation};
use serde::{Deserialize, Serialize};

use crate::{
    conversions::{chunk_coord_to_world_pos, flatten_index, world_pos_to_chunk_coord},
    data_loader::file_loader::{
        ChunkDataFileReadWrite, ChunkIndexFile, ChunkIndexMap, create_chunk_file_data,
        load_chunk_data,
    },
    player::player::PLAYER_SPAWN,
    sparse_voxel_octree::ChunkSvo,
    terrain::{
        chunk_generator::{chunk_contains_surface, generate_densities},
        terrain_material::TerrainMaterial,
    },
};

pub const SAMPLES_PER_CHUNK_DIM: usize = 32; // Number of voxel sample points
pub const CHUNK_SIZE: f32 = 8.0; //in world units
pub const SAMPLES_PER_CHUNK: usize =
    SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const Z0_RADIUS: f32 = 15.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z0_RADIUS_SQUARED: f32 = Z0_RADIUS * Z0_RADIUS;
pub const Z1_RADIUS: f32 = 90.0; //in world units. Distance where chunks are loaded but not physically simulated.
pub const Z1_RADIUS_SQUARED: f32 = Z1_RADIUS * Z1_RADIUS;
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32;

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub Arc<GeneratorWrapper<SafeNode>>);

#[derive(Resource)]
pub struct TerrainMaterialHandle(pub Handle<TerrainMaterial>);

#[derive(Resource)]
pub struct TextureAtlasHandle(pub Handle<Image>);

#[derive(Component, Serialize, Deserialize, Debug)]
pub struct TerrainChunk {
    pub densities: Box<[i16]>,
    pub materials: Box<[u8]>,
}

impl TerrainChunk {
    pub fn new(chunk_coord: (i16, i16, i16), fbm: &GeneratorWrapper<SafeNode>) -> (Self, bool) {
        let (densities, materials, has_surface) = generate_densities(&chunk_coord, fbm);
        (
            Self {
                densities,
                materials,
            },
            has_surface,
        )
    }

    pub fn set_density(&mut self, x: u32, y: u32, z: u32, density: i16) {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM);
        self.densities[index as usize] = density;
    }

    pub fn get_density(&self, x: u32, y: u32, z: u32) -> i16 {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM);
        self.densities[index as usize]
    }

    pub fn get_mut_density(&mut self, x: u32, y: u32, z: u32) -> &mut i16 {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM);
        &mut self.densities[index as usize]
    }

    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_density(x, y, z) < 0
    }
}

pub fn setup_map(
    mut commands: Commands,
    mut materials: ResMut<Assets<TerrainMaterial>>,
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
    let standard_terrain_material_handle = materials.add(TerrainMaterial {
        texture: atlas_texture_handle.clone(),
        scale: 0.5,
    });
    commands.insert_resource(ChunkSvo::new());
    commands.insert_resource(TerrainMaterialHandle(standard_terrain_material_handle));
    commands.insert_resource(TextureAtlasHandle(atlas_texture_handle));
}

pub fn spawn_initial_chunks(
    mut commands: Commands,
    chunk_index_map: Res<ChunkIndexMap>,
    mut svo: ResMut<ChunkSvo>,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_material: Res<TerrainMaterialHandle>,
    fbm: Res<NoiseFunction>,
    index_file: ResMut<ChunkIndexFile>,
    chunk_data_file: Res<ChunkDataFileReadWrite>,
) {
    let player_chunk = world_pos_to_chunk_coord(&PLAYER_SPAWN);
    let min_chunk = (
        player_chunk.0 - Z0_RADIUS as i16,
        player_chunk.1 - Z0_RADIUS as i16,
        player_chunk.2 - Z0_RADIUS as i16,
    );
    let max_chunk = (
        player_chunk.0 + Z0_RADIUS as i16,
        player_chunk.1 + Z0_RADIUS as i16,
        player_chunk.2 + Z0_RADIUS as i16,
    );
    let mut index_map_lock = chunk_index_map.0.lock().unwrap();
    let mut data_file_lock = chunk_data_file.0.lock().unwrap();
    let mut index_file_lock = index_file.0.lock().unwrap();
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let distance_squared =
                    chunk_coord_to_world_pos(&chunk_coord).distance_squared(PLAYER_SPAWN);
                if distance_squared < Z0_RADIUS_SQUARED {
                    let file_offset = index_map_lock.get(&chunk_coord);
                    let (terrain_chunk, contains_surface) = if let Some(offset) = file_offset {
                        let chunk = load_chunk_data(&mut data_file_lock, *offset);
                        let contains_surface = chunk_contains_surface(&chunk);
                        (chunk, contains_surface)
                    } else {
                        let (chunk, contains_surface) = TerrainChunk::new(chunk_coord, &fbm.0);
                        create_chunk_file_data(
                            &chunk,
                            &chunk_coord,
                            &mut index_map_lock,
                            &mut data_file_lock,
                            &mut index_file_lock,
                        );
                        (chunk, contains_surface)
                    };
                    if contains_surface {
                        let mut mesh_buffers = MeshBuffers::new();
                        mc_mesh_generation(
                            &mut mesh_buffers,
                            &terrain_chunk.densities,
                            &terrain_chunk.materials,
                            SAMPLES_PER_CHUNK_DIM,
                            HALF_CHUNK,
                        );
                        let mesh = generate_bevy_mesh(mesh_buffers);
                        let transform =
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord));
                        let collider = Collider::from_bevy_mesh(
                            &mesh,
                            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                        )
                        .unwrap();
                        let id = commands
                            .spawn((
                                Mesh3d(meshes.add(mesh)),
                                MeshMaterial3d(standard_material.0.clone()),
                                ChunkTag,
                                transform,
                                collider,
                            ))
                            .id();
                        svo.root
                            .insert(chunk_coord, Some(id), Some(terrain_chunk), 0);
                    } else {
                        svo.root.insert(chunk_coord, None, Some(terrain_chunk), 0);
                    }
                }
            }
        }
    }
}

//writes data to disk for a large amount of chunks without saving to memory
// 900GB written in 8 minutes HEHE
pub fn generate_large_map_utility(
    chunk_index_map: Res<ChunkIndexMap>,
    fbm: Res<NoiseFunction>,
    index_file: ResMut<ChunkIndexFile>,
    chunk_data_file: Res<ChunkDataFileReadWrite>,
) {
    const CREATION_RADIUS: f32 = 150.0;
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
                    let mut locked_index_map = chunk_index_map.0.lock().unwrap();
                    if !locked_index_map.contains_key(&chunk_coord) {
                        let (chunk, _) = TerrainChunk::new(chunk_coord, &fbm.0);
                        let mut data_file = chunk_data_file.0.lock().unwrap();
                        let mut index_file = index_file.0.lock().unwrap();
                        create_chunk_file_data(
                            &chunk,
                            &chunk_coord,
                            &mut locked_index_map,
                            &mut data_file,
                            &mut index_file,
                        );
                        drop(data_file);
                        drop(index_file);
                    };
                    drop(locked_index_map);
                }
            }
        }
    }
    println!("Finished generating large map.");
    exit(0);
}

pub fn generate_bevy_mesh(mesh_buffers: MeshBuffers) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let MeshBuffers {
        positions,
        normals,
        indices,
        uvs,
    } = mesh_buffers;
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh
}
