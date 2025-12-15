use std::process::exit;

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageLoaderSettings, ImageSampler},
    mesh::{Indices, PrimitiveTopology},
    pbr::ExtendedMaterial,
    prelude::*,
    render::render_resource::{AddressMode, SamplerDescriptor},
};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use serde::{Deserialize, Serialize};

use crate::{
    conversions::{chunk_coord_to_world_pos, flatten_index, world_pos_to_chunk_coord},
    data_loader::file_loader::{
        ChunkDataFileReadWrite, ChunkIndexFile, ChunkIndexMap, create_chunk_file_data,
        get_project_root,
    },
    player::player::PLAYER_SPAWN,
    terrain::{chunk_generator::generate_densities, terrain_material::TerrainMaterial},
};

pub const SAMPLES_PER_CHUNK_DIM: usize = 32; // Number of voxel sample points
pub const CHUNK_SIZE: f32 = 8.0; //in world units
pub const SAMPLES_PER_CHUNK: usize =
    SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const Z0_RADIUS: f32 = 35.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z0_RADIUS_SQUARED: f32 = Z0_RADIUS * Z0_RADIUS;
pub const Z1_RADIUS: f32 = 100.0; //in world units. Distance where chunks are loaded at full res but not stored in memory.
pub const Z1_RADIUS_SQUARED: f32 = Z1_RADIUS * Z1_RADIUS;
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
pub const Z2_RADIUS: f32 = 1200.0;
pub const Z2_RADIUS_SQUARED: f32 = Z2_RADIUS * Z2_RADIUS;
pub const MAX_RADIUS: f32 = Z0_RADIUS.max(Z1_RADIUS).max(Z2_RADIUS);
pub const MAX_RADIUS_SQUARED: f32 = MAX_RADIUS * MAX_RADIUS;

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

#[derive(Resource)]
pub struct TerrainMaterialHandle(pub Handle<ExtendedMaterial<StandardMaterial, TerrainMaterial>>);

#[derive(Resource)]
pub struct TextureAtlasHandle(pub Handle<Image>);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UniformChunk {
    NonUniform,
    Dirt,
    Air,
}

#[derive(Component, Serialize, Deserialize, Debug, Clone)]
pub struct TerrainChunk {
    pub densities: Box<[i16]>,
    pub materials: Box<[u8]>,
    pub is_uniform: UniformChunk,
}

impl TerrainChunk {
    pub fn new(chunk_coord: (i16, i16, i16), fbm: &GeneratorWrapper<SafeNode>) -> Self {
        let (densities, materials, is_uniform) = generate_densities(&chunk_coord, fbm);
        //if it does not have a surface it must be uniform dirt or air
        let is_uniform = if !is_uniform {
            UniformChunk::NonUniform
        } else {
            if materials[0] == 1 {
                UniformChunk::Dirt
            } else if materials[0] == 0 {
                UniformChunk::Air
            } else {
                println!("materials[0]: {}", materials[0]);
                panic!("Generated uniform chunk with unknown material type!");
            }
        };
        Self {
            densities,
            materials,
            is_uniform,
        }
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
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainMaterial>>>,
    asset_server: Res<AssetServer>,
) {
    let root = get_project_root();
    let atlas_texture_handle: Handle<Image> = asset_server
        .load_with_settings::<Image, ImageLoaderSettings>(
            root.join("assets/texture_atlas.ktx2"),
            |settings| {
                settings.sampler = ImageSampler::Descriptor(
                    SamplerDescriptor {
                        address_mode_u: AddressMode::ClampToEdge,
                        address_mode_v: AddressMode::ClampToEdge,
                        address_mode_w: AddressMode::ClampToEdge,
                        lod_min_clamp: 0.0,
                        lod_max_clamp: 5.0,
                        ..Default::default()
                    }
                    .into(),
                );
            },
        );
    let standard_terrain_material_handle = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            perceptual_roughness: 0.8,
            ..Default::default()
        },
        extension: TerrainMaterial {
            texture: atlas_texture_handle.clone(),
            scale: 0.5,
        },
    });
    commands.insert_resource(TerrainMaterialHandle(standard_terrain_material_handle));
    commands.insert_resource(TextureAtlasHandle(atlas_texture_handle));
}

//writes data to disk for a large amount of chunks without saving to memory
// 900GB written in 8 minutes HEHE
pub fn generate_large_map_utility(
    chunk_index_map: Res<ChunkIndexMap>,
    fbm: Res<NoiseFunction>,
    chunk_index_file: ResMut<ChunkIndexFile>,
    chunk_data_file: ResMut<ChunkDataFileReadWrite>,
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
                        let chunk = TerrainChunk::new(chunk_coord, &fbm.0);
                        let mut chunk_data_file_locked = chunk_data_file.0.lock().unwrap();
                        let mut chunk_index_file_locked = chunk_index_file.0.lock().unwrap();
                        create_chunk_file_data(
                            &chunk,
                            &chunk_coord,
                            &mut locked_index_map,
                            &mut chunk_data_file_locked,
                            &mut chunk_index_file_locked,
                        );
                        drop(chunk_index_file_locked);
                        drop(chunk_data_file_locked);
                    };
                    drop(locked_index_map);
                }
            }
        }
    }
    println!("Finished generating large map.");
    exit(0);
}

pub fn generate_bevy_mesh(
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
