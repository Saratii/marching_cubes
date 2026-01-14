use std::sync::Arc;

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
    conversions::flatten_index,
    data_loader::file_loader::get_project_root,
    terrain::{
        chunk_generator::{HEIGHT_MAP_GRID_SIZE, generate_densities},
        terrain_material::{ATTRIBUTE_MATERIAL_ID, TerrainMaterialExtension},
    },
};

pub const SAMPLES_PER_CHUNK_DIM: usize = 50; // Number of voxel sample points
pub const CHUNK_SIZE: f32 = 12.5; //in world units
pub const CLUSTER_SIZE: usize = 5; //number of chunks along one edge of a cluster
pub const SAMPLES_PER_CHUNK: usize =
    SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
pub const SAMPLES_PER_CHUNK_DIM_M1: usize = SAMPLES_PER_CHUNK_DIM - 1;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const Z0_RADIUS: f32 = 80.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z0_RADIUS_SQUARED: f32 = Z0_RADIUS * Z0_RADIUS;
pub const Z1_RADIUS: f32 = 100.0; //in world units. Distance where chunks are loaded at full res but not stored in memory.
pub const Z1_RADIUS_SQUARED: f32 = Z1_RADIUS * Z1_RADIUS;
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / SAMPLES_PER_CHUNK_DIM_M1 as f32;
pub const Z2_RADIUS: f32 = 1700.0;
pub const Z2_RADIUS_SQUARED: f32 = Z2_RADIUS * Z2_RADIUS;
pub const MAX_RADIUS: f32 = Z0_RADIUS.max(Z1_RADIUS).max(Z2_RADIUS);
pub const MAX_RADIUS_SQUARED: f32 = MAX_RADIUS * MAX_RADIUS;

#[derive(Component)]
pub struct ChunkTag;

#[derive(Resource)]
pub struct NoiseFunction(pub GeneratorWrapper<SafeNode>);

#[derive(Resource)]
pub struct TerrainMaterialHandle(
    pub Handle<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>,
);

#[derive(Resource)]
pub struct TextureAtlasHandle(pub Handle<Image>);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
pub enum Uniformity {
    NonUniform,
    Dirt,
    Air,
}

#[derive(Component, Debug, Clone)]
pub struct TerrainChunk {
    pub densities: Arc<[i16; SAMPLES_PER_CHUNK]>, //arc, so the write thread can read them
    pub materials: Arc<[u8; SAMPLES_PER_CHUNK]>,
    pub is_uniform: Uniformity,
}

impl TerrainChunk {
    pub fn new(densities: Arc<[i16; SAMPLES_PER_CHUNK]>, materials: Arc<[u8; SAMPLES_PER_CHUNK]>, is_uniform: Uniformity) -> Self {
        Self {
            densities,
            materials,
            is_uniform,
        }
    }

    pub fn get_density(&self, x: u32, y: u32, z: u32) -> i16 {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM);
        self.densities[index as usize]
    }

    pub fn get_mut_density(&mut self, x: u32, y: u32, z: u32) -> &mut i16 {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM);
        let densities = Arc::make_mut(&mut self.densities);
        &mut densities[index as usize]
    }

    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_density(x, y, z) < 0
    }
}

pub fn generate_chunk_into_buffers(
    fbm: &GeneratorWrapper<SafeNode>,
    first_sample_reuse: f32,
    chunk_start: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [u8],
    heightmap_buffer: &mut [f32; HEIGHT_MAP_GRID_SIZE],
) -> Uniformity {
    let is_uniform = generate_densities(
        fbm,
        first_sample_reuse,
        chunk_start,
        density_buffer,
        material_buffer,
        heightmap_buffer,
    );
    //if it does not have a surface it must be uniform dirt or air
    let uniformity = if !is_uniform {
        Uniformity::NonUniform
    } else {
        if material_buffer[0] == 1 {
            Uniformity::Dirt
        } else if material_buffer[0] == 0 {
            Uniformity::Air
        } else {
            println!("materials[0]: {}", material_buffer[0]);
            panic!("Generated uniform chunk with unknown material type!");
        }
    };
    uniformity
}

pub fn setup_map(
    mut commands: Commands,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>>,
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
        extension: TerrainMaterialExtension {
            base_texture: atlas_texture_handle.clone(),
            scale: 0.5,
        },
    });
    commands.insert_resource(TerrainMaterialHandle(standard_terrain_material_handle));
    commands.insert_resource(TextureAtlasHandle(atlas_texture_handle));
}

pub fn generate_bevy_mesh(
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    material_ids: Vec<u32>,
    indices: Vec<u32>,
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_ID, material_ids);
    mesh
}
