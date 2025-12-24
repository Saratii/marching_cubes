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
    terrain::{chunk_generator::generate_densities, terrain_material::TerrainMaterial},
};

pub const SAMPLES_PER_CHUNK_DIM: usize = 50; // Number of voxel sample points
pub const CHUNK_SIZE: f32 = 12.5; //in world units
pub const SAMPLES_PER_CHUNK: usize =
    SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
pub const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;
pub const Z0_RADIUS: f32 = 50.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const Z0_RADIUS_SQUARED: f32 = Z0_RADIUS * Z0_RADIUS;
pub const Z1_RADIUS: f32 = 100.0; //in world units. Distance where chunks are loaded at full res but not stored in memory.
pub const Z1_RADIUS_SQUARED: f32 = Z1_RADIUS * Z1_RADIUS;
pub const VOXEL_SIZE: f32 = CHUNK_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
pub const Z2_RADIUS: f32 = 1400.0;
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
