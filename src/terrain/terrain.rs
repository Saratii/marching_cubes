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
    constants::SAMPLES_PER_CHUNK_DIM,
    conversions::flatten_index,
    data_loader::file_loader::get_project_root,
    terrain::{
        ATTRIBUTE_MATERIAL_ID,
        chunk_generator::{fill_voxel_densities, generate_terrain_heights},
        terrain_material::TerrainMaterialExtension,
    },
};

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

#[derive(Clone)]
pub struct NonUniformTerrainChunk {
    pub densities: Arc<[i16]>, //arc, so the write thread can read them
    pub materials: Arc<[u8]>,
}

pub enum TerrainChunk {
    UniformDirt,
    UniformAir,
    NonUniformTerrainChunk(NonUniformTerrainChunk),
}

impl TerrainChunk {
    #[inline(always)]
    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        match self {
            TerrainChunk::UniformDirt => true,
            TerrainChunk::UniformAir => false,
            TerrainChunk::NonUniformTerrainChunk(chunk) => chunk.is_solid(x, y, z),
        }
    }
}

impl NonUniformTerrainChunk {
    pub fn new(densities: Arc<[i16]>, materials: Arc<[u8]>) -> Self {
        Self {
            densities,
            materials,
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
    chunk_start: Vec3,
    density_buffer: &mut [i16],
    material_buffer: &mut [u8],
    heightmap_buffer: &mut [f32],
    samples_per_chunk_dim: usize,
) -> Uniformity {
    generate_terrain_heights(
        chunk_start.x,
        chunk_start.z,
        fbm,
        heightmap_buffer,
        samples_per_chunk_dim,
    );
    let is_uniform = fill_voxel_densities(
        density_buffer,
        material_buffer,
        &chunk_start,
        heightmap_buffer,
        samples_per_chunk_dim,
    );
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
