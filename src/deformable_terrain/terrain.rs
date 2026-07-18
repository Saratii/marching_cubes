use std::sync::Arc;

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageLoaderSettings, ImageSampler},
    mesh::{Indices, MeshVertexAttribute, PrimitiveTopology},
    pbr::ExtendedMaterial,
    prelude::*,
    render::render_resource::{AddressMode, SamplerDescriptor},
};
use wgpu::VertexFormat;

use crate::{
    constants::SAMPLES_PER_CHUNK_DIM_PADDED,
    conversions::flatten_index,
    deformable_terrain::{
        chunk_generator::MaterialCode, file_loader::get_project_root,
        terrain_material::TerrainMaterialExtension,
    },
};

pub(crate) const ATTRIBUTE_MATERIAL_ID: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialId", 988540918, VertexFormat::Uint32);

#[derive(Resource)]
pub struct TerrainMaterialHandle(
    pub Handle<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>,
);

#[derive(Clone)]
pub(crate) struct NonUniformTerrainChunk {
    pub(crate) densities: Arc<[i16]>, //arc, so the write thread can read them
    pub(crate) materials: Arc<[MaterialCode]>,
}

pub(crate) enum TerrainChunk {
    UniformDirt,
    UniformAir,
    NonUniformTerrainChunk(NonUniformTerrainChunk),
}

impl TerrainChunk {
    #[inline(always)]
    pub(crate) fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        match self {
            TerrainChunk::UniformDirt => true,
            TerrainChunk::UniformAir => false,
            TerrainChunk::NonUniformTerrainChunk(chunk) => chunk.is_solid(x, y, z),
        }
    }
}

impl NonUniformTerrainChunk {
    pub(crate) fn get_density(&self, x: u32, y: u32, z: u32) -> i16 {
        let index = flatten_index(x, y, z, SAMPLES_PER_CHUNK_DIM_PADDED);
        self.densities[index as usize]
    }

    pub(crate) fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_density(x, y, z) < 0
    }
}

pub(crate) fn setup_map(
    mut commands: Commands,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>>,
    asset_server: Res<AssetServer>,
) {
    let root = get_project_root();
    let texture_array_handle: Handle<Image> = asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
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
        })
        .load(root.join("assets/texture_array.ktx2"));
    let standard_terrain_material_handle = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            perceptual_roughness: 0.8,
            ..Default::default()
        },
        extension: TerrainMaterialExtension {
            base_texture: texture_array_handle.clone(),
            scale: 1.5,
        },
    });
    commands.insert_resource(TerrainMaterialHandle(standard_terrain_material_handle));
}

pub(crate) fn generate_bevy_mesh(
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
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
