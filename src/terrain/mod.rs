use bevy::mesh::MeshVertexAttribute;
use wgpu::VertexFormat;

pub const ATTRIBUTE_MATERIAL_ID: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialId", 988540918, VertexFormat::Uint32);

pub mod chunk_compute_pipeline;
pub mod chunk_generator;
pub mod heightmap_compute_pipeline;
pub mod terrain;
pub mod terrain_material;
