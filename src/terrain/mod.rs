use bevy::mesh::MeshVertexAttribute;
use wgpu::VertexFormat;

pub const ATTRIBUTE_TRIANGLE_INDEX: MeshVertexAttribute =
    MeshVertexAttribute::new("TriangleIndex", 988776655, VertexFormat::Uint32);

pub mod chunk_compute_pipeline;
pub mod chunk_generator;
pub mod heightmap_compute_pipeline;
pub mod terrain;
pub mod terrain_material;
