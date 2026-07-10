use bevy::mesh::MeshVertexAttribute;
use wgpu::VertexFormat;

pub const ATTRIBUTE_MATERIAL_ID: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialId", 988540918, VertexFormat::Uint32);
