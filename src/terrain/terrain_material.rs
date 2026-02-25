use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError},
        storage::ShaderStorageBuffer,
    },
    shader::ShaderRef,
};

use crate::terrain::ATTRIBUTE_TRIANGLE_INDEX;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriData {
    pub ids0: [u32; 3],
    pub _pad0: u32,
    pub ids1: [u32; 3],
    pub _pad1: u32,
    pub ids2: [u32; 3],
    pub _pad2: u32,
    pub w0: [f32; 3],
    pub _pad3: f32,
    pub w1: [f32; 3],
    pub _pad4: f32,
    pub w2: [f32; 3],
    pub _pad5: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainMaterialExtension {
    #[texture(103, dimension = "2d_array")]
    #[sampler(104)]
    pub base_texture: Handle<Image>,
    #[uniform(105)]
    pub scale: f32,
    #[storage(200, read_only)]
    pub tri_buffer: Handle<ShaderStorageBuffer>,
}

impl MaterialExtension for TerrainMaterialExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/triplanar.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/triplanar.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_TRIANGLE_INDEX.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
