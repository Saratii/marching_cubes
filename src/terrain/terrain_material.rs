use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainMaterialExtension {
    #[texture(103, dimension = "2d_array")]
    #[sampler(104)]
    pub base_texture: Handle<Image>,
    #[uniform(105)]
    pub scale: f32,
    #[texture(106, dimension = "3d", sample_type = "u_int")]
    pub material_field: Handle<Image>,
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
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}
