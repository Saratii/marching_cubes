use bevy::{
    asset::{Asset, Handle},
    image::Image,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey},
    reflect::TypePath,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct TerrainMaterial {
    #[texture(3)]
    #[sampler(4)]
    pub texture: Handle<Image>,
    #[uniform(5)]
    pub scale: f32,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/triplanar.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
