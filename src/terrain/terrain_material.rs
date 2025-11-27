use bevy::{
    asset::{Asset, Handle},
    image::Image,
    pbr::MaterialExtension,
    reflect::Reflect,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct TerrainMaterial {
    #[texture(103)]
    #[sampler(104)]
    pub texture: Handle<Image>,
    #[uniform(105)]
    pub scale: f32,
}

impl MaterialExtension for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/triplanar.wgsl".into()
    }
}
