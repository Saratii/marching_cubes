#import bevy_pbr::{
    mesh_view_bindings::view,
    forward_io::VertexOutput,
}

@group(3) @binding(3) var base_texture: texture_2d<f32>;
@group(3) @binding(4) var base_sampler: sampler;
@group(3) @binding(5) var<uniform> scale: f32;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = in.world_position.xyz;
    let world_normal = normalize(in.world_normal);
    var blend = abs(world_normal);
    blend = pow(blend, vec3(4.0));
    blend = blend / (blend.x + blend.y + blend.z);
    let material = u32(in.uv.x);
    let material_offset_u = select(0.0, 0.5, material == 2u);
    let uv_x = fract(world_pos.yz * scale);
    let uv_y = fract(world_pos.xz * scale);
    let uv_z = fract(world_pos.xy * scale);
    let atlas_uv_x = vec2(uv_x.x * 0.5 + material_offset_u, uv_x.y);
    let atlas_uv_y = vec2(uv_y.x * 0.5 + material_offset_u, uv_y.y);
    let atlas_uv_z = vec2(uv_z.x * 0.5 + material_offset_u, uv_z.y);
    let color_x = textureSample(base_texture, base_sampler, atlas_uv_x).rgb;
    let color_y = textureSample(base_texture, base_sampler, atlas_uv_y).rgb;
    let color_z = textureSample(base_texture, base_sampler, atlas_uv_z).rgb;
    let final_color = color_x * blend.x + color_y * blend.y + color_z * blend.z;
    return vec4<f32>(final_color, 1.0);
}