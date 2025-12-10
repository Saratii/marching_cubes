#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}

@group(3) @binding(103) var base_texture: texture_2d<f32>;
@group(3) @binding(104) var base_sampler: sampler;
@group(3) @binding(105) var<uniform> scale: f32;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let world_pos = in.world_position.xyz;
    let world_normal = normalize(in.world_normal);
    var blend = abs(world_normal);
    blend = pow(blend, vec3(4.0));
    blend = blend / (blend.x + blend.y + blend.z);
    let eps = 0.1;
    let raw = in.uv.x;
    let is_grass = abs(raw - 2.0) < eps;
    let material_offset_u = select(0.0, 0.5, is_grass);
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
    pbr_input.material.base_color = vec4<f32>(final_color, 1.0);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    //let g = world_pos.y * 0.01;
    //out.color = vec4(g, 0.2 + g, 0.8 - g, 1.0);
    return out;
}