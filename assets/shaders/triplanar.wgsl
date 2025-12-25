#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_functions::{get_world_from_local, mesh_position_local_to_clip},
}

@group(3) @binding(103) var base_texture: texture_2d<f32>;
@group(3) @binding(104) var base_sampler: sampler;
@group(3) @binding(105) var<uniform> scale: f32;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) material_id: u32,
}

struct CustomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_id: u32,
}

@vertex
fn vertex(vertex: Vertex) -> CustomVertexOutput {
    var out: CustomVertexOutput;
    let world_from_local = get_world_from_local(vertex.instance_index);
    out.world_position = world_from_local * vec4<f32>(vertex.position, 1.0);
    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(vertex.position, 1.0));
    out.world_normal = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz
    ) * vertex.normal;
    out.material_id = vertex.material_id;
    return out;
}

@fragment
fn fragment(
    in: CustomVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var standard_in: VertexOutput;
    standard_in.position = in.clip_position;
    standard_in.world_position = in.world_position;
    standard_in.world_normal = in.world_normal;
    var pbr_input = pbr_input_from_standard_material(standard_in, is_front);
    let world_pos = in.world_position.xyz;
    let world_normal = normalize(in.world_normal);
    var blend = abs(world_normal);
    blend = pow(blend, vec3(4.0));
    blend = blend / (blend.x + blend.y + blend.z);
    let id = i32(in.material_id);
    var material_offset_u = 0.0;  // dirt = 1
    if (id == 2) {
        material_offset_u = 0.333333;  // grass
    } else if (id == 3) {
        material_offset_u = 0.666666;  // sand
    }
    let scale_vec = vec2(scale);
    let uv_x = fract(world_pos.yz * scale_vec);
    let uv_y = fract(world_pos.xz * scale_vec);
    let uv_z = fract(world_pos.xy * scale_vec);
    let color_x = textureSample(base_texture, base_sampler, vec2(uv_x.x * 0.333333 + material_offset_u, uv_x.y)).rgb;
    let color_y = textureSample(base_texture, base_sampler, vec2(uv_y.x * 0.333333 + material_offset_u, uv_y.y)).rgb;
    let color_z = textureSample(base_texture, base_sampler, vec2(uv_z.x * 0.333333 + material_offset_u, uv_z.y)).rgb;
    let final_color = color_x * blend.x + color_y * blend.y + color_z * blend.z;
    pbr_input.material.base_color = vec4<f32>(final_color, 1.0);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}