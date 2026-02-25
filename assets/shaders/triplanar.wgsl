#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_functions::{get_world_from_local, mesh_position_local_to_clip},
}

struct TriData {
    ids0: vec3<u32>,
    ids1: vec3<u32>,
    ids2: vec3<u32>,
    w0: vec3<f32>,
    w1: vec3<f32>,
    w2: vec3<f32>,
};

@group(3) @binding(200)
var<storage, read> tri_data: array<TriData>;

@group(3) @binding(103) var base_texture: texture_2d_array<f32>;
@group(3) @binding(104) var base_sampler: sampler;
@group(3) @binding(105) var<uniform> scale: f32;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) triangle_index: u32
}

struct CustomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) triangle_index: u32,
}

@vertex
fn vertex(v: Vertex) -> CustomVertexOutput {
    var out: CustomVertexOutput;
    let world_from_local = get_world_from_local(v.instance_index);
    out.world_position = world_from_local * vec4<f32>(v.position, 1.0);
    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(v.position, 1.0));
    out.world_normal = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz
    ) * v.normal;
    out.triangle_index = v.triangle_index;
    return out;
}

fn triplanar_sample(
    world_pos: vec3<f32>,
    blend: vec3<f32>,
    layer: i32,
    scale_val: f32,
) -> vec3<f32> {
    let s = vec2(scale_val);
    let uv_x_raw = world_pos.yz * s;
    let uv_y_raw = world_pos.xz * s;
    let uv_z_raw = world_pos.xy * s;
    let color_x = textureSampleGrad(
        base_texture, base_sampler,
        fract(uv_x_raw), layer,
        dpdx(uv_x_raw), dpdy(uv_x_raw)
    ).rgb;
    let color_y = textureSampleGrad(
        base_texture, base_sampler,
        fract(uv_y_raw), layer,
        dpdx(uv_y_raw), dpdy(uv_y_raw)
    ).rgb;
    let color_z = textureSampleGrad(
        base_texture, base_sampler,
        fract(uv_z_raw), layer,
        dpdx(uv_z_raw), dpdy(uv_z_raw)
    ).rgb;
    return color_x * blend.x + color_y * blend.y + color_z * blend.z;
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

    let t = tri_data[in.triangle_index];

    var best_weight: f32 = t.w0.x;
    var best_id: u32 = t.ids0.x;

    if (t.w0.y > best_weight) { best_weight = t.w0.y; best_id = t.ids0.y; }
    if (t.w0.z > best_weight) { best_weight = t.w0.z; best_id = t.ids0.z; }

    if (t.w1.x > best_weight) { best_weight = t.w1.x; best_id = t.ids1.x; }
    if (t.w1.y > best_weight) { best_weight = t.w1.y; best_id = t.ids1.y; }
    if (t.w1.z > best_weight) { best_weight = t.w1.z; best_id = t.ids1.z; }

    if (t.w2.x > best_weight) { best_weight = t.w2.x; best_id = t.ids2.x; }
    if (t.w2.y > best_weight) { best_weight = t.w2.y; best_id = t.ids2.y; }
    if (t.w2.z > best_weight) {                     best_id = t.ids2.z; }
    // ---------------------------------------------------------------------------

    let layer: i32 = i32(best_id);

    let scale_vec = vec2(scale);
    let uv_x_raw = world_pos.yz * scale_vec;
    let uv_y_raw = world_pos.xz * scale_vec;
    let uv_z_raw = world_pos.xy * scale_vec;

    let uv_x = fract(uv_x_raw);
    let uv_y = fract(uv_y_raw);
    let uv_z = fract(uv_z_raw);

    let duvdx_x = dpdx(uv_x_raw);
    let duvdy_x = dpdy(uv_x_raw);
    let duvdx_y = dpdx(uv_y_raw);
    let duvdy_y = dpdy(uv_y_raw);
    let duvdx_z = dpdx(uv_z_raw);
    let duvdy_z = dpdy(uv_z_raw);

    let color_x = textureSampleGrad(base_texture, base_sampler, uv_x, layer, duvdx_x, duvdy_x).rgb;
    let color_y = textureSampleGrad(base_texture, base_sampler, uv_y, layer, duvdx_y, duvdy_y).rgb;
    let color_z = textureSampleGrad(base_texture, base_sampler, uv_z, layer, duvdx_z, duvdy_z).rgb;

    let final_color = color_x * blend.x + color_y * blend.y + color_z * blend.z;

    pbr_input.material.base_color = vec4<f32>(final_color, 1.0);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}