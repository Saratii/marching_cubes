#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_functions::{get_world_from_local, mesh_position_local_to_clip},
}

const CHUNK_WORLD_SIZE: f32 = 12.0;
const HALF_CHUNK: f32 = CHUNK_WORLD_SIZE * 0.5;
const VOXEL_WORLD_SIZE: f32 = CHUNK_WORLD_SIZE / 63.0;

@group(3) @binding(103) var base_texture: texture_2d_array<f32>;
@group(3) @binding(104) var base_sampler: sampler;
@group(3) @binding(105) var<uniform> scale: f32;
@group(3) @binding(106) var material_field: texture_3d<u32>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct CustomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
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
    var out: FragmentOutput;

    let wp = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let axis_sum = abs(n.x) + abs(n.y) + abs(n.z);          // 1..~1.732
    let bias = VOXEL_WORLD_SIZE * 0.45 * axis_sum;          // bigger on steep/diagonal
    let wp_biased = wp - n * bias;
    let chunk_coord = world_pos_to_chunk_coord(wp_biased);
    let local_index = world_pos_to_voxel_index(wp_biased, chunk_coord);
    let voxel_i = vec3<i32>(local_index);
    let value: u32 = textureLoad(material_field, voxel_i, 0).x;

    if (value == 0) {
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else if (value == 2) {
        out.color = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    } else if (value == 3) {
        out.color = vec4<f32>(0.0, 1.0, 1.0, 1.0);
    } else {
        out.color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
        return out;
    }

fn world_pos_to_chunk_coord(world_pos: vec3<f32>) -> vec3<i32> {
    let offset_pos = world_pos + vec3<f32>(HALF_CHUNK);
    let chunk = floor(offset_pos / CHUNK_WORLD_SIZE);
    return vec3<i32>(chunk);
}

fn world_pos_to_voxel_index(
    world_pos: vec3<f32>,
    chunk_coord: vec3<i32>,
) -> vec3<u32> {
    let chunk_world_center = vec3<f32>(chunk_coord) * CHUNK_WORLD_SIZE;
    let chunk_world_min = chunk_world_center - vec3<f32>(HALF_CHUNK);
    let relative_pos =
        world_pos - chunk_world_min;
    let voxel = floor(relative_pos / VOXEL_WORLD_SIZE);
    return vec3<u32>(voxel);
}