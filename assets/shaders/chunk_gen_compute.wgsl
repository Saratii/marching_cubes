const SAMPLES_PER_CHUNK_DIM: u32 = 32u;
const SAMPLES_PER_CHUNK: u32 = SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;

struct Params {
    chunk_start : vec3<f32>,
    _pad : f32,
};

struct Output {
    densities : array<i32, SAMPLES_PER_CHUNK>,
    materials : array<u32, SAMPLES_PER_CHUNK>,
};

@group(0) @binding(0)
var<storage, read> params : Params;

@group(0) @binding(1)
var<storage, read_write> output : Output;

@compute @workgroup_size(64)
fn generate_terrain(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_index = global_id.x;
    if (sample_index >= SAMPLES_PER_CHUNK) {
        return;
    }
    let chunk_index = global_id.y;
    let output_index = chunk_index * SAMPLES_PER_CHUNK + sample_index;
    output.densities[output_index] = i32(sample_index) + 1;
    output.materials[output_index] = 1u;
}