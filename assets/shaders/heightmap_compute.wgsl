const SAMPLES_PER_CHUNK_DIM: u32 = 50u;
const SAMPLES_PER_CHUNK: u32 = SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;
const NOISE_SAMPLES_DIM: u32 = 9u;  // 9x9 noise samples per chunk
const NOISE_SAMPLES_PER_CHUNK: u32 = NOISE_SAMPLES_DIM * NOISE_SAMPLES_DIM;
const CHUNKS_PER_CLUSTER_DIM: u32 = 5u;

struct Params {
    chunk_start: vec2<f32>,
    _pad: vec2<f32>,
};

struct Output {
    heights: array<f32, NOISE_SAMPLES_PER_CHUNK>,
};

struct ClusterParams {
    cluster_lower_chunk: vec3<i32>,
    _pad: i32,
};

struct ClusterOutput {
    heights: array<f32>,
};

// New: batch cluster structures
struct BatchClusterParams {
    cluster_count: u32,
    _pad: vec3<u32>,
    cluster_coords: array<vec2<i32>>,
};

// Single chunk generation bindings
@group(0) @binding(0)
var<storage, read> single_params: Params;

@group(0) @binding(1)
var<storage, read_write> single_output: Output;

// Single cluster generation bindings (reuse group 0)
@group(1) @binding(0)
var<storage, read> cluster_params: ClusterParams;

@group(1) @binding(1)
var<storage, read_write> cluster_output: ClusterOutput;

// Batch cluster generation bindings
@group(2) @binding(0)
var<storage, read> batch_cluster_params: BatchClusterParams;

@group(2) @binding(1)
var<storage, read_write> batch_cluster_output: ClusterOutput;

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));
    //return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    return 0.0;
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        value += amplitude * noise(pos * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

@compute @workgroup_size(64)
fn generate_heightmap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_index = global_id.x;
    if (sample_index >= NOISE_SAMPLES_PER_CHUNK) {
        return;
    }
    let x = sample_index % NOISE_SAMPLES_DIM;
    let z = sample_index / NOISE_SAMPLES_DIM;
    
    // Sample spacing to cover the chunk with 9x9 grid
    let spacing = f32(SAMPLES_PER_CHUNK_DIM) / f32(NOISE_SAMPLES_DIM - 1);
    let world_pos = single_params.chunk_start + vec2<f32>(f32(x) * spacing, f32(z) * spacing);
    
    let height = fbm(world_pos * 0.01) * 100.0;
    single_output.heights[sample_index] = height;
}

@compute @workgroup_size(64)
fn generate_cluster_heightmap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_index = global_id.y;
    let sample_index = global_id.x;
    if (sample_index >= NOISE_SAMPLES_PER_CHUNK || chunk_index >= CHUNKS_PER_CLUSTER_DIM * CHUNKS_PER_CLUSTER_DIM) {
        return;
    }
    let local_x = chunk_index % CHUNKS_PER_CLUSTER_DIM;
    let local_z = chunk_index / CHUNKS_PER_CLUSTER_DIM;
    let chunk_x = cluster_params.cluster_lower_chunk.x + i32(local_x);
    let chunk_z = cluster_params.cluster_lower_chunk.z + i32(local_z);
    let chunk_start = vec2<f32>(f32(chunk_x) * f32(SAMPLES_PER_CHUNK_DIM), f32(chunk_z) * f32(SAMPLES_PER_CHUNK_DIM));
    
    let x = sample_index % NOISE_SAMPLES_DIM;
    let z = sample_index / NOISE_SAMPLES_DIM;
    let spacing = f32(SAMPLES_PER_CHUNK_DIM) / f32(NOISE_SAMPLES_DIM - 1);
    let world_pos = chunk_start + vec2<f32>(f32(x) * spacing, f32(z) * spacing);
    
    let height = fbm(world_pos * 0.01) * 100.0;
    let output_index = chunk_index * NOISE_SAMPLES_PER_CHUNK + sample_index;
    cluster_output.heights[output_index] = height;
}

@compute @workgroup_size(64)
fn generate_batch_clusters(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cluster_index = global_id.z;
    let chunk_in_cluster_index = global_id.y;
    let sample_index = global_id.x;
    
    if (cluster_index >= batch_cluster_params.cluster_count || 
        chunk_in_cluster_index >= CHUNKS_PER_CLUSTER_DIM * CHUNKS_PER_CLUSTER_DIM ||
        sample_index >= NOISE_SAMPLES_PER_CHUNK) {
        return;
    }
    
    // Get the base cluster coordinate
    let cluster_coord = batch_cluster_params.cluster_coords[cluster_index];
    
    // Calculate chunk position within cluster
    let local_x = chunk_in_cluster_index % CHUNKS_PER_CLUSTER_DIM;
    let local_z = chunk_in_cluster_index / CHUNKS_PER_CLUSTER_DIM;
    
    let chunk_x = cluster_coord.x + i32(local_x);
    let chunk_z = cluster_coord.y + i32(local_z);
    
    let chunk_start = vec2<f32>(
        f32(chunk_x) * f32(SAMPLES_PER_CHUNK_DIM), 
        f32(chunk_z) * f32(SAMPLES_PER_CHUNK_DIM)
    );
    
    // Calculate sample position
    let x = sample_index % NOISE_SAMPLES_DIM;
    let z = sample_index / NOISE_SAMPLES_DIM;
    let spacing = f32(SAMPLES_PER_CHUNK_DIM) / f32(NOISE_SAMPLES_DIM - 1);
    let world_pos = chunk_start + vec2<f32>(f32(x) * spacing, f32(z) * spacing);
    
    // Generate height
    let height = fbm(world_pos * 0.01) * 100.0;
    
    // Calculate output index
    let chunks_per_cluster = CHUNKS_PER_CLUSTER_DIM * CHUNKS_PER_CLUSTER_DIM;
    let output_index = cluster_index * chunks_per_cluster * NOISE_SAMPLES_PER_CHUNK + 
                      chunk_in_cluster_index * NOISE_SAMPLES_PER_CHUNK + 
                      sample_index;
    
    batch_cluster_output.heights[output_index] = height;
}