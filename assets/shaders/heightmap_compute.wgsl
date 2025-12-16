// 2D Heightmap Generator Compute Shader
// Generates height values for a 2D grid

const SAMPLES_PER_CHUNK_DIM: u32 = 32u;
const SAMPLES_PER_CHUNK: u32 = SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM;

struct Params {
    chunk_start : vec2<f32>,  // Now 2D position
    _pad : vec2<f32>,
};

struct Output {
    heights : array<f32, SAMPLES_PER_CHUNK>,
};

@group(0) @binding(0)
var<storage, read> params : Params;

@group(0) @binding(1)
var<storage, read_write> output : Output;

// Simple noise function for heightmap generation
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
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
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
    if (sample_index >= SAMPLES_PER_CHUNK) {
        return;
    }
    
    // Convert 1D index to 2D coordinates
    let x = sample_index % SAMPLES_PER_CHUNK_DIM;
    let z = sample_index / SAMPLES_PER_CHUNK_DIM;
    
    // Calculate world position
    let world_pos = params.chunk_start + vec2<f32>(f32(x), f32(z));
    
    // Generate height using noise
    let height = fbm(world_pos * 0.01) * 100.0;
    
    output.heights[sample_index] = height;
}