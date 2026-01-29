use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use marching_cubes::terrain::{
    chunk_generator::{HEIGHT_MAP_GRID_SIZE, sample_fbm},
    terrain::{CHUNK_SIZE, SAMPLES_PER_CHUNK_DIM},
};

fn benchmark_full_chunk_noise(c: &mut Criterion) {
    //call noise on every sample in the chunk
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("full_chunk_noise", |b| {
        b.iter(|| {
            let start_x = 0.0;
            let start_z = 0.0;
            let step = CHUNK_SIZE as f32 / SAMPLES_PER_CHUNK_DIM as f32;
            let mut roller = 0;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let sample_x = start_x + (x as f32 * step);
                for z in 0..SAMPLES_PER_CHUNK_DIM {
                    let sample_z = start_z + (z as f32 * step);
                    let height = sample_fbm(&noise_function, sample_x, sample_z);
                    heightmap_buffer[roller] = height;
                    roller += 1;
                }
            }
            black_box(&heightmap_buffer);
        })
    });
}

fn benchmark_corner_lerp_noise(c: &mut Criterion) {
    //only call noise on the 4 corners and lerp in between
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("corner_lerp_noise", |b| {
        b.iter(|| {
            let start_x = 0.0;
            let start_z = 0.0;
            let top_left = sample_fbm(&noise_function, start_x, start_z);
            let top_right = sample_fbm(&noise_function, start_x + CHUNK_SIZE as f32, start_z);
            let bottom_left = sample_fbm(&noise_function, start_x, start_z + CHUNK_SIZE as f32);
            let bottom_right = sample_fbm(
                &noise_function,
                start_x + CHUNK_SIZE as f32,
                start_z + CHUNK_SIZE as f32,
            );
            let inv_samples = 1.0 / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
            let mut roller = 0;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let t_x = x as f32 * inv_samples;
                let top = lerp(top_left, top_right, t_x);
                let bottom = lerp(bottom_left, bottom_right, t_x);
                for z in 0..SAMPLES_PER_CHUNK_DIM {
                    let t_z = z as f32 * inv_samples;
                    let height = lerp(top, bottom, t_z);
                    heightmap_buffer[roller] = height;
                    roller += 1;
                }
            }
            black_box(&heightmap_buffer);
        })
    });
}

fn benchmark_grid_3x3_noise(c: &mut Criterion) {
    //sample the 4 corners, plus the midpoints of each edge and the center, then bilerp between them
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("grid_3x3_noise", |b| {
        b.iter(|| {
            let start_x = 0.0;
            let start_z = 0.0;
            let half_chunk = CHUNK_SIZE as f32 / 2.0;
            let grid = [
                [
                    sample_fbm(&noise_function, start_x, start_z),
                    sample_fbm(&noise_function, start_x + half_chunk, start_z),
                    sample_fbm(&noise_function, start_x + CHUNK_SIZE as f32, start_z),
                ],
                [
                    sample_fbm(&noise_function, start_x, start_z + half_chunk),
                    sample_fbm(&noise_function, start_x + half_chunk, start_z + half_chunk),
                    sample_fbm(
                        &noise_function,
                        start_x + CHUNK_SIZE as f32,
                        start_z + half_chunk,
                    ),
                ],
                [
                    sample_fbm(&noise_function, start_x, start_z + CHUNK_SIZE as f32),
                    sample_fbm(
                        &noise_function,
                        start_x + half_chunk,
                        start_z + CHUNK_SIZE as f32,
                    ),
                    sample_fbm(
                        &noise_function,
                        start_x + CHUNK_SIZE as f32,
                        start_z + CHUNK_SIZE as f32,
                    ),
                ],
            ];
            let inv_samples = 1.0 / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
            let mut roller = 0;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let t_x = x as f32 * inv_samples;
                let grid_x = t_x * 2.0;
                let grid_x_idx = grid_x.floor() as usize;
                let grid_x_next = (grid_x_idx + 1).min(2);
                let local_t_x = grid_x - grid_x_idx as f32;
                for z in 0..SAMPLES_PER_CHUNK_DIM {
                    let t_z = z as f32 * inv_samples;
                    let grid_z = t_z * 2.0;
                    let grid_z_idx = grid_z.floor() as usize;
                    let grid_z_next = (grid_z_idx + 1).min(2);
                    let local_t_z = grid_z - grid_z_idx as f32;
                    let top = lerp(
                        grid[grid_z_idx][grid_x_idx],
                        grid[grid_z_idx][grid_x_next],
                        local_t_x,
                    );
                    let bottom = lerp(
                        grid[grid_z_next][grid_x_idx],
                        grid[grid_z_next][grid_x_next],
                        local_t_x,
                    );
                    let height = lerp(top, bottom, local_t_z);
                    heightmap_buffer[roller] = height;
                    roller += 1;
                }
            }
            black_box(&heightmap_buffer);
        })
    });
}

fn benchmark_corner_bicubic_noise(c: &mut Criterion) {
    //sample a 4x4 grid with corners at inner positions, then bicubic interpolate
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("corner_bicubic_noise", |b| {
        b.iter(|| {
            let start_x = 0.0;
            let start_z = 0.0;
            let step = CHUNK_SIZE as f32;
            let mut grid = [[0.0f32; 4]; 4];
            for gz in 0..4 {
                let sample_z = start_z + ((gz as i32 - 1) as f32 * step);
                for gx in 0..4 {
                    let sample_x = start_x + ((gx as i32 - 1) as f32 * step);
                    grid[gz][gx] = sample_fbm(&noise_function, sample_x, sample_z);
                }
            }
            let inv_samples = 1.0 / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
            let mut roller = 0;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let t_x = x as f32 * inv_samples;
                let local_t_x = t_x + 1.0;
                for z in 0..SAMPLES_PER_CHUNK_DIM {
                    let t_z = z as f32 * inv_samples;
                    let local_t_z = t_z + 1.0;
                    let height = bicubic_interp(&grid, local_t_x, local_t_z);
                    heightmap_buffer[roller] = height;
                    roller += 1;
                }
            }
            black_box(&heightmap_buffer);
        })
    });
}

fn benchmark_grid_bicubic_noise(c: &mut Criterion) {
    //sample a 5x5 evenly spaced grid, then bicubic interpolate between points
    let noise_function =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut heightmap_buffer = [0.0; HEIGHT_MAP_GRID_SIZE];
    c.bench_function("grid_bicubic_noise", |b| {
        b.iter(|| {
            let start_x = 0.0;
            let start_z = 0.0;
            let step = CHUNK_SIZE as f32 / 4.0;
            let mut grid = [[0.0f32; 5]; 5];
            for gz in 0..5 {
                let sample_z = start_z + (gz as f32 * step);
                for gx in 0..5 {
                    let sample_x = start_x + (gx as f32 * step);
                    grid[gz][gx] = sample_fbm(&noise_function, sample_x, sample_z);
                }
            }
            let inv_samples = 1.0 / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
            let mut roller = 0;
            for x in 0..SAMPLES_PER_CHUNK_DIM {
                let t_x = x as f32 * inv_samples;
                let grid_x = t_x * 4.0;
                let grid_x_idx = grid_x.floor() as usize;
                let gx0 = grid_x_idx.saturating_sub(1).min(4);
                let gx1 = grid_x_idx.min(4);
                let gx2 = (grid_x_idx + 1).min(4);
                let gx3 = (grid_x_idx + 2).min(4);
                let local_t_x = grid_x - grid_x_idx as f32;
                for z in 0..SAMPLES_PER_CHUNK_DIM {
                    let t_z = z as f32 * inv_samples;
                    let grid_z = t_z * 4.0;
                    let grid_z_idx = grid_z.floor() as usize;
                    let gz0 = grid_z_idx.saturating_sub(1).min(4);
                    let gz1 = grid_z_idx.min(4);
                    let gz2 = (grid_z_idx + 1).min(4);
                    let gz3 = (grid_z_idx + 2).min(4);
                    let local_t_z = grid_z - grid_z_idx as f32;
                    let sub_grid = [
                        [
                            grid[gz0][gx0],
                            grid[gz0][gx1],
                            grid[gz0][gx2],
                            grid[gz0][gx3],
                        ],
                        [
                            grid[gz1][gx0],
                            grid[gz1][gx1],
                            grid[gz1][gx2],
                            grid[gz1][gx3],
                        ],
                        [
                            grid[gz2][gx0],
                            grid[gz2][gx1],
                            grid[gz2][gx2],
                            grid[gz2][gx3],
                        ],
                        [
                            grid[gz3][gx0],
                            grid[gz3][gx1],
                            grid[gz3][gx2],
                            grid[gz3][gx3],
                        ],
                    ];
                    let height = bicubic_interp(&sub_grid, local_t_x, local_t_z);
                    heightmap_buffer[roller] = height;
                    roller += 1;
                }
            }
            black_box(&heightmap_buffer);
        })
    });
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn cubic_interp(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let a0 = p3 - p2 - p0 + p1;
    let a1 = p0 - p1 - a0;
    let a2 = p2 - p0;
    a0 * t * t2 + a1 * t2 + a2 * t + p1
}

fn bicubic_interp(grid: &[[f32; 4]; 4], t_x: f32, t_z: f32) -> f32 {
    let col0 = cubic_interp(grid[0][0], grid[0][1], grid[0][2], grid[0][3], t_x);
    let col1 = cubic_interp(grid[1][0], grid[1][1], grid[1][2], grid[1][3], t_x);
    let col2 = cubic_interp(grid[2][0], grid[2][1], grid[2][2], grid[2][3], t_x);
    let col3 = cubic_interp(grid[3][0], grid[3][1], grid[3][2], grid[3][3], t_x);
    cubic_interp(col0, col1, col2, col3, t_z)
}

criterion_group!(
    benches,
    benchmark_full_chunk_noise,
    benchmark_corner_lerp_noise,
    benchmark_grid_3x3_noise,
    benchmark_corner_bicubic_noise,
    benchmark_grid_bicubic_noise,
);
criterion_main!(benches);

//cargo bench --bench noise_bench -- full_chunk_noise
