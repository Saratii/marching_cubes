use std::hint::black_box;

use bevy::math::Vec3;
use criterion::{Criterion, criterion_group, criterion_main};
use marching_cubes::{constants::MAX_RENDER_RADIUS_SQUARED, sparse_voxel_octree::ChunkSvo};
use rustc_hash::FxHashSet;

fn benchmark_svo_first_traversal(c: &mut Criterion) {
    c.bench_function("bench_create_new_svo", |b| {
        b.iter(|| {
            let mut svo = ChunkSvo::new();
            svo.root.fill_missing_chunks_in_radius(
                &Vec3::ZERO,
                MAX_RENDER_RADIUS_SQUARED,
                &mut FxHashSet::default(),
                &mut Vec::new(),
            );
            black_box(svo);
        })
    });
}

criterion_group!(benches, benchmark_svo_first_traversal);
criterion_main!(benches);
