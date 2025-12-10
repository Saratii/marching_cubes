use std::{collections::HashSet, hint::black_box};

use bevy::math::Vec3;
use criterion::{Criterion, criterion_group, criterion_main};
use marching_cubes::{sparse_voxel_octree::ChunkSvo, terrain::terrain::MAX_RADIUS};

fn benchmark_svo_first_traversal(c: &mut Criterion) {
    c.bench_function("bench_create_new_svo", |b| {
        b.iter(|| {
            let mut svo = ChunkSvo::new();
            svo.root.fill_missing_chunks_in_radius(
                &Vec3::ZERO,
                MAX_RADIUS,
                &mut HashSet::new(),
                &mut Vec::new(),
            );
            black_box(svo);
        })
    });
}

criterion_group!(benches, benchmark_svo_first_traversal);
criterion_main!(benches);
