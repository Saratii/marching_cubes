use std::hint::black_box;

use bevy::math::Vec3;
use criterion::{Criterion, criterion_group, criterion_main};

use marching_cubes::{
    deformable_terrain::sparse_voxel_octree::SvoNode,
    ui::configurable_settings::DEFAULT_RENDER_RADIUS_SQUARED,
};
use rustc_hash::FxHashSet;

fn benchmark_svo_first_traversal(c: &mut Criterion) {
    c.bench_function("bench_create_new_svo", |b| {
        b.iter(|| {
            let mut svo = SvoNode::world_root();
            svo.fill_missing_chunks_in_radius(
                &Vec3::ZERO,
                DEFAULT_RENDER_RADIUS_SQUARED,
                &mut FxHashSet::default(),
                &mut Vec::new(),
            );
            black_box(svo);
        })
    });
}

criterion_group!(benches, benchmark_svo_first_traversal);
criterion_main!(benches);
