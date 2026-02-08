use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use marching_cubes::{data_loader::column_range_map::ColumnRangeMap, terrain::terrain::Uniformity};

fn benchmark_insert_sequential(c: &mut Criterion) {
    c.bench_function("insert_sequential", |b| {
        b.iter(|| {
            let mut map = ColumnRangeMap::new();
            for y in -100..100 {
                black_box(map.insert(black_box((0, y, 0)), black_box(Uniformity::Air)));
            }
            black_box(map);
        })
    });
}

fn benchmark_insert_random_order(c: &mut Criterion) {
    let ys: Vec<i16> = (0..200).map(|i| (i * 7919) % 200 - 100).collect();
    c.bench_function("insert_random_order", |b| {
        b.iter(|| {
            let mut map = ColumnRangeMap::new();
            for &y in &ys {
                black_box(map.insert(black_box((0, y, 0)), black_box(Uniformity::Air)));
            }
            black_box(map);
        })
    });
}

fn benchmark_insert_many_columns(c: &mut Criterion) {
    c.bench_function("insert_many_columns", |b| {
        b.iter(|| {
            let mut map = ColumnRangeMap::new();
            for x in -10..10 {
                for z in -10..10 {
                    for y in -5..5 {
                        black_box(map.insert(black_box((x, y, z)), black_box(Uniformity::Air)));
                    }
                }
            }
            black_box(map);
        })
    });
}

fn benchmark_insert_interleaved_uniformities(c: &mut Criterion) {
    c.bench_function("insert_interleaved_uniformities", |b| {
        b.iter(|| {
            let mut map = ColumnRangeMap::new();
            for y in -50..50 {
                let uniformity = if y < 0 {
                    Uniformity::Dirt
                } else {
                    Uniformity::Air
                };
                black_box(map.insert(black_box((0, y, 0)), black_box(uniformity)));
            }
            black_box(map);
        })
    });
}

fn benchmark_contains_hit(c: &mut Criterion) {
    let mut map = ColumnRangeMap::new();
    for x in -10..10 {
        for z in -10..10 {
            for y in -1000..100 {
                map.insert((x, y, z), Uniformity::Air);
            }
        }
    }
    c.bench_function("contains_hit", |b| {
        b.iter(|| {
            for y in -100..100 {
                black_box(map.contains(black_box((0, y, 0))));
            }
        })
    });
}

fn benchmark_contains_miss(c: &mut Criterion) {
    let mut map = ColumnRangeMap::new();
    for x in -10..10 {
        for z in -10..10 {
            for y in -1000..100 {
                map.insert((x, y, z), Uniformity::Air);
            }
        }
    }
    c.bench_function("contains_miss", |b| {
        b.iter(|| {
            for y in 100..200 {
                black_box(map.contains(black_box((0, y, 0))));
            }
        })
    });
}

fn benchmark_contains_many_columns(c: &mut Criterion) {
    let mut map = ColumnRangeMap::new();
    for x in -10..10 {
        for z in -10..10 {
            for y in -5..5 {
                map.insert((x, y, z), Uniformity::Air);
            }
        }
    }
    c.bench_function("contains_many_columns", |b| {
        b.iter(|| {
            for x in -10..10 {
                for z in -10..10 {
                    for y in -5..5 {
                        black_box(map.contains(black_box((x, y, z))));
                    }
                }
            }
        })
    });
}

fn benchmark_mixed_operations(c: &mut Criterion) {
    c.bench_function("mixed_operations", |b| {
        b.iter(|| {
            let mut map = ColumnRangeMap::new();
            for y in -50..50 {
                black_box(map.insert(black_box((0, y, 0)), black_box(Uniformity::Air)));
                if y % 5 == 0 {
                    black_box(map.contains(black_box((0, y - 10, 0))));
                }
            }
            black_box(map);
        })
    });
}

criterion_group!(
    benches,
    benchmark_insert_sequential,
    benchmark_insert_random_order,
    benchmark_insert_many_columns,
    benchmark_insert_interleaved_uniformities,
    benchmark_contains_hit,
    benchmark_contains_miss,
    benchmark_contains_many_columns,
    benchmark_mixed_operations,
);
criterion_main!(benches);

//cargo bench --bench column_range_map -- contains_miss
