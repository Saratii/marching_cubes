use std::{collections::HashSet, fs::OpenOptions, sync::Arc};

use bevy::{
    asset::{Assets, Handle},
    ecs::{
        bundle::{Bundle, DynamicBundle, NoBundleEffect},
        entity::Entity,
        event::{EventReader, EventWriter},
        query::{With, Without},
        resource::Resource,
        system::{Command, Commands, Res, ResMut, Single},
        world::{Mut, World},
    },
    math::{Affine3A, Vec3},
    pbr::{MeshMaterial3d, StandardMaterial},
    render::{
        mesh::{Mesh, Mesh3d},
        primitives::{Aabb, Frustum},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on},
    transform::components::Transform,
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::chunk_loader::{
        ChunkDataFile, ChunkIndexFile, ChunkIndexMap, create_chunk_file_data, load_chunk_data,
    },
    marching_cubes::march_cubes,
    player::player::{MainCameraTag, PlayerTag},
    terrain::{
        chunk_generator::{GenerateChunkEvent, LoadChunksEvent},
        terrain::{
            CHUNK_SIZE, CUBES_PER_CHUNK_DIM, ChunkMap, ChunkTag, L1_RADIUS_SQUARED, L2_RADIUS,
            L2_RADIUS_SQUARED, NoiseFunction, SDF_VALUES_PER_CHUNK_DIM,
            StandardTerrainMaterialHandle, TerrainChunk,
        },
    },
};
use fastnoise2::SafeNode;
use fastnoise2::generator::GeneratorWrapper;
const MAX_CHUNKS_PER_TASK: usize = 10000000;

#[derive(Resource)]
pub struct MyMapGenTasks {
    pub generation_tasks: Vec<
        Task<(
            Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform, Collider)>,
            Vec<((i16, i16, i16), TerrainChunk, Transform)>,
        )>,
    >,
    pub chunks_being_generated: HashSet<(i16, i16, i16)>,
}

#[derive(Resource)]
pub struct LoadChunkTasks {
    pub loading_tasks: Vec<
        Task<(
            Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform, Collider)>,
            Vec<((i16, i16, i16), TerrainChunk, Transform)>,
        )>,
    >,
    pub chunks_being_loaded: HashSet<(i16, i16, i16)>,
}

struct ChunkSpawnCommand<B> {
    pub bundles: Vec<((i16, i16, i16), TerrainChunk, B)>,
}

impl<B> Command for ChunkSpawnCommand<B>
where
    B: Bundle,
    <B as DynamicBundle>::Effect: NoBundleEffect,
{
    fn apply(self, world: &mut World) {
        let mut coords_chunks = Vec::with_capacity(self.bundles.len());
        let bundles: Vec<B> = self
            .bundles
            .into_iter()
            .map(|(coord, chunk_data, bundle)| {
                coords_chunks.push((coord, chunk_data));
                bundle
            })
            .collect();
        let entities: Vec<Entity> = world.spawn_batch(bundles).collect();
        world.resource_scope(|world, mut chunk_map: Mut<ChunkMap>| {
            world.resource_scope(|world, mut chunk_data_file: Mut<ChunkDataFile>| {
                world.resource_scope(|world, chunk_index_map: Mut<ChunkIndexMap>| {
                    let mut index_file = world.get_resource_mut::<ChunkIndexFile>().unwrap();
                    for ((coord, chunk_data), entity) in coords_chunks.into_iter().zip(entities) {
                        let mut locked_index_map = chunk_index_map.0.lock().unwrap();
                        create_chunk_file_data(
                            &chunk_data,
                            coord,
                            &mut locked_index_map,
                            &mut chunk_data_file.0,
                            &mut index_file.0,
                        );
                        drop(locked_index_map);
                        chunk_map.0.insert(coord, (entity, chunk_data));
                    }
                });
            });
        });
    }
}

struct ChunkLoadCommand<B> {
    pub bundles: Vec<((i16, i16, i16), TerrainChunk, B)>,
}

impl<B> Command for ChunkLoadCommand<B>
where
    B: Bundle,
    <B as DynamicBundle>::Effect: NoBundleEffect,
{
    fn apply(self, world: &mut World) {
        #[cfg(feature = "timers")]
        let s = std::time::Instant::now();
        // Split into coords/chunks and bundles while keeping everything in one pass
        let mut coords_chunks = Vec::with_capacity(self.bundles.len());
        let bundles: Vec<B> = self
            .bundles
            .into_iter()
            .map(|(coord, chunk, bundle)| {
                coords_chunks.push((coord, chunk));
                bundle
            })
            .collect();
        // Batch spawn bundles
        let entities: Vec<Entity> = world.spawn_batch(bundles).collect();
        let mut chunk_map = world.get_resource_mut::<ChunkMap>().unwrap();
        //preallocated hashsmap
        chunk_map.0.reserve(entities.len());
        // Pair entities with coords/chunks without separate zipping of pre-existing vecs
        for ((coord, chunk), entity) in coords_chunks.into_iter().zip(entities) {
            let a = chunk_map.0.insert(coord, (entity, chunk));

            assert!(a.is_none(), "Inserted chunk at occupied coord");
        }
        #[cfg(feature = "timers")]
        {
            let duration = s.elapsed();
            println!("spent {:?} in ChunkLoadCommand", duration);
        }
    }
}

pub fn catch_chunk_generation_request(
    mut chunk_generation_events: EventReader<GenerateChunkEvent>,
    fbm: Res<NoiseFunction>,
    mut my_tasks: ResMut<MyMapGenTasks>,
) {
    let noise_gen = fbm.0.clone();
    let task_pool = AsyncComputeTaskPool::get();
    for event in chunk_generation_events.read() {
        for chunk_batch in event.chunk_coords.chunks(MAX_CHUNKS_PER_TASK) {
            let chunk_coords = chunk_batch.to_vec();
            let noise_gen_clone = noise_gen.clone();
            let task = task_pool.spawn(async move {
                let results = chunk_coords
                    .par_iter()
                    .map(|coord| generate_chunk_data(coord, &noise_gen_clone))
                    .collect::<Vec<_>>();
                let mut with_collider = Vec::new();
                let mut without_collider = Vec::new();
                for (coord, chunk, mesh_option, transform, collider_opt) in results {
                    if let (Some(collider), Some(mesh)) = (collider_opt, mesh_option) {
                        with_collider.push((coord, chunk, mesh, transform, collider));
                    } else {
                        without_collider.push((coord, chunk, transform));
                    }
                }
                (with_collider, without_collider)
            });
            my_tasks.generation_tasks.push(task);
        }
    }
}

pub fn catch_load_generation_request(
    mut chunk_load_events: EventReader<LoadChunksEvent>,
    mut my_tasks: ResMut<LoadChunkTasks>,
    chunk_index_map: Res<ChunkIndexMap>,
) {
    let task_pool = AsyncComputeTaskPool::get();
    for event in chunk_load_events.read() {
        let chunk_data_file = OpenOptions::new()
            .read(true)
            .open("data/chunk_data.txt")
            .unwrap();
        let chunk_index_map = Arc::clone(&chunk_index_map.0);
        let chunk_coords = event.chunk_coords.clone();
        let task = task_pool.spawn(async move {
            let results = chunk_coords
                .iter()
                .map(|coord| {
                    let chunk_index_map = chunk_index_map.lock().unwrap();
                    let terrain_chunk = load_chunk_data(&chunk_data_file, &chunk_index_map, coord);
                    drop(chunk_index_map);
                    let mesh = march_cubes(
                        &terrain_chunk.sdfs,
                        CUBES_PER_CHUNK_DIM,
                        SDF_VALUES_PER_CHUNK_DIM,
                    );
                    let transform = Transform::from_translation(chunk_coord_to_world_pos(coord));
                    let (collider, return_mesh) = if mesh.count_vertices() > 0 {
                        (
                            Collider::from_bevy_mesh(
                                &mesh,
                                &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                            ),
                            Some(mesh),
                        )
                    } else {
                        (None, None)
                    };
                    (*coord, terrain_chunk, return_mesh, transform, collider)
                })
                .collect::<Vec<_>>();
            let mut with_collider = Vec::new();
            let mut without_collider = Vec::new();
            for (coord, chunk, mesh, transform, collider_opt) in results {
                if let (Some(collider), Some(mesh)) = (collider_opt, mesh) {
                    with_collider.push((coord, chunk, mesh, transform, collider));
                } else {
                    without_collider.push((coord, chunk, transform));
                }
            }
            (with_collider, without_collider)
        });
        my_tasks.loading_tasks.push(task);
    }
}

pub fn generate_chunk_data(
    coord: &(i16, i16, i16),
    noise_gen: &GeneratorWrapper<SafeNode>,
) -> (
    (i16, i16, i16),
    TerrainChunk,
    Option<Mesh>,
    Transform,
    Option<Collider>,
) {
    let terrain_chunk = TerrainChunk::new(*coord, noise_gen);
    let mesh = march_cubes(
        &terrain_chunk.sdfs,
        CUBES_PER_CHUNK_DIM,
        SDF_VALUES_PER_CHUNK_DIM,
    );
    let transform = Transform::from_translation(chunk_coord_to_world_pos(coord));
    let (collider, return_mesh) = if mesh.count_vertices() > 0 {
        (
            Collider::from_bevy_mesh(
                &mesh,
                &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
            ),
            Some(mesh),
        )
    } else {
        (None, None)
    };
    (*coord, terrain_chunk, return_mesh, transform, collider)
}

pub fn spawn_generated_chunks(
    mut my_tasks: ResMut<MyMapGenTasks>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    frustum: Single<&Frustum, With<MainCameraTag>>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        if task.is_finished() {
            let (bundle_data_with_collider, bundle_data_without_collider) = block_on(task);
            let (final_bundles_without_collider, final_bundles_with_collider) =
                spawn_chunks_from_source_task(
                    bundle_data_with_collider,
                    bundle_data_without_collider,
                    &player_transform.translation,
                    &mut chunk_coords_to_be_removed,
                    &mut meshes,
                    &material_handle.0,
                    &frustum,
                );
            let chunk_spawner = ChunkSpawnCommand {
                bundles: final_bundles_without_collider,
            };
            let chunk_spawner_with_collider = ChunkSpawnCommand {
                bundles: final_bundles_with_collider,
            };
            commands.queue(chunk_spawner);
            commands.queue(chunk_spawner_with_collider);
            return false;
        }
        true
    });
    for coord in chunk_coords_to_be_removed {
        my_tasks.chunks_being_generated.remove(&coord);
    }
}

pub fn finish_chunk_loading(
    mut my_tasks: ResMut<LoadChunkTasks>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
    frustum: Single<&Frustum, With<MainCameraTag>>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.loading_tasks.retain_mut(|task| {
        if task.is_finished() {
            let (bundle_data_with_collider, bundle_data_without_collider) = block_on(task);
            let (final_bundles_without_collider, final_bundles_with_collider) =
                spawn_chunks_from_source_task(
                    bundle_data_with_collider,
                    bundle_data_without_collider,
                    &player_transform.translation,
                    &mut chunk_coords_to_be_removed,
                    &mut meshes,
                    &material_handle.0,
                    &frustum,
                );
            let chunk_spawner = ChunkLoadCommand {
                bundles: final_bundles_without_collider,
            };
            let chunk_spawner_with_collider = ChunkLoadCommand {
                bundles: final_bundles_with_collider,
            };
            commands.queue(chunk_spawner);
            commands.queue(chunk_spawner_with_collider);
            return false;
        }
        true
    });
    for coord in chunk_coords_to_be_removed {
        my_tasks.chunks_being_loaded.remove(&coord);
    }
}

fn spawn_chunks_from_source_task(
    bundle_data_with_collider: Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform, Collider)>,
    bundle_data_without_collider: Vec<((i16, i16, i16), TerrainChunk, Transform)>,
    player_translation: &Vec3,
    chunk_coords_to_be_removed: &mut Vec<(i16, i16, i16)>,
    meshes: &mut ResMut<Assets<Mesh>>,
    material_handle: &Handle<StandardMaterial>,
    frustum: &Frustum,
) -> (
    Vec<((i16, i16, i16), TerrainChunk, (ChunkTag, Transform))>,
    Vec<(
        (i16, i16, i16),
        TerrainChunk,
        (
            Mesh3d,
            ChunkTag,
            Transform,
            MeshMaterial3d<StandardMaterial>,
            Collider,
        ),
    )>,
) {
    #[cfg(feature = "timers")]
    let start = std::time::Instant::now();
    let current_player_chunk = world_pos_to_chunk_coord(player_translation);
    let player_chunk_world_pos = chunk_coord_to_world_pos(&current_player_chunk);
    //reserve space
    chunk_coords_to_be_removed
        .reserve(bundle_data_with_collider.len() + bundle_data_without_collider.len());
    chunk_coords_to_be_removed.extend(bundle_data_with_collider.iter().map(|(coord, ..)| *coord));
    chunk_coords_to_be_removed.extend(
        bundle_data_without_collider
            .iter()
            .map(|(coord, ..)| *coord),
    );
    let mut aabb = Aabb {
        center: Vec3::ZERO.into(),
        half_extents: Vec3::splat(CHUNK_SIZE).into(),
    };
    let material = MeshMaterial3d(material_handle.clone());
    let final_bundles_with_collider: Vec<_> = bundle_data_with_collider
        .into_iter()
        .filter_map(|(coord, chunk, mesh, transform, collider)| {
            aabb.center = transform.translation.into();
            let distance_squared = transform
                .translation
                .distance_squared(player_chunk_world_pos);
            (distance_squared <= L1_RADIUS_SQUARED
                || frustum.intersects_obb(&aabb, &Affine3A::IDENTITY, true, true)
                    && distance_squared <= L2_RADIUS_SQUARED)
                .then(|| {
                    (
                        coord,
                        chunk,
                        (
                            Mesh3d(meshes.add(mesh)),
                            ChunkTag,
                            transform,
                            material.clone(),
                            collider,
                        ),
                    )
                })
        })
        .collect();
    let final_bundles_without_collider: Vec<_> = bundle_data_without_collider
        .into_iter()
        .filter_map(|(coord, chunk, transform)| {
            (transform
                .translation
                .distance_squared(player_chunk_world_pos)
                <= L1_RADIUS_SQUARED)
                .then(|| (coord, chunk, (ChunkTag, transform)))
        })
        .collect();
    #[cfg(feature = "timers")]
    {
        let duration = start.elapsed();
        println!("finished loading chunks in {:?}", duration);
    }
    (final_bundles_without_collider, final_bundles_with_collider)
}

//load chunks within L2 range that are also in the frustum. Triggered by changing frustum angle.

pub fn l2_chunk_load(
    chunk_map: Res<ChunkMap>,
    player_transform: Single<&mut Transform, (With<PlayerTag>, Without<MainCameraTag>)>,
    mut chunk_generation_events: EventWriter<GenerateChunkEvent>,
    mut chunk_load_event_writer: EventWriter<LoadChunksEvent>,
    mut map_gen_tasks: ResMut<MyMapGenTasks>,
    mut load_chunk_tasks: ResMut<LoadChunkTasks>,
    chunk_index_map: Res<ChunkIndexMap>,
    frustum: Single<&Frustum, With<MainCameraTag>>,
) {
    let mut chunks_coords_to_generate = Vec::new();
    let mut chunk_coords_to_load = Vec::new();
    let origin_chunk = world_pos_to_chunk_coord(&player_transform.translation);
    let origin_chunk_world_pos = chunk_coord_to_world_pos(&origin_chunk);
    let min_world_pos = origin_chunk_world_pos - Vec3::splat(L2_RADIUS);
    let max_world_pos = origin_chunk_world_pos + Vec3::splat(L2_RADIUS);
    let min_chunk = world_pos_to_chunk_coord(&min_world_pos);
    let max_chunk = world_pos_to_chunk_coord(&max_world_pos);
    let mut aabb = Aabb {
        center: Vec3::ZERO.into(),
        half_extents: Vec3::splat(CHUNK_SIZE).into(),
    };
    for chunk_x in min_chunk.0..=max_chunk.0 {
        for chunk_z in min_chunk.2..=max_chunk.2 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                let chunk_coord = (chunk_x, chunk_y, chunk_z);
                let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
                let distance_squared = chunk_world_pos.distance_squared(origin_chunk_world_pos);
                aabb.center = chunk_world_pos.into();
                if distance_squared < L2_RADIUS_SQUARED
                    && frustum.intersects_obb(&aabb, &Affine3A::IDENTITY, true, true)
                {
                    if !chunk_map.0.contains_key(&chunk_coord)
                        && !map_gen_tasks.chunks_being_generated.contains(&chunk_coord)
                        && !load_chunk_tasks.chunks_being_loaded.contains(&chunk_coord)
                    {
                        let chunk_index_map = chunk_index_map.0.lock().unwrap();
                        if chunk_index_map.contains_key(&chunk_coord) {
                            chunk_coords_to_load.push(chunk_coord);
                            load_chunk_tasks.chunks_being_loaded.insert(chunk_coord);
                        } else {
                            chunks_coords_to_generate.push(chunk_coord);
                            map_gen_tasks.chunks_being_generated.insert(chunk_coord);
                        }
                        drop(chunk_index_map);
                    }
                }
            }
        }
    }
    if chunks_coords_to_generate.len() > 0 {
        chunk_generation_events.write(GenerateChunkEvent {
            chunk_coords: chunks_coords_to_generate,
        });
    }
    if chunk_coords_to_load.len() > 0 {
        chunk_load_event_writer.write(LoadChunksEvent {
            chunk_coords: chunk_coords_to_load,
        });
    }
}
