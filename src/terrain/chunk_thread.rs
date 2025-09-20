use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    sync::Arc,
};

use bevy::{
    asset::{Assets, Handle},
    ecs::{
        bundle::{Bundle, DynamicBundle, NoBundleEffect},
        entity::Entity,
        event::EventReader,
        query::With,
        resource::Resource,
        system::{Command, Commands, Res, ResMut, Single},
        world::{Mut, World},
    },
    math::Vec3,
    pbr::{MeshMaterial3d, StandardMaterial},
    render::mesh::{Mesh, Mesh3d},
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
    player::player::PlayerTag,
    terrain::{
        chunk_generator::{GenerateChunkEvent, LoadChunksEvent},
        terrain::{
            CHUNK_CREATION_RADIUS_SQUARED, CUBES_PER_CHUNK_DIM, ChunkMap, ChunkTag, NoiseFunction,
            SDF_VALUES_PER_CHUNK_DIM, StandardTerrainMaterialHandle, TerrainChunk,
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
            Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform)>,
        )>,
    >,
    pub chunks_being_generated: HashSet<(i16, i16, i16)>,
}

#[derive(Resource)]
pub struct LoadChunkTasks {
    pub loading_tasks: Vec<
        Task<(
            Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform, Collider)>,
            Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform)>,
        )>,
    >,
    pub chunks_being_loaded: HashSet<(i16, i16, i16)>,
}

#[derive(Resource)]
pub struct StagedChunksLoaded(pub HashMap<(i16, i16, i16), (Entity, TerrainChunk)>);

struct ChunkSpawnCommand<B> {
    pub bundles: Vec<B>,
    pub chunk_datas: Vec<TerrainChunk>,
    pub chunk_coords: Vec<(i16, i16, i16)>,
}

impl<B> Command for ChunkSpawnCommand<B>
where
    B: Bundle,
    <B as DynamicBundle>::Effect: NoBundleEffect,
{
    fn apply(self, world: &mut World) {
        let entities: Vec<Entity> = world.spawn_batch(self.bundles).collect();
        world.resource_scope(|world, mut chunk_map: Mut<ChunkMap>| {
            world.resource_scope(|world, mut chunk_data_file: Mut<ChunkDataFile>| {
                world.resource_scope(|world, chunk_index_map: Mut<ChunkIndexMap>| {
                    let mut index_file = world.get_resource_mut::<ChunkIndexFile>().unwrap();
                    for ((coord, chunk_data), entity) in self
                        .chunk_coords
                        .into_iter()
                        .zip(self.chunk_datas.into_iter())
                        .zip(entities.into_iter())
                    {
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
    pub bundles: Vec<B>,
    pub chunk_datas: Vec<TerrainChunk>,
    pub chunk_coords: Vec<(i16, i16, i16)>,
}

impl<B> Command for ChunkLoadCommand<B>
where
    B: Bundle,
    <B as DynamicBundle>::Effect: NoBundleEffect,
{
    fn apply(self, world: &mut World) {
        let entities: Vec<Entity> = world.spawn_batch(self.bundles).collect();
        let mut staged = world.get_resource_mut::<StagedChunksLoaded>().unwrap();
        for ((coord, chunk_data), entity) in self
            .chunk_coords
            .into_iter()
            .zip(self.chunk_datas)
            .zip(entities)
        {
            staged.0.insert(coord, (entity, chunk_data));
        }
    }
}

pub fn flush_staged_chunks(
    mut staged: ResMut<StagedChunksLoaded>,
    mut chunk_map: ResMut<ChunkMap>,
) {
    for (coord, (entity, chunk)) in staged.0.drain() {
        chunk_map.0.insert(coord, (entity, chunk));
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
                for (coord, chunk, mesh, transform, collider_opt) in results {
                    match collider_opt {
                        Some(collider) => {
                            with_collider.push((coord, chunk, mesh, transform, collider));
                        }
                        None => {
                            without_collider.push((coord, chunk, mesh, transform));
                        }
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
                        &terrain_chunk.densities,
                        CUBES_PER_CHUNK_DIM,
                        SDF_VALUES_PER_CHUNK_DIM,
                    );
                    let transform = Transform::from_translation(chunk_coord_to_world_pos(coord));
                    let collider = if mesh.count_vertices() > 0 {
                        Collider::from_bevy_mesh(
                            &mesh,
                            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                        )
                    } else {
                        None
                    };
                    (*coord, terrain_chunk, mesh, transform, collider)
                })
                .collect::<Vec<_>>();
            let mut with_collider = Vec::new();
            let mut without_collider = Vec::new();
            for (coord, chunk, mesh, transform, collider_opt) in results {
                match collider_opt {
                    Some(collider) => {
                        with_collider.push((coord, chunk, mesh, transform, collider));
                    }
                    None => {
                        without_collider.push((coord, chunk, mesh, transform));
                    }
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
    Mesh,
    Transform,
    Option<Collider>,
) {
    let terrain_chunk = TerrainChunk::new(*coord, noise_gen);
    let mesh = march_cubes(
        &terrain_chunk.densities,
        CUBES_PER_CHUNK_DIM,
        SDF_VALUES_PER_CHUNK_DIM,
    );
    let transform = Transform::from_translation(chunk_coord_to_world_pos(coord));
    let collider = if mesh.count_vertices() > 0 {
        Collider::from_bevy_mesh(
            &mesh,
            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
        )
    } else {
        None
    };
    (*coord, terrain_chunk, mesh, transform, collider)
}

pub fn spawn_generated_chunks(
    mut my_tasks: ResMut<MyMapGenTasks>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        if task.is_finished() {
            let (bundle_data_with_collider, bundle_data_without_collider) = block_on(task);
            let (
                final_bundles_without_collider,
                chunk_data_without_collider,
                coords_without_collider,
                final_bundles_with_collider,
                chunk_data_with_collider,
                coords_with_collider,
            ) = spawn_chunks_from_source_task(
                bundle_data_with_collider,
                bundle_data_without_collider,
                &player_transform.translation,
                &mut chunk_coords_to_be_removed,
                &mut meshes,
                &material_handle.0,
            );
            let chunk_spawner = ChunkSpawnCommand {
                bundles: final_bundles_without_collider,
                chunk_datas: chunk_data_without_collider,
                chunk_coords: coords_without_collider,
            };
            let chunk_spawner_with_collider = ChunkSpawnCommand {
                bundles: final_bundles_with_collider,
                chunk_datas: chunk_data_with_collider,
                chunk_coords: coords_with_collider,
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
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.loading_tasks.retain_mut(|task| {
        if task.is_finished() {
            let (bundle_data_with_collider, bundle_data_without_collider) = block_on(task);
            let (
                final_bundles_without_collider,
                chunk_data_without_collider,
                coords_without_collider,
                final_bundles_with_collider,
                chunk_data_with_collider,
                coords_with_collider,
            ) = spawn_chunks_from_source_task(
                bundle_data_with_collider,
                bundle_data_without_collider,
                &player_transform.translation,
                &mut chunk_coords_to_be_removed,
                &mut meshes,
                &material_handle.0,
            );
            let chunk_spawner = ChunkLoadCommand {
                bundles: final_bundles_without_collider,
                chunk_datas: chunk_data_without_collider,
                chunk_coords: coords_without_collider,
            };
            let chunk_spawner_with_collider = ChunkLoadCommand {
                bundles: final_bundles_with_collider,
                chunk_datas: chunk_data_with_collider,
                chunk_coords: coords_with_collider,
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
    mut bundle_data_with_collider: Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform, Collider)>,
    mut bundle_data_without_collider: Vec<((i16, i16, i16), TerrainChunk, Mesh, Transform)>,
    player_translation: &Vec3,
    chunk_coords_to_be_removed: &mut Vec<(i16, i16, i16)>,
    meshes: &mut ResMut<Assets<Mesh>>,
    material_handle: &Handle<StandardMaterial>,
) -> (
    Vec<(
        Mesh3d,
        ChunkTag,
        Transform,
        MeshMaterial3d<StandardMaterial>,
    )>,
    Vec<TerrainChunk>,
    Vec<(i16, i16, i16)>,
    Vec<(
        Mesh3d,
        ChunkTag,
        Transform,
        MeshMaterial3d<StandardMaterial>,
        Collider,
    )>,
    Vec<TerrainChunk>,
    Vec<(i16, i16, i16)>,
) {
    let start = std::time::Instant::now();
    let current_player_chunk = world_pos_to_chunk_coord(player_translation);
    let player_chunk_world_pos = chunk_coord_to_world_pos(&current_player_chunk);
    chunk_coords_to_be_removed.extend(bundle_data_with_collider.iter().map(|(coord, ..)| *coord));
    chunk_coords_to_be_removed.extend(
        bundle_data_without_collider
            .iter()
            .map(|(coord, ..)| *coord),
    );
    bundle_data_with_collider.retain(|(_chunk_coord, _chunkj, _mesh, transform, _collider)| {
        transform
            .translation
            .distance_squared(player_chunk_world_pos)
            <= CHUNK_CREATION_RADIUS_SQUARED
    });
    bundle_data_without_collider.retain(|(_chunk_coord, _chunk, _mesh, transform)| {
        transform
            .translation
            .distance_squared(player_chunk_world_pos)
            <= CHUNK_CREATION_RADIUS_SQUARED
    });
    let mut coords_with_collider = Vec::with_capacity(bundle_data_with_collider.len());
    let mut chunk_data_with_collider = Vec::with_capacity(bundle_data_with_collider.len());
    let mut coords_without_collider = Vec::with_capacity(bundle_data_without_collider.len());
    let mut chunk_data_without_collider = Vec::with_capacity(bundle_data_without_collider.len());
    let mut final_bundles_with_collider = Vec::with_capacity(bundle_data_with_collider.len());
    let material = MeshMaterial3d(material_handle.clone());
    for (coords, chunk, mesh, transform, collider) in bundle_data_with_collider.into_iter() {
        coords_with_collider.push(coords);
        chunk_data_with_collider.push(chunk);
        final_bundles_with_collider.push((
            Mesh3d(meshes.add(mesh)),
            ChunkTag,
            transform,
            material.clone(),
            collider,
        ));
    }
    let mut final_bundles_without_collider = Vec::with_capacity(bundle_data_without_collider.len());
    for (coords, chunk, mesh, transform) in bundle_data_without_collider.into_iter() {
        coords_without_collider.push(coords);
        chunk_data_without_collider.push(chunk);
        final_bundles_without_collider.push((
            Mesh3d(meshes.add(mesh)),
            ChunkTag,
            transform,
            material.clone(),
        ));
    }
    let duration = start.elapsed();
    println!("finished loading chunks in {:?}", duration);
    (
        final_bundles_without_collider,
        chunk_data_without_collider,
        coords_without_collider,
        final_bundles_with_collider,
        chunk_data_with_collider,
        coords_with_collider,
    )
}
