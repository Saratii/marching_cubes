use std::{collections::HashSet, fs::OpenOptions, sync::Arc};

use bevy::{
    asset::Assets,
    ecs::{
        event::EventReader,
        query::With,
        resource::Resource,
        system::{Commands, Res, ResMut, Single},
    },
    render::mesh::Mesh,
    tasks::{AsyncComputeTaskPool, Task, block_on},
    transform::components::Transform,
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::chunk_loader::{
        create_chunk_file_data, load_chunk_data, ChunkDataFile, ChunkIndexFile, ChunkIndexMap
    },
    marching_cubes::march_cubes,
    player::player::PlayerTag,
    terrain::{
        chunk_generator::{GenerateChunkEvent, LoadChunksEvent},
        terrain::{
            spawn_chunk, ChunkMap, NoiseFunction, StandardTerrainMaterialHandle, TerrainChunk, CHUNK_CREATION_RADIUS_SQUARED, CUBES_PER_CHUNK_DIM, SDF_VALUES_PER_CHUNK_DIM
        },
    },
};
use fastnoise2::SafeNode;
use fastnoise2::generator::GeneratorWrapper;
const MAX_CHUNKS_PER_TASK: usize = 10000000;

#[derive(Resource)]
pub struct MyMapGenTasks {
    pub generation_tasks: Vec<
        Task<
            Vec<(
                (i16, i16, i16),
                TerrainChunk,
                Mesh,
                Transform,
                Option<Collider>,
            )>,
        >,
    >,
    pub chunks_being_generated: HashSet<(i16, i16, i16)>,
}

#[derive(Resource)]
pub struct LoadChunkTasks {
    pub generation_tasks: Vec<
        Task<
            Vec<(
                (i16, i16, i16),
                TerrainChunk,
                Mesh,
                Transform,
                Option<Collider>,
            )>,
        >,
    >,
    pub chunks_being_loaded: HashSet<(i16, i16, i16)>,
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
                chunk_coords
                    .par_iter()
                    .map(|coord| generate_chunk_data(coord, &noise_gen_clone))
                    .collect::<Vec<_>>()
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
            chunk_coords
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
                .collect::<Vec<_>>()
        });
        my_tasks.generation_tasks.push(task);
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

fn should_chunk_exist(chunk_coord: (i16, i16, i16), player_chunk: &(i16, i16, i16)) -> bool {
    let player_chunk_world_pos = chunk_coord_to_world_pos(player_chunk);
    let chunk_world_pos = chunk_coord_to_world_pos(&chunk_coord);
    chunk_world_pos.distance_squared(player_chunk_world_pos) <= CHUNK_CREATION_RADIUS_SQUARED
}

pub fn spawn_generated_chunks(
    mut my_tasks: ResMut<MyMapGenTasks>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    chunk_index_map: Res<ChunkIndexMap>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
    mut index_file: ResMut<ChunkIndexFile>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        if task.is_finished() {
            let chunk_data = block_on(task);
            let num_chunks = chunk_data.len();
            let start_index = chunk_coords_to_be_removed.len();
            chunk_coords_to_be_removed
                .resize(chunk_coords_to_be_removed.len() + num_chunks, (0, 0, 0));
            let current_player_chunk = world_pos_to_chunk_coord(&player_transform.translation);
            for (i, (chunk_coord, terrain_chunk, mesh, transform, collider)) in
                chunk_data.into_iter().enumerate()
            {
                if should_chunk_exist(chunk_coord, &current_player_chunk) {
                    let mut locked_index_map = chunk_index_map.0.lock().unwrap();
                    create_chunk_file_data(
                        &terrain_chunk,
                        chunk_coord,
                        &mut locked_index_map,
                        &mut chunk_data_file.0,
                        &mut index_file.0,
                    );
                    drop(locked_index_map);
                    let entity = spawn_chunk(
                        &mut commands,
                        &mut meshes,
                        material_handle.0.clone(),
                        mesh,
                        transform,
                        collider,
                    );
                    chunk_map.0.insert(chunk_coord, (entity, terrain_chunk));
                }
                chunk_coords_to_be_removed[start_index + i] = chunk_coord;
            }
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
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material_handle: Res<StandardTerrainMaterialHandle>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        let current_player_chunk = world_pos_to_chunk_coord(&player_transform.translation);
        if task.is_finished() {
            let chunk_data = block_on(task);
            let num_chunks = chunk_data.len();
            let start_index = chunk_coords_to_be_removed.len();
            chunk_coords_to_be_removed
                .resize(chunk_coords_to_be_removed.len() + num_chunks, (0, 0, 0));
            for (i, (chunk_coord, terrain_chunk, mesh, transform, collider)) in
                chunk_data.into_iter().enumerate()
            {
                if should_chunk_exist(chunk_coord, &current_player_chunk) {
                    let entity = spawn_chunk(
                        &mut commands,
                        &mut meshes,
                        material_handle.0.clone(),
                        mesh,
                        transform,
                        collider,
                    );
                    chunk_map.0.insert(chunk_coord, (entity, terrain_chunk));
                }
                chunk_coords_to_be_removed[start_index + i] = chunk_coord;
            }
            return false;
        }
        true
    });
    for coord in chunk_coords_to_be_removed {
        my_tasks.chunks_being_loaded.remove(&coord);
    }
}
