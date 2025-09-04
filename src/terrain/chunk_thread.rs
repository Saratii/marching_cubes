use std::collections::HashSet;

use bevy::{
    asset::{Assets, Handle},
    ecs::{
        event::EventReader,
        system::{Commands, Res, ResMut},
    },
    pbr::StandardMaterial,
    render::mesh::Mesh,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    transform::components::Transform,
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    conversions::chunk_coord_to_world_pos,
    data_loader::chunk_loader::{
        ChunkDataFile, ChunkIndexFile, ChunkIndexMap, create_chunk_file_data,
    },
    marching_cubes::march_cubes,
    terrain::{
        chunk_generator::GenerateChunkEvent,
        terrain::{ChunkMap, NoiseFunction, StandardTerrainMaterialHandle, TerrainChunk},
    },
};
use fastnoise2::SafeNode;
use fastnoise2::generator::GeneratorWrapper;
const MAX_CHUNKS_PER_TASK: usize = 10000000;

#[derive(bevy::ecs::resource::Resource)]
pub struct MyMapGenTasks {
    pub generation_tasks: Vec<
        Task<
            Vec<(
                (i16, i16, i16),
                TerrainChunk,
                Mesh,
                Transform,
                Option<Collider>,
                Handle<StandardMaterial>,
            )>,
        >,
    >,
    pub chunks_being_generated: HashSet<(i16, i16, i16)>,
}

pub fn catch_chunk_generation_request(
    mut chunk_generation_events: EventReader<GenerateChunkEvent>,
    fbm: Res<NoiseFunction>,
    mut my_tasks: ResMut<MyMapGenTasks>,
    standard_terrain_material_handle: Res<StandardTerrainMaterialHandle>,
) {
    let noise_gen = fbm.0.clone();
    let task_pool = AsyncComputeTaskPool::get();
    for event in chunk_generation_events.read() {
        for chunk_batch in event.chunk_coords.chunks(MAX_CHUNKS_PER_TASK) {
            let chunk_coords = chunk_batch.to_vec();
            let noise_gen_clone = noise_gen.clone();
            let material_handle = standard_terrain_material_handle.0.clone();
            let task = task_pool.spawn(async move {
                chunk_coords
                    .par_iter()
                    .map(|coord| {
                        generate_chunk_data(coord, &noise_gen_clone, material_handle.clone())
                    })
                    .collect::<Vec<_>>()
            });

            my_tasks.generation_tasks.push(task);
        }
    }
}

pub fn generate_chunk_data(
    coord: &(i16, i16, i16),
    noise_gen: &GeneratorWrapper<SafeNode>,
    standard_terrain_material_handle: Handle<StandardMaterial>,
) -> (
    (i16, i16, i16),
    TerrainChunk,
    Mesh,
    Transform,
    Option<Collider>,
    Handle<StandardMaterial>,
) {
    let terrain_chunk = TerrainChunk::new(*coord, noise_gen);
    let mesh = march_cubes(&terrain_chunk.densities);
    let transform = Transform::from_translation(chunk_coord_to_world_pos(*coord));
    let collider = if mesh.count_vertices() > 0 {
        Collider::from_bevy_mesh(
            &mesh,
            &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
        )
    } else {
        None
    };
    (
        *coord,
        terrain_chunk,
        mesh,
        transform,
        collider,
        standard_terrain_material_handle,
    )
}

pub fn spawn_generated_chunks(
    mut my_tasks: ResMut<MyMapGenTasks>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_index_map: ResMut<ChunkIndexMap>,
    mut chunk_data_file: ResMut<ChunkDataFile>,
    mut index_file: ResMut<ChunkIndexFile>,
) {
    let mut chunk_coords_to_be_removed = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        let status = block_on(future::poll_once(task));
        let retain = status.is_none();
        if let Some(chunk_data) = status {
            let num_chunks = chunk_data.len();
            let start_index = chunk_coords_to_be_removed.len();
            chunk_coords_to_be_removed
                .resize(chunk_coords_to_be_removed.len() + num_chunks, (0, 0, 0));
            for (i, (chunk_coord, terrain_chunk, mesh, transform, collider, material_handle)) in
                chunk_data.into_iter().enumerate()
            {
                create_chunk_file_data(
                    &terrain_chunk,
                    chunk_coord,
                    &mut chunk_index_map.0,
                    &mut chunk_data_file.0,
                    &mut index_file.0,
                );
                chunk_coords_to_be_removed[start_index + i] = chunk_coord;
                let entity = chunk_map.spawn_chunk(
                    &mut commands,
                    &mut meshes,
                    material_handle,
                    mesh,
                    transform,
                    collider,
                );
                chunk_map.0.insert(chunk_coord, (entity, terrain_chunk));
            }
        }
        retain
    });
    for coord in chunk_coords_to_be_removed {
        my_tasks.chunks_being_generated.remove(&coord);
    }
}
