use std::collections::HashSet;

use bevy::{
    asset::Assets,
    ecs::{
        event::EventReader,
        system::{Commands, Res, ResMut},
    },
    render::mesh::Mesh,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    transform::components::Transform,
};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    conversions::chunk_coord_to_world_pos,
    marching_cubes::march_cubes,
    terrain::{
        chunk_generator::GenerateChunkEvent,
        terrain::{ChunkMap, NoiseFunction, StandardTerrainMaterialHandle, TerrainChunk},
    },
};

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
            )>,
        >,
    >,
    pub chunks_being_generated: HashSet<(i16, i16, i16)>,
}

pub fn catch_chunk_generation_request(
    mut chunk_generation_events: EventReader<GenerateChunkEvent>,
    fbm: Res<NoiseFunction>,
    mut my_tasks: ResMut<MyMapGenTasks>,
) {
    let noise_gen = fbm.0.clone();
    for event in chunk_generation_events.read() {
        for (chunk_coord, _) in &event.chunk_data {
            my_tasks.chunks_being_generated.insert(*chunk_coord);
        }
        let task_pool = AsyncComputeTaskPool::get();
        let chunk_coords = event.chunk_data.clone();
        let noise_gen_clone = noise_gen.clone();
        let task = task_pool.spawn(async move {
            chunk_coords
                .par_iter()
                .map(|(coord, needs_noise)| {
                    let terrain_chunk = TerrainChunk::new(*coord, &noise_gen_clone, *needs_noise);
                    let mesh = march_cubes(&terrain_chunk.densities);
                    let chunk_center = chunk_coord_to_world_pos(*coord);
                    let transform = Transform::from_translation(chunk_center);
                    let collider = if *needs_noise {
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

pub fn spawn_generated_chunks(
    mut my_tasks: ResMut<MyMapGenTasks>,
    mut chunk_map: ResMut<ChunkMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_terrain_material_handle: Res<StandardTerrainMaterialHandle>,
) {
    let mut completed_chunks = Vec::new();
    my_tasks.generation_tasks.retain_mut(|task| {
        let status = block_on(future::poll_once(task));
        let retain = status.is_none();
        if let Some(chunk_data) = status {
            completed_chunks.extend(chunk_data);
        }
        retain
    });
    for (chunk_coord, terrain_chunk, mesh, transform, collider) in completed_chunks {
        my_tasks.chunks_being_generated.remove(&chunk_coord);
        let entity = chunk_map.spawn_chunk(
            &mut commands,
            &mut meshes,
            &standard_terrain_material_handle.0,
            terrain_chunk,
            mesh,
            transform,
            collider,
        );
        chunk_map.0.insert(chunk_coord, entity);
    }
}
