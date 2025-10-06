use std::{
    collections::HashMap,
    fs::OpenOptions,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use crossbeam_channel::{Receiver, Sender, unbounded};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};

use crate::{
    conversions::chunk_coord_to_world_pos,
    data_loader::file_loader::{ChunkIndexMap, create_chunk_file_data, load_chunk_data},
    marching_cubes::march_cubes,
    player::player::PlayerTag,
    sparse_voxel_octree::ChunkSvo,
    terrain::{
        lod_zones::in_zone_1,
        terrain::{
            CUBES_PER_CHUNK_DIM, ChunkTag, SDF_VALUES_PER_CHUNK_DIM, StandardTerrainMaterialHandle,
            TerrainChunk, Z2_RADIUS_SQUARED,
        },
    },
};

#[derive(Resource)]
pub struct ChunkChannels {
    pub requests: Sender<ChunkRequest>,
    pub results: Receiver<ChunkResult>,
}

#[derive(Resource)]
pub struct ChunksBeingLoaded(pub HashMap<(i16, i16, i16), Arc<AtomicBool>>);

//level 0: full detail, returns TerrainChunk and collider
//level n: full/(2^n) detail, no persistant data besides mesh
pub struct ChunkRequest {
    pub position: (i16, i16, i16),
    pub level: u8,
    pub canceled: Arc<AtomicBool>,
}

pub struct ChunkResult {
    data: Option<TerrainChunk>,
    mesh: Option<Mesh>,
    collider: Option<Collider>,
    transform: Transform,
    chunk_coord: (i16, i16, i16),
}

pub fn setup_loading_thread(mut commands: Commands, index_map: Res<ChunkIndexMap>) {
    let (req_tx, req_rx) = unbounded::<ChunkRequest>();
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let index_map_arc: Arc<Mutex<std::collections::HashMap<(i16, i16, i16), u64>>> =
        Arc::clone(&index_map.0);
    let chunks_being_loaded = HashMap::new();
    std::thread::spawn(move || {
        let mut data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("data/chunk_data.txt")
            .unwrap();
        let index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("data/chunk_index_data.txt")
            .unwrap();
        let fbm = || -> GeneratorWrapper<SafeNode> {
            (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build()
        }();
        for req in req_rx.iter() {
            if req.canceled.load(Ordering::Relaxed) {
                continue;
            }
            let chunk_coord = req.position;
            let file_offset = {
                let index_map_lock = index_map_arc.lock().unwrap();
                index_map_lock.get(&chunk_coord).copied()
            };
            let chunk_sdfs = if let Some(offset) = file_offset {
                load_chunk_data(&mut data_file, offset)
            } else {
                let chunk = TerrainChunk::new(chunk_coord, &fbm);
                let mut index_map_lock = index_map_arc.lock().unwrap();
                create_chunk_file_data(
                    &chunk,
                    &chunk_coord,
                    &mut index_map_lock,
                    &data_file,
                    &index_file,
                );
                chunk
            };
            let mesh = march_cubes(
                &chunk_sdfs.sdfs,
                CUBES_PER_CHUNK_DIM,
                SDF_VALUES_PER_CHUNK_DIM,
            );
            let vertex_count = mesh.count_vertices();
            let collider = if req.level == 0 && vertex_count > 0 {
                Some(
                    Collider::from_bevy_mesh(
                        &mesh,
                        &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            let data = if req.level == 0 {
                Some(chunk_sdfs)
            } else {
                None
            };
            let mesh = if vertex_count > 0 { Some(mesh) } else { None };
            let _ = res_tx.send(ChunkResult {
                data,
                mesh,
                collider,
                transform: Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                chunk_coord,
            });
        }
    });
    commands.insert_resource(ChunkChannels {
        requests: req_tx,
        results: res_rx,
    });
    commands.insert_resource(ChunksBeingLoaded(chunks_being_loaded));
}

pub fn chunk_reciever(
    channels: Res<ChunkChannels>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    standard_material: Res<StandardTerrainMaterialHandle>,
    mut svo: ResMut<ChunkSvo>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
) {
    while let Ok(result) = channels.results.try_recv() {
        let entity = if let (Some(mesh), Some(collider)) = (result.mesh, result.collider) {
            commands
                .spawn((
                    Mesh3d(meshes.add(mesh)),
                    collider,
                    ChunkTag,
                    result.transform,
                    MeshMaterial3d(standard_material.0.clone()),
                ))
                .id()
        } else {
            commands.spawn((ChunkTag, result.transform)).id()
        };
        if let Some(data) = result.data {
            svo.root.insert(result.chunk_coord, entity, data);
        } else {
        }
        chunks_being_loaded.0.remove(&result.chunk_coord);
    }
}

pub fn validate_loading_queue(
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let player_position = player_transform.translation;
    chunks_being_loaded.0.retain(|chunk_coord, canceled| {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        if !in_zone_1(&player_position, &chunk_world_pos) {
            let distance_squared = chunk_world_pos.distance_squared(player_position);
            if distance_squared <= Z2_RADIUS_SQUARED {
                canceled.store(true, Ordering::Relaxed);
                return false;
            }
        }
        true
    });
}
