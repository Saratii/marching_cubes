use std::{
    collections::HashMap,
    sync::{
        Arc,
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
    data_loader::file_loader::{
        DataBaseEnvHandle, DataBaseHandle, LoadedChunkKeys, deserialize_chunk_data,
        serialize_chunk_data,
    },
    marching_cubes::march_cubes,
    player::player::PlayerTag,
    sparse_voxel_octree::ChunkSvo,
    terrain::terrain::{
        CUBES_PER_CHUNK_DIM, ChunkTag, SDF_VALUES_PER_CHUNK_DIM, StandardTerrainMaterialHandle,
        TerrainChunk, Z2_RADIUS_SQUARED,
    },
};

#[derive(Resource)]
pub struct ChunkChannels {
    pub requests: Sender<ChunkRequest>,
    pub results: Receiver<ChunkResult>,
}

#[derive(Resource)]
pub struct ChunksBeingLoaded(
    pub HashMap<(i16, i16, i16), (Arc<AtomicBool>, u64)>,
    pub u64,
);

//level 0: full detail, returns TerrainChunk and collider
//level n: full/(2^n) detail, no persistant data besides mesh
pub struct ChunkRequest {
    pub position: (i16, i16, i16),
    pub level: u8,
    pub canceled: Arc<AtomicBool>,
    pub request_id: u64,
}

pub struct ChunkResult {
    data: Option<TerrainChunk>,
    mesh: Option<Mesh>,
    collider: Option<Collider>,
    transform: Transform,
    chunk_coord: (i16, i16, i16),
    request_id: u64,
}

pub fn setup_loading_thread(
    mut commands: Commands,
    loaded_chunk_keys: Res<LoadedChunkKeys>,
    database_env: Res<DataBaseEnvHandle>,
    database: Res<DataBaseHandle>,
) {
    let (req_tx, req_rx) = unbounded::<ChunkRequest>();
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let loaded_chunk_keys = Arc::clone(&loaded_chunk_keys.0);
    let chunks_being_loaded = HashMap::new();
    let env = database_env.0.clone();
    let db = database.0.clone();
    std::thread::spawn(move || {
        let fbm = || -> GeneratorWrapper<SafeNode> {
            (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build()
        }();
        for req in req_rx.iter() {
            if req.canceled.load(Ordering::Relaxed) {
                continue;
            }
            let chunk_coord = req.position;
            let rtxn = env.read_txn().unwrap();
            let mut loaded_chunk_keys_locked = loaded_chunk_keys.lock().unwrap();
            let chunk_sdfs = if loaded_chunk_keys_locked.contains(&chunk_coord) {
                drop(loaded_chunk_keys_locked);
                let bytes = db.get(&rtxn, &chunk_coord).unwrap().unwrap();
                deserialize_chunk_data(&bytes)
            } else {
                loaded_chunk_keys_locked.insert(chunk_coord);
                drop(loaded_chunk_keys_locked);
                let chunk = TerrainChunk::new(chunk_coord, &fbm);
                let mut wtxn = env.write_txn().unwrap();
                let bytes = serialize_chunk_data(&chunk);
                db.put(&mut wtxn, &chunk_coord, &bytes).unwrap();
                wtxn.commit().unwrap();
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
                request_id: req.request_id,
            });
        }
    });
    commands.insert_resource(ChunkChannels {
        requests: req_tx,
        results: res_rx,
    });
    commands.insert_resource(ChunksBeingLoaded(chunks_being_loaded, 1));
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
        match chunks_being_loaded.0.get(&result.chunk_coord) {
            Some((_, expected_id)) if *expected_id == result.request_id => {
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
                }
                chunks_being_loaded.0.remove(&result.chunk_coord);
            }
            _ => {
                // Stale/unknown result: drop it. Do NOT remove any current in-flight entry
                // (the map may contain a newer id now). Also do not spawn an entity for it.
                continue;
            }
        }
    }
}

pub fn validate_loading_queue(
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
    player_transform: Single<&Transform, With<PlayerTag>>,
) {
    let player_position = player_transform.translation;
    chunks_being_loaded.0.retain(|chunk_coord, canceled| {
        let chunk_world_pos = chunk_coord_to_world_pos(chunk_coord);
        let distance_squared = player_position.distance_squared(chunk_world_pos);
        if distance_squared > Z2_RADIUS_SQUARED {
            canceled.0.store(true, Ordering::Relaxed);
            return false;
        }
        true
    });
}
