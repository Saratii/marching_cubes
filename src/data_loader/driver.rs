use std::{
    collections::{BinaryHeap, HashMap},
    fs::OpenOptions,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use crossbeam_channel::{Receiver, Sender, unbounded};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
use isomesh::marching_cubes::mc::{MeshBuffers, mc_mesh_generation};

use crate::{
    conversions::chunk_coord_to_world_pos,
    data_loader::file_loader::{ChunkIndexMap, create_chunk_file_data, load_chunk_data},
    player::player::PlayerTag,
    sparse_voxel_octree::ChunkSvo,
    terrain::{
        chunk_generator::chunk_contains_surface,
        terrain::{
            ChunkTag, HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk, TerrainMaterialHandle,
            Z2_RADIUS_SQUARED, generate_bevy_mesh,
        },
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
    pub load_status: u8,
    pub canceled: Arc<AtomicBool>,
    pub request_id: u64,
    pub upgrade: bool,
    pub distance_squared: u32,
}

impl PartialEq for ChunkRequest {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && self.load_status == other.load_status
            && self.request_id == other.request_id
            && self.upgrade == other.upgrade
            && self.distance_squared == other.distance_squared
    }
}

impl Eq for ChunkRequest {}

impl Ord for ChunkRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance_squared.cmp(&self.distance_squared)
    }
}

impl PartialOrd for ChunkRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ChunkResult {
    data: Option<TerrainChunk>,
    mesh: Option<Mesh>,
    collider: Option<Collider>,
    transform: Transform,
    chunk_coord: (i16, i16, i16),
    request_id: u64,
    load_status: u8,
    upgrade: bool,
}

pub fn setup_loading_thread(mut commands: Commands, index_map: Res<ChunkIndexMap>) {
    let (req_tx, req_rx) = unbounded::<ChunkRequest>();
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let index_map_arc: Arc<Mutex<std::collections::HashMap<(i16, i16, i16), u64>>> =
        Arc::clone(&index_map.0);
    let chunks_being_loaded = HashMap::new();
    thread::spawn(move || {
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
        let mut priority_queue = BinaryHeap::new();
        loop {
            while let Ok(req) = req_rx.try_recv() {
                if !req.canceled.load(Ordering::Relaxed) {
                    priority_queue.push(req);
                }
            }
            if let Some(req) = priority_queue.pop() {
                if req.canceled.load(Ordering::Relaxed) {
                    continue;
                }
                let chunk_coord = req.position;
                let file_offset = {
                    let index_map_lock = index_map_arc.lock().unwrap();
                    index_map_lock.get(&chunk_coord).copied()
                };
                let (chunk_sdfs, contains_surface) = if let Some(offset) = file_offset {
                    let chunk = load_chunk_data(&mut data_file, offset);
                    let contains_surface = chunk_contains_surface(&chunk);
                    (chunk, contains_surface)
                } else {
                    let (chunk, contains_surface) = TerrainChunk::new(chunk_coord, &fbm);
                    let mut index_map_lock = index_map_arc.lock().unwrap();
                    create_chunk_file_data(
                        &chunk,
                        &chunk_coord,
                        &mut index_map_lock,
                        &data_file,
                        &index_file,
                    );
                    (chunk, contains_surface)
                };
                let (collider, mesh) = if contains_surface {
                    let mut mesh_buffers = MeshBuffers::new();
                    mc_mesh_generation(
                        &mut mesh_buffers,
                        &chunk_sdfs.densities,
                        &chunk_sdfs.materials,
                        SAMPLES_PER_CHUNK_DIM,
                        HALF_CHUNK,
                    );
                    let mesh = generate_bevy_mesh(mesh_buffers);
                    let collider = if req.load_status == 0 {
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
                    (collider, Some(mesh))
                } else {
                    (None, None)
                };
                let data = if req.load_status == 0 {
                    Some(chunk_sdfs)
                } else {
                    None
                };
                let _ = res_tx.send(ChunkResult {
                    data,
                    mesh,
                    collider,
                    transform: Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                    chunk_coord,
                    request_id: req.request_id,
                    load_status: req.load_status,
                    upgrade: req.upgrade,
                });
            } else {
                if let Ok(req) = req_rx.recv() {
                    if !req.canceled.load(Ordering::Relaxed) {
                        priority_queue.push(req);
                    }
                } else {
                    break;
                }
            }
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
    standard_material: Res<TerrainMaterialHandle>,
    mut svo: ResMut<ChunkSvo>,
    mut chunks_being_loaded: ResMut<ChunksBeingLoaded>,
) {
    while let Ok(result) = channels.results.try_recv() {
        match chunks_being_loaded.0.get(&result.chunk_coord) {
            Some((_, expected_id)) if *expected_id == result.request_id => {
                match result.collider {
                    //if has collider
                    Some(collider) => {
                        match result.upgrade {
                            //if upgrading
                            true => {
                                //get spawned entity
                                let entity =
                                    svo.root.get_mut(result.chunk_coord).unwrap().0.unwrap();
                                //insert collider
                                commands.entity(entity).insert(collider);
                                //update data and load status
                                let (_, data, load_status) =
                                    svo.root.get_mut(result.chunk_coord).unwrap();
                                *data = result.data;
                                *load_status = result.load_status;
                            }
                            //if not upgrading
                            false => {
                                //spawn it with collider
                                let entity = commands
                                    .spawn((
                                        Mesh3d(meshes.add(result.mesh.unwrap())),
                                        collider,
                                        ChunkTag,
                                        result.transform,
                                        MeshMaterial3d(standard_material.0.clone()),
                                    ))
                                    .id();
                                //insert into svo
                                svo.root.insert(
                                    result.chunk_coord,
                                    Some(entity),
                                    result.data,
                                    result.load_status,
                                );
                            }
                        }
                    }
                    //if does not have collider
                    None => match result.upgrade {
                        //if upgrading
                        true => {
                            match result.load_status {
                                //if loading status 0
                                0 => {
                                    svo.root.insert(
                                        result.chunk_coord,
                                        None,
                                        result.data,
                                        result.load_status,
                                    );
                                }
                                //if load status not 0
                                _ => {
                                    panic!("this shouldnt be happening yet");
                                }
                            }
                        }
                        //if not upgrading
                        false => match result.mesh {
                            //if has mesh
                            Some(mesh) => match result.load_status {
                                //if has mesh
                                0 => {
                                    //if load status 0 spawn and insert data
                                    let entity = commands
                                        .spawn((
                                            Mesh3d(meshes.add(mesh)),
                                            ChunkTag,
                                            result.transform,
                                            MeshMaterial3d(standard_material.0.clone()),
                                        ))
                                        .id();
                                    svo.root.insert(
                                        result.chunk_coord,
                                        Some(entity),
                                        result.data,
                                        result.load_status,
                                    );
                                }
                                //if load status not 0 spawn and insert no data
                                _ => {
                                    let entity = commands
                                        .spawn((
                                            Mesh3d(meshes.add(mesh)),
                                            ChunkTag,
                                            result.transform,
                                            MeshMaterial3d(standard_material.0.clone()),
                                        ))
                                        .id();
                                    svo.root.insert(
                                        result.chunk_coord,
                                        Some(entity),
                                        None,
                                        result.load_status,
                                    );
                                }
                            },
                            //if no mesh
                            None => match result.load_status {
                                //if has mesh
                                0 => {
                                    //if load status 0insert data
                                    svo.root.insert(
                                        result.chunk_coord,
                                        None,
                                        result.data,
                                        result.load_status,
                                    );
                                }
                                //if load status not 0 insert no data
                                _ => {
                                    svo.root.insert(
                                        result.chunk_coord,
                                        None,
                                        None,
                                        result.load_status,
                                    );
                                }
                            },
                        },
                    },
                }
                chunks_being_loaded.0.remove(&result.chunk_coord);
            }
            _ => {
                continue;
                // panic!("what happened here");
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
