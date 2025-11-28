use std::{
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fs::File,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
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
        ChunkDataFileReadWrite, ChunkEntityMap, ChunkIndexFile, ChunkIndexMap,
        CompressionFileHandles, UniformChunkMap, create_chunk_file_data, load_chunk_data,
        write_uniform_chunk,
    },
    marching_cubes::mc::{MeshBuffers, mc_mesh_generation},
    player::player::PLAYER_SPAWN,
    sparse_voxel_octree::ChunkSvo,
    terrain::{
        chunk_generator::chunk_contains_surface,
        terrain::{
            ChunkTag, HALF_CHUNK, MAX_RADIUS, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM,
            TerrainChunk, TerrainMaterialHandle, UniformChunk, Z0_RADIUS, generate_bevy_mesh,
        },
    },
};

//I dont like this but, block player movement until first chunk load happens
pub static INITIAL_CHUNKS_LOADED: AtomicBool = AtomicBool::new(false);

#[derive(Resource)]
pub struct LogicalProcesors(pub usize);

#[derive(Resource)]
pub struct PlayerTranslationMutexHandle(pub Arc<Mutex<Vec3>>);

//stores the data for all chunks in Z0 radius on the bevy thread. Chunk loader can write to the mutex and bevy can modify it for digging operations.
#[derive(Resource)]
pub struct TerrainChunkMap(pub Arc<Mutex<HashMap<(i16, i16, i16), TerrainChunk>>>);

struct ChunkSpawnResult {
    to_spawn: Vec<((i16, i16, i16), Option<Collider>, Option<Mesh>, bool)>, //bool indicates if spawning new mesh (true) or just inserting collider
    to_despawn: Vec<(i16, i16, i16)>,
}

#[derive(Resource)]
pub struct ChunkSpawnReciever(Receiver<ChunkSpawnResult>);

pub struct ChunksBeingLoaded {
    pub chunks: HashMap<(i16, i16, i16), u16>,
    pub request_id: u16,
}

//level 0: full detail, returns TerrainChunk and collider
//level n: full/(2^n) detail, no persistant data besides mesh
pub struct ChunkRequest {
    pub position: (i16, i16, i16),
    pub load_status: u8,
    pub request_id: u16,
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
    chunk_coord: (i16, i16, i16),
    request_id: u16,
    load_status: u8,
    upgrade: bool,
}

pub fn setup_chunk_driver(
    mut commands: Commands,
    index_map: Res<ChunkIndexMap>,
    player_translation_mutex_handle: Res<PlayerTranslationMutexHandle>,
    uniform_chunks_map: Res<UniformChunkMap>,
    chunk_data_file: Res<ChunkDataFileReadWrite>,
    chunk_index_file: Res<ChunkIndexFile>,
    compression_files: Res<CompressionFileHandles>,
) {
    let num_processors = thread::available_parallelism().unwrap().get();
    info!("Number of Available Processors: {}", num_processors);
    commands.insert_resource(LogicalProcesors(num_processors));
    let chunks_being_loaded = Arc::new(Mutex::new(ChunksBeingLoaded {
        chunks: HashMap::new(),
        request_id: 1,
    }));
    let player_translation_arc = Arc::clone(&player_translation_mutex_handle.0);
    let (chunk_spawn_sender, chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    let terrain_chunk_map = Arc::new(Mutex::new(HashMap::new()));
    let terrain_chunk_map_arc = Arc::clone(&terrain_chunk_map);
    let (req_tx, req_rx) = unbounded::<ChunkRequest>();
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    commands.insert_resource(TerrainChunkMap(terrain_chunk_map));
    commands.insert_resource(ChunkSpawnReciever(chunk_spawn_reciever));
    for _ in 0..1.max(num_processors - 2) {
        //leave one processor free for main thread and one for svo manager <- might be wrong
        let index_map_arc = Arc::clone(&index_map.0);
        let uniform_air_chunks_arc = Arc::clone(&uniform_chunks_map.air_chunks);
        let uniform_dirt_chunks_arc = Arc::clone(&uniform_chunks_map.dirt_chunks);
        let chunk_data_file_arc = Arc::clone(&chunk_data_file.0);
        let chunk_index_file_arc = Arc::clone(&chunk_index_file.0);
        let air_compression_file_arc = Arc::clone(&compression_files.air_file);
        let dirt_compression_file_arc = Arc::clone(&compression_files.dirt_file);
        let req_rx_clone = req_rx.clone();
        let res_tx_clone = res_tx.clone();
        thread::spawn(move || {
            chunk_loader_thread(
                req_rx_clone,
                res_tx_clone,
                index_map_arc,
                uniform_air_chunks_arc,
                uniform_dirt_chunks_arc,
                chunk_data_file_arc,
                chunk_index_file_arc,
                air_compression_file_arc,
                dirt_compression_file_arc,
            );
        });
    }
    thread::spawn(move || {
        svo_manager_thread(
            res_rx,
            chunks_being_loaded,
            player_translation_arc,
            chunk_spawn_sender,
            req_tx,
            terrain_chunk_map_arc,
        );
    });
}

//compute thread for loading chunks from disk
//recieves chunk load requests from svo_manager_thread and returns the data
fn chunk_loader_thread(
    req_rx: Receiver<ChunkRequest>,
    res_tx: Sender<ChunkResult>,
    index_map_arc: Arc<Mutex<HashMap<(i16, i16, i16), u64>>>,
    uniform_air_chunks: Arc<Mutex<(HashSet<(i16, i16, i16)>, VecDeque<u64>)>>,
    uniform_dirt_chunks: Arc<Mutex<(HashSet<(i16, i16, i16)>, VecDeque<u64>)>>,
    chunk_data_file: Arc<Mutex<File>>,
    chunk_index_file: Arc<Mutex<File>>,
    air_compression_file: Arc<Mutex<File>>,
    dirt_compression_file: Arc<Mutex<File>>,
) {
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    let mut priority_queue = BinaryHeap::new();
    #[cfg(feature = "timers")]
    let mut chunks_generated = 0;
    #[cfg(feature = "timers")]
    let mut last_debug_print_time = Instant::now();
    #[cfg(feature = "timers")]
    let mut uniform_chunks_generated = 0;
    loop {
        if priority_queue.is_empty() {
            priority_queue.push(req_rx.recv().expect("channel closed"));
        }
        while let Ok(req) = req_rx.try_recv() {
            priority_queue.push(req);
        }
        if let Some(req) = priority_queue.pop() {
            let chunk_coord = req.position;
            let uniform_air_chunks_lock = uniform_air_chunks.lock().unwrap();
            let uniform_dirt_chunks_lock = uniform_dirt_chunks.lock().unwrap();
            if uniform_air_chunks_lock.0.contains(&chunk_coord) {
                let data = if req.load_status == 0 {
                    Some(TerrainChunk {
                        densities: Box::new([i16::MAX; SAMPLES_PER_CHUNK]),
                        materials: Box::new([0; SAMPLES_PER_CHUNK]),
                        is_uniform: UniformChunk::Air,
                    })
                } else {
                    None
                };
                let _ = res_tx.send(ChunkResult {
                    data,
                    mesh: None,
                    collider: None,
                    chunk_coord,
                    request_id: req.request_id,
                    load_status: req.load_status,
                    upgrade: req.upgrade,
                });
                #[cfg(feature = "timers")]
                {
                    uniform_chunks_generated += 1;
                }
                continue;
            } else if uniform_dirt_chunks_lock.0.contains(&chunk_coord) {
                let data = if req.load_status == 0 {
                    Some(TerrainChunk {
                        densities: Box::new([i16::MIN; SAMPLES_PER_CHUNK]),
                        materials: Box::new([1; SAMPLES_PER_CHUNK]),
                        is_uniform: UniformChunk::Dirt,
                    })
                } else {
                    None
                };
                let _ = res_tx.send(ChunkResult {
                    data,
                    mesh: None,
                    collider: None,
                    chunk_coord,
                    request_id: req.request_id,
                    load_status: req.load_status,
                    upgrade: req.upgrade,
                });
                #[cfg(feature = "timers")]
                {
                    uniform_chunks_generated += 1;
                }
                continue;
            }
            drop(uniform_air_chunks_lock);
            drop(uniform_dirt_chunks_lock);
            let file_offset: Option<u64> = {
                let index_map_lock = index_map_arc.lock().unwrap();
                index_map_lock.get(&chunk_coord).copied()
            };
            let (chunk_sdfs, contains_surface) = if let Some(offset) = file_offset {
                let mut chunk_data_file_lock = chunk_data_file.lock().unwrap();
                let chunk = load_chunk_data(&mut chunk_data_file_lock, offset);
                drop(chunk_data_file_lock);
                let contains_surface = chunk_contains_surface(&chunk); //this could be potentially wrong on edge case chunks
                (chunk, contains_surface)
            } else {
                let chunk = TerrainChunk::new(chunk_coord, &fbm);
                match chunk.is_uniform {
                    UniformChunk::Air => {
                        let mut uniform_air_chunks_lock = uniform_air_chunks.lock().unwrap();
                        uniform_air_chunks_lock.0.insert(chunk_coord);
                        let mut air_compression_file_lock = air_compression_file.lock().unwrap();
                        write_uniform_chunk(
                            &chunk_coord,
                            &mut air_compression_file_lock,
                            &mut uniform_air_chunks_lock.1,
                        );
                        drop(air_compression_file_lock);
                        drop(uniform_air_chunks_lock);
                    }
                    UniformChunk::Dirt => {
                        let mut uniform_dirt_chunks_lock = uniform_dirt_chunks.lock().unwrap();
                        uniform_dirt_chunks_lock.0.insert(chunk_coord);
                        let mut dirt_compression_file_lock = dirt_compression_file.lock().unwrap();
                        write_uniform_chunk(
                            &chunk_coord,
                            &mut dirt_compression_file_lock,
                            &mut uniform_dirt_chunks_lock.1,
                        );
                        drop(dirt_compression_file_lock);
                        drop(uniform_dirt_chunks_lock);
                    }
                    UniformChunk::NonUniform => {
                        let mut index_map_lock = index_map_arc.lock().unwrap();
                        let mut chunk_data_file_lock = chunk_data_file.lock().unwrap();
                        let mut chunk_index_file_lock = chunk_index_file.lock().unwrap();
                        create_chunk_file_data(
                            &chunk,
                            &chunk_coord,
                            &mut index_map_lock,
                            &mut chunk_data_file_lock,
                            &mut chunk_index_file_lock,
                        );
                        drop(chunk_index_file_lock);
                        drop(chunk_data_file_lock);
                        drop(index_map_lock);
                    }
                }
                let contains_surface = chunk_contains_surface(&chunk);
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
                chunk_coord,
                request_id: req.request_id,
                load_status: req.load_status,
                upgrade: req.upgrade,
            });
            #[cfg(feature = "timers")]
            {
                chunks_generated += 1;
                if Instant::now()
                    .duration_since(last_debug_print_time)
                    .as_secs()
                    >= 10
                {
                    println!(
                        "Chunk Loader Time: {:?} ms -> {} chunks and {} uniform chunks. Priority Queue Size: {}",
                        Instant::now()
                            .duration_since(last_debug_print_time)
                            .as_millis(),
                        chunks_generated,
                        uniform_chunks_generated,
                        priority_queue.len()
                    );
                    chunks_generated = 0;
                    last_debug_print_time = Instant::now();
                    uniform_chunks_generated = 0;
                }
            }
        }
    }
}

//owns the main svo
//recieves and handles modification requests
//produces chunk load requests for chunk_loader_thread and recieves the data
//sends chunks to be spawned to main thread
fn svo_manager_thread(
    results_channel: Receiver<ChunkResult>,
    chunks_being_loaded: Arc<Mutex<ChunksBeingLoaded>>,
    player_translation: Arc<Mutex<Vec3>>,
    chunk_spawn_channel: Sender<ChunkSpawnResult>,
    load_request_channel: Sender<ChunkRequest>,
    mut terrain_chunk_map: Arc<Mutex<HashMap<(i16, i16, i16), TerrainChunk>>>,
) {
    let mut svo = ChunkSvo::new();
    let mut request_buffer = Vec::new();
    //do initial z0 load so player can start moving
    let mut chunks_being_loaded_lock = chunks_being_loaded.lock().unwrap();
    let player_translation_lock = player_translation.lock().unwrap();
    drop(player_translation_lock);
    let start = Instant::now();
    svo.root.fill_missing_chunks_in_radius(
        &PLAYER_SPAWN,
        Z0_RADIUS,
        &mut chunks_being_loaded_lock,
        &mut request_buffer,
    );
    drop(chunks_being_loaded_lock);
    for request in request_buffer.drain(..) {
        let _ = load_request_channel.send(request);
    }
    println!(
        "Loaded octree: {:?} ms",
        Instant::now().duration_since(start).as_millis()
    );
    loop {
        let mut chunks_being_loaded = chunks_being_loaded.lock().unwrap();
        let to_spawn = recieve_loaded_chunks(
            &results_channel,
            &mut svo,
            &mut chunks_being_loaded,
            &mut terrain_chunk_map,
        );
        let mut to_despawn = Vec::new();
        let player_translation_lock = player_translation.lock().unwrap();
        let player_translation = player_translation_lock.clone();
        drop(player_translation_lock);
        let mut chunks_to_deallocate: Vec<((i16, i16, i16), bool)> = Vec::new();
        svo.root
            .query_chunks_outside_sphere(&player_translation, &mut chunks_to_deallocate);
        for (chunk_coord, has_entity) in &chunks_to_deallocate {
            if *has_entity {
                to_despawn.push(*chunk_coord);
            }
            chunks_being_loaded.chunks.remove(chunk_coord);
            svo.root.delete(*chunk_coord);
        }
        svo.root.fill_missing_chunks_in_radius(
            &player_translation,
            MAX_RADIUS,
            &mut chunks_being_loaded,
            &mut request_buffer,
        );
        drop(chunks_being_loaded);
        if !to_spawn.is_empty() || !to_despawn.is_empty() {
            let response = ChunkSpawnResult {
                to_spawn,
                to_despawn,
            };
            let _ = chunk_spawn_channel.send(response);
        }
        for request in request_buffer.drain(..) {
            let _ = load_request_channel.send(request);
        }
    }
}

fn recieve_loaded_chunks(
    results_channel: &Receiver<ChunkResult>,
    svo: &mut ChunkSvo,
    chunks_being_loaded: &mut ChunksBeingLoaded,
    terrain_chunk_map: &mut Arc<Mutex<HashMap<(i16, i16, i16), TerrainChunk>>>,
) -> Vec<((i16, i16, i16), Option<Collider>, Option<Mesh>, bool)> {
    let mut chunks_to_spawn = Vec::new();
    if results_channel.is_empty() {
        return chunks_to_spawn;
    }
    while let Ok(result) = results_channel.try_recv() {
        match chunks_being_loaded.chunks.get(&result.chunk_coord) {
            Some(expected_id) if *expected_id == result.request_id => {
                match result.collider {
                    //if has collider
                    Some(collider) => {
                        match result.upgrade {
                            //if upgrading
                            true => {
                                //insert collider
                                chunks_to_spawn.push((
                                    result.chunk_coord,
                                    Some(collider),
                                    None,
                                    false,
                                ));
                                //update data and load status
                                let (_has_entity, load_status) =
                                    svo.root.get_mut(result.chunk_coord).unwrap();
                                let mut terrain_chunk_map_lock = terrain_chunk_map.lock().unwrap();
                                terrain_chunk_map_lock
                                    .insert(result.chunk_coord, result.data.unwrap());
                                drop(terrain_chunk_map_lock);
                                *load_status = result.load_status;
                            }
                            //if not upgrading
                            false => {
                                //spawn it with collider
                                //insert into svo
                                svo.root
                                    .insert(result.chunk_coord, result.load_status, true);
                                let mut terrain_chunk_map_lock = terrain_chunk_map.lock().unwrap();
                                terrain_chunk_map_lock
                                    .insert(result.chunk_coord, result.data.unwrap());
                                drop(terrain_chunk_map_lock);
                                chunks_to_spawn.push((
                                    result.chunk_coord,
                                    Some(collider),
                                    result.mesh,
                                    true,
                                ));
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
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, false);
                                    let mut terrain_chunk_map_lock =
                                        terrain_chunk_map.lock().unwrap();
                                    terrain_chunk_map_lock
                                        .insert(result.chunk_coord, result.data.unwrap());
                                    drop(terrain_chunk_map_lock);
                                }
                                //if load status not 0
                                _ => {
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, false);
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
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, true);
                                    let mut terrain_chunk_map_lock =
                                        terrain_chunk_map.lock().unwrap();
                                    terrain_chunk_map_lock
                                        .insert(result.chunk_coord, result.data.unwrap());
                                    drop(terrain_chunk_map_lock);
                                    chunks_to_spawn.push((
                                        result.chunk_coord,
                                        None,
                                        Some(mesh),
                                        true,
                                    ));
                                }
                                //if load status not 0 spawn and insert no data
                                _ => {
                                    chunks_to_spawn.push((
                                        result.chunk_coord,
                                        None,
                                        Some(mesh),
                                        true,
                                    ));
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, true);
                                }
                            },
                            //if no mesh
                            None => match result.load_status {
                                //if has mesh
                                0 => {
                                    //if load status 0insert data
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, false);
                                    let mut terrain_chunk_map_lock =
                                        terrain_chunk_map.lock().unwrap();
                                    terrain_chunk_map_lock
                                        .insert(result.chunk_coord, result.data.unwrap());
                                    drop(terrain_chunk_map_lock);
                                }
                                //if load status not 0 insert no data
                                _ => {
                                    svo.root
                                        .insert(result.chunk_coord, result.load_status, false);
                                }
                            },
                        },
                    },
                }
                chunks_being_loaded.chunks.remove(&result.chunk_coord);
            }
            _ => {
                panic!("what happened here");
            }
        }
    }
    chunks_to_spawn
}

//recieves chunks that were loaded and need spawning from the manager
pub fn chunk_spawn_reciever(
    mut commands: Commands,
    standard_material: Res<TerrainMaterialHandle>,
    mut meshes: ResMut<Assets<Mesh>>,
    req_rx: Res<ChunkSpawnReciever>,
    mut chunk_entity_map: ResMut<ChunkEntityMap>,
    terrain_chunk_map: Res<TerrainChunkMap>,
) {
    while let Ok(req) = req_rx.0.try_recv() {
        #[cfg(feature = "timers")]
        let t0 = Instant::now();
        if !INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed) {
            INITIAL_CHUNKS_LOADED.store(true, Ordering::Relaxed);
        } //wasteful
        for (chunk, collider_opt, mesh_opt, spawn) in req.to_spawn {
            if spawn {
                let entity = match collider_opt {
                    Some(collider) => commands
                        .spawn((
                            Mesh3d(meshes.add(mesh_opt.unwrap())),
                            collider,
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id(),
                    None => commands
                        .spawn((
                            Mesh3d(meshes.add(mesh_opt.unwrap())),
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id(),
                };
                chunk_entity_map.0.insert(chunk, entity);
            } else {
                let entity = chunk_entity_map.0.get(&chunk).unwrap();
                commands.entity(*entity).insert(collider_opt.unwrap());
            }
        }
        if !req.to_despawn.is_empty() {
            let mut terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
            for chunk_coord in req.to_despawn {
                commands
                    .entity(chunk_entity_map.0.remove(&chunk_coord).unwrap())
                    .despawn();
                terrain_chunk_map_lock.remove(&chunk_coord);
            }
            drop(terrain_chunk_map_lock);
        }
        #[cfg(feature = "timers")]
        {
            println!(
                "chunk_spawn_reciever {:?} ms",
                Instant::now().duration_since(t0).as_millis()
            );
        }
    }
}
