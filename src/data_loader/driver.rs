use crate::data_loader::file_loader::{
    ChunkIndexMapDelta, get_project_root, load_chunk_index_map, load_uniform_chunks,
    remove_uniform_chunk, update_chunk_file_data,
};
use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use crossbeam_channel::{Receiver, Sender, unbounded};
use fastnoise2::{
    SafeNode,
    generator::{Generator, GeneratorWrapper, simplex::opensimplex2},
};
#[cfg(feature = "timers")]
use std::io::Write;
#[cfg(feature = "timers")]
use std::time::Instant;
use std::{
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fs::{File, OpenOptions},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::file_loader::{
        ChunkEntityMap, ChunkIndexMapRead, create_chunk_file_data, load_chunk_data,
        write_uniform_chunk,
    },
    marching_cubes::mc::mc_mesh_generation,
    player::player::PlayerTag,
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

#[derive(Resource)]
pub struct NoiseGenerator(pub GeneratorWrapper<SafeNode>);

//stores the data for all chunks in Z0 radius on the bevy thread. Chunk loader can write to the mutex and bevy can modify it for digging operations.
#[derive(Resource)]
pub struct TerrainChunkMap(pub Arc<Mutex<HashMap<(i16, i16, i16), TerrainChunk>>>);

pub enum ChunkSpawnResult {
    ToSpawn(((i16, i16, i16), Option<Collider>, Mesh)),
    ToDespawn((i16, i16, i16)),
    ToGiveCollider(((i16, i16, i16), Collider)),
}

pub enum WriteCmd {
    WriteNonUniform {
        chunk: TerrainChunk,
        coord: (i16, i16, i16),
    },
    UpdateNonUniform {
        offset: u64,
        chunk: TerrainChunk,
    },
    WriteUniformAir {
        coord: (i16, i16, i16),
    },
    WriteUniformDirt {
        coord: (i16, i16, i16),
    },
    RemoveUniformAir {
        coord: (i16, i16, i16),
    },
    RemoveUniformDirt {
        coord: (i16, i16, i16),
    },
}

#[derive(Resource)]
pub struct ChunkSpawnReciever(Receiver<ChunkSpawnResult>);

//level 0: full detail, returns TerrainChunk and collider
//level n: full/(2^n) detail, no persistant data besides mesh
pub struct ChunkRequest {
    pub position: (i16, i16, i16),
    pub load_status: u8,
    pub upgrade: bool,
    pub distance_squared: u32,
}

impl PartialEq for ChunkRequest {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && self.load_status == other.load_status
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

struct ChunkResult {
    has_entity: bool,
    chunk_coord: (i16, i16, i16),
    load_status: u8,
}

#[derive(Resource)]
pub struct WriteCmdSender(pub Sender<WriteCmd>);

pub fn setup_chunk_driver(
    mut commands: Commands,
    index_map_delta: Res<ChunkIndexMapDelta>,
    player_translation_mutex_handle: Res<PlayerTranslationMutexHandle>,
) {
    #[cfg(feature = "timers")]
    {
        std::fs::create_dir_all("plots").unwrap();
    }
    let num_processors = thread::available_parallelism().unwrap().get();
    info!("Number of Available Processors: {}", num_processors);
    commands.insert_resource(LogicalProcesors(num_processors));
    let player_translation_arc = Arc::clone(&player_translation_mutex_handle.0);
    let (chunk_spawn_sender, chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    let terrain_chunk_map = Arc::new(Mutex::new(HashMap::new()));
    let terrain_chunk_map_arc = Arc::clone(&terrain_chunk_map);
    let (req_tx, req_rx) = unbounded::<ChunkRequest>();
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let svo = Arc::new(Mutex::new(ChunkSvo::new()));
    commands.insert_resource(TerrainChunkMap(terrain_chunk_map));
    commands.insert_resource(ChunkSpawnReciever(chunk_spawn_reciever));
    let root = get_project_root();
    let mut air_compression_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/air_compression_data.txt"))
        .unwrap();
    let mut dirt_compression_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/dirt_compression_data.txt"))
        .unwrap();
    let (uniform_air_chunks, empty_air_offsets) = load_uniform_chunks(&mut air_compression_file);
    let uniform_air_chunks = Arc::new(uniform_air_chunks);
    let (uniform_dirt_chunks, empty_dirt_offsets) = load_uniform_chunks(&mut dirt_compression_file);
    let uniform_dirt_chunks = Arc::new(uniform_dirt_chunks);
    println!("Loaded {} compressed air chunks", uniform_air_chunks.len());
    println!(
        "Loaded {} compressed dirt chunks",
        uniform_dirt_chunks.len()
    );
    let fbm =
        || -> GeneratorWrapper<SafeNode> { (opensimplex2().fbm(0.0000000, 0.5, 1, 2.5)).build() }();
    commands.insert_resource(NoiseGenerator(fbm.clone()));
    let uniform_air_chunk_deltas = Arc::new(Mutex::new(HashSet::new()));
    let uniform_dirt_chunk_deltas = Arc::new(Mutex::new(HashSet::new()));
    let (write_tx, write_rx) = crossbeam_channel::unbounded();
    let index_map_delta_arc = Arc::clone(&index_map_delta.0);
    let data_file_write = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/chunk_data.txt"))
        .unwrap();
    let mut chunk_index_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(root.join("data/chunk_index_data.txt"))
        .unwrap();
    let index_map_read = Arc::new(load_chunk_index_map(&mut chunk_index_file));
    println!("Loaded {} chunks into index map", index_map_read.len());
    thread::spawn(move || {
        dedicated_write_thread(
            write_rx,
            index_map_delta_arc,
            data_file_write,
            chunk_index_file,
            air_compression_file,
            dirt_compression_file,
            empty_air_offsets,
            empty_dirt_offsets,
        );
    });
    for thread_idx in 0..num_processors.saturating_sub(2) {
        //leave one processor free for main thread and one for svo manager <- might be wrong
        let index_map_read = Arc::clone(&index_map_read);
        let index_map_delta = Arc::clone(&index_map_delta.0);
        let uniform_air_chunks_delta = Arc::clone(&uniform_air_chunk_deltas);
        let uniform_dirt_chunks_delta = Arc::clone(&uniform_dirt_chunk_deltas);
        let chunk_data_file_read = OpenOptions::new()
            .read(true)
            .open(root.join("data/chunk_data.txt"))
            .unwrap();
        let req_rx_clone = req_rx.clone();
        let res_tx_clone = res_tx.clone();
        let terrain_chunk_map_arc = Arc::clone(&terrain_chunk_map_arc);
        let svo_arc = Arc::clone(&svo);
        let chunk_spawn_channel = chunk_spawn_sender.clone();
        let fbm_clone = fbm.clone();
        let uniform_air_chunks_read_only = Arc::clone(&uniform_air_chunks);
        let uniform_dirt_chunks_read_only = Arc::clone(&uniform_dirt_chunks);
        let write_sender_clone = write_tx.clone();
        thread::spawn(move || {
            chunk_loader_thread(
                thread_idx,
                req_rx_clone,
                res_tx_clone,
                index_map_read,
                index_map_delta,
                uniform_air_chunks_delta,
                uniform_dirt_chunks_delta,
                chunk_data_file_read,
                terrain_chunk_map_arc,
                svo_arc,
                chunk_spawn_channel,
                fbm_clone,
                uniform_air_chunks_read_only,
                uniform_dirt_chunks_read_only,
                write_sender_clone,
            );
        });
    }
    thread::spawn(move || {
        svo_manager_thread(
            res_rx,
            player_translation_arc,
            chunk_spawn_sender,
            req_tx,
            svo,
        );
    });
    commands.insert_resource(WriteCmdSender(write_tx));
    commands.insert_resource(ChunkIndexMapRead(index_map_read));
}

fn dedicated_write_thread(
    rx: Receiver<WriteCmd>,
    index_map_delta: Arc<Mutex<HashMap<(i16, i16, i16), u64>>>,
    mut chunk_data_file: File,
    mut chunk_index_file: File,
    mut air_file: File,
    mut dirt_file: File,
    mut air_empty_offsets: VecDeque<u64>,
    mut dirt_empty_offsets: VecDeque<u64>,
) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            WriteCmd::WriteNonUniform { chunk, coord } => {
                let mut index_map = index_map_delta.lock().unwrap();
                create_chunk_file_data(
                    &chunk,
                    &coord,
                    &mut index_map,
                    &mut chunk_data_file,
                    &mut chunk_index_file,
                );
            }
            WriteCmd::UpdateNonUniform { offset, chunk } => {
                update_chunk_file_data(offset, &chunk, &mut chunk_data_file);
            }
            WriteCmd::WriteUniformAir { coord } => {
                write_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
            }
            WriteCmd::WriteUniformDirt { coord } => {
                write_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
            }
            WriteCmd::RemoveUniformAir { coord } => {
                remove_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
            }
            WriteCmd::RemoveUniformDirt { coord } => {
                remove_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
            }
        }
    }
}

//compute thread for loading chunks from disk
//recieves chunk load requests from svo_manager_thread and returns the data
fn chunk_loader_thread(
    #[cfg_attr(not(feature = "timers"), allow(unused_variables))] thread_idx: usize,
    req_rx: Receiver<ChunkRequest>,
    res_tx: Sender<ChunkResult>,
    index_map_read: Arc<HashMap<(i16, i16, i16), u64>>,
    index_map_delta: Arc<Mutex<HashMap<(i16, i16, i16), u64>>>,
    uniform_air_chunks_delta: Arc<Mutex<HashSet<(i16, i16, i16)>>>,
    uniform_dirt_chunks_delta: Arc<Mutex<HashSet<(i16, i16, i16)>>>,
    mut chunk_data_file_read: File,
    terrain_chunk_map: Arc<Mutex<HashMap<(i16, i16, i16), TerrainChunk>>>,
    svo: Arc<Mutex<ChunkSvo>>,
    chunk_spawn_channel: Sender<ChunkSpawnResult>,
    fbm: GeneratorWrapper<SafeNode>,
    uniform_air_chunks_read_only: Arc<HashSet<(i16, i16, i16)>>,
    uniform_dirt_chunks_read_only: Arc<HashSet<(i16, i16, i16)>>,
    write_sender: Sender<WriteCmd>,
) {
    let mut priority_queue = BinaryHeap::new();
    #[cfg(feature = "timers")]
    let mut chunks_generated = 0;
    #[cfg(feature = "timers")]
    let mut chunks_with_entities = 0;
    #[cfg(feature = "timers")]
    let mut last_record_time = Instant::now();
    #[cfg(feature = "timers")]
    let start_time = Instant::now();
    #[cfg(feature = "timers")]
    let mut throughput_file =
        File::create(format!("plots/latest/throughput_thread_{}.csv", thread_idx)).unwrap();
    #[cfg(feature = "timers")]
    writeln!(
        throughput_file,
        "time_seconds,chunks_per_second,entity_chunks_per_second"
    )
    .unwrap();
    #[cfg(feature = "timers")]
    let mut queue_size_file =
        File::create(format!("plots/latest/queue_size_thread_{}.csv", thread_idx)).unwrap();
    #[cfg(feature = "timers")]
    writeln!(queue_size_file, "time_seconds,queue_size").unwrap();
    loop {
        if priority_queue.is_empty() {
            priority_queue.push(req_rx.recv().expect("channel closed"));
        }
        while let Ok(req) = req_rx.try_recv() {
            priority_queue.push(req);
        }
        if let Some(req) = priority_queue.pop() {
            let chunk_coord = req.position;
            //first search the read only without locking
            let mut is_uniform_air = uniform_air_chunks_read_only.contains(&chunk_coord);
            let mut is_uniform_dirt = if is_uniform_air {
                false
            } else {
                uniform_dirt_chunks_read_only.contains(&chunk_coord)
            };
            //if neither, lock and double check the runtime uniform deltas
            if !is_uniform_air && !is_uniform_dirt {
                let uniform_air_chunks_lock = uniform_air_chunks_delta.lock().unwrap();
                is_uniform_air = uniform_air_chunks_lock.contains(&chunk_coord);
                drop(uniform_air_chunks_lock);
                if !is_uniform_air {
                    let uniform_dirt_chunks_lock = uniform_dirt_chunks_delta.lock().unwrap();
                    is_uniform_dirt = uniform_dirt_chunks_lock.contains(&chunk_coord);
                    drop(uniform_dirt_chunks_lock);
                }
            }
            let has_mesh = if is_uniform_air || is_uniform_dirt {
                if req.load_status == 0 {
                    let data = if is_uniform_air {
                        TerrainChunk {
                            densities: Box::new([i16::MAX; SAMPLES_PER_CHUNK]),
                            materials: Box::new([0; SAMPLES_PER_CHUNK]),
                            is_uniform: UniformChunk::Air,
                        }
                    } else {
                        TerrainChunk {
                            densities: Box::new([i16::MIN; SAMPLES_PER_CHUNK]),
                            materials: Box::new([1; SAMPLES_PER_CHUNK]),
                            is_uniform: UniformChunk::Dirt,
                        }
                    };
                    terrain_chunk_map.lock().unwrap().insert(chunk_coord, data);
                }
                false
            } else {
                //first check read only initial copy
                let file_offset = {
                    let offset = index_map_read.get(&chunk_coord).cloned();
                    //if index isnt there maybe its in delta per chance
                    let offset = if offset.is_none() {
                        let index_map_lock = index_map_delta.lock().unwrap();
                        index_map_lock.get(&chunk_coord).cloned().clone()
                    } else {
                        offset
                    };
                    offset
                };
                let chunk_sdfs = if let Some(offset) = file_offset {
                    load_chunk_data(&mut chunk_data_file_read, offset)
                } else {
                    let chunk = TerrainChunk::new(chunk_coord, &fbm);
                    match chunk.is_uniform {
                        UniformChunk::Air => {
                            let mut uniform_air_chunks_lock =
                                uniform_air_chunks_delta.lock().unwrap();
                            uniform_air_chunks_lock.insert(chunk_coord);
                            drop(uniform_air_chunks_lock);
                            write_sender
                                .send(WriteCmd::WriteUniformAir { coord: chunk_coord })
                                .unwrap();
                        }
                        UniformChunk::Dirt => {
                            let mut uniform_dirt_chunks_lock =
                                uniform_dirt_chunks_delta.lock().unwrap();
                            uniform_dirt_chunks_lock.insert(chunk_coord);
                            drop(uniform_dirt_chunks_lock);
                            write_sender
                                .send(WriteCmd::WriteUniformDirt { coord: chunk_coord })
                                .unwrap();
                        }
                        UniformChunk::NonUniform => {
                            write_sender
                                .send(WriteCmd::WriteNonUniform {
                                    chunk: chunk.clone(),
                                    coord: chunk_coord,
                                })
                                .unwrap();
                        }
                    }
                    chunk
                };
                let has_mesh = chunk_contains_surface(&chunk_sdfs);
                let has_collider = has_mesh && req.load_status == 0;
                let (collider, mesh) = if has_mesh {
                    let (vertices, normals, material_ids, indices) = mc_mesh_generation(
                        &chunk_sdfs.densities,
                        &chunk_sdfs.materials,
                        SAMPLES_PER_CHUNK_DIM,
                        HALF_CHUNK,
                    );
                    let mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
                    let collider = if has_collider {
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
                if req.load_status == 0 {
                    let mut terrain_chunk_map_lock = terrain_chunk_map.lock().unwrap();
                    terrain_chunk_map_lock.insert(chunk_coord, chunk_sdfs);
                    drop(terrain_chunk_map_lock);
                }
                if has_collider {
                    if req.upgrade {
                        let mut svo_lock = svo.lock().unwrap();
                        let (_has_entity, load_status) =
                            svo_lock.root.get_mut(chunk_coord).unwrap();
                        *load_status = req.load_status;
                        drop(svo_lock);
                        chunk_spawn_channel
                            .send(ChunkSpawnResult::ToGiveCollider((
                                chunk_coord,
                                collider.unwrap(),
                            )))
                            .unwrap();
                    } else {
                        chunk_spawn_channel
                            .send(ChunkSpawnResult::ToSpawn((
                                chunk_coord,
                                collider,
                                mesh.unwrap(),
                            )))
                            .unwrap();
                    }
                } else {
                    if !req.upgrade {
                        if has_mesh {
                            chunk_spawn_channel
                                .send(ChunkSpawnResult::ToSpawn((
                                    chunk_coord,
                                    None,
                                    mesh.unwrap(),
                                )))
                                .unwrap();
                        }
                    }
                }
                has_mesh
            };
            let _ = res_tx.send(ChunkResult {
                has_entity: has_mesh,
                chunk_coord,
                load_status: req.load_status,
            });
            #[cfg(feature = "timers")]
            {
                chunks_generated += 1;
                if has_mesh {
                    chunks_with_entities += 1;
                }
                let elapsed = Instant::now().duration_since(last_record_time);
                if elapsed.as_secs() >= 1 {
                    let time_seconds = Instant::now().duration_since(start_time).as_secs_f64();
                    let chunks_per_second = chunks_generated as f64 / elapsed.as_secs_f64();
                    let entity_chunks_per_second =
                        chunks_with_entities as f64 / elapsed.as_secs_f64();
                    let queue_size = priority_queue.len();
                    writeln!(
                        &mut throughput_file,
                        "{},{},{}",
                        time_seconds, chunks_per_second, entity_chunks_per_second
                    )
                    .unwrap();
                    writeln!(&mut queue_size_file, "{},{}", time_seconds, queue_size).unwrap();
                    chunks_generated = 0;
                    chunks_with_entities = 0;
                    last_record_time = Instant::now();
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
    player_translation: Arc<Mutex<Vec3>>,
    chunk_spawn_channel: Sender<ChunkSpawnResult>,
    load_request_channel: Sender<ChunkRequest>,
    svo: Arc<Mutex<ChunkSvo>>,
) {
    #[cfg(feature = "timers")]
    let t0 = Instant::now();
    #[cfg(feature = "timers")]
    let mut first_completion_printed = false;
    let mut request_buffer = Vec::new();
    //do initial z0 load so player can start moving
    let mut chunks_being_loaded = HashSet::new();
    let mut svo_lock = svo.lock().unwrap();
    let player_translation_lock = player_translation.lock().unwrap();
    let initial_player_translation = *player_translation_lock;
    drop(player_translation_lock);
    svo_lock.root.fill_missing_chunks_in_radius(
        &initial_player_translation,
        Z0_RADIUS,
        &mut chunks_being_loaded,
        &mut request_buffer,
    );
    drop(svo_lock);
    for request in request_buffer.drain(..) {
        let _ = load_request_channel.send(request);
    }
    let mut results_batch = Vec::new();
    loop {
        let svo_lock = svo.lock().unwrap();
        drop(svo_lock);
        while let Ok(result) = results_channel.try_recv() {
            //this channel creates a 1000x improvement in performance at high thread count cause of magic
            results_batch.push(result);
        }
        if !results_batch.is_empty() {
            let mut svo_lock = svo.lock().unwrap();
            for result in results_batch.drain(..) {
                svo_lock
                    .root
                    .insert(result.chunk_coord, result.load_status, result.has_entity);
                chunks_being_loaded.remove(&result.chunk_coord);
            }
            drop(svo_lock);
        }
        let player_translation_lock = player_translation.lock().unwrap();
        let player_translation = *player_translation_lock;
        drop(player_translation_lock);
        let mut chunks_to_deallocate = Vec::new();
        let mut svo_lock = svo.lock().unwrap();
        svo_lock
            .root
            .query_chunks_outside_sphere(&player_translation, &mut chunks_to_deallocate);
        for (chunk_coord, _) in &chunks_to_deallocate {
            svo_lock.root.delete(*chunk_coord);
        }
        drop(svo_lock);
        for (chunk_coord, has_entity) in &chunks_to_deallocate {
            if *has_entity {
                chunk_spawn_channel
                    .send(ChunkSpawnResult::ToDespawn(*chunk_coord))
                    .unwrap();
            }
            chunks_being_loaded.remove(chunk_coord);
        }
        let mut svo_lock = svo.lock().unwrap();
        svo_lock.root.fill_missing_chunks_in_radius(
            &player_translation,
            MAX_RADIUS,
            &mut chunks_being_loaded,
            &mut request_buffer,
        );
        drop(svo_lock);
        for request in request_buffer.drain(..) {
            let _ = load_request_channel.send(request);
        }
        #[cfg(feature = "timers")]
        {
            if !first_completion_printed && chunks_being_loaded.is_empty() {
                let run_time = Instant::now().duration_since(t0).as_millis();
                println!("SVO Manager First Completion Time: {:?} ms", run_time);
                first_completion_printed = true;
                INITIAL_CHUNKS_LOADED.store(true, Ordering::Relaxed);
            }
        }
    }
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
    let mut count = 0;
    while let Ok(req) = req_rx.0.try_recv() {
        #[cfg(feature = "timers")]
        let t0 = Instant::now();
        count += 1;
        match req {
            ChunkSpawnResult::ToSpawn((chunk, collider_opt, mesh)) => {
                let entity = match collider_opt {
                    Some(collider) => commands
                        .spawn((
                            Mesh3d(meshes.add(mesh)),
                            collider,
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id(),
                    None => commands
                        .spawn((
                            Mesh3d(meshes.add(mesh)),
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id(),
                };
                chunk_entity_map.0.insert(chunk, entity);
            }
            ChunkSpawnResult::ToGiveCollider((chunk, collider)) => {
                let entity = chunk_entity_map.0.get(&chunk).unwrap();
                commands.entity(*entity).insert(collider);
            }
            ChunkSpawnResult::ToDespawn(chunk_coord) => {
                commands
                    .entity(chunk_entity_map.0.remove(&chunk_coord).unwrap())
                    .despawn();
                let mut terrain_chunk_map_lock = terrain_chunk_map.0.lock().unwrap();
                terrain_chunk_map_lock.remove(&chunk_coord);
                drop(terrain_chunk_map_lock);
            }
        }
        if count >= 40 {
            //this is stupid but bevy is slow
            return;
        }
        #[cfg(feature = "timers")]
        {
            let run_time = Instant::now().duration_since(t0).as_millis();
            if run_time >= 10 {
                println!("chunk_spawn_reciever: {:?} ms", run_time);
            }
        }
    }
}

//wait for player's chunk to be spawned
pub fn project_downward(
    chunk_entity_map: Res<ChunkEntityMap>,
    player_position: Single<&Transform, (With<PlayerTag>, Without<ChunkTag>)>,
    spawned_chunks_query: Query<(), With<ChunkTag>>,
) {
    let player_chunk = world_pos_to_chunk_coord(&player_position.translation);
    if let Some(entity) = chunk_entity_map.0.get(&player_chunk) {
        if spawned_chunks_query.get(*entity).is_ok() {
            INITIAL_CHUNKS_LOADED.store(true, Ordering::Relaxed);
        }
    }
}
