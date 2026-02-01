use crate::{
    conversions::cluster_coord_to_min_chunk_coord,
    data_loader::file_loader::{
        CHUNK_SERIALIZED_SIZE, get_project_root, load_chunk, load_chunk_index_map,
        load_uniform_chunks, remove_uniform_chunk, update_chunk, write_chunk,
    },
    terrain::{
        chunk_generator::{
            NOISE_AMPLITUDE, NOISE_FREQUENCY, NOISE_SEED, calculate_chunk_start, get_fbm,
        },
        terrain::{
            CHUNK_SIZE, CLUSTER_SIZE, HEIGHT_MAP_GRID_SIZE, MAX_RADIUS_SQUARED,
            NonUniformTerrainChunk, Z0_RADIUS_SQUARED, generate_chunk_into_buffers,
        },
    },
};
use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use crossbeam_channel::{Receiver, Sender, unbounded};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(feature = "timers")]
use std::io::Write;
use std::{
    collections::{BinaryHeap, VecDeque},
    fs::{File, OpenOptions},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::file_loader::{ChunkEntityMap, write_uniform_chunk},
    marching_cubes::mc::mc_mesh_generation,
    player::player::PlayerTag,
    sparse_voxel_octree::ChunkSvo,
    terrain::{
        chunk_generator::chunk_contains_surface,
        terrain::{
            ChunkTag, HALF_CHUNK, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk,
            TerrainMaterialHandle, Uniformity, generate_bevy_mesh,
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
pub struct TerrainChunkMap(pub Arc<Mutex<FxHashMap<(i16, i16, i16), TerrainChunk>>>);

pub enum ChunkSpawnResult {
    ToSpawn(((i16, i16, i16), Option<Collider>, Mesh)),
    ToDespawn((i16, i16, i16)),
    ToGiveCollider(((i16, i16, i16), Collider)),
}

pub enum WriteCmd {
    UpdateNonUniform {
        densities: Arc<[i16]>,
        materials: Arc<[u8]>,
        coord: (i16, i16, i16),
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
#[derive(Debug)]
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
    has_entity: [bool; CLUSTER_SIZE * CLUSTER_SIZE * CLUSTER_SIZE],
    cluster_coord: (i16, i16, i16),
    load_status: u8,
}

#[derive(Resource)]
pub struct WriteCmdSender(pub Sender<WriteCmd>);

pub fn setup_chunk_driver(
    mut commands: Commands,
    player_translation_mutex_handle: Res<PlayerTranslationMutexHandle>,
) {
    #[cfg(feature = "timers")]
    {
        std::fs::create_dir_all("plots").unwrap();
    }
    let index_map_delta = Arc::new(RwLock::new(FxHashMap::default()));
    let num_processors = thread::available_parallelism().unwrap().get();
    info!("Number of Available Processors: {}", num_processors);
    commands.insert_resource(LogicalProcesors(num_processors));
    let player_translation_arc = Arc::clone(&player_translation_mutex_handle.0);
    let (chunk_spawn_sender, chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    let terrain_chunk_map = Arc::new(Mutex::new(FxHashMap::default()));
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let svo = ChunkSvo::new();
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
    let mut t0 = Instant::now();
    let (uniform_air_chunks, empty_air_offsets) = load_uniform_chunks(&mut air_compression_file);
    println!(
        "Loaded {} compressed air chunks in {} ms.",
        uniform_air_chunks.len(),
        t0.elapsed().as_millis()
    );
    let uniform_air_chunks = Arc::new(uniform_air_chunks);
    t0 = Instant::now();
    let (uniform_dirt_chunks, empty_dirt_offsets) = load_uniform_chunks(&mut dirt_compression_file);
    println!(
        "Loaded {} compressed dirt chunks in {} ms.",
        uniform_dirt_chunks.len(),
        t0.elapsed().as_millis()
    );
    let uniform_dirt_chunks = Arc::new(uniform_dirt_chunks);
    let fbm = get_fbm();
    commands.insert_resource(NoiseGenerator(fbm.clone()));
    let (write_tx, write_rx) = crossbeam_channel::unbounded();
    let index_map_delta_arc = Arc::clone(&index_map_delta);
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
    let t0 = Instant::now();
    let index_map_read = Arc::new(load_chunk_index_map(&mut chunk_index_file));
    let index_map_read_arc = Arc::clone(&index_map_read);
    let (terrain_chunk_map_insert_sender, terrain_chunk_map_insert_reciever) =
        crossbeam_channel::unbounded();
    println!(
        "Loaded {} chunks into index map in {} ms.",
        index_map_read.len(),
        t0.elapsed().as_millis()
    );
    let uniform_air_set = uniform_air_chunks.as_ref().clone();
    let uniform_dirt_set = uniform_dirt_chunks.as_ref().clone();
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
            index_map_read_arc,
            uniform_air_set,
            uniform_dirt_set,
        );
    });
    let priority_queue = Arc::new((Mutex::new(BinaryHeap::new()), Condvar::new()));
    for thread_idx in 0..num_processors.saturating_sub(4) {
        //leave one processor free for main thread and one for svo manager <- might be wrong
        let index_map_read = Arc::clone(&index_map_read);
        let index_map_delta = Arc::clone(&index_map_delta);
        let chunk_data_file_read = OpenOptions::new()
            .read(true)
            .open(root.join("data/chunk_data.txt"))
            .unwrap();
        let res_tx_clone = res_tx.clone();
        let chunk_spawn_channel = chunk_spawn_sender.clone();
        let fbm_clone = fbm.clone();
        let uniform_air_chunks_read_only = Arc::clone(&uniform_air_chunks);
        let uniform_dirt_chunks_read_only = Arc::clone(&uniform_dirt_chunks);
        let write_sender_clone = write_tx.clone();
        let priority_queue_arc = Arc::clone(&priority_queue);
        let terrain_chunk_map_insert_sender_clone = terrain_chunk_map_insert_sender.clone();
        thread::spawn(move || {
            chunk_loader_thread(
                thread_idx,
                res_tx_clone,
                index_map_read,
                index_map_delta,
                chunk_data_file_read,
                chunk_spawn_channel,
                fbm_clone,
                uniform_air_chunks_read_only,
                uniform_dirt_chunks_read_only,
                write_sender_clone,
                priority_queue_arc,
                terrain_chunk_map_insert_sender_clone,
            );
        });
    }
    let terrain_chunk_map_arc = Arc::clone(&terrain_chunk_map);
    thread::spawn(move || {
        svo_manager_thread(
            res_rx,
            player_translation_arc,
            chunk_spawn_sender,
            svo,
            priority_queue,
            terrain_chunk_map_arc,
            terrain_chunk_map_insert_reciever,
        );
    });
    commands.insert_resource(WriteCmdSender(write_tx));
    commands.insert_resource(TerrainChunkMap(terrain_chunk_map));
}

fn dedicated_write_thread(
    rx: Receiver<WriteCmd>,
    index_map_delta: Arc<RwLock<FxHashMap<(i16, i16, i16), u64>>>,
    mut chunk_data_file: File,
    mut chunk_index_file: File,
    mut air_file: File,
    mut dirt_file: File,
    mut air_empty_offsets: VecDeque<u64>,
    mut dirt_empty_offsets: VecDeque<u64>,
    chunk_index_map_read: Arc<FxHashMap<(i16, i16, i16), u64>>,
    mut uniform_air_set: FxHashSet<(i16, i16, i16)>, //track locally to avoid duplicate writes
    mut uniform_dirt_set: FxHashSet<(i16, i16, i16)>, //these are real time sets whereas thee worker thread sets are initial snapshots. It may be worth using the arc here and storing a delta instead
) {
    let mut chunk_write_reuse = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    let mut serial_buffer = [0; CHUNK_SERIALIZED_SIZE];
    while let Ok(cmd) = rx.recv() {
        match cmd {
            WriteCmd::UpdateNonUniform {
                densities,
                materials,
                coord,
            } => {
                //offset lookup must be async to avoid situation where we try to update a chunk that isnt written
                //because the channel is ordered, the write should always process before the update
                let offset = chunk_index_map_read
                    .get(&coord)
                    .cloned()
                    .or_else(|| index_map_delta.read().get(&coord).cloned());
                match offset {
                    Some(offset) => {
                        update_chunk(
                            offset,
                            &densities,
                            &materials,
                            &mut chunk_data_file,
                            &mut serial_buffer,
                        );
                    }
                    None => {
                        let mut index_map = index_map_delta.write();
                        write_chunk(
                            &densities,
                            &materials,
                            &coord,
                            &mut index_map,
                            &mut chunk_data_file,
                            &mut chunk_index_file,
                            &mut chunk_write_reuse,
                            &mut serial_buffer,
                        );
                    }
                }
            }
            WriteCmd::WriteUniformAir { coord } => {
                if !uniform_air_set.contains(&coord) {
                    write_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
                    uniform_air_set.insert(coord);
                }
            }
            WriteCmd::WriteUniformDirt { coord } => {
                if !uniform_dirt_set.contains(&coord) {
                    write_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
                    uniform_dirt_set.insert(coord);
                }
            }
            WriteCmd::RemoveUniformAir { coord } => {
                remove_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
                uniform_air_set.remove(&coord);
            }
            WriteCmd::RemoveUniformDirt { coord } => {
                remove_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
                uniform_dirt_set.remove(&coord);
            }
        }
    }
}

//compute thread for loading chunks from disk
//recieves chunk load requests from svo_manager_thread and returns the data
// - no uniform clusters should enter this thread
fn chunk_loader_thread(
    #[cfg_attr(not(feature = "timers"), allow(unused_variables))] thread_idx: usize,
    res_tx: Sender<ChunkResult>,
    index_map_read: Arc<FxHashMap<(i16, i16, i16), u64>>,
    index_map_delta: Arc<RwLock<FxHashMap<(i16, i16, i16), u64>>>,
    mut chunk_data_file_read: File,
    chunk_spawn_channel: Sender<ChunkSpawnResult>,
    fbm: GeneratorWrapper<SafeNode>,
    uniform_air_chunks_read_only: Arc<FxHashSet<(i16, i16, i16)>>,
    uniform_dirt_chunks_read_only: Arc<FxHashSet<(i16, i16, i16)>>,
    write_sender: Sender<WriteCmd>,
    priority_queue: Arc<(Mutex<BinaryHeap<ChunkRequest>>, Condvar)>,
    terrain_chunk_map_insert_sender: Sender<((i16, i16, i16), TerrainChunk)>,
) {
    const REDUCTION_FACTOR: usize = 1;
    const REDUCED_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / REDUCTION_FACTOR;
    const REDUCED_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / REDUCTION_FACTOR.pow(3);
    const REDUCED_HEIGHTMAP_GRID_SIZE: usize = HEIGHT_MAP_GRID_SIZE / REDUCTION_FACTOR.pow(2);
    let mut internal_queue = Vec::with_capacity(32);
    let mut density_buffer = [0; REDUCED_SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; REDUCED_SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; REDUCED_HEIGHTMAP_GRID_SIZE];
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
    const PROCESS_BATCH_SIZE: usize = 64;
    loop {
        let (lock, cv) = &*priority_queue;
        let mut heap = lock.lock().unwrap();
        while heap.is_empty() {
            heap = cv.wait(heap).unwrap();
        }
        let bound = heap.len().min(PROCESS_BATCH_SIZE);
        for _ in 0..bound {
            internal_queue.push(heap.pop().unwrap());
        }
        drop(heap);
        for request in internal_queue.drain(..) {
            let mut has_entity_buffer = [false; CLUSTER_SIZE * CLUSTER_SIZE * CLUSTER_SIZE];
            let mut rolling = 0;
            let min_chunk = cluster_coord_to_min_chunk_coord(&request.position);
            for chunk_x in min_chunk.0..min_chunk.0 + CLUSTER_SIZE as i16 {
                for chunk_y in min_chunk.1..min_chunk.1 + CLUSTER_SIZE as i16 {
                    for chunk_z in min_chunk.2..min_chunk.2 + CLUSTER_SIZE as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        let (found_in_uniform_air, found_in_uniform_dirt) = search_uniform_maps(
                            &chunk_coord,
                            &uniform_air_chunks_read_only,
                            &uniform_dirt_chunks_read_only,
                        );
                        let has_surface = if found_in_uniform_air || found_in_uniform_dirt {
                            if request.load_status == 0 {
                                let terrain_chunk = if found_in_uniform_air {
                                    TerrainChunk::UniformAir
                                } else {
                                    TerrainChunk::UniformDirt
                                };
                                terrain_chunk_map_insert_sender
                                    .send((chunk_coord, terrain_chunk))
                                    .unwrap();
                            }
                            false
                        } else {
                            //first check read only initial copy
                            let file_offset = index_map_read
                                .get(&chunk_coord)
                                .copied()
                                .or_else(|| index_map_delta.read().get(&chunk_coord).copied());
                            //check if we even need the sdfs
                            if let Some(offset) = file_offset {
                                load_chunk(
                                    &mut chunk_data_file_read,
                                    offset,
                                    &mut density_buffer,
                                    &mut material_buffer,
                                );
                                if request.load_status == 0 {
                                    terrain_chunk_map_insert_sender
                                        .send((
                                            chunk_coord,
                                            TerrainChunk::NonUniformTerrainChunk(
                                                NonUniformTerrainChunk {
                                                    //allocation here
                                                    densities: Arc::new(density_buffer),
                                                    materials: Arc::new(material_buffer),
                                                },
                                            ),
                                        ))
                                        .unwrap();
                                }
                                chunk_contains_surface(&density_buffer)
                            } else {
                                let chunk_start = calculate_chunk_start(&chunk_coord);
                                let chunk_center_sample = fbm.gen_single_2d(
                                    (chunk_start.x + HALF_CHUNK) * NOISE_FREQUENCY,
                                    (chunk_start.z + HALF_CHUNK) * NOISE_FREQUENCY,
                                    NOISE_SEED,
                                ) * NOISE_AMPLITUDE;
                                //conservative heuristic: if the surface height at the first sample is greater than one chunk above the bottom of the chunk, we assume it is uniform air
                                if chunk_center_sample + CHUNK_SIZE * 3.0 < chunk_start.y {
                                    write_sender
                                        .send(WriteCmd::WriteUniformAir { coord: chunk_coord })
                                        .unwrap();
                                    false
                                } else {
                                    //generate new chunk
                                    let uniformity = generate_chunk_into_buffers(
                                        &fbm,
                                        chunk_start,
                                        &mut density_buffer,
                                        &mut material_buffer,
                                        &mut heightmap_buffer,
                                        REDUCED_SAMPLES_PER_CHUNK_DIM,
                                    );
                                    match uniformity {
                                        Uniformity::Air => {
                                            write_sender
                                                .send(WriteCmd::WriteUniformAir {
                                                    coord: chunk_coord,
                                                })
                                                .unwrap();
                                        }
                                        Uniformity::Dirt => {
                                            //send even if the write could be a duplicate and handle later
                                            write_sender
                                                .send(WriteCmd::WriteUniformDirt {
                                                    coord: chunk_coord,
                                                })
                                                .unwrap();
                                        }
                                        Uniformity::NonUniform => { // write_sender
                                            //     .send(WriteCmd::WriteNonUniform {
                                            //         densities: Arc::clone(&densities),
                                            //         materials: Arc::clone(&materials),
                                            //         coord: chunk_coord,
                                            //     })
                                            //     .unwrap();
                                        }
                                    }
                                    if request.load_status == 0 {
                                        let terrain_chunk = match uniformity {
                                            Uniformity::Air => TerrainChunk::UniformAir,
                                            Uniformity::Dirt => TerrainChunk::UniformDirt,
                                            Uniformity::NonUniform => {
                                                TerrainChunk::NonUniformTerrainChunk(
                                                    NonUniformTerrainChunk {
                                                        //allocation here
                                                        densities: Arc::new(density_buffer), //allocation here
                                                        materials: Arc::new(material_buffer), //allocation here
                                                    },
                                                )
                                            }
                                        };
                                        terrain_chunk_map_insert_sender
                                            .send((chunk_coord, terrain_chunk))
                                            .unwrap();
                                    }
                                    let has_surface = match uniformity {
                                        Uniformity::Air | Uniformity::Dirt => false,
                                        Uniformity::NonUniform => {
                                            chunk_contains_surface(&density_buffer) //must not call on a potentially corrupted buffer from uniform generation
                                        }
                                    };
                                    has_surface
                                }
                            }
                        };
                        let has_collider = has_surface && request.load_status == 0;
                        let (collider, mesh) = if has_surface {
                            let (vertices, normals, material_ids, indices) = mc_mesh_generation(
                                &density_buffer,
                                &material_buffer,
                                REDUCED_SAMPLES_PER_CHUNK_DIM,
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
                        if has_collider {
                            if request.upgrade {
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
                        } else if !request.upgrade {
                            if has_surface {
                                chunk_spawn_channel
                                    .send(ChunkSpawnResult::ToSpawn((
                                        chunk_coord,
                                        None,
                                        mesh.unwrap(),
                                    )))
                                    .unwrap();
                            }
                        }
                        has_entity_buffer[rolling] = has_surface;
                        rolling += 1;
                        #[cfg(feature = "timers")]
                        {
                            if has_surface {
                                chunks_with_entities += 1;
                            }
                            chunks_generated += 1;
                        }
                    }
                }
            }
            let _ = res_tx.send(ChunkResult {
                has_entity: has_entity_buffer,
                cluster_coord: request.position,
                load_status: request.load_status,
            });
            #[cfg(feature = "timers")]
            {
                let elapsed = Instant::now().duration_since(last_record_time);
                if elapsed.as_secs() >= 1 {
                    let time_seconds = Instant::now().duration_since(start_time).as_secs_f64();
                    let chunks_per_second = chunks_generated as f64 / elapsed.as_secs_f64();
                    let entity_chunks_per_second =
                        chunks_with_entities as f64 / elapsed.as_secs_f64();
                    let queue_size = {
                        let (lock, _cv) = &*priority_queue;
                        lock.lock().unwrap().len()
                    };
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
    mut svo: ChunkSvo,
    priority_queue: Arc<(Mutex<BinaryHeap<ChunkRequest>>, Condvar)>,
    terrain_chunk_map: Arc<Mutex<FxHashMap<(i16, i16, i16), TerrainChunk>>>,
    terrain_chunk_map_insert_reciever: Receiver<((i16, i16, i16), TerrainChunk)>,
) {
    #[cfg(feature = "timers")]
    let t0 = Instant::now();
    #[cfg(feature = "timers")]
    let mut first_completion_printed = false;
    let mut request_buffer = Vec::new();
    //do initial z0 load so player can start moving
    let mut chunks_being_loaded = FxHashSet::default();
    let player_translation_lock = player_translation.lock().unwrap();
    let initial_player_translation = *player_translation_lock;
    drop(player_translation_lock);
    svo.root.fill_missing_chunks_in_radius(
        &initial_player_translation,
        Z0_RADIUS_SQUARED,
        &mut chunks_being_loaded,
        &mut request_buffer,
    );
    let (lock, cv) = &*priority_queue;
    {
        let mut heap = lock.lock().unwrap();
        for req in request_buffer.drain(..) {
            heap.push(req);
        }
    }
    cv.notify_all();
    loop {
        //get z0 chunk allocations from loader threads and insert them
        let mut terrain_map_lock = terrain_chunk_map.lock().unwrap();
        while let Ok((chunk_coord, terrain_chunk)) = terrain_chunk_map_insert_reciever.try_recv() {
            terrain_map_lock.insert(chunk_coord, terrain_chunk);
        }
        drop(terrain_map_lock);
        while let Ok(result) = results_channel.try_recv() {
            svo.root
                .insert(result.cluster_coord, result.load_status, result.has_entity);
            chunks_being_loaded.remove(&result.cluster_coord);
        }
        let player_translation_lock = player_translation.lock().unwrap();
        let player_translation = *player_translation_lock;
        drop(player_translation_lock);
        let mut chunks_to_deallocate = Vec::new();
        svo.root
            .query_chunks_outside_sphere(&player_translation, &mut chunks_to_deallocate);
        for (chunk_coord, _) in &chunks_to_deallocate {
            svo.root.delete(*chunk_coord);
        }
        let mut roller = 0;
        let mut terrain_map_lock = terrain_chunk_map.lock().unwrap();
        for (cluster_coord, has_entity) in &chunks_to_deallocate {
            let min_chunk = cluster_coord_to_min_chunk_coord(cluster_coord);
            for chunk_x in min_chunk.0..min_chunk.0 + CLUSTER_SIZE as i16 {
                for chunk_y in min_chunk.1..min_chunk.1 + CLUSTER_SIZE as i16 {
                    for chunk_z in min_chunk.2..min_chunk.2 + CLUSTER_SIZE as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        if has_entity[roller] {
                            chunk_spawn_channel
                                .send(ChunkSpawnResult::ToDespawn(chunk_coord))
                                .unwrap();
                        }
                        terrain_map_lock.remove(&chunk_coord);
                        roller += 1;
                    }
                }
            }
            chunks_being_loaded.remove(cluster_coord);
            roller = 0;
        }
        drop(terrain_map_lock);
        svo.root.fill_missing_chunks_in_radius(
            &player_translation,
            MAX_RADIUS_SQUARED,
            &mut chunks_being_loaded,
            &mut request_buffer,
        );
        let (lock, cv) = &*priority_queue;
        {
            let mut heap = lock.lock().unwrap();
            for req in request_buffer.drain(..) {
                heap.push(req);
            }
        }
        cv.notify_all();
        #[cfg(feature = "timers")]
        {
            if !first_completion_printed && chunks_being_loaded.is_empty() {
                let run_time = Instant::now().duration_since(t0).as_millis();
                println!("SVO Manager First Completion Time: {:?} ms", run_time);
                first_completion_printed = true;
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
    for chunk_y in (player_chunk.1 - 10..=player_chunk.1).rev() {
        if let Some(entity) = chunk_entity_map
            .0
            .get(&(player_chunk.0, chunk_y, player_chunk.2))
        {
            if spawned_chunks_query.get(*entity).is_ok() {
                INITIAL_CHUNKS_LOADED.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
}

fn search_uniform_maps(
    chunk_coord: &(i16, i16, i16),
    uniform_air_chunks_read_only: &Arc<FxHashSet<(i16, i16, i16)>>,
    uniform_dirt_chunks_read_only: &Arc<FxHashSet<(i16, i16, i16)>>,
) -> (bool, bool) {
    //first search the read only without locking
    let is_uniform_air = uniform_air_chunks_read_only.contains(&chunk_coord);
    let is_uniform_dirt = if is_uniform_air {
        false
    } else {
        uniform_dirt_chunks_read_only.contains(&chunk_coord)
    };
    (is_uniform_air, is_uniform_dirt)
}
