use crate::{
    constants::{
        CHUNK_WORLD_SIZE, CHUNKS_PER_CLUSTER, CHUNKS_PER_CLUSTER_DIM, HALF_CHUNK,
        MAX_RENDER_RADIUS_SQUARED, NOISE_AMPLITUDE, NOISE_FREQUENCY, NOISE_SEED,
        PLAYER_CUBOID_SIZE, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_2D, SAMPLES_PER_CHUNK_DIM,
        SIMULATION_RADIUS_SQUARED,
    },
    conversions::cluster_coord_to_min_chunk_coord,
    data_loader::{
        column_range_map::ColumnRangeMap,
        file_loader::{
            CHUNK_SERIALIZED_SIZE, get_project_root, load_chunk, load_chunk_index_map,
            load_uniform_chunks, remove_uniform_chunk, update_chunk, write_chunk,
        },
    },
    terrain::{
        chunk_generator::{calculate_chunk_start, downscale, get_fbm},
        terrain::{NonUniformTerrainChunk, generate_chunk_into_buffers},
    },
};
use bevy::{camera::primitives::MeshAabb, prelude::*};
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
        terrain::{ChunkTag, TerrainChunk, TerrainMaterialHandle, Uniformity, generate_bevy_mesh},
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

#[repr(u8)]
#[derive(Debug, PartialEq)]
pub enum LoadStateTransition {
    ToFull,              //transition to full LOD with no collider
    ToLod1,              //transition to Lod1 with no collider
    ToLod2,              //transition to Lod2 with no collider
    ToLod3,              //transition to Lod3 with no collider
    ToLod4,              //transition to Lod4 with no collider
    ToLod5,              //transition to Lod5 with no collider
    ToFullWithCollider,  //transition to full LOD and collider is needed
    NoChangeAddCollider, //LOD did not change but collider is needed
}

impl LoadStateTransition {
    fn to_state(&self) -> LoadState {
        match self {
            LoadStateTransition::ToFull => LoadState::Full,
            LoadStateTransition::ToLod1 => LoadState::Lod1,
            LoadStateTransition::ToLod2 => LoadState::Lod2,
            LoadStateTransition::ToLod3 => LoadState::Lod3,
            LoadStateTransition::ToLod4 => LoadState::Lod4,
            LoadStateTransition::ToLod5 => LoadState::Lod5,
            LoadStateTransition::NoChangeAddCollider => LoadState::FullWithCollider,
            LoadStateTransition::ToFullWithCollider => LoadState::FullWithCollider,
        }
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LoadState {
    FullWithCollider,
    Full,
    Lod1,
    Lod2,
    Lod3,
    Lod4,
    Lod5,
}

pub enum ChunkSpawnResult {
    ToSpawn(((i16, i16, i16), Mesh)), //when a chunk is spawned without a collider
    ToSpawnWithCollider(((i16, i16, i16), Collider, Mesh)), //when a chunk is spawned with a collider
    ToDespawn((i16, i16, i16)),
    ToGiveCollider(((i16, i16, i16), Collider)), //same lod but now needs a collider
    ToChangeLod(((i16, i16, i16), Mesh)), //change mesh, assume it has no collider and doesnt need one
    ToChangeLodAddCollider(((i16, i16, i16), Mesh, Collider)), //when its both changing LOD and now needs a collider
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

#[derive(Debug)]
pub struct ClusterRequest {
    pub position: (i16, i16, i16),
    pub distance_squared: f32, //distance to cluster center in world units
    pub load_state_transition: LoadStateTransition,
    pub prev_has_entity: Option<[bool; CHUNKS_PER_CLUSTER]>, //this is needed for the off chance that resolution change causes there to be a new surface
}

impl PartialEq for ClusterRequest {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && self.distance_squared == other.distance_squared
            && self.load_state_transition == other.load_state_transition
    }
}

impl Eq for ClusterRequest {}

impl Ord for ClusterRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance_squared
            .partial_cmp(&self.distance_squared)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for ClusterRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct ChunkResult {
    has_entity: [bool; CHUNKS_PER_CLUSTER],
    cluster_coord: (i16, i16, i16),
    load_state: LoadState,
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
    let t0 = Instant::now();
    let mut column_range_map = ColumnRangeMap::new();
    let empty_air_offsets = load_uniform_chunks(
        &mut air_compression_file,
        Uniformity::Air,
        &mut column_range_map,
    );
    let empty_dirt_offsets = load_uniform_chunks(
        &mut dirt_compression_file,
        Uniformity::Dirt,
        &mut column_range_map,
    );
    println!(
        "Loaded ColumnRangeMap with {} bytes in {} ms.",
        column_range_map.size_in_bytes(),
        t0.elapsed().as_millis()
    );
    let column_range_map_clone = column_range_map.clone();
    let column_range_map = Arc::new(column_range_map);
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
            column_range_map_clone,
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
        let column_range_map_read_only = Arc::clone(&column_range_map);
        let write_sender_clone = write_tx.clone();
        let priority_queue_arc = Arc::clone(&priority_queue);
        let terrain_chunk_map_insert_sender_clone = terrain_chunk_map_insert_sender.clone();
        let _handle = thread::Builder::new()
            .name(format!("chunk_loader_{thread_idx}"))
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                chunk_loader_thread(
                    thread_idx,
                    res_tx_clone,
                    index_map_read,
                    index_map_delta,
                    chunk_data_file_read,
                    chunk_spawn_channel,
                    fbm_clone,
                    column_range_map_read_only,
                    write_sender_clone,
                    priority_queue_arc,
                    terrain_chunk_map_insert_sender_clone,
                );
            })
            .expect("failed to spawn chunk loader thread");
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
    mut column_range_map: ColumnRangeMap, //track locally to avoid duplicate writes. //these are real time sets whereas the worker thread sets are initial snapshots. It may be worth using the arc here and storing a delta instead
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
                let uniformity = column_range_map.contains(coord);
                if uniformity.is_none() {
                    write_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
                    column_range_map.insert(coord, Uniformity::Air);
                }
            }
            WriteCmd::WriteUniformDirt { coord } => {
                let uniformity = column_range_map.contains(coord);
                if uniformity.is_none() {
                    write_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
                    column_range_map.insert(coord, Uniformity::Dirt);
                }
            }
            WriteCmd::RemoveUniformAir { coord } => {
                remove_uniform_chunk(&coord, &mut air_file, &mut air_empty_offsets);
                column_range_map.remove(coord, Uniformity::Air);
            }
            WriteCmd::RemoveUniformDirt { coord } => {
                remove_uniform_chunk(&coord, &mut dirt_file, &mut dirt_empty_offsets);
                column_range_map.remove(coord, Uniformity::Dirt);
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
    column_range_map_read_only: Arc<ColumnRangeMap>,
    write_sender: Sender<WriteCmd>,
    priority_queue: Arc<(Mutex<BinaryHeap<ClusterRequest>>, Condvar)>,
    terrain_chunk_map_insert_sender: Sender<((i16, i16, i16), TerrainChunk)>,
) {
    const RF1: usize = 2;
    const RF2: usize = 4;
    const RF3: usize = 8;
    const RF4: usize = 16;
    const RF5: usize = 32;
    const RF1_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF1;
    const RF1_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / RF1.pow(3);
    const RF2_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF2;
    const RF2_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / RF2.pow(3);
    const RF3_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF3;
    const RF3_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / RF3.pow(3);
    const RF4_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF4;
    const RF4_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / RF4.pow(3);
    const RF5_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF5;
    const RF5_SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK / RF5.pow(3);
    const PROCESS_BATCH_SIZE: usize = 64;
    let mut internal_queue = Vec::with_capacity(32);
    let mut density_buffer = [0; SAMPLES_PER_CHUNK];
    let mut material_buffer = [0; SAMPLES_PER_CHUNK];
    let mut heightmap_buffer = [0.0; SAMPLES_PER_CHUNK_2D];
    let mut density_buffer_r1 = [0; RF1_SAMPLES_PER_CHUNK];
    let mut material_buffer_r1 = [0; RF1_SAMPLES_PER_CHUNK];
    let mut density_buffer_r2 = [0; RF2_SAMPLES_PER_CHUNK];
    let mut material_buffer_r2 = [0; RF2_SAMPLES_PER_CHUNK];
    let mut density_buffer_r3 = [0; RF3_SAMPLES_PER_CHUNK];
    let mut material_buffer_r3 = [0; RF3_SAMPLES_PER_CHUNK];
    let mut density_buffer_r4 = [0; RF4_SAMPLES_PER_CHUNK];
    let mut material_buffer_r4 = [0; RF4_SAMPLES_PER_CHUNK];
    let mut density_buffer_r5 = [0; RF5_SAMPLES_PER_CHUNK];
    let mut material_buffer_r5 = [0; RF5_SAMPLES_PER_CHUNK];
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
            let mut has_entity_buffer = [false; CHUNKS_PER_CLUSTER];
            let mut rolling = 0;
            let min_chunk = cluster_coord_to_min_chunk_coord(&request.position);
            for chunk_x in min_chunk.0..min_chunk.0 + CHUNKS_PER_CLUSTER_DIM as i16 {
                for chunk_y in min_chunk.1..min_chunk.1 + CHUNKS_PER_CLUSTER_DIM as i16 {
                    for chunk_z in min_chunk.2..min_chunk.2 + CHUNKS_PER_CLUSTER_DIM as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        let uniformity_option = column_range_map_read_only.contains(chunk_coord);
                        let uniformity = if uniformity_option.is_none() {
                            //wasnt in the cache, check if in non uniform index map
                            let file_offset = index_map_read
                                .get(&chunk_coord)
                                .copied()
                                .or_else(|| index_map_delta.read().get(&chunk_coord).copied());
                            let uniformity = if let Some(offset) = file_offset {
                                //if its found here its not uniform
                                load_chunk(
                                    &mut chunk_data_file_read,
                                    offset,
                                    &mut density_buffer,
                                    &mut material_buffer,
                                );
                                //propagate uniformity
                                Uniformity::NonUniform
                            } else {
                                //not in non uniform file and not in uniform cache, must generate
                                let chunk_start = calculate_chunk_start(&chunk_coord);
                                //conservative heuristic, if chunk is way up in the air assume its dirt and skip generation. Other wise generate normally
                                let chunk_center_sample = fbm.gen_single_2d(
                                    (chunk_start.x + HALF_CHUNK) * NOISE_FREQUENCY,
                                    (chunk_start.z + HALF_CHUNK) * NOISE_FREQUENCY,
                                    NOISE_SEED,
                                ) * NOISE_AMPLITUDE;
                                let uniformity = if chunk_center_sample + CHUNK_WORLD_SIZE * 3.0
                                    < chunk_start.y
                                {
                                    Uniformity::Air
                                } else {
                                    let uniformity = generate_chunk_into_buffers(
                                        &fbm,
                                        chunk_start,
                                        &mut density_buffer,
                                        &mut material_buffer,
                                        &mut heightmap_buffer,
                                        SAMPLES_PER_CHUNK_DIM,
                                    );
                                    uniformity
                                };
                                //if we find a uniform chunk we send a write command
                                match uniformity {
                                    Uniformity::Air => {
                                        write_sender
                                            .send(WriteCmd::WriteUniformAir { coord: chunk_coord })
                                            .unwrap();
                                    }
                                    Uniformity::Dirt => {
                                        write_sender
                                            .send(WriteCmd::WriteUniformDirt { coord: chunk_coord })
                                            .unwrap();
                                    }
                                    Uniformity::NonUniform => {}
                                }
                                uniformity
                            };
                            uniformity
                        } else {
                            uniformity_option.unwrap()
                        };
                        //at this point we know the uniformity, the buffer is filled if non uniform, and the uniform update write was sent
                        let has_surface = match uniformity {
                            Uniformity::Air => false,
                            Uniformity::Dirt => false,
                            Uniformity::NonUniform => {
                                let has_surface = match request.load_state_transition {
                                    LoadStateTransition::ToLod5 => {
                                        downscale(
                                            &density_buffer,
                                            &material_buffer,
                                            SAMPLES_PER_CHUNK_DIM,
                                            &mut density_buffer_r5,
                                            &mut material_buffer_r5,
                                            RF5_SAMPLES_PER_CHUNK_DIM,
                                        );
                                        let has_surface =
                                            if chunk_contains_surface(&density_buffer_r5) {
                                                let (vertices, normals, material_ids, indices) =
                                                    mc_mesh_generation(
                                                        &density_buffer_r5,
                                                        &material_buffer_r5,
                                                        RF5_SAMPLES_PER_CHUNK_DIM,
                                                        HALF_CHUNK,
                                                    );
                                                let mesh = generate_bevy_mesh(
                                                    vertices,
                                                    normals,
                                                    material_ids,
                                                    indices,
                                                );
                                                let had_entity_before = request
                                                    .prev_has_entity
                                                    .as_ref()
                                                    .map(|a| a[rolling])
                                                    .unwrap_or(false);
                                                if had_entity_before {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToChangeLod((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                } else {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToSpawn((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                }
                                                true
                                            } else {
                                                false
                                            };
                                        has_surface
                                    }
                                    LoadStateTransition::ToLod4 => {
                                        downscale(
                                            &density_buffer,
                                            &material_buffer,
                                            SAMPLES_PER_CHUNK_DIM,
                                            &mut density_buffer_r4,
                                            &mut material_buffer_r4,
                                            RF4_SAMPLES_PER_CHUNK_DIM,
                                        );
                                        let has_surface =
                                            if chunk_contains_surface(&density_buffer_r4) {
                                                let (vertices, normals, material_ids, indices) =
                                                    mc_mesh_generation(
                                                        &density_buffer_r4,
                                                        &material_buffer_r4,
                                                        RF4_SAMPLES_PER_CHUNK_DIM,
                                                        HALF_CHUNK,
                                                    );
                                                let mesh = generate_bevy_mesh(
                                                    vertices,
                                                    normals,
                                                    material_ids,
                                                    indices,
                                                );
                                                let had_entity_before = request
                                                    .prev_has_entity
                                                    .as_ref()
                                                    .map(|a| a[rolling])
                                                    .unwrap_or(false);
                                                if had_entity_before {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToChangeLod((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                } else {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToSpawn((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                }
                                                true
                                            } else {
                                                false
                                            };
                                        has_surface
                                    }
                                    LoadStateTransition::ToLod3 => {
                                        downscale(
                                            &density_buffer,
                                            &material_buffer,
                                            SAMPLES_PER_CHUNK_DIM,
                                            &mut density_buffer_r3,
                                            &mut material_buffer_r3,
                                            RF3_SAMPLES_PER_CHUNK_DIM,
                                        );
                                        let has_surface =
                                            if chunk_contains_surface(&density_buffer_r3) {
                                                let (vertices, normals, material_ids, indices) =
                                                    mc_mesh_generation(
                                                        &density_buffer_r3,
                                                        &material_buffer_r3,
                                                        RF3_SAMPLES_PER_CHUNK_DIM,
                                                        HALF_CHUNK,
                                                    );
                                                let mesh = generate_bevy_mesh(
                                                    vertices,
                                                    normals,
                                                    material_ids,
                                                    indices,
                                                );
                                                let had_entity_before = request
                                                    .prev_has_entity
                                                    .as_ref()
                                                    .map(|a| a[rolling])
                                                    .unwrap_or(false);
                                                if had_entity_before {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToChangeLod((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                } else {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToSpawn((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                }
                                                true
                                            } else {
                                                false
                                            };
                                        has_surface
                                    }
                                    LoadStateTransition::ToLod2 => {
                                        downscale(
                                            &density_buffer,
                                            &material_buffer,
                                            SAMPLES_PER_CHUNK_DIM,
                                            &mut density_buffer_r2,
                                            &mut material_buffer_r2,
                                            RF2_SAMPLES_PER_CHUNK_DIM,
                                        );
                                        let has_surface =
                                            if chunk_contains_surface(&density_buffer_r2) {
                                                let (vertices, normals, material_ids, indices) =
                                                    mc_mesh_generation(
                                                        &density_buffer_r2,
                                                        &material_buffer_r2,
                                                        RF2_SAMPLES_PER_CHUNK_DIM,
                                                        HALF_CHUNK,
                                                    );
                                                let mesh = generate_bevy_mesh(
                                                    vertices,
                                                    normals,
                                                    material_ids,
                                                    indices,
                                                );
                                                let had_entity_before = request
                                                    .prev_has_entity
                                                    .as_ref()
                                                    .map(|a| a[rolling])
                                                    .unwrap_or(false);
                                                if had_entity_before {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToChangeLod((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                } else {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToSpawn((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                }
                                                true
                                            } else {
                                                false
                                            };
                                        has_surface
                                    }
                                    LoadStateTransition::ToLod1 => {
                                        downscale(
                                            &density_buffer,
                                            &material_buffer,
                                            SAMPLES_PER_CHUNK_DIM,
                                            &mut density_buffer_r1,
                                            &mut material_buffer_r1,
                                            RF1_SAMPLES_PER_CHUNK_DIM,
                                        );
                                        let has_surface =
                                            if chunk_contains_surface(&density_buffer_r1) {
                                                let (vertices, normals, material_ids, indices) =
                                                    mc_mesh_generation(
                                                        &density_buffer_r1,
                                                        &material_buffer_r1,
                                                        RF1_SAMPLES_PER_CHUNK_DIM,
                                                        HALF_CHUNK,
                                                    );
                                                let mesh = generate_bevy_mesh(
                                                    vertices,
                                                    normals,
                                                    material_ids,
                                                    indices,
                                                );
                                                let had_entity_before = request
                                                    .prev_has_entity
                                                    .as_ref()
                                                    .map(|a| a[rolling])
                                                    .unwrap_or(false);
                                                if had_entity_before {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToChangeLod((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                } else {
                                                    chunk_spawn_channel
                                                        .send(ChunkSpawnResult::ToSpawn((
                                                            chunk_coord,
                                                            mesh,
                                                        )))
                                                        .unwrap();
                                                }
                                                true
                                            } else {
                                                false
                                            };
                                        has_surface
                                    }
                                    LoadStateTransition::ToFull => {
                                        let has_surface = if chunk_contains_surface(&density_buffer)
                                        {
                                            let (vertices, normals, material_ids, indices) =
                                                mc_mesh_generation(
                                                    &density_buffer,
                                                    &material_buffer,
                                                    SAMPLES_PER_CHUNK_DIM,
                                                    HALF_CHUNK,
                                                );
                                            let mesh = generate_bevy_mesh(
                                                vertices,
                                                normals,
                                                material_ids,
                                                indices,
                                            );
                                            let had_entity_before = request
                                                .prev_has_entity
                                                .as_ref()
                                                .map(|a| a[rolling])
                                                .unwrap_or(false);
                                            if had_entity_before {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToChangeLod((
                                                        chunk_coord,
                                                        mesh,
                                                    )))
                                                    .unwrap();
                                            } else {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToSpawn((
                                                        chunk_coord,
                                                        mesh,
                                                    )))
                                                    .unwrap();
                                            }
                                            true
                                        } else {
                                            false
                                        };
                                        has_surface
                                    }
                                    LoadStateTransition::ToFullWithCollider => {
                                        let has_surface = if chunk_contains_surface(&density_buffer)
                                        {
                                            let (vertices, normals, material_ids, indices) =
                                                mc_mesh_generation(
                                                    &density_buffer,
                                                    &material_buffer,
                                                    SAMPLES_PER_CHUNK_DIM,
                                                    HALF_CHUNK,
                                                );
                                            let mesh = generate_bevy_mesh(
                                                vertices,
                                                normals,
                                                material_ids,
                                                indices,
                                            );
                                            let collider = Collider::from_bevy_mesh(
                                                &mesh,
                                                &ComputedColliderShape::TriMesh(
                                                    TriMeshFlags::default(),
                                                ),
                                            )
                                            .unwrap();
                                            let had_entity_before = request
                                                .prev_has_entity
                                                .as_ref()
                                                .map(|a| a[rolling])
                                                .unwrap_or(false);
                                            if had_entity_before {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToChangeLodAddCollider(
                                                        (chunk_coord, mesh, collider),
                                                    ))
                                                    .unwrap();
                                            } else {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToSpawnWithCollider((
                                                        chunk_coord,
                                                        collider,
                                                        mesh,
                                                    )))
                                                    .unwrap();
                                            }
                                            true
                                        } else {
                                            false
                                        };
                                        has_surface
                                    }
                                    LoadStateTransition::NoChangeAddCollider => {
                                        let has_surface = if chunk_contains_surface(&density_buffer)
                                        {
                                            let (vertices, normals, material_ids, indices) =
                                                mc_mesh_generation(
                                                    &density_buffer,
                                                    &material_buffer,
                                                    SAMPLES_PER_CHUNK_DIM,
                                                    HALF_CHUNK,
                                                );
                                            let mesh = generate_bevy_mesh(
                                                vertices,
                                                normals,
                                                material_ids,
                                                indices,
                                            );
                                            let collider = Collider::from_bevy_mesh(
                                                &mesh,
                                                &ComputedColliderShape::TriMesh(
                                                    TriMeshFlags::default(),
                                                ),
                                            )
                                            .unwrap();
                                            let had_entity_before = request
                                                .prev_has_entity
                                                .as_ref()
                                                .map(|a| a[rolling])
                                                .unwrap_or(false);
                                            if had_entity_before {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToGiveCollider((
                                                        chunk_coord,
                                                        collider,
                                                    )))
                                                    .unwrap();
                                            } else {
                                                chunk_spawn_channel
                                                    .send(ChunkSpawnResult::ToSpawnWithCollider((
                                                        chunk_coord,
                                                        collider,
                                                        mesh,
                                                    )))
                                                    .unwrap();
                                            }
                                            true
                                        } else {
                                            false
                                        };
                                        has_surface
                                    }
                                };
                                has_surface
                            }
                        };
                        //we have determined if a surface exists at the given LOD, sent the requests to spawn the lod, and have all the correct lod buffer loaded
                        //if simulated store the densities
                        let in_simulated = matches!(
                            request.load_state_transition,
                            LoadStateTransition::ToFullWithCollider
                                | LoadStateTransition::NoChangeAddCollider
                        );
                        if in_simulated {
                            match uniformity {
                                Uniformity::Air => {
                                    terrain_chunk_map_insert_sender
                                        .send((chunk_coord, TerrainChunk::UniformAir))
                                        .unwrap();
                                }
                                Uniformity::Dirt => {
                                    terrain_chunk_map_insert_sender
                                        .send((chunk_coord, TerrainChunk::UniformDirt))
                                        .unwrap();
                                }
                                Uniformity::NonUniform => {
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
            let new_state = request.load_state_transition.to_state();
            let _ = res_tx.send(ChunkResult {
                has_entity: has_entity_buffer,
                cluster_coord: request.position,
                load_state: new_state,
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
    priority_queue: Arc<(Mutex<BinaryHeap<ClusterRequest>>, Condvar)>,
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
        SIMULATION_RADIUS_SQUARED,
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
            //store the last state transition used for load
            svo.root
                .insert(result.cluster_coord, result.has_entity, result.load_state);
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
            for chunk_x in min_chunk.0..min_chunk.0 + CHUNKS_PER_CLUSTER_DIM as i16 {
                for chunk_y in min_chunk.1..min_chunk.1 + CHUNKS_PER_CLUSTER_DIM as i16 {
                    for chunk_z in min_chunk.2..min_chunk.2 + CHUNKS_PER_CLUSTER_DIM as i16 {
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
            MAX_RENDER_RADIUS_SQUARED,
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
    mut mesh_handles: ResMut<Assets<Mesh>>,
    req_rx: Res<ChunkSpawnReciever>,
    mut chunk_entity_map: ResMut<ChunkEntityMap>,
    mut rendered_chunks_query: Query<&mut Mesh3d, With<ChunkTag>>,
) {
    let mut count = 0;
    while let Ok(req) = req_rx.0.try_recv() {
        #[cfg(feature = "timers")]
        let t0 = Instant::now();
        count += 1;
        match req {
            ChunkSpawnResult::ToSpawn((chunk_coord, mesh)) => {
                let entity = commands
                    .spawn((
                        Mesh3d(mesh_handles.add(mesh)),
                        ChunkTag,
                        Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                        MeshMaterial3d(standard_material.0.clone()),
                    ))
                    .id();
                chunk_entity_map.0.insert(chunk_coord, entity);
            }
            ChunkSpawnResult::ToGiveCollider((chunk_coord, collider)) => {
                let entity = chunk_entity_map.0.get(&chunk_coord).unwrap();
                commands.entity(*entity).insert(collider);
            }
            ChunkSpawnResult::ToDespawn(chunk_coord) => {
                commands
                    .entity(chunk_entity_map.0.remove(&chunk_coord).unwrap())
                    .despawn();
            }
            ChunkSpawnResult::ToChangeLodAddCollider((chunk_coord, new_mesh, new_collider)) => {
                let entity = chunk_entity_map.0.get(&chunk_coord).unwrap();
                let mut mesh_handle = rendered_chunks_query.get_mut(*entity).unwrap();
                mesh_handles.remove(&mesh_handle.0);
                if let Some(aabb) = new_mesh.compute_aabb() {
                    commands.entity(*entity).insert(aabb);
                }
                *mesh_handle = Mesh3d(mesh_handles.add(new_mesh));
                commands.entity(*entity).insert(new_collider);
            }
            ChunkSpawnResult::ToChangeLod((chunk_coord, new_mesh)) => {
                let entity = chunk_entity_map.0.get(&chunk_coord).unwrap();
                let mut mesh_handle = rendered_chunks_query.get_mut(*entity).unwrap();
                mesh_handles.remove(&mesh_handle.0);
                if let Some(aabb) = new_mesh.compute_aabb() {
                    commands.entity(*entity).insert(aabb);
                }
                *mesh_handle = Mesh3d(mesh_handles.add(new_mesh));
            }
            ChunkSpawnResult::ToSpawnWithCollider((chunk_coord, collider, mesh)) => {
                let entity = commands
                    .spawn((
                        Mesh3d(mesh_handles.add(mesh)),
                        collider,
                        ChunkTag,
                        Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                        MeshMaterial3d(standard_material.0.clone()),
                    ))
                    .id();
                chunk_entity_map.0.insert(chunk_coord, entity);
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
    mut player_position_query: Query<&mut Transform, (With<PlayerTag>, Without<ChunkTag>)>,
    spawned_chunks_query: Query<(), (With<ChunkTag>, With<Collider>)>,
    terrain_chunk_map: Res<TerrainChunkMap>,
) {
    let mut player_position = player_position_query.iter_mut().next().unwrap();
    let player_chunk = world_pos_to_chunk_coord(&player_position.translation);
    for chunk_y in (player_chunk.1 - 10..=player_chunk.1).rev() {
        if let Some(entity) = chunk_entity_map
            .0
            .get(&(player_chunk.0, chunk_y, player_chunk.2))
        {
            if spawned_chunks_query.get(*entity).is_ok() {
                INITIAL_CHUNKS_LOADED.store(true, Ordering::Relaxed);
                validate_player_spawn(
                    &(player_chunk.0, chunk_y, player_chunk.2),
                    &terrain_chunk_map,
                    &mut player_position,
                );
                break;
            }
        }
    }
}

//move player up gradually until not stuck in floor
fn validate_player_spawn(
    chunk_coord: &(i16, i16, i16),
    terrain_chunk_map: &TerrainChunkMap,
    player_transform: &mut Transform,
) {
    let terrain_map = terrain_chunk_map.0.lock().unwrap();
    let chunk = terrain_map.get(chunk_coord).unwrap();
    let TerrainChunk::NonUniformTerrainChunk(non_uniform) = chunk else {
        return;
    };
    let chunk_start = calculate_chunk_start(chunk_coord);
    let local_x = SAMPLES_PER_CHUNK_DIM / 2;
    let local_z = SAMPLES_PER_CHUNK_DIM / 2;
    let voxel_size = CHUNK_WORLD_SIZE / SAMPLES_PER_CHUNK_DIM as f32;
    let original_y = player_transform.translation.y;
    let player_local_y = ((original_y - chunk_start.y) / voxel_size).floor() as usize;
    let start_y = player_local_y.min(SAMPLES_PER_CHUNK_DIM - 1);
    for y in start_y..SAMPLES_PER_CHUNK_DIM {
        let mut all_air = true;
        for dx in 0..=(PLAYER_CUBOID_SIZE.x / voxel_size).ceil() as usize {
            for dz in 0..=(PLAYER_CUBOID_SIZE.z / voxel_size).ceil() as usize {
                let check_x = (local_x + dx).min(SAMPLES_PER_CHUNK_DIM - 1);
                let check_z = (local_z + dz).min(SAMPLES_PER_CHUNK_DIM - 1);
                let idx = y * SAMPLES_PER_CHUNK_DIM * SAMPLES_PER_CHUNK_DIM
                    + check_z * SAMPLES_PER_CHUNK_DIM
                    + check_x;
                if non_uniform.densities[idx] > 0 {
                    all_air = false;
                    break;
                }
            }
            if !all_air {
                break;
            }
        }
        if all_air {
            let new_y = chunk_start.y + (y as f32 * voxel_size);
            if new_y != original_y {
                player_transform.translation.y = new_y;
                println!("Moved player up from {} to {}", original_y, new_y);
            }
            return;
        }
    }
}
