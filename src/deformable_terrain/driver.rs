use crate::constants::SAMPLES_PER_CHUNK_PADDED;
use crate::conversions::{chunk_coord_to_cluster_coord, cluster_coord_to_world_center};
use crate::deformable_terrain::chunk_entity_map::ChunkEntityMap;
use crate::deformable_terrain::chunk_generator::{
    MaterialCode, calculate_chunk_start, chunk_contains_surface, compute_heightmap_gradients,
    downscale, fast_get_uniformity, generate_chunk_into_buffers, generate_noise_height_samples,
    generate_terrain_heights, get_fbm, padded_chunk_contains_surface,
};
use crate::deformable_terrain::column_range_map::ColumnRangeMap;
#[cfg(feature = "debug")]
use crate::deformable_terrain::driver_debug_ui::{
    CHUNK_SPAWN_RECEIVER_QUEUE_SIZE, CLUSTERS_PROCESSED, INTERNAL_QUEUE_SIZES,
};
use crate::deformable_terrain::file_loader::{
    CHUNK_SERIALIZED_SIZE, get_project_root, load_chunk, load_chunk_index_map, load_uniform_chunks,
    remove_uniform_chunk, update_chunk, write_chunk, write_uniform_chunk,
};
use crate::deformable_terrain::marching_cubes::mc::mc_mesh_generation;
use crate::deformable_terrain::plugin::{ChunkTag, MoveableCenter, Uniformity};
use crate::deformable_terrain::sparse_voxel_octree::SvoNode;
use crate::deformable_terrain::terrain::{
    NonUniformTerrainChunk, TerrainChunk, TerrainMaterialHandle, generate_bevy_mesh,
};

use crate::{
    constants::{
        CHUNKS_PER_CLUSTER, CHUNKS_PER_CLUSTER_DIM, SAMPLES_PER_CHUNK, SAMPLES_PER_CHUNK_2D_PADDED,
        SAMPLES_PER_CHUNK_DIM, SIMULATION_RADIUS_SQUARED,
    },
    conversions::cluster_coord_to_min_chunk_coord,
};
use bevy::{camera::primitives::MeshAabb, prelude::*};
use bevy_rapier3d::prelude::{Collider, ComputedColliderShape, TriMeshFlags};
use crossbeam_channel::{Receiver, Sender, unbounded};
use fastnoise2::{SafeNode, generator::GeneratorWrapper};
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering::Equal;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::{
    collections::{BinaryHeap, VecDeque},
    fs::{File, OpenOptions},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::conversions::chunk_coord_to_world_pos;

pub(crate) const RF1: usize = 2;
pub(crate) const RF2: usize = 4;
pub(crate) const RF3: usize = 8;
pub(crate) const RF4: usize = 16;
pub(crate) const RF5: usize = 32;
pub(crate) const RF1_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF1;
pub(crate) const RF2_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF2;
pub(crate) const RF3_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF3;
pub(crate) const RF4_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF4;
pub(crate) const RF5_SAMPLES_PER_CHUNK_DIM: usize = SAMPLES_PER_CHUNK_DIM / RF5;
const PRIORITY_QUEUE_MAX_SIZE: usize = 10000;
const INTERNAL_WORKER_QUEUE_SIZE: usize = 64;

//I dont like this but, block player movement until first chunk load happens
pub static INITIAL_CHUNKS_LOADED: AtomicBool = AtomicBool::new(false);
pub static QUEUE_SIZE: AtomicUsize = AtomicUsize::new(0);
pub static RENDER_RADIUS_SQUARED: AtomicU32 = AtomicU32::new(0);

#[repr(u8)]
pub enum FullLodMode {
    NoCollider,
    WithCollider,
    AddColliderToExisting,
}

#[repr(u8)]
pub(crate) enum TerrainChunkMapModification {
    Insert((i16, i16, i16), TerrainChunk),
    Remove((i16, i16, i16)),
}

pub struct ChunkBuffers {
    pub density: [i16; SAMPLES_PER_CHUNK_PADDED],
    pub material: [MaterialCode; SAMPLES_PER_CHUNK],
    pub heightmap: [f32; SAMPLES_PER_CHUNK_2D_PADDED],
    pub dhdx: [f32; SAMPLES_PER_CHUNK_2D_PADDED],
    pub dhdz: [f32; SAMPLES_PER_CHUNK_2D_PADDED],
}

impl ChunkBuffers {
    pub fn new() -> Box<Self> {
        //boxed at the struct level for better cache locality
        unsafe { Box::new_zeroed().assume_init() } //unsafe to avoid stack overflow on debug builds 
    }
}

pub struct LodBuffers {
    pub density_r1: [i16; SAMPLES_PER_CHUNK / RF1.pow(3)],
    pub material_r1: [MaterialCode; SAMPLES_PER_CHUNK / RF1.pow(3)],
    pub density_r2: [i16; SAMPLES_PER_CHUNK / RF2.pow(3)],
    pub material_r2: [MaterialCode; SAMPLES_PER_CHUNK / RF2.pow(3)],
    pub density_r3: [i16; SAMPLES_PER_CHUNK / RF3.pow(3)],
    pub material_r3: [MaterialCode; SAMPLES_PER_CHUNK / RF3.pow(3)],
    pub density_r4: [i16; SAMPLES_PER_CHUNK / RF4.pow(3)],
    pub material_r4: [MaterialCode; SAMPLES_PER_CHUNK / RF4.pow(3)],
    pub density_r5: [i16; SAMPLES_PER_CHUNK / RF5.pow(3)],
    pub material_r5: [MaterialCode; SAMPLES_PER_CHUNK / RF5.pow(3)],
}

impl LodBuffers {
    pub fn new() -> Box<Self> {
        //boxed at the struct level for better cache locality
        unsafe { Box::new_zeroed().assume_init() } //unsafe to avoid stack overflow on debug builds 
    }
}

#[derive(Resource)]
pub struct FrameStart(pub Instant);

#[derive(Resource)]
pub struct LogicalProcesors(pub usize);

#[derive(Resource)]
pub struct NoiseGenerator(pub GeneratorWrapper<SafeNode>);

//stores the data for all chunks in Z0 radius on the bevy thread. Chunk loader can write to the mutex and bevy can modify it for digging operations.
#[derive(Resource)]
pub struct TerrainChunkMap(pub(crate) Arc<Mutex<FxHashMap<(i16, i16, i16), TerrainChunk>>>);

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
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
    ToChangeLodRemoveCollider(((i16, i16, i16), Mesh)), //had collider and becoming lod therefor no longer needs collider
    ToRemoveCollider((i16, i16, i16)), //was full, still full except no longer needs collider
}

pub enum WriteCmd {
    UpdateNonUniform {
        densities: Arc<[i16]>,
        materials: Arc<[MaterialCode]>,
        chunk_coord: (i16, i16, i16),
    },
    WriteUniformAir {
        chunk_coord: (i16, i16, i16),
    },
    WriteUniformDirt {
        chunk_coord: (i16, i16, i16),
    },
    RemoveUniformAir {
        chunk_coord: (i16, i16, i16),
    },
    RemoveUniformDirt {
        chunk_coord: (i16, i16, i16),
    },
}

#[derive(Resource)]
pub struct ChunkSpawnReciever(Receiver<ChunkSpawnResult>);

#[derive(Debug)]
pub struct ClusterRequest {
    pub position: (i16, i16, i16),
    pub distance_squared: f32, //distance to cluster center in world units
    pub load_state_transition: LoadStateTransition,
    pub prev_has_entity: Option<[bool; CHUNKS_PER_CLUSTER]>,
    pub prev_in_simulation_radius: bool, //if in sim radius and had entity, it also had a collider
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

impl ClusterRequest {
    fn had_entity(&self, idx: usize) -> bool {
        self.prev_has_entity.map_or(false, |a| a[idx])
    }
}

struct ChunkResult {
    has_entity: [bool; CHUNKS_PER_CLUSTER],
    cluster_coord: (i16, i16, i16),
    load_state: LoadState,
}

#[derive(Resource)]
pub struct WriteCmdSender(pub Sender<WriteCmd>);

#[derive(Resource)]
pub(crate) struct Lods(pub(crate) bool);

pub(crate) fn setup_chunk_driver(
    mut commands: Commands,
    moveable_center: Res<MoveableCenter>,
    lods: Res<Lods>,
) {
    let lods: bool = lods.0;
    commands.remove_resource::<Lods>();
    #[cfg(feature = "timers")]
    {
        std::fs::create_dir_all("plots").unwrap();
    }
    let index_map_delta = Arc::new(RwLock::new(FxHashMap::default()));
    let num_processors = thread::available_parallelism().unwrap().get();
    info!("Number of Available Processors: {}", num_processors);
    commands.insert_resource(LogicalProcesors(num_processors));
    #[cfg(feature = "debug")]
    INTERNAL_QUEUE_SIZES.get_or_init(|| {
        (0..num_processors.saturating_sub(4))
            .map(|_| AtomicUsize::new(0))
            .collect()
    });
    let moveable_center_arc = Arc::clone(&moveable_center.center_mutex);
    let (chunk_spawn_sender, chunk_spawn_reciever) = unbounded::<ChunkSpawnResult>();
    let terrain_chunk_map = Arc::new(Mutex::new(FxHashMap::default()));
    let (res_tx, res_rx) = unbounded::<ChunkResult>();
    let svo = SvoNode::world_root();
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
    info!(
        "Loaded ColumnRangeMap with {} bytes in {} ms.",
        column_range_map.size_in_bytes(),
        t0.elapsed().as_millis()
    );
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
    let (terrain_chunk_map_modification_sender, terrain_chunk_map_modification_reciever) =
        crossbeam_channel::unbounded();
    info!(
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
        let terrain_chunk_map_modification_sender_clone =
            terrain_chunk_map_modification_sender.clone();
        let _handle = thread::Builder::new()
            .name(format!("chunk_loader_{thread_idx}"))
            .spawn(move || {
                if lods {
                    lod_chunk_loader_thread(
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
                        terrain_chunk_map_modification_sender_clone,
                    );
                } else {
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
                        terrain_chunk_map_modification_sender_clone,
                    );
                }
            })
            .expect("failed to spawn chunk loader thread");
    }
    let terrain_chunk_map_arc = Arc::clone(&terrain_chunk_map);
    thread::spawn(move || {
        svo_manager_thread(
            res_rx,
            moveable_center_arc,
            chunk_spawn_sender,
            svo,
            priority_queue,
            terrain_chunk_map_arc,
            terrain_chunk_map_modification_reciever,
            terrain_chunk_map_modification_sender,
            lods,
        );
    });
    commands.insert_resource(WriteCmdSender(write_tx));
    commands.insert_resource(TerrainChunkMap(terrain_chunk_map));
}

//assume duplicate writes are impossible otherwise something went wrong
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
) {
    let mut chunk_write_reuse = Vec::with_capacity(14); //sizeof (i16, i16, i16, u64)
    let mut serial_buffer = [0; CHUNK_SERIALIZED_SIZE];
    while let Ok(cmd) = rx.recv() {
        match cmd {
            WriteCmd::UpdateNonUniform {
                densities,
                materials,
                chunk_coord,
            } => {
                //offset lookup must be async to avoid situation where we try to update a chunk that isnt written
                //because the channel is ordered, the write should always process before the update
                let offset = chunk_index_map_read
                    .get(&chunk_coord)
                    .cloned()
                    .or_else(|| index_map_delta.read().get(&chunk_coord).cloned());
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
                            &chunk_coord,
                            &mut index_map,
                            &mut chunk_data_file,
                            &mut chunk_index_file,
                            &mut chunk_write_reuse,
                            &mut serial_buffer,
                        );
                    }
                }
            }
            WriteCmd::WriteUniformAir { chunk_coord } => {
                write_uniform_chunk(&chunk_coord, &mut air_file, &mut air_empty_offsets);
            }
            WriteCmd::WriteUniformDirt { chunk_coord } => {
                write_uniform_chunk(&chunk_coord, &mut dirt_file, &mut dirt_empty_offsets);
            }
            WriteCmd::RemoveUniformAir { chunk_coord } => {
                remove_uniform_chunk(&chunk_coord, &mut air_file, &mut air_empty_offsets);
            }
            WriteCmd::RemoveUniformDirt { chunk_coord } => {
                remove_uniform_chunk(&chunk_coord, &mut dirt_file, &mut dirt_empty_offsets);
            }
        }
    }
}

//compute thread for loading or generating chunks
//recieves chunk load requests from svo_manager_thread and returns the data
//uses a fast uniformity check to skip most of the chunk calculation on uniform chunks
fn lod_chunk_loader_thread(
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
    terrain_chunk_map_modification_sender: Sender<TerrainChunkMapModification>,
) {
    let mut lod_buffers = LodBuffers::new();
    let mut chunk_buffers = ChunkBuffers::new();
    let mut internal_queue = Vec::with_capacity(INTERNAL_WORKER_QUEUE_SIZE);
    loop {
        let (binary_heap_lock, condvar) = &*priority_queue;
        let mut binary_heap = binary_heap_lock.lock().unwrap();
        while binary_heap.is_empty() {
            binary_heap = condvar.wait(binary_heap).unwrap();
        }
        let num_to_pop = binary_heap.len().min(INTERNAL_WORKER_QUEUE_SIZE);
        for _ in 0..num_to_pop {
            internal_queue.push(binary_heap.pop().unwrap());
        }
        QUEUE_SIZE.store(binary_heap.len(), Ordering::Relaxed);
        drop(binary_heap);
        #[cfg(feature = "debug")]
        INTERNAL_QUEUE_SIZES.get().unwrap()[thread_idx]
            .store(internal_queue.len(), Ordering::Relaxed);
        for cluster_request in internal_queue.drain(..) {
            #[cfg(feature = "debug")]
            INTERNAL_QUEUE_SIZES.get().unwrap()[thread_idx].fetch_sub(1, Ordering::Relaxed);
            let mut has_entity_buffer = [false; CHUNKS_PER_CLUSTER];
            let mut rolling = 0;
            let in_simulation_range = matches!(
                cluster_request.load_state_transition,
                LoadStateTransition::ToFullWithCollider | LoadStateTransition::NoChangeAddCollider
            );
            let min_chunk = cluster_coord_to_min_chunk_coord(cluster_request.position);
            for chunk_x in min_chunk.0..min_chunk.0 + CHUNKS_PER_CLUSTER_DIM as i16 {
                for chunk_z in min_chunk.2..min_chunk.2 + CHUNKS_PER_CLUSTER_DIM as i16 {
                    let mut has_heightmap_been_calculated = false;
                    let column_cache = column_range_map_read_only.get_column(chunk_x, chunk_z);
                    for chunk_y in min_chunk.1..min_chunk.1 + CHUNKS_PER_CLUSTER_DIM as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        let mut uniformity = column_cache.uniformity_at_y(chunk_y);
                        if uniformity == Uniformity::Air {
                            //cache hit
                            if in_simulation_range {
                                let _ = terrain_chunk_map_modification_sender.send(
                                    TerrainChunkMapModification::Insert(
                                        chunk_coord,
                                        TerrainChunk::UniformAir,
                                    ),
                                );
                            }
                            rolling += 1;
                            continue;
                        } else if uniformity == Uniformity::Dirt {
                            //cache hit
                            if in_simulation_range {
                                let _ = terrain_chunk_map_modification_sender.send(
                                    TerrainChunkMapModification::Insert(
                                        chunk_coord,
                                        TerrainChunk::UniformDirt,
                                    ),
                                );
                            }
                            rolling += 1;
                            continue;
                        }
                        let mut loaded_from_disk = false;
                        if uniformity == Uniformity::Unknown {
                            uniformity = try_load_chunk(
                                chunk_coord,
                                &index_map_read,
                                &index_map_delta,
                                &mut chunk_data_file_read,
                                &mut chunk_buffers,
                            );
                            if uniformity == Uniformity::NonUniform {
                                loaded_from_disk = true;
                            }
                        }
                        let chunk_start = calculate_chunk_start(&chunk_coord);
                        if uniformity == Uniformity::Unknown {
                            if !has_heightmap_been_calculated {
                                let noise_samples = generate_noise_height_samples(
                                    chunk_start.x,
                                    chunk_start.z,
                                    &fbm,
                                );
                                generate_terrain_heights(
                                    &mut chunk_buffers.heightmap,
                                    &noise_samples,
                                );
                                compute_heightmap_gradients(
                                    &mut chunk_buffers.dhdx,
                                    &mut chunk_buffers.dhdz,
                                    &noise_samples,
                                );
                                has_heightmap_been_calculated = true;
                            }
                            uniformity = fast_get_uniformity(
                                &chunk_buffers.heightmap,
                                &chunk_buffers.dhdx,
                                &chunk_buffers.dhdz,
                                &chunk_start,
                            );
                        }
                        match uniformity {
                            Uniformity::Air => {
                                let _ =
                                    write_sender.send(WriteCmd::WriteUniformAir { chunk_coord });
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::UniformAir,
                                        ),
                                    );
                                }
                            }
                            Uniformity::Dirt => {
                                let _ =
                                    write_sender.send(WriteCmd::WriteUniformDirt { chunk_coord });
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::UniformDirt,
                                        ),
                                    );
                                }
                            }
                            Uniformity::NonUniform => {
                                if !loaded_from_disk {
                                    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
                                }
                                let has_surface = lod_resolve_has_surface(
                                    &cluster_request,
                                    &chunk_buffers,
                                    &mut lod_buffers,
                                    chunk_coord,
                                    rolling,
                                    &chunk_spawn_channel,
                                );
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::NonUniformTerrainChunk(
                                                NonUniformTerrainChunk {
                                                    //allocation here
                                                    densities: Arc::from(
                                                        &chunk_buffers.density[..],
                                                    ),
                                                    materials: Arc::from(
                                                        &chunk_buffers.material[..],
                                                    ),
                                                },
                                            ),
                                        ),
                                    );
                                }
                                has_entity_buffer[rolling] = has_surface;
                            }
                            Uniformity::Unknown => unreachable!(),
                        };
                        rolling += 1;
                    }
                }
            }
            let new_state = cluster_request.load_state_transition.to_state();
            let _ = res_tx.send(ChunkResult {
                has_entity: has_entity_buffer,
                cluster_coord: cluster_request.position,
                load_state: new_state,
            });
            #[cfg(feature = "debug")]
            CLUSTERS_PROCESSED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

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
    terrain_chunk_map_modification_sender: Sender<TerrainChunkMapModification>,
) {
    let mut chunk_buffers = ChunkBuffers::new();
    let mut internal_queue = Vec::with_capacity(INTERNAL_WORKER_QUEUE_SIZE);
    loop {
        let (binary_heap_lock, condvar) = &*priority_queue;
        let mut binary_heap = binary_heap_lock.lock().unwrap();
        while binary_heap.is_empty() {
            binary_heap = condvar.wait(binary_heap).unwrap();
        }
        let num_to_pop = binary_heap.len().min(INTERNAL_WORKER_QUEUE_SIZE);
        for _ in 0..num_to_pop {
            internal_queue.push(binary_heap.pop().unwrap());
        }
        QUEUE_SIZE.store(binary_heap.len(), Ordering::Relaxed);
        drop(binary_heap);
        #[cfg(feature = "debug")]
        INTERNAL_QUEUE_SIZES.get().unwrap()[thread_idx]
            .store(internal_queue.len(), Ordering::Relaxed);
        for cluster_request in internal_queue.drain(..) {
            #[cfg(feature = "debug")]
            INTERNAL_QUEUE_SIZES.get().unwrap()[thread_idx].fetch_sub(1, Ordering::Relaxed);
            let mut has_entity_buffer = [false; CHUNKS_PER_CLUSTER];
            let mut rolling = 0;
            let in_simulation_range = matches!(
                cluster_request.load_state_transition,
                LoadStateTransition::ToFullWithCollider | LoadStateTransition::NoChangeAddCollider
            );
            let min_chunk = cluster_coord_to_min_chunk_coord(cluster_request.position);
            for chunk_x in min_chunk.0..min_chunk.0 + CHUNKS_PER_CLUSTER_DIM as i16 {
                for chunk_z in min_chunk.2..min_chunk.2 + CHUNKS_PER_CLUSTER_DIM as i16 {
                    let mut has_heightmap_been_calculated = false;
                    let column_cache = column_range_map_read_only.get_column(chunk_x, chunk_z);
                    for chunk_y in min_chunk.1..min_chunk.1 + CHUNKS_PER_CLUSTER_DIM as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        let mut uniformity = column_cache.uniformity_at_y(chunk_y);
                        if uniformity == Uniformity::Air {
                            //cache hit
                            if in_simulation_range {
                                let _ = terrain_chunk_map_modification_sender.send(
                                    TerrainChunkMapModification::Insert(
                                        chunk_coord,
                                        TerrainChunk::UniformAir,
                                    ),
                                );
                            }
                            rolling += 1;
                            continue;
                        } else if uniformity == Uniformity::Dirt {
                            //cache hit
                            if in_simulation_range {
                                let _ = terrain_chunk_map_modification_sender.send(
                                    TerrainChunkMapModification::Insert(
                                        chunk_coord,
                                        TerrainChunk::UniformDirt,
                                    ),
                                );
                            }
                            rolling += 1;
                            continue;
                        }
                        let mut loaded_from_disk = false;
                        if uniformity == Uniformity::Unknown {
                            uniformity = try_load_chunk(
                                chunk_coord,
                                &index_map_read,
                                &index_map_delta,
                                &mut chunk_data_file_read,
                                &mut chunk_buffers,
                            );
                            if uniformity == Uniformity::NonUniform {
                                loaded_from_disk = true;
                            }
                        }
                        let chunk_start = calculate_chunk_start(&chunk_coord);
                        if uniformity == Uniformity::Unknown {
                            if !has_heightmap_been_calculated {
                                let noise_samples = generate_noise_height_samples(
                                    chunk_start.x,
                                    chunk_start.z,
                                    &fbm,
                                );
                                generate_terrain_heights(
                                    &mut chunk_buffers.heightmap,
                                    &noise_samples,
                                );
                                compute_heightmap_gradients(
                                    &mut chunk_buffers.dhdx,
                                    &mut chunk_buffers.dhdz,
                                    &noise_samples,
                                );
                                has_heightmap_been_calculated = true;
                            }
                            uniformity = fast_get_uniformity(
                                &chunk_buffers.heightmap,
                                &chunk_buffers.dhdx,
                                &chunk_buffers.dhdz,
                                &chunk_start,
                            );
                        }
                        match uniformity {
                            Uniformity::Air => {
                                let _ =
                                    write_sender.send(WriteCmd::WriteUniformAir { chunk_coord });
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::UniformAir,
                                        ),
                                    );
                                }
                            }
                            Uniformity::Dirt => {
                                let _ =
                                    write_sender.send(WriteCmd::WriteUniformDirt { chunk_coord });
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::UniformDirt,
                                        ),
                                    );
                                }
                            }
                            Uniformity::NonUniform => {
                                if !loaded_from_disk {
                                    generate_chunk_into_buffers(chunk_start, &mut chunk_buffers);
                                }
                                let has_surface = resolve_has_surface(
                                    &cluster_request,
                                    &chunk_buffers,
                                    chunk_coord,
                                    rolling,
                                    &chunk_spawn_channel,
                                );
                                if in_simulation_range {
                                    let _ = terrain_chunk_map_modification_sender.send(
                                        TerrainChunkMapModification::Insert(
                                            chunk_coord,
                                            TerrainChunk::NonUniformTerrainChunk(
                                                NonUniformTerrainChunk {
                                                    //allocation here
                                                    densities: Arc::from(
                                                        &chunk_buffers.density[..],
                                                    ),
                                                    materials: Arc::from(
                                                        &chunk_buffers.material[..],
                                                    ),
                                                },
                                            ),
                                        ),
                                    );
                                }
                                has_entity_buffer[rolling] = has_surface;
                            }
                            Uniformity::Unknown => unreachable!(),
                        };
                        rolling += 1;
                    }
                }
            }
            let new_state = cluster_request.load_state_transition.to_state();
            let _ = res_tx.send(ChunkResult {
                has_entity: has_entity_buffer,
                cluster_coord: cluster_request.position,
                load_state: new_state,
            });
            #[cfg(feature = "debug")]
            CLUSTERS_PROCESSED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

//owns the main svo
//recieves and handles modification requests
//produces chunk load requests for chunk_loader_thread and recieves the data
//sends chunks to be spawned to main thread
fn svo_manager_thread(
    results_channel: Receiver<ChunkResult>,
    moveable_center: Arc<Mutex<Vec3>>,
    chunk_spawn_channel: Sender<ChunkSpawnResult>,
    mut svo: SvoNode,
    priority_queue: Arc<(Mutex<BinaryHeap<ClusterRequest>>, Condvar)>,
    terrain_chunk_map: Arc<Mutex<FxHashMap<(i16, i16, i16), TerrainChunk>>>,
    terrain_chunk_map_modification_reciever: Receiver<TerrainChunkMapModification>,
    terrain_chunk_map_modification_sender: Sender<TerrainChunkMapModification>,
    lods: bool,
) {
    #[cfg(feature = "timers")]
    let t0 = Instant::now();
    #[cfg(feature = "timers")]
    let mut first_completion_printed = false;
    let mut request_buffer = Vec::new();
    let mut chunks_being_loaded = FxHashSet::default();
    let moveable_center_lock = moveable_center.lock().unwrap();
    let initial_moveable_center = *moveable_center_lock;
    drop(moveable_center_lock);
    if lods {
        svo.lod_fill_missing_chunks_in_radius(
            &initial_moveable_center,
            SIMULATION_RADIUS_SQUARED,
            &chunks_being_loaded,
            &mut request_buffer,
        );
    } else {
        svo.fill_missing_chunks_in_radius(
            &initial_moveable_center,
            SIMULATION_RADIUS_SQUARED,
            &chunks_being_loaded,
            &mut request_buffer,
        );
    }
    request_buffer.sort_unstable_by(|a, b| {
        a.distance_squared
            .partial_cmp(&b.distance_squared)
            .unwrap_or(Equal)
    });
    request_buffer.truncate(10000);
    for request in &request_buffer {
        chunks_being_loaded.insert(request.position);
    }
    let (binary_heap_lock, condvar) = &*priority_queue;
    {
        let mut binary_heap = binary_heap_lock.lock().unwrap();
        for request in request_buffer.drain(..) {
            binary_heap.push(request);
        }
        QUEUE_SIZE.store(binary_heap.len(), Ordering::Relaxed);
    }
    condvar.notify_all();
    let mut clusters_to_deallocate = Vec::new();
    loop {
        let moveable_center_lock = moveable_center.lock().unwrap();
        let moveable_center = *moveable_center_lock;
        drop(moveable_center_lock);
        let mut terrain_map_lock = terrain_chunk_map.lock().unwrap();
        while let Ok(modification) = terrain_chunk_map_modification_reciever.try_recv() {
            match modification {
                TerrainChunkMapModification::Insert(chunk_coord, terrain_chunk) => {
                    terrain_map_lock.insert(chunk_coord, terrain_chunk);
                }
                TerrainChunkMapModification::Remove(chunk_coord) => {
                    terrain_map_lock.remove(&chunk_coord);
                }
            }
        }
        for chunk_coord in terrain_map_lock.keys() {
            let lower_cluster_coord = chunk_coord_to_cluster_coord(chunk_coord);
            let distance_squared = moveable_center
                .distance_squared(cluster_coord_to_world_center(&lower_cluster_coord));
            if distance_squared > SIMULATION_RADIUS_SQUARED {
                let _ = terrain_chunk_map_modification_sender
                    .send(TerrainChunkMapModification::Remove(*chunk_coord));
            }
        }
        drop(terrain_map_lock);
        while let Ok(result) = results_channel.try_recv() {
            svo.insert(result.cluster_coord, result.has_entity, result.load_state);
            chunks_being_loaded.remove(&result.cluster_coord);
        }
        svo.query_chunks_outside_sphere(&moveable_center, &mut clusters_to_deallocate);
        for (chunk_coord, _) in &clusters_to_deallocate {
            svo.delete(*chunk_coord);
        }
        let mut roller = 0;
        let mut terrain_map_lock = terrain_chunk_map.lock().unwrap();
        for (cluster_coord, has_entity) in clusters_to_deallocate.drain(..) {
            let min_chunk = cluster_coord_to_min_chunk_coord(cluster_coord);
            for chunk_x in min_chunk.0..min_chunk.0 + CHUNKS_PER_CLUSTER_DIM as i16 {
                for chunk_z in min_chunk.2..min_chunk.2 + CHUNKS_PER_CLUSTER_DIM as i16 {
                    for chunk_y in min_chunk.1..min_chunk.1 + CHUNKS_PER_CLUSTER_DIM as i16 {
                        let chunk_coord = (chunk_x, chunk_y, chunk_z);
                        if has_entity[roller] {
                            let _ =
                                chunk_spawn_channel.send(ChunkSpawnResult::ToDespawn(chunk_coord));
                        }
                        terrain_map_lock.remove(&chunk_coord);
                        roller += 1;
                    }
                }
            }
            chunks_being_loaded.remove(&cluster_coord);
            roller = 0;
        }
        drop(terrain_map_lock);
        if QUEUE_SIZE.load(Ordering::Relaxed) < PRIORITY_QUEUE_MAX_SIZE {
            if lods {
                svo.lod_fill_missing_chunks_in_radius(
                    &moveable_center,
                    f32::from_bits(RENDER_RADIUS_SQUARED.load(Ordering::Relaxed)),
                    &chunks_being_loaded,
                    &mut request_buffer,
                );
            } else {
                svo.fill_missing_chunks_in_radius(
                    &moveable_center,
                    f32::from_bits(RENDER_RADIUS_SQUARED.load(Ordering::Relaxed)),
                    &chunks_being_loaded,
                    &mut request_buffer,
                );
            }
            request_buffer.sort_unstable_by(|a, b| {
                a.distance_squared
                    .partial_cmp(&b.distance_squared)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let cap = 10000usize.saturating_sub(QUEUE_SIZE.load(Ordering::Relaxed));
            request_buffer.truncate(cap);
            for request in &request_buffer {
                chunks_being_loaded.insert(request.position);
            }
            let (binary_heap_lock, condvar) = &*priority_queue;
            {
                let mut binary_heap = binary_heap_lock.lock().unwrap();
                for request in request_buffer.drain(..) {
                    binary_heap.push(request);
                }
                QUEUE_SIZE.store(binary_heap.len(), Ordering::Relaxed);
            }
            condvar.notify_all();
        }
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
    frame_start: Res<FrameStart>,
) {
    const TARGET_FRAME_TIME: Duration = Duration::from_nanos(1_000_000_000 / 90);
    while let Ok(request) = req_rx.0.try_recv() {
        match request {
            ChunkSpawnResult::ToSpawn((chunk_coord, mesh)) => {
                //use option in case a chunk is spawned, despawned, and spawned again but the second spawn comes before the despawn
                if chunk_entity_map.get_option(chunk_coord).is_none() {
                    let mesh_handle = mesh_handles.add(mesh);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle.clone()),
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id();
                    chunk_entity_map.insert(chunk_coord, (entity, mesh_handle));
                }
            }
            ChunkSpawnResult::ToGiveCollider((chunk_coord, collider)) => {
                let (entity, _) = chunk_entity_map.get(chunk_coord);
                commands.entity(entity).insert(collider);
            }
            ChunkSpawnResult::ToRemoveCollider(chunk_coord) => {
                let (entity, _) = chunk_entity_map.get(chunk_coord);
                commands.entity(entity).remove::<Collider>();
            }
            ChunkSpawnResult::ToDespawn(chunk_coord) => {
                //use option in case the corresponding ToSpawn was skipped due to a duplicate, leaving nothing to remove
                if let Some((entity, mesh_handle)) = chunk_entity_map.get_option(chunk_coord) {
                    let entity = *entity;
                    let mesh_handle = mesh_handle.clone();
                    chunk_entity_map.remove(chunk_coord);
                    mesh_handles.remove(&mesh_handle);
                    commands.entity(entity).despawn();
                }
            }
            ChunkSpawnResult::ToChangeLodAddCollider((chunk_coord, new_mesh, new_collider)) => {
                //use option to handle the case where the chunk was despawned while the LOD change was in flight
                if let Some((entity, mesh_handle)) = chunk_entity_map.get_option(chunk_coord) {
                    if let Some(aabb) = new_mesh.compute_aabb() {
                        commands.entity(*entity).insert(aabb);
                    }
                    mesh_handles.insert(mesh_handle, new_mesh).unwrap();
                    commands.entity(*entity).insert(new_collider);
                }
            }
            ChunkSpawnResult::ToChangeLod((chunk_coord, new_mesh)) => {
                //use option to handle the case where the chunk was despawned while the LOD change was in flight
                if let Some((entity, mesh_handle)) = chunk_entity_map.get_option(chunk_coord) {
                    if let Some(aabb) = new_mesh.compute_aabb() {
                        commands.entity(*entity).insert(aabb);
                    }
                    mesh_handles.insert(mesh_handle, new_mesh).unwrap();
                }
            }
            ChunkSpawnResult::ToChangeLodRemoveCollider((chunk_coord, new_mesh)) => {
                let (entity, mesh_handle) = chunk_entity_map.get(chunk_coord);
                if let Some(aabb) = new_mesh.compute_aabb() {
                    commands.entity(entity).insert(aabb);
                }
                mesh_handles.insert(&mesh_handle, new_mesh).unwrap();
                commands.entity(entity).remove::<Collider>();
            }
            ChunkSpawnResult::ToSpawnWithCollider((chunk_coord, collider, mesh)) => {
                //use option in case a chunk is spawned, despawned, and spawned again but the second spawn comes before the despawn
                if chunk_entity_map.get_option(chunk_coord).is_none() {
                    let mesh_handle = mesh_handles.add(mesh);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle.clone()),
                            collider,
                            ChunkTag,
                            Transform::from_translation(chunk_coord_to_world_pos(&chunk_coord)),
                            MeshMaterial3d(standard_material.0.clone()),
                        ))
                        .id();
                    chunk_entity_map.insert(chunk_coord, (entity, mesh_handle));
                }
            }
        }
        if frame_start.0.elapsed() >= TARGET_FRAME_TIME {
            return; //if this fn would cause fps to drop below a certain threshold, wait until next frame to continue processing requests
        }
    }
    #[cfg(feature = "debug")]
    CHUNK_SPAWN_RECEIVER_QUEUE_SIZE.store(req_rx.0.len(), Ordering::Relaxed);
}

//downscales to new resolution from full resolution
//searches downscaled densities for surface
//if has surface runs marching cubes, generates mesh, and sends either a spawn command or a change lod command based on if it was previously loaded or not
//if going from full within simulation radius to an lod, send remove from chunk map command
//returns if the reduced chunk contains a surface
fn process_lod(
    density_buffer: &[i16],
    material_buffer: &[MaterialCode],
    chunk_spawn_channel: &Sender<ChunkSpawnResult>,
    chunk_coord: (i16, i16, i16),
    reduced_density_buffer: &mut [i16],
    reduced_material_buffer: &mut [MaterialCode],
    out_samples_per_chunk_dim: usize,
    had_entity: bool,
    prev_in_simulation_radius: bool,
) -> bool {
    downscale(
        density_buffer,
        material_buffer,
        reduced_density_buffer,
        reduced_material_buffer,
        out_samples_per_chunk_dim,
    );
    //must recheck surface incase the reduction eliminated the surface. Additionally filters out the false positive state from calling chunk_contains_surface on a padded buffer preventing empty geometry.
    if !chunk_contains_surface(reduced_density_buffer) {
        if had_entity {
            let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToDespawn(chunk_coord));
        }
        return false;
    }
    let (vertices, normals, material_ids, indices) = mc_mesh_generation(
        reduced_density_buffer,
        reduced_material_buffer,
        out_samples_per_chunk_dim,
        false,
        &density_buffer,
    );
    let mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
    if had_entity {
        if prev_in_simulation_radius {
            let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToChangeLodRemoveCollider((
                chunk_coord,
                mesh,
            )));
        } else {
            let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToChangeLod((chunk_coord, mesh)));
        }
    } else {
        let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToSpawn((chunk_coord, mesh)));
    }
    true
}

//try to get file offset from index map or index map delta (delta requiring read lock)
//if offset found, load chunk from file and return uniformity
pub fn try_load_chunk(
    chunk_coord: (i16, i16, i16),
    index_map_read: &FxHashMap<(i16, i16, i16), u64>,
    index_map_delta: &RwLock<FxHashMap<(i16, i16, i16), u64>>,
    chunk_data_file_read: &mut File,
    chunk_buffers: &mut ChunkBuffers,
) -> Uniformity {
    let file_offset = index_map_read
        .get(&chunk_coord)
        .copied()
        .or_else(|| index_map_delta.read().get(&chunk_coord).copied());
    if let Some(offset) = file_offset {
        load_chunk(
            chunk_data_file_read,
            offset,
            &mut chunk_buffers.density,
            &mut chunk_buffers.material,
        );
        return Uniformity::NonUniform;
    }
    Uniformity::Unknown
}

//run fast surface check for early exit
//else process lod or process full both eventually double checking that it has a surface before submitting spawn chunk command
//potentially builds mesh and submits spawn chunk command
pub fn lod_resolve_has_surface(
    cluster_request: &ClusterRequest,
    chunk_buffers: &ChunkBuffers,
    lod_buffers: &mut LodBuffers,
    chunk_coord: (i16, i16, i16),
    rolling: usize,
    chunk_spawn_channel: &Sender<ChunkSpawnResult>,
) -> bool {
    if cluster_request.prev_in_simulation_radius
        && cluster_request.load_state_transition == LoadStateTransition::ToFull
    {
        let had_entity = cluster_request.had_entity(rolling);
        if had_entity {
            let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToRemoveCollider(chunk_coord));
        }
        return had_entity;
    }
    if !chunk_contains_surface(&chunk_buffers.density) {
        return false; //surface scan is faster than downscaling and early exit will apply to the majority of chunks. Produces false positives handled later. 
    }
    //must be non-uniform (including false positive state caused by padding)
    match cluster_request.load_state_transition {
        LoadStateTransition::ToLod5 => {
            let had_entity = cluster_request.had_entity(rolling);
            process_lod(
                &chunk_buffers.density,
                &chunk_buffers.material,
                &chunk_spawn_channel,
                chunk_coord,
                &mut lod_buffers.density_r5,
                &mut lod_buffers.material_r5,
                RF5_SAMPLES_PER_CHUNK_DIM,
                had_entity,
                cluster_request.prev_in_simulation_radius,
            )
        }
        LoadStateTransition::ToLod4 => {
            let had_entity = cluster_request.had_entity(rolling);
            process_lod(
                &chunk_buffers.density,
                &chunk_buffers.material,
                &chunk_spawn_channel,
                chunk_coord,
                &mut lod_buffers.density_r4,
                &mut lod_buffers.material_r4,
                RF4_SAMPLES_PER_CHUNK_DIM,
                had_entity,
                cluster_request.prev_in_simulation_radius,
            )
        }
        LoadStateTransition::ToLod3 => {
            let had_entity = cluster_request.had_entity(rolling);
            process_lod(
                &chunk_buffers.density,
                &chunk_buffers.material,
                &chunk_spawn_channel,
                chunk_coord,
                &mut lod_buffers.density_r3,
                &mut lod_buffers.material_r3,
                RF3_SAMPLES_PER_CHUNK_DIM,
                had_entity,
                cluster_request.prev_in_simulation_radius,
            )
        }
        LoadStateTransition::ToLod2 => {
            let had_entity = cluster_request.had_entity(rolling);
            process_lod(
                &chunk_buffers.density,
                &chunk_buffers.material,
                &chunk_spawn_channel,
                chunk_coord,
                &mut lod_buffers.density_r2,
                &mut lod_buffers.material_r2,
                RF2_SAMPLES_PER_CHUNK_DIM,
                had_entity,
                cluster_request.prev_in_simulation_radius,
            )
        }
        LoadStateTransition::ToLod1 => {
            let had_entity = cluster_request.had_entity(rolling);
            process_lod(
                &chunk_buffers.density,
                &chunk_buffers.material,
                &chunk_spawn_channel,
                chunk_coord,
                &mut lod_buffers.density_r1,
                &mut lod_buffers.material_r1,
                RF1_SAMPLES_PER_CHUNK_DIM,
                had_entity,
                cluster_request.prev_in_simulation_radius,
            )
        }
        LoadStateTransition::ToFull => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::NoCollider,
        ),
        LoadStateTransition::ToFullWithCollider => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::WithCollider,
        ),
        LoadStateTransition::NoChangeAddCollider => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::AddColliderToExisting,
        ),
    }
}

pub fn resolve_has_surface(
    cluster_request: &ClusterRequest,
    chunk_buffers: &ChunkBuffers,
    chunk_coord: (i16, i16, i16),
    rolling: usize,
    chunk_spawn_channel: &Sender<ChunkSpawnResult>,
) -> bool {
    if cluster_request.prev_in_simulation_radius
        && cluster_request.load_state_transition == LoadStateTransition::ToFull
    {
        let had_entity = cluster_request.had_entity(rolling);
        if had_entity {
            let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToRemoveCollider(chunk_coord));
        }
        return had_entity;
    }
    if !chunk_contains_surface(&chunk_buffers.density) {
        return false; //surface scan is faster than downscaling and early exit will apply to the majority of chunks. Produces false positives handled later. 
    }
    //must be non-uniform (including false positive state caused by padding)
    match cluster_request.load_state_transition {
        LoadStateTransition::ToFull => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::NoCollider,
        ),
        LoadStateTransition::ToFullWithCollider => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::WithCollider,
        ),
        LoadStateTransition::NoChangeAddCollider => build_full_mesh_and_spawn(
            &chunk_buffers.density,
            &chunk_buffers.material,
            chunk_coord,
            cluster_request,
            rolling,
            chunk_spawn_channel,
            FullLodMode::AddColliderToExisting,
        ),
        _ => unreachable!(),
    }
}

//searches for if the chunk contains a surface
//if it has a surface:
//run marching cubes
//generate a mesh
//check if the chunk previously had an entity
//match on the needed collider mode
//send proper chunk spawn command with or without collider
//return if the chunk contains a surface
pub fn build_full_mesh_and_spawn(
    density_buffer: &[i16],
    material_buffer: &[MaterialCode],
    chunk_coord: (i16, i16, i16),
    cluster_request: &ClusterRequest,
    rolling: usize,
    chunk_spawn_channel: &Sender<ChunkSpawnResult>,
    mode: FullLodMode,
) -> bool {
    //slower surface check to eliminate false possitive state to prevent empty geometry.
    padded_chunk_contains_surface(density_buffer) && {
        let (vertices, normals, material_ids, indices) = mc_mesh_generation(
            density_buffer,
            material_buffer,
            SAMPLES_PER_CHUNK_DIM,
            true,
            density_buffer,
        );
        #[cfg(feature = "debug")]
        assert!(
            !vertices.is_empty(),
            "padded_chunk_contains_surface returned true but MC produced no geometry for {:?}",
            chunk_coord
        );
        #[cfg(feature = "debug")]
        assert!(
            !indices.is_empty(),
            "MC produced vertices but empty indices for {:?}",
            chunk_coord
        );
        let mesh = generate_bevy_mesh(vertices, normals, material_ids, indices);
        let had_entity = cluster_request.had_entity(rolling);
        match mode {
            FullLodMode::NoCollider => {
                if had_entity {
                    let _ = chunk_spawn_channel
                        .send(ChunkSpawnResult::ToChangeLod((chunk_coord, mesh)));
                } else {
                    let _ =
                        chunk_spawn_channel.send(ChunkSpawnResult::ToSpawn((chunk_coord, mesh)));
                }
            }
            FullLodMode::WithCollider => {
                let collider = Collider::from_bevy_mesh(
                    &mesh,
                    &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                )
                .unwrap();
                if had_entity {
                    let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToChangeLodAddCollider((
                        chunk_coord,
                        mesh,
                        collider,
                    )));
                } else {
                    let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToSpawnWithCollider((
                        chunk_coord,
                        collider,
                        mesh,
                    )));
                }
            }
            FullLodMode::AddColliderToExisting => {
                let collider = Collider::from_bevy_mesh(
                    &mesh,
                    &ComputedColliderShape::TriMesh(TriMeshFlags::default()),
                )
                .unwrap();
                if had_entity {
                    let _ = chunk_spawn_channel
                        .send(ChunkSpawnResult::ToGiveCollider((chunk_coord, collider)));
                } else {
                    let _ = chunk_spawn_channel.send(ChunkSpawnResult::ToSpawnWithCollider((
                        chunk_coord,
                        collider,
                        mesh,
                    )));
                }
            }
        }
        true
    }
}

pub fn record_frame_start(mut frame_start: ResMut<FrameStart>) {
    //record frame start time so a thread can yield if its taking too long
    frame_start.0 = Instant::now();
}

pub(crate) fn info_print() {
    info!("fma: {}", std::is_x86_feature_detected!("fma"));
    info!("avx2: {}", std::is_x86_feature_detected!("avx2"));
    info!("sse2: {}", std::is_x86_feature_detected!("sse2"));
}
