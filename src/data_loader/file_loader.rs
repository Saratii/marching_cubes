use crate::terrain::terrain::{TerrainChunk, VOXELS_PER_CHUNK, VoxelData};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use heed::types::SerdeBincode;
use heed::{Database, Env, EnvOpenOptions};
use std::fs::create_dir_all;
use std::sync::{Arc, Mutex};

const BYTES_PER_VOXEL: usize = std::mem::size_of::<f32>() + std::mem::size_of::<u8>();
const CHUNK_SERIALIZED_SIZE: usize = VOXELS_PER_CHUNK * BYTES_PER_VOXEL;

#[derive(Resource)]
pub struct LoadedChunkKeys(pub Arc<Mutex<HashSet<(i16, i16, i16)>>>);

#[derive(Resource)]
pub struct DataBaseHandle(pub Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>>);

#[derive(Resource)]
pub struct DataBaseEnvHandle(pub Env);

// Binary format layout:
// - Number of voxels: u32 (4 bytes)
// - SDF values: num_voxels * f32 (4 bytes each)
// - Material values: num_voxels * u8 (1 byte each)

pub fn serialize_chunk_data(chunk: &TerrainChunk) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(CHUNK_SERIALIZED_SIZE);
    for voxel in chunk.sdfs.iter() {
        buffer.extend_from_slice(&voxel.sdf.to_le_bytes());
    }
    for voxel in chunk.sdfs.iter() {
        buffer.push(voxel.material);
    }
    buffer
}

pub fn deserialize_chunk_data(data: &[u8]) -> TerrainChunk {
    let mut sdfs = Vec::with_capacity(VOXELS_PER_CHUNK);
    let (sdf_bytes, material_bytes) = data.split_at(VOXELS_PER_CHUNK * 4);
    for (sdf_bytes, &material) in sdf_bytes.chunks_exact(4).zip(material_bytes) {
        let sdf = f32::from_le_bytes(sdf_bytes.try_into().unwrap());
        sdfs.push(VoxelData { sdf, material });
    }
    TerrainChunk {
        sdfs: sdfs.into_boxed_slice(),
    }
}

pub fn setup_chunk_loading(mut commands: Commands) {
    create_dir_all("data/chunk_db").unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024) // 10 GB max size
            .max_dbs(1)
            .open("data/chunk_db")
            .unwrap()
    };
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<SerdeBincode<(i16, i16, i16)>, SerdeBincode<Vec<u8>>> =
        env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();
    let keys = {
        let rtxn = env.read_txn().unwrap();
        let keys: HashSet<(i16, i16, i16)> = db
            .iter(&rtxn)
            .unwrap()
            .map(|result| result.unwrap().0)
            .collect();
        keys
    };
    commands.insert_resource(LoadedChunkKeys(Arc::new(Mutex::new(keys))));
    commands.insert_resource(DataBaseHandle(db));
    commands.insert_resource(DataBaseEnvHandle(env));
}
