use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};

use bevy::prelude::*;

use crate::conversions::chunk_coord_to_world_pos;
use crate::terrain::terrain::{CHUNK_CREATION_RADIUS_SQUARED, TerrainChunk};

#[derive(Resource)]
pub struct ChunkIndexFile(pub File);

#[derive(Resource)]
pub struct ChunkDataFile(pub File);

#[derive(Resource)]
pub struct ChunkIndexMap(pub HashMap<(i16, i16, i16), u64>);

pub fn create_chunk_file_data(
    chunk: &TerrainChunk,
    chunk_coord: (i16, i16, i16),
    index_map: &mut HashMap<(i16, i16, i16), u64>,
    mut data_file: &File,
    mut index_file: &File,
) {
    let chunk_data = serde_json::to_vec(chunk).unwrap();
    let index = data_file.seek(SeekFrom::End(0)).unwrap();
    data_file.write_all(&chunk_data).unwrap();
    data_file.write_all(b"\n").unwrap();
    writeln!(
        index_file,
        "{},{},{},{}",
        chunk_coord.0, chunk_coord.1, chunk_coord.2, index
    )
    .unwrap();
    index_map.insert(chunk_coord, index);
}

pub fn update_chunk_file_data(
    index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_coord: (i16, i16, i16),
    chunk: &TerrainChunk,
    mut data_file: &File,
) {
    let chunk_data = serde_json::to_vec(chunk).unwrap();
    if let Some(&index) = index_map.get(&chunk_coord) {
        data_file.seek(SeekFrom::Start(index)).unwrap();
        data_file.write_all(&chunk_data).unwrap();
        data_file.write_all(b"\n").unwrap();
    }
}

pub fn load_chunk_data(
    data_file: &mut File,
    index_map: &HashMap<(i16, i16, i16), u64>,
    chunk_coord: (i16, i16, i16),
) -> TerrainChunk {
    let index = index_map.get(&chunk_coord).unwrap();
    data_file.seek(SeekFrom::Start(*index)).unwrap();
    let reader = BufReader::new(data_file);
    let line = reader.lines().next().unwrap().unwrap();
    serde_json::from_str(&line).unwrap()
}

pub fn load_chunk_index_map(index_file: &File) -> HashMap<(i16, i16, i16), u64> {
    let mut index_map = HashMap::new();
    let reader = BufReader::new(index_file);
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 4 {
            if let (Ok(x), Ok(y), Ok(z), Ok(index)) = (
                parts[0].parse::<i16>(),
                parts[1].parse::<i16>(),
                parts[2].parse::<i16>(),
                parts[3].parse::<u64>(),
            ) {
                index_map.insert((x, y, z), index);
            }
        }
    }
    println!("loaded {:?} chunk indexes from file", index_map.len());
    index_map
}

pub fn setup_chunk_loading(mut commands: Commands) {
    let index_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/chunk_index_data.txt")
        .unwrap();
    let data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("data/chunk_data.txt")
        .unwrap();
    commands.insert_resource(ChunkDataFile(data_file));
    commands.insert_resource(ChunkIndexMap(load_chunk_index_map(&index_file)));
    commands.insert_resource(ChunkIndexFile(index_file));
}

pub fn deallocate_chunks(
    player_chunk: (i16, i16, i16),
    chunk_map: &mut HashMap<(i16, i16, i16), (Entity, TerrainChunk)>,
    commands: &mut Commands,
) {
    let player_chunk_world_pos = chunk_coord_to_world_pos(player_chunk);
    let mut chunks_to_remove = Vec::new();
    for (chunk_coord, chunk) in chunk_map.iter() {
        let world_pos = chunk_coord_to_world_pos(*chunk_coord);
        if world_pos.distance_squared(player_chunk_world_pos) > CHUNK_CREATION_RADIUS_SQUARED {
            commands.entity(chunk.0).despawn();
            chunks_to_remove.push(*chunk_coord);
        }
    }
    for chunk_coord in chunks_to_remove {
        chunk_map.remove(&chunk_coord);
    }
}
