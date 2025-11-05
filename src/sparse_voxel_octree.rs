use bevy::prelude::*;

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    data_loader::driver::{ChunkChannels, ChunkRequest, ChunksBeingLoaded},
    terrain::{
        chunk_generator::{dequantize_i16_to_f32, quantize_f32_to_i16},
        lod_zones::Z2_RADIUS_SQUARED,
        terrain::{HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, TerrainChunk, VOXEL_SIZE, Z1_RADIUS_SQUARED},
    },
};

const MAX_WORLD_SIZE: i16 = 2048;
const CHUNK_WORLD_SIZE: f32 = SAMPLES_PER_CHUNK_DIM as f32 * VOXEL_SIZE;

#[derive(Resource)]
pub struct ChunkSvo {
    pub root: SvoNode,
}

impl ChunkSvo {
    pub fn new() -> Self {
        Self {
            root: SvoNode::new(
                (-MAX_WORLD_SIZE, -MAX_WORLD_SIZE, -MAX_WORLD_SIZE),
                2 * MAX_WORLD_SIZE,
            ),
        }
    }
}

#[derive(Debug)]
pub struct SvoNode {
    pub position: (i16, i16, i16), // lower corner in chunk coordinates
    pub size: i16,                 // region size in chunks (power of 2)
    pub children: Option<Box<[Option<SvoNode>; 8]>>,
    pub chunk: Option<(Option<Entity>, Option<TerrainChunk>, u8)>,
    pub node_min: Vec3,
    pub node_max: Vec3,
}

impl SvoNode {
    pub fn new(position: (i16, i16, i16), size: i16) -> Self {
        let node_min = Vec3::new(
            position.0 as f32 * CHUNK_WORLD_SIZE,
            position.1 as f32 * CHUNK_WORLD_SIZE,
            position.2 as f32 * CHUNK_WORLD_SIZE,
        );
        Self {
            position,
            size,
            children: None,
            chunk: None,
            node_min,
            node_max: node_min + Vec3::splat(size as f32 * CHUNK_WORLD_SIZE),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    fn child_index(&self, chunk_coord: &(i16, i16, i16)) -> usize {
        let half = self.size / 2;
        let mut index = 0usize;
        if chunk_coord.0 >= self.position.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.position.1 + half {
            index |= 2;
        }
        if chunk_coord.2 >= self.position.2 + half {
            index |= 4;
        }
        index
    }

    pub fn get(
        &self,
        coord: (i16, i16, i16),
    ) -> Option<&(Option<Entity>, Option<TerrainChunk>, u8)> {
        if self.size == 1 {
            return self.chunk.as_ref();
        }
        let children = self.children.as_ref()?;
        let idx = self.child_index(&coord);
        children[idx].as_ref()?.get(coord)
    }

    pub fn get_mut(
        &mut self,
        coord: (i16, i16, i16),
    ) -> Option<&mut (Option<Entity>, Option<TerrainChunk>, u8)> {
        if self.size == 1 {
            return self.chunk.as_mut();
        }
        let idx = self.child_index(&coord);
        let children = self.children.as_mut()?;
        children[idx].as_mut()?.get_mut(coord)
    }

    pub fn insert(
        &mut self,
        coord: (i16, i16, i16),
        entity: Option<Entity>,
        chunk: Option<TerrainChunk>,
        load_status: u8,
    ) {
        if self.size == 1 {
            debug_assert!(
                self.chunk.is_none(),
                "Overwriting existing chunk at {:?}",
                coord
            );
            self.chunk = Some((entity, chunk, load_status));
            return;
        }
        let index = self.child_index(&coord);
        if self.children.is_none() {
            self.children = Some(Box::new([None, None, None, None, None, None, None, None]));
        }
        let children = self.children.as_mut().unwrap();
        if children[index].is_none() {
            let half = self.size / 2;
            let child_pos = (
                self.position.0 + if (index & 1) != 0 { half } else { 0 },
                self.position.1 + if (index & 2) != 0 { half } else { 0 },
                self.position.2 + if (index & 4) != 0 { half } else { 0 },
            );
            children[index] = Some(SvoNode::new(child_pos, self.size / 2));
        }
        children[index]
            .as_mut()
            .unwrap()
            .insert(coord, entity, chunk, load_status);
    }

    pub fn contains(&self, chunk_coord: &(i16, i16, i16)) -> bool {
        if chunk_coord.0 < self.position.0
            || chunk_coord.1 < self.position.1
            || chunk_coord.2 < self.position.2
            || chunk_coord.0 >= self.position.0 + self.size
            || chunk_coord.1 >= self.position.1 + self.size
            || chunk_coord.2 >= self.position.2 + self.size
        {
            return false;
        }

        if self.is_leaf() {
            if self.size == 1 {
                return self.chunk.is_some() && self.position == *chunk_coord;
            }
            return false;
        }

        let index = self.child_index(chunk_coord);
        match &self.children {
            Some(children) => {
                if let Some(child) = &children[index] {
                    child.contains(chunk_coord)
                } else {
                    false
                }
            }
            None => false,
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &SvoNode> + '_> {
        if self.is_leaf() {
            Box::new(std::iter::once(self))
        } else if let Some(children) = &self.children {
            Box::new(
                children
                    .iter()
                    .filter_map(|c| c.as_ref())
                    .flat_map(|c| c.iter()),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }

    pub fn remove_leaf(&mut self, chunk_coord: (i16, i16, i16)) -> bool {
        if chunk_coord.0 < self.position.0
            || chunk_coord.1 < self.position.1
            || chunk_coord.2 < self.position.2
            || chunk_coord.0 >= self.position.0 + self.size
            || chunk_coord.1 >= self.position.1 + self.size
            || chunk_coord.2 >= self.position.2 + self.size
        {
            return self.chunk.is_some() || self.children.is_some();
        }

        if self.is_leaf() {
            if self.size == 1 && self.position == chunk_coord {
                self.chunk = None;
            }
            return self.chunk.is_some();
        }

        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.position.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.position.1 + half {
            index |= 2;
        }
        if chunk_coord.2 >= self.position.2 + half {
            index |= 4;
        }

        if let Some(children) = self.children.as_mut() {
            if let Some(child) = &mut children[index] {
                if !child.remove_leaf(chunk_coord) {
                    children[index] = None;
                }
            }
        }

        self.chunk.is_some()
            || self
                .children
                .as_ref()
                .map_or(false, |c| c.iter().any(|x| x.is_some()))
    }

    pub fn delete(&mut self, coord: (i16, i16, i16)) -> bool {
        if self.size == 1 {
            let had_chunk = self.chunk.take().is_some();
            return had_chunk;
        }

        let half = self.size / 2;
        let (x, y, z) = self.position;
        let (cx, cy, cz) = coord;

        let ix = if cx >= x + half { 1 } else { 0 };
        let iy = if cy >= y + half { 1 } else { 0 };
        let iz = if cz >= z + half { 1 } else { 0 };
        let child_index = ix | (iy << 1) | (iz << 2);

        if let Some(children) = self.children.as_mut() {
            if let Some(child) = children[child_index].as_mut() {
                if child.delete(coord) {
                    if child.children.is_none() && child.chunk.is_none() {
                        children[child_index] = None;
                    }
                    if children.iter().all(|c| c.is_none()) {
                        self.children = None;
                    }
                    return true;
                }
            }
        }

        false
    }

    pub fn dig_sphere(&mut self, center: Vec3, radius: f32, strength: f32) -> Vec<(i16, i16, i16)> {
        let mut modified_chunks = Vec::new();
        let min_world = center - Vec3::splat(radius);
        let max_world = center + Vec3::splat(radius);
        let min_chunk = world_pos_to_chunk_coord(&min_world);
        let max_chunk = world_pos_to_chunk_coord(&max_world);
        for chunk_x in min_chunk.0..=max_chunk.0 {
            for chunk_y in min_chunk.1..=max_chunk.1 {
                for chunk_z in min_chunk.2..=max_chunk.2 {
                    let chunk_coord = (chunk_x, chunk_y, chunk_z);
                    let chunk_modified =
                        self.modify_chunk_voxels(chunk_coord, center, radius, strength);
                    if chunk_modified && !modified_chunks.contains(&chunk_coord) {
                        modified_chunks.push(chunk_coord);
                    }
                }
            }
        }
        modified_chunks
    }

    pub fn modify_chunk_voxels(
        &mut self,
        chunk_coord: (i16, i16, i16),
        dig_center: Vec3,
        radius: f32,
        strength: f32,
    ) -> bool {
        if chunk_coord.0 < self.position.0
            || chunk_coord.1 < self.position.1
            || chunk_coord.2 < self.position.2
            || chunk_coord.0 >= self.position.0 + self.size
            || chunk_coord.1 >= self.position.1 + self.size
            || chunk_coord.2 >= self.position.2 + self.size
        {
            return false;
        }
        if self.size == 1 {
            if self.chunk.is_none() {
                return false;
            }
            let (_entity, chunk_option, _load_status) = self.chunk.as_mut().unwrap();
            let chunk = chunk_option.as_mut().unwrap();
            let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
            let chunk_world_size = SAMPLES_PER_CHUNK_DIM as f32 * VOXEL_SIZE;
            let node_min = Vec3::new(
                chunk_center.x - HALF_CHUNK,
                chunk_center.y - HALF_CHUNK,
                chunk_center.z - HALF_CHUNK,
            );
            let node_max = node_min + Vec3::splat(chunk_world_size);
            if !sphere_intersects_aabb(&dig_center, radius, &node_min, &node_max) {
                return false;
            }
            let mut chunk_modified = false;
            for z in 0..SAMPLES_PER_CHUNK_DIM {
                for y in 0..SAMPLES_PER_CHUNK_DIM {
                    for x in 0..SAMPLES_PER_CHUNK_DIM {
                        let world_x = chunk_center.x - HALF_CHUNK + x as f32 * VOXEL_SIZE;
                        let world_y = chunk_center.y - HALF_CHUNK + y as f32 * VOXEL_SIZE;
                        let world_z = chunk_center.z - HALF_CHUNK + z as f32 * VOXEL_SIZE;
                        let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                        let distance = voxel_world_pos.distance(dig_center);
                        if distance <= radius {
                            let falloff = 1.0 - (distance / radius).clamp(0.0, 1.0);
                            let dig_amount = strength * falloff;
                            let current_density =
                                chunk.get_mut_density(x as u32, y as u32, z as u32);
                            if *current_density < 0 {
                                let sdf_f32 = dequantize_i16_to_f32(*current_density);
                                let new_sdf = (sdf_f32 + dig_amount).clamp(-10.0, 10.0);
                                *current_density = quantize_f32_to_i16(new_sdf);
                                chunk_modified = true;
                            }
                        }
                    }
                }
            }
            return chunk_modified;
        } else {
            let index = self.child_index(&chunk_coord);
            if let Some(children) = self.children.as_mut() {
                if let Some(child) = &mut children[index] {
                    return child.modify_chunk_voxels(chunk_coord, dig_center, radius, strength);
                }
            }
            return false;
        }
    }

    pub fn fill_missing_chunks_in_radius(
        &mut self,
        center: &Vec3,
        radius: f32,
        chunks_being_loaded: &mut ChunksBeingLoaded,
        chunk_channels: &ChunkChannels,
    ) {
        if !sphere_intersects_aabb(center, radius, &self.node_min, &self.node_max) {
            return;
        }
        if self.is_leaf() {
            if self.size == 1 {
                let chunk_coord = self.position;
                if self.chunk.is_none() && !chunks_being_loaded.0.contains_key(&chunk_coord) {
                    let request_id = chunks_being_loaded.1;
                    chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                    chunks_being_loaded.0.insert(chunk_coord, request_id);
                    let distance_squared =
                        center.distance_squared(chunk_coord_to_world_pos(&chunk_coord));
                    if distance_squared <= Z1_RADIUS_SQUARED {
                        let _ = chunk_channels.requests.send(ChunkRequest {
                            position: chunk_coord,
                            load_status: 1,
                            request_id,
                            upgrade: false,
                            distance_squared: distance_squared as u32,
                        });
                    } else if distance_squared <= Z2_RADIUS_SQUARED {
                        let _ = chunk_channels.requests.send(ChunkRequest {
                            position: chunk_coord,
                            load_status: 2,
                            request_id,
                            upgrade: false,
                            distance_squared: distance_squared as u32,
                        });
                    }
                }
                return;
            } else {
                self.children = Some(Box::new([None, None, None, None, None, None, None, None]));
            }
        }
        if let Some(children) = &mut self.children {
            let half = self.size / 2;
            for i in 0..8 {
                let child_pos = (
                    self.position.0 + if (i & 1) != 0 { half } else { 0 },
                    self.position.1 + if (i & 2) != 0 { half } else { 0 },
                    self.position.2 + if (i & 4) != 0 { half } else { 0 },
                );
                if children[i].is_none() {
                    let child_min = Vec3::new(
                        child_pos.0 as f32 * CHUNK_WORLD_SIZE,
                        child_pos.1 as f32 * CHUNK_WORLD_SIZE,
                        child_pos.2 as f32 * CHUNK_WORLD_SIZE,
                    );
                    let child_max = child_min + Vec3::splat(half as f32 * CHUNK_WORLD_SIZE);
                    let child_intersects =
                        sphere_intersects_aabb(center, radius, &child_min, &child_max);
                    if child_intersects {
                        children[i] = Some(SvoNode::new(child_pos, half));
                    }
                }
                if let Some(child) = &mut children[i] {
                    child.fill_missing_chunks_in_radius(
                        center,
                        radius,
                        chunks_being_loaded,
                        chunk_channels,
                    );
                }
            }
        }
    }

    /// Query all chunks that are completely outside the given sphere.
    /// Returns coordinates and entity IDs.
    pub fn query_chunks_outside_sphere(
        &self,
        center: &Vec3,
        radius: f32,
        results: &mut Vec<((i16, i16, i16), Option<Entity>)>,
    ) {
        let chunk_world_size = SAMPLES_PER_CHUNK_DIM as f32 * VOXEL_SIZE;
        let node_min = Vec3::new(
            self.position.0 as f32 * chunk_world_size,
            self.position.1 as f32 * chunk_world_size,
            self.position.2 as f32 * chunk_world_size,
        );
        let node_max = node_min + Vec3::splat(self.size as f32 * chunk_world_size);
        let intersects = sphere_intersects_aabb(center, radius, &node_min, &node_max);
        if self.is_leaf() && self.size == 1 {
            if !intersects {
                if let Some((entity, _, _)) = &self.chunk {
                    results.push((self.position, *entity));
                }
            }
            return;
        }
        if !intersects {
            self.collect_all_chunks(results);
            return;
        }
        if let Some(children) = &self.children {
            for child in children.iter().filter_map(|c| c.as_ref()) {
                child.query_chunks_outside_sphere(center, radius, results);
            }
        }
    }

    fn collect_all_chunks(&self, results: &mut Vec<((i16, i16, i16), Option<Entity>)>) {
        if self.is_leaf() && self.size == 1 {
            if let Some((entity, _, _)) = &self.chunk {
                results.push((self.position, *entity));
            }
            return;
        }
        if let Some(children) = &self.children {
            for child in children.iter().filter_map(|c| c.as_ref()) {
                child.collect_all_chunks(results);
            }
        }
    }
}

fn sphere_intersects_aabb(center: &Vec3, radius: f32, min: &Vec3, max: &Vec3) -> bool {
    let radius_sq = radius * radius;
    let mut d = 0.0;
    let v = center.x;
    if v < min.x {
        d += (min.x - v) * (min.x - v);
    } else if v > max.x {
        d += (v - max.x) * (v - max.x);
    }
    let v = center.y;
    if v < min.y {
        d += (min.y - v) * (min.y - v);
    } else if v > max.y {
        d += (v - max.y) * (v - max.y);
    }
    let v = center.z;
    if v < min.z {
        d += (min.z - v) * (min.z - v);
    } else if v > max.z {
        d += (v - max.z) * (v - max.z);
    }
    d <= radius_sq
}
