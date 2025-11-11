use bevy::prelude::*;

use crate::{
    conversions::chunk_coord_to_world_pos,
    data_loader::driver::{ChunkRequest, ChunksBeingLoaded},
    terrain::terrain::{
            HALF_CHUNK, MAX_RADIUS_SQUARED, Z0_RADIUS_SQUARED, Z1_RADIUS_SQUARED, Z2_RADIUS_SQUARED
        },
};

const MAX_WORLD_SIZE: i16 = 512; //in chunks

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
    pub lower_chunk_coord: (i16, i16, i16), // chunk coord of lower corner
    pub size: i16,                 // region size in chunks (power of 2)
    pub children: Option<Box<[Option<SvoNode>; 8]>>,
    pub chunk: Option<(bool, u8)>, // (has_entity, chunk data, load status)
    pub node_min: Vec3,
    pub node_max: Vec3,
}

impl SvoNode {
    pub fn new(lower_chunk_coord: (i16, i16, i16), size: i16) -> Self {
        let chunk_center = chunk_coord_to_world_pos(&lower_chunk_coord);
        let node_min = chunk_center - Vec3::splat(HALF_CHUNK);
        let node_max = node_min + Vec3::splat(size as f32 * (HALF_CHUNK * 2.0));
        Self {
            lower_chunk_coord,
            size,
            children: None,
            chunk: None,
            node_min,
            node_max,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    fn child_index(&self, chunk_coord: &(i16, i16, i16)) -> usize {
        let half = self.size / 2;
        let mut index = 0usize;
        if chunk_coord.0 >= self.lower_chunk_coord.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.lower_chunk_coord.1 + half {
            index |= 2;
        }
        if chunk_coord.2 >= self.lower_chunk_coord.2 + half {
            index |= 4;
        }
        index
    }

    pub fn get(&self, coord: (i16, i16, i16)) -> Option<&(bool, u8)> {
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
    ) -> Option<&mut (bool, u8)> {
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
        load_status: u8,
        has_entity: bool,
    ) {
        if self.size == 1 {
            debug_assert!(
                self.chunk.is_none(),
                "Overwriting existing chunk at {:?}",
                coord
            );
            self.chunk = Some((has_entity, load_status));
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
                self.lower_chunk_coord.0 + if (index & 1) != 0 { half } else { 0 },
                self.lower_chunk_coord.1 + if (index & 2) != 0 { half } else { 0 },
                self.lower_chunk_coord.2 + if (index & 4) != 0 { half } else { 0 },
            );
            children[index] = Some(SvoNode::new(child_pos, self.size / 2));
        }
        children[index]
            .as_mut()
            .unwrap()
            .insert(coord, load_status, has_entity);
    }

    pub fn contains(&self, chunk_coord: &(i16, i16, i16)) -> bool {
        if chunk_coord.0 < self.lower_chunk_coord.0
            || chunk_coord.1 < self.lower_chunk_coord.1
            || chunk_coord.2 < self.lower_chunk_coord.2
            || chunk_coord.0 >= self.lower_chunk_coord.0 + self.size
            || chunk_coord.1 >= self.lower_chunk_coord.1 + self.size
            || chunk_coord.2 >= self.lower_chunk_coord.2 + self.size
        {
            return false;
        }

        if self.is_leaf() {
            if self.size == 1 {
                return self.chunk.is_some() && self.lower_chunk_coord == *chunk_coord;
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
        if chunk_coord.0 < self.lower_chunk_coord.0
            || chunk_coord.1 < self.lower_chunk_coord.1
            || chunk_coord.2 < self.lower_chunk_coord.2
            || chunk_coord.0 >= self.lower_chunk_coord.0 + self.size
            || chunk_coord.1 >= self.lower_chunk_coord.1 + self.size
            || chunk_coord.2 >= self.lower_chunk_coord.2 + self.size
        {
            return self.chunk.is_some() || self.children.is_some();
        }

        if self.is_leaf() {
            if self.size == 1 && self.lower_chunk_coord == chunk_coord {
                self.chunk = None;
            }
            return self.chunk.is_some();
        }

        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.lower_chunk_coord.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.lower_chunk_coord.1 + half {
            index |= 2;
        }
        if chunk_coord.2 >= self.lower_chunk_coord.2 + half {
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
        let (x, y, z) = self.lower_chunk_coord;
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

    pub fn fill_missing_chunks_in_radius(
        &mut self,
        center: &Vec3,
        radius: f32,
        chunks_being_loaded: &mut ChunksBeingLoaded,
        request_buffer: &mut Vec<ChunkRequest>,
    ) {
        if !sphere_intersects_aabb(center, radius, &self.node_min, &self.node_max) {
            return;
        }
        if self.is_leaf() {
            if self.size == 1 {
                let chunk_coord = self.lower_chunk_coord;
                if self.chunk.is_none() && !chunks_being_loaded.0.contains_key(&chunk_coord) {
                    let distance_squared =
                        center.distance_squared(chunk_coord_to_world_pos(&chunk_coord));
                    if distance_squared > MAX_RADIUS_SQUARED {
                        return; //skip chunks where the sphere intersects but chunk cenbter is outside max radius
                    }
                    let request_id = chunks_being_loaded.1;
                    chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                    chunks_being_loaded.0.insert(chunk_coord, request_id);
                    let load_priority = get_load_priority(distance_squared);
                    if request_buffer.contains(&ChunkRequest {
                        position: chunk_coord,
                        load_status: load_priority,
                        request_id,
                        upgrade: false,
                        distance_squared: distance_squared.round() as u32,
                    }) {
                        panic!("Duplicate chunk load request for chunk {:?}", chunk_coord);
                    }
                    request_buffer.push(ChunkRequest {
                        position: chunk_coord,
                        load_status: load_priority,
                        request_id,
                        upgrade: false,
                        distance_squared: distance_squared.round() as u32,
                    });
                } else if !chunks_being_loaded.0.contains_key(&chunk_coord) {
                    let current_load_status = self.chunk.as_ref().unwrap().1;
                    let distance_squared =
                        center.distance_squared(chunk_coord_to_world_pos(&chunk_coord));
                    let desired_load_status = get_load_priority(distance_squared);
                    if desired_load_status < current_load_status {
                        let request_id = chunks_being_loaded.1;
                        chunks_being_loaded.1 = chunks_being_loaded.1.wrapping_add(1);
                        chunks_being_loaded.0.insert(chunk_coord, request_id);
                        if request_buffer.contains(&ChunkRequest {
                            position: chunk_coord,
                            load_status: desired_load_status,
                            request_id,
                            upgrade: true,
                            distance_squared: distance_squared.round() as u32,
                        }) {
                            panic!("Duplicate chunk load request for chunk {:?}", chunk_coord);
                        }
                        request_buffer.push(ChunkRequest {
                            position: chunk_coord,
                            load_status: desired_load_status,
                            request_id,
                            upgrade: true,
                            distance_squared: distance_squared.round() as u32,
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
                    self.lower_chunk_coord.0 + if (i & 1) != 0 { half } else { 0 },
                    self.lower_chunk_coord.1 + if (i & 2) != 0 { half } else { 0 },
                    self.lower_chunk_coord.2 + if (i & 4) != 0 { half } else { 0 },
                );
                if children[i].is_none() {
                    let child_center = chunk_coord_to_world_pos(&child_pos);
                    let child_min = child_center - Vec3::splat(HALF_CHUNK);
                    let child_max = child_min + Vec3::splat(half as f32 * (HALF_CHUNK * 2.0));
                    if sphere_intersects_aabb(center, radius, &child_min, &child_max) {
                        children[i] = Some(SvoNode::new(child_pos, half));
                    }
                }
                if let Some(child) = &mut children[i] {
                    child.fill_missing_chunks_in_radius(
                        center,
                        radius,
                        chunks_being_loaded,
                        request_buffer,
                    );
                }
            }
        }
    }

    /// Query all chunks that are completely outside the given sphere.
    /// Returns coordinates and entity IDs.
    /// This may need to change to base on distance instead of intersection
    pub fn query_chunks_outside_sphere(
        &self,
        center: &Vec3,
        results: &mut Vec<((i16, i16, i16), bool)>,
    ) {
        // Quick prune: if the node’s nearest point is still beyond MAX_RADIUS, skip this node entirely.
        let node_center_to_sphere = {
            let mut sq_dist = 0.0;
            let mut v = center.x;
            if v < self.node_min.x {
                sq_dist += (self.node_min.x - v) * (self.node_min.x - v);
            } else if v > self.node_max.x {
                sq_dist += (v - self.node_max.x) * (v - self.node_max.x);
            }
            v = center.y;
            if v < self.node_min.y {
                sq_dist += (self.node_min.y - v) * (self.node_min.y - v);
            } else if v > self.node_max.y {
                sq_dist += (v - self.node_max.y) * (v - self.node_max.y);
            }
            v = center.z;
            if v < self.node_min.z {
                sq_dist += (self.node_min.z - v) * (self.node_min.z - v);
            } else if v > self.node_max.z {
                sq_dist += (v - self.node_max.z) * (v - self.node_max.z);
            }
            sq_dist
        };
        //if this entire node is beyond MAX_RADIUS, collect all chunks inside it
        if node_center_to_sphere > MAX_RADIUS_SQUARED {
            self.collect_all_chunks(results);
            return;
        }
        if self.is_leaf() && self.size == 1 {
            if let Some((has_entity, _)) = &self.chunk {
                let chunk_center = chunk_coord_to_world_pos(&self.lower_chunk_coord);
                let dist_sq = center.distance_squared(chunk_center);
                if dist_sq > MAX_RADIUS_SQUARED {
                    results.push((self.lower_chunk_coord, *has_entity));
                }
            }
            return;
        }
        if let Some(children) = &self.children {
            for child in children.iter().filter_map(|c| c.as_ref()) {
                child.query_chunks_outside_sphere(center, results);
            }
        }
    }

    fn collect_all_chunks(&self, results: &mut Vec<((i16, i16, i16), bool)>) {
        if self.is_leaf() && self.size == 1 {
            if let Some((has_entity, _)) = &self.chunk {
                results.push((self.lower_chunk_coord, *has_entity));
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

pub fn sphere_intersects_aabb(center: &Vec3, radius: f32, min: &Vec3, max: &Vec3) -> bool {
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

fn get_load_priority(distance_squared: f32) -> u8 {
    if distance_squared <= Z0_RADIUS_SQUARED {
        0
    } else if distance_squared <= Z1_RADIUS_SQUARED {
        1
    } else if distance_squared <= Z2_RADIUS_SQUARED {
        2
    } else {
        3
    }
}
