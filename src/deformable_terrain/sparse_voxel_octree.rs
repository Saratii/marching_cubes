use std::sync::atomic::Ordering;

use crate::deformable_terrain::driver::RENDER_RADIUS_SQUARED;
use crate::{
    constants::{
        CHUNK_WORLD_SIZE, CHUNKS_PER_CLUSTER, CHUNKS_PER_CLUSTER_DIM, CLUSTER_WORLD_LENGTH,
        REDUCED_LOD_1_RADIUS_SQUARED, REDUCED_LOD_2_RADIUS_SQUARED, REDUCED_LOD_3_RADIUS_SQUARED,
        REDUCED_LOD_4_RADIUS_SQUARED, REDUCED_LOD_5_RADIUS_SQUARED, SIMULATION_RADIUS_SQUARED,
    },
    conversions::{cluster_coord_to_world_center, cluster_coord_to_world_pos},
    deformable_terrain::driver::{ClusterRequest, LoadState, LoadStateTransition},
};
use bevy::prelude::*;
use rustc_hash::FxHashSet;

const MAX_WORLD_SIZE: i16 = 512; //in chunks

#[derive(Debug)]
pub struct SvoNode {
    pub lower_cluster_coord: (i16, i16, i16), // cluster coord of lower corner (RENAMED)
    pub size: i16,                            // region size in clusters (power of 2)
    pub children: Option<Box<[Option<SvoNode>; 8]>>,
    pub chunk: Option<([bool; CHUNKS_PER_CLUSTER], LoadState)>,
    pub node_min: Vec3,
    pub node_max: Vec3,
}

impl SvoNode {
    pub fn world_root() -> Self {
        SvoNode::new(
            (-MAX_WORLD_SIZE, -MAX_WORLD_SIZE, -MAX_WORLD_SIZE),
            2 * MAX_WORLD_SIZE,
        )
    }

    fn new(lower_cluster_coord: (i16, i16, i16), size: i16) -> Self {
        let node_min = cluster_coord_to_world_pos(&lower_cluster_coord);
        let node_max = node_min + Vec3::splat(size as f32 * CLUSTER_WORLD_LENGTH);
        Self {
            lower_cluster_coord,
            size,
            children: None,
            chunk: None,
            node_min,
            node_max,
        }
    }

    #[inline(always)]
    fn child_index(&self, chunk_coord: &(i16, i16, i16)) -> usize {
        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.lower_cluster_coord.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.lower_cluster_coord.1 + half {
            index |= 2;
        }
        if chunk_coord.2 >= self.lower_cluster_coord.2 + half {
            index |= 4;
        }
        index
    }

    pub fn insert(
        &mut self,
        coord: (i16, i16, i16),
        has_entity: [bool; CHUNKS_PER_CLUSTER],
        load_state: LoadState,
    ) {
        if self.size == 1 {
            self.chunk = Some((has_entity, load_state));
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
                self.lower_cluster_coord.0 + if (index & 1) != 0 { half } else { 0 },
                self.lower_cluster_coord.1 + if (index & 2) != 0 { half } else { 0 },
                self.lower_cluster_coord.2 + if (index & 4) != 0 { half } else { 0 },
            );
            children[index] = Some(SvoNode::new(child_pos, self.size / 2));
        }
        children[index]
            .as_mut()
            .unwrap()
            .insert(coord, has_entity, load_state);
    }

    pub fn delete(&mut self, coord: (i16, i16, i16)) -> bool {
        if self.size == 1 {
            let had_chunk = self.chunk.take().is_some();
            return had_chunk;
        }
        let child_index = self.child_index(&coord);
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
        radius_squared: f32,
        chunks_being_loaded: &FxHashSet<(i16, i16, i16)>,
        request_buffer: &mut Vec<ClusterRequest>,
    ) {
        if !sphere_intersects_aabb(center, radius_squared, &self.node_min, &self.node_max) {
            return;
        }
        if self.children.is_none() {
            if self.size == 1 {
                if self.chunk.is_none() && !chunks_being_loaded.contains(&self.lower_cluster_coord)
                {
                    //chunk did not already exist
                    let distance_squared = center
                        .distance_squared(cluster_coord_to_world_center(&self.lower_cluster_coord));
                    if distance_squared
                        > f32::from_bits(RENDER_RADIUS_SQUARED.load(Ordering::Relaxed))
                    {
                        return; //skip chunks where the sphere intersects but chunk center is outside max radius
                    }
                    let load_state_transition =
                        get_load_state_transition(None, get_desired_state(distance_squared));
                    request_buffer.push(ClusterRequest {
                        position: self.lower_cluster_coord,
                        distance_squared,
                        load_state_transition,
                        prev_has_entity: None,
                        prev_in_simulation_radius: false,
                    });
                } else if !chunks_being_loaded.contains(&self.lower_cluster_coord) {
                    //chunk already existed
                    let current_load_state = self.chunk.as_ref().unwrap().1;
                    let distance_squared = center
                        .distance_squared(cluster_coord_to_world_center(&self.lower_cluster_coord));
                    let desired_load_state = get_desired_state(distance_squared);
                    if desired_load_state != current_load_state {
                        let load_state_transition =
                            get_load_state_transition(Some(current_load_state), desired_load_state);
                        let prev_has_entity = self.chunk.as_ref().unwrap().0;
                        let prev_in_simulation_radius =
                            current_load_state == LoadState::FullWithCollider;
                        request_buffer.push(ClusterRequest {
                            position: self.lower_cluster_coord,
                            distance_squared,
                            load_state_transition,
                            prev_has_entity: Some(prev_has_entity),
                            prev_in_simulation_radius,
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
                    self.lower_cluster_coord.0 + if (i & 1) != 0 { half } else { 0 },
                    self.lower_cluster_coord.1 + if (i & 2) != 0 { half } else { 0 },
                    self.lower_cluster_coord.2 + if (i & 4) != 0 { half } else { 0 },
                );
                if children[i].is_none() {
                    let child_center = cluster_coord_to_world_pos(&child_pos);
                    let cluster_size_world = CHUNK_WORLD_SIZE * CHUNKS_PER_CLUSTER_DIM as f32;
                    let half_cluster = cluster_size_world * 0.5;
                    let child_min = child_center - Vec3::splat(half_cluster);
                    let child_max = child_min + Vec3::splat(half as f32 * cluster_size_world);
                    if sphere_intersects_aabb(center, radius_squared, &child_min, &child_max) {
                        children[i] = Some(SvoNode::new(child_pos, half));
                    }
                }
                if let Some(child) = &mut children[i] {
                    child.fill_missing_chunks_in_radius(
                        center,
                        radius_squared,
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
        results: &mut Vec<((i16, i16, i16), [bool; CHUNKS_PER_CLUSTER])>,
    ) {
        // Quick prune: if the node’s nearest point is still beyond MAX_RENDER_RADIUS, skip this node entirely.
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
        //if this entire node is beyond MAX_RENDER_RADIUS, collect all chunks inside it
        if node_center_to_sphere > f32::from_bits(RENDER_RADIUS_SQUARED.load(Ordering::Relaxed)) {
            self.collect_all_chunks(results);
            return;
        }
        if self.size == 1 {
            if let Some((has_entity, _)) = &self.chunk {
                let chunk_center = cluster_coord_to_world_center(&self.lower_cluster_coord);
                let dist_sq = center.distance_squared(chunk_center);
                if dist_sq > f32::from_bits(RENDER_RADIUS_SQUARED.load(Ordering::Relaxed)) {
                    results.push((self.lower_cluster_coord, *has_entity));
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

    fn collect_all_chunks(&self, results: &mut Vec<((i16, i16, i16), [bool; CHUNKS_PER_CLUSTER])>) {
        if self.size == 1 {
            if let Some((has_entity, _)) = &self.chunk {
                results.push((self.lower_cluster_coord, *has_entity));
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

pub fn sphere_intersects_aabb(center: &Vec3, radius_squared: f32, min: &Vec3, max: &Vec3) -> bool {
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
    d <= radius_squared
}

#[inline(always)]
fn get_load_state_transition(
    current: Option<LoadState>,
    desired: LoadState,
) -> LoadStateTransition {
    match (current, desired) {
        (Some(LoadState::Full), LoadState::FullWithCollider) => {
            LoadStateTransition::NoChangeAddCollider
        }
        (_, LoadState::FullWithCollider) => LoadStateTransition::ToFullWithCollider,
        (_, LoadState::Full) => LoadStateTransition::ToFull,
        (_, LoadState::Lod1) => LoadStateTransition::ToLod1,
        (_, LoadState::Lod2) => LoadStateTransition::ToLod2,
        (_, LoadState::Lod3) => LoadStateTransition::ToLod3,
        (_, LoadState::Lod4) => LoadStateTransition::ToLod4,
        (_, LoadState::Lod5) => LoadStateTransition::ToLod5,
    }
}

#[inline(always)]
pub fn get_desired_state(distance_squared: f32) -> LoadState {
    if distance_squared > REDUCED_LOD_5_RADIUS_SQUARED {
        LoadState::Lod5
    } else if distance_squared > REDUCED_LOD_4_RADIUS_SQUARED {
        LoadState::Lod4
    } else if distance_squared > REDUCED_LOD_3_RADIUS_SQUARED {
        LoadState::Lod3
    } else if distance_squared > REDUCED_LOD_2_RADIUS_SQUARED {
        LoadState::Lod2
    } else if distance_squared > REDUCED_LOD_1_RADIUS_SQUARED {
        LoadState::Lod1
    } else if distance_squared <= SIMULATION_RADIUS_SQUARED {
        LoadState::FullWithCollider
    } else {
        //in full lod but out of simulation range
        LoadState::Full
    }
}
