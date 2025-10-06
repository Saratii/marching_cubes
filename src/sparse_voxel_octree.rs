use bevy::prelude::*;

use crate::{
    conversions::{chunk_coord_to_world_pos, world_pos_to_chunk_coord},
    terrain::terrain::{HALF_CHUNK, SDF_VALUES_PER_CHUNK_DIM, TerrainChunk, VOXEL_SIZE},
};

const MAX_WORLD_SIZE: i16 = 2048;

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

pub struct SvoNode {
    pub position: (i16, i16, i16), // lower-left corner in chunk coordinates
    pub size: i16,                 // side length in chunks (power of 2)
    pub children: Option<Box<[Option<SvoNode>; 8]>>, // 8 octants
    pub chunks: Vec<(Entity, TerrainChunk)>, // only stored at leaf nodes
}

impl SvoNode {
    pub fn new(position: (i16, i16, i16), size: i16) -> Self {
        Self {
            position,
            size,
            children: None,
            chunks: Vec::new(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn get(&self, chunk_coord: &(i16, i16, i16)) -> &(Entity, TerrainChunk) {
        if self.is_leaf() {
            &self.chunks[0]
        } else {
            let half = self.size / 2;
            let mut index = 0;
            if chunk_coord.0 >= self.position.0 + half {
                index |= 1;
            }
            if chunk_coord.1 >= self.position.1 + half {
                index |= 4;
            }
            if chunk_coord.2 >= self.position.2 + half {
                index |= 2;
            }
            self.children.as_ref().unwrap()[index]
                .as_ref()
                .unwrap()
                .get(chunk_coord)
        }
    }

    pub fn insert(&mut self, chunk_coord: (i16, i16, i16), entity: Entity, chunk: TerrainChunk) {
        if self.size == 1 {
            self.chunks.push((entity, chunk));
            return;
        }
        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.position.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.position.1 + half {
            index |= 4;
        }
        if chunk_coord.2 >= self.position.2 + half {
            index |= 2;
        }
        if self.children.is_none() {
            self.children = Some(Box::new([None, None, None, None, None, None, None, None]));
        }
        let children = self.children.as_mut().unwrap();
        if children[index].is_none() {
            let child_pos = (
                self.position.0 + if (index & 1) != 0 { half } else { 0 },
                self.position.1 + if (index & 4) != 0 { half } else { 0 },
                self.position.2 + if (index & 2) != 0 { half } else { 0 },
            );
            children[index] = Some(SvoNode::new(child_pos, half));
        }
        children[index]
            .as_mut()
            .unwrap()
            .insert(chunk_coord, entity, chunk);
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
            self.chunks.iter().any(|(_, _)| {
                let center = (chunk_coord.0, chunk_coord.1, chunk_coord.2);
                center == *chunk_coord
            })
        } else {
            let half = self.size / 2;
            let mut index = 0;
            if chunk_coord.0 >= self.position.0 + half {
                index |= 1;
            }
            if chunk_coord.1 >= self.position.1 + half {
                index |= 4;
            }
            if chunk_coord.2 >= self.position.2 + half {
                index |= 2;
            }
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
        if self.is_leaf() {
            if self.chunks.iter().any(|(_entity, _)| {
                let center = (chunk_coord.0, chunk_coord.1, chunk_coord.2);
                center == chunk_coord
            }) {
                self.chunks.clear();
            }
            return !self.chunks.is_empty();
        }
        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.position.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.position.1 + half {
            index |= 4;
        }
        if chunk_coord.2 >= self.position.2 + half {
            index |= 2;
        }
        if let Some(children) = self.children.as_mut() {
            if let Some(child) = &mut children[index] {
                if !child.remove_leaf(chunk_coord) {
                    children[index] = None;
                }
            }
        }
        !self.chunks.is_empty()
            || self
                .children
                .as_ref()
                .map_or(false, |c| c.iter().any(|x| x.is_some()))
    }

    pub fn delete(&mut self, chunk_coord: (i16, i16, i16), commands: &mut Commands) -> bool {
        if self.size == 1 {
            self.chunks.retain(|(_entity, _chunk)| {
                let keep = false;
                keep
            });
            return !self.chunks.is_empty();
        }
        let half = self.size / 2;
        let mut index = 0;
        if chunk_coord.0 >= self.position.0 + half {
            index |= 1;
        }
        if chunk_coord.1 >= self.position.1 + half {
            index |= 4;
        }
        if chunk_coord.2 >= self.position.2 + half {
            index |= 2;
        }
        if let Some(children) = self.children.as_mut() {
            if let Some(child) = &mut children[index] {
                if !child.delete(chunk_coord, commands) {
                    children[index] = None;
                }
            }
        }
        let has_chunks = !self.chunks.is_empty();
        let has_children = self
            .children
            .as_ref()
            .map_or(false, |c| c.iter().any(|x| x.is_some()));
        has_chunks || has_children
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
        let mut chunk_modified = false;
        let (_, chunk) = if self.size == 1 {
            &mut self.chunks[0]
        } else {
            let half = self.size / 2;
            let mut index = 0;
            if chunk_coord.0 >= self.position.0 + half {
                index |= 1;
            }
            if chunk_coord.1 >= self.position.1 + half {
                index |= 4;
            }
            if chunk_coord.2 >= self.position.2 + half {
                index |= 2;
            }
            self.children.as_mut().unwrap()[index]
                .as_mut()
                .unwrap()
                .modify_chunk_voxels_get_mut(chunk_coord)
        };
        let chunk_center = chunk_coord_to_world_pos(&chunk_coord);
        for z in 0..SDF_VALUES_PER_CHUNK_DIM {
            for y in 0..SDF_VALUES_PER_CHUNK_DIM {
                for x in 0..SDF_VALUES_PER_CHUNK_DIM {
                    let world_x = chunk_center.x - HALF_CHUNK + x as f32 * VOXEL_SIZE;
                    let world_y = chunk_center.y - HALF_CHUNK + y as f32 * VOXEL_SIZE;
                    let world_z = chunk_center.z - HALF_CHUNK + z as f32 * VOXEL_SIZE;
                    let voxel_world_pos = Vec3::new(world_x, world_y, world_z);
                    let distance = voxel_world_pos.distance(dig_center);
                    if distance <= radius {
                        let falloff = 1.0 - (distance / radius).clamp(0.0, 1.0);
                        let dig_amount = strength * falloff;
                        let current_density = chunk.get_mut_density(x as u32, y as u32, z as u32);
                        if current_density.sdf > 0.0 {
                            current_density.sdf -= dig_amount;
                            chunk_modified = true;
                        }
                    }
                }
            }
        }
        chunk_modified
    }

    fn modify_chunk_voxels_get_mut(
        &mut self,
        chunk_coord: (i16, i16, i16),
    ) -> &mut (Entity, TerrainChunk) {
        if self.size == 1 {
            &mut self.chunks[0]
        } else {
            let half = self.size / 2;
            let mut index = 0;
            if chunk_coord.0 >= self.position.0 + half {
                index |= 1;
            }
            if chunk_coord.1 >= self.position.1 + half {
                index |= 4;
            }
            if chunk_coord.2 >= self.position.2 + half {
                index |= 2;
            }
            self.children.as_mut().unwrap()[index]
                .as_mut()
                .unwrap()
                .modify_chunk_voxels_get_mut(chunk_coord)
        }
    }
}
