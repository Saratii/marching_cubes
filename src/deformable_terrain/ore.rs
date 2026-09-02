use bevy::prelude::*;

use crate::{
    constants::{
        CHUNK_WORLD_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK_DIM, VOXEL_WORLD_SIZE, WORLD_SEED,
    },
    conversions::flatten_index,
    deformable_terrain::chunk_generator::MaterialCode,
};

// Ore is a pure function of world position rather than stored world state, so
// it can be evaluated on demand. That keeps deep chunks UniformDirt: a lump
// buried in solid rock is invisible until something digs to it, and the dig
// materializes the chunk anyway.

/// Edge of one lattice cell in world units. Each cell holds at most one lump.
const ORE_CELL_SIZE: f32 = 8.0;

/// Fraction of cells that hold a lump.
const ORE_CELL_CHANCE: f32 = 0.5;

/// Lump radius range, in world units.
const ORE_RADIUS: (f32, f32) = (0.35, 0.85);

/// Per-axis stretch, so lumps read as veined blobs rather than beads.
const ORE_STRETCH: (f32, f32) = (0.7, 1.5);

/// Widest a lump can reach from its center on any axis.
const ORE_MAX_REACH: f32 = ORE_RADIUS.1 * ORE_STRETCH.1;

struct OrePatch {
    center: Vec3,
    radius_squared: f32,
    inverse_stretch: Vec3,
    reach: f32,
}

impl OrePatch {
    #[inline(always)]
    fn contains(&self, pos: Vec3) -> bool {
        ((pos - self.center) * self.inverse_stretch).length_squared() < self.radius_squared
    }

    #[inline(always)]
    fn overlaps(&self, min: Vec3, max: Vec3) -> bool {
        (self.center - Vec3::splat(self.reach)).cmple(max).all()
            && (self.center + Vec3::splat(self.reach)).cmpge(min).all()
    }
}

#[inline(always)]
fn hash_cell(cell: (i32, i32, i32), salt: u32) -> u32 {
    let mut hash = (cell.0 as u32).wrapping_mul(0x8DA6_B343)
        ^ (cell.1 as u32).wrapping_mul(0xD871_9C61)
        ^ (cell.2 as u32).wrapping_mul(0xF1B0_9C79)
        ^ (WORLD_SEED as u32).wrapping_mul(0x2545_F491)
        ^ salt.wrapping_mul(0x27D4_EB2D);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7FEB_352D);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846C_A68B);
    hash ^= hash >> 16;
    hash
}

#[inline(always)]
fn hash_unit(cell: (i32, i32, i32), salt: u32) -> f32 {
    (hash_cell(cell, salt) >> 8) as f32 / (1u32 << 24) as f32
}

#[inline(always)]
fn lerp_unit(cell: (i32, i32, i32), salt: u32, range: (f32, f32)) -> f32 {
    range.0 + hash_unit(cell, salt) * (range.1 - range.0)
}

fn cell_patch(cell: (i32, i32, i32)) -> Option<OrePatch> {
    if hash_unit(cell, 0) >= ORE_CELL_CHANCE {
        return None;
    }
    let center = Vec3::new(
        (cell.0 as f32 + hash_unit(cell, 1)) * ORE_CELL_SIZE,
        (cell.1 as f32 + hash_unit(cell, 2)) * ORE_CELL_SIZE,
        (cell.2 as f32 + hash_unit(cell, 3)) * ORE_CELL_SIZE,
    );
    let radius = lerp_unit(cell, 4, ORE_RADIUS);
    let stretch = Vec3::new(
        lerp_unit(cell, 5, ORE_STRETCH),
        lerp_unit(cell, 6, ORE_STRETCH),
        lerp_unit(cell, 7, ORE_STRETCH),
    );
    Some(OrePatch {
        center,
        radius_squared: radius * radius,
        inverse_stretch: Vec3::ONE / stretch,
        reach: radius * stretch.max_element(),
    })
}

fn for_each_patch(min: Vec3, max: Vec3, mut f: impl FnMut(&OrePatch)) {
    let low = ((min - Vec3::splat(ORE_MAX_REACH)) / ORE_CELL_SIZE).floor();
    let high = ((max + Vec3::splat(ORE_MAX_REACH)) / ORE_CELL_SIZE).floor();
    for cell_z in low.z as i32..=high.z as i32 {
        for cell_y in low.y as i32..=high.y as i32 {
            for cell_x in low.x as i32..=high.x as i32 {
                if let Some(patch) = cell_patch((cell_x, cell_y, cell_z))
                    && patch.overlaps(min, max)
                {
                    f(&patch);
                }
            }
        }
    }
}

/// World coordinate of an unpadded sample along one axis, derived from the
/// sample's global index. Two chunks sharing a border sample derive the same
/// integer, so the lump test lands on the same side in both copies.
#[inline(always)]
fn sample_world_coord(chunk_coord: i16, sample: usize) -> f32 {
    let global = chunk_coord as i32 * (SAMPLES_PER_CHUNK_DIM as i32 - 1) + sample as i32;
    global as f32 * VOXEL_WORLD_SIZE - HALF_CHUNK
}

/// Stamp ore over the dirt of one chunk's unpadded material array. Only samples
/// already marked Dirt are converted, so ore never breaks through the grass or
/// sand skin — it is something you find by digging. Costs nothing in the chunks
/// no lump reaches.
pub fn apply_ore(materials: &mut [MaterialCode], chunk_coord: (i16, i16, i16)) {
    let chunk_start = Vec3::new(
        sample_world_coord(chunk_coord.0, 0),
        sample_world_coord(chunk_coord.1, 0),
        sample_world_coord(chunk_coord.2, 0),
    );
    let chunk_end = chunk_start + Vec3::splat(CHUNK_WORLD_SIZE);
    let last_sample = (SAMPLES_PER_CHUNK_DIM - 1) as f32;
    for_each_patch(chunk_start, chunk_end, |patch| {
        // a sample either side of the exact bound, so rounding cannot drop a
        // sample from one chunk's range that its neighbour keeps
        let low = ((patch.center - Vec3::splat(patch.reach) - chunk_start) / VOXEL_WORLD_SIZE
            - Vec3::ONE)
            .ceil()
            .max(Vec3::ZERO);
        let high = ((patch.center + Vec3::splat(patch.reach) - chunk_start) / VOXEL_WORLD_SIZE
            + Vec3::ONE)
            .floor()
            .min(Vec3::splat(last_sample));
        if low.cmpgt(high).any() {
            return;
        }
        for z in low.z as usize..=high.z as usize {
            let world_z = sample_world_coord(chunk_coord.2, z);
            for y in low.y as usize..=high.y as usize {
                let world_y = sample_world_coord(chunk_coord.1, y);
                for x in low.x as usize..=high.x as usize {
                    let index =
                        flatten_index(x as u32, y as u32, z as u32, SAMPLES_PER_CHUNK_DIM) as usize;
                    if materials[index] != MaterialCode::Dirt {
                        continue;
                    }
                    let world_x = sample_world_coord(chunk_coord.0, x);
                    if patch.contains(Vec3::new(world_x, world_y, world_z)) {
                        materials[index] = MaterialCode::Ore;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::SAMPLES_PER_CHUNK;

    fn ore_count(chunk_coord: (i16, i16, i16)) -> usize {
        let mut materials = vec![MaterialCode::Dirt; SAMPLES_PER_CHUNK];
        apply_ore(&mut materials, chunk_coord);
        materials
            .iter()
            .filter(|m| **m == MaterialCode::Ore)
            .count()
    }

    /// Rarity is about how much of the ground is ore, not how many chunks a
    /// lump grazes — lumps straddle borders, so at any interesting rate nearly
    /// every chunk holds a few samples of one.
    #[test]
    fn ore_is_rare_but_present() {
        let mut ore = 0;
        let mut total = 0;
        for x in -4..4 {
            for y in -4..4 {
                for z in -4..4 {
                    ore += ore_count((x, y, z));
                    total += SAMPLES_PER_CHUNK;
                }
            }
        }
        assert!(ore > 0, "no ore anywhere in {total} samples");
        let fraction = ore as f32 / total as f32;
        assert!(
            fraction < 0.02,
            "ore is {:.2}% of the ground, which is not rare",
            fraction * 100.0
        );
    }

    /// A lump straddling a chunk border must be stamped identically into both
    /// chunks, or the two meshes disagree about the material at the seam.
    #[test]
    fn neighbouring_chunks_agree_on_shared_samples() {
        let span = SAMPLES_PER_CHUNK_DIM - 1;
        for x in -3..3 {
            for y in -3..3 {
                for z in -3..3 {
                    let mut here = vec![MaterialCode::Dirt; SAMPLES_PER_CHUNK];
                    let mut next = vec![MaterialCode::Dirt; SAMPLES_PER_CHUNK];
                    apply_ore(&mut here, (x, y, z));
                    apply_ore(&mut next, (x + 1, y, z));
                    for sample_z in 0..SAMPLES_PER_CHUNK_DIM {
                        for sample_y in 0..SAMPLES_PER_CHUNK_DIM {
                            let here_index = flatten_index(
                                span as u32,
                                sample_y as u32,
                                sample_z as u32,
                                SAMPLES_PER_CHUNK_DIM,
                            ) as usize;
                            let next_index = flatten_index(
                                0,
                                sample_y as u32,
                                sample_z as u32,
                                SAMPLES_PER_CHUNK_DIM,
                            ) as usize;
                            assert_eq!(here[here_index], next[next_index]);
                        }
                    }
                }
            }
        }
    }
}
