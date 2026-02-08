use rustc_hash::FxHashMap;

use crate::terrain::terrain::Uniformity;

//store ranges of chunk columns that match uniformity instead of storing individial chunks
#[derive(Clone)]
pub struct ColumnRangeMap {
    map: FxHashMap<u32, Vec<ColumnRange>>,
}

impl ColumnRangeMap {
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }

    //given a chunk coordinate, return whether it is contained in any of the column ranges and if so its uniformity
    #[inline(always)]
    pub fn contains(&self, chunk_coord: (i16, i16, i16)) -> Option<Uniformity> {
        self.map
            .get(&pack_xz(chunk_coord.0, chunk_coord.2))?
            .iter()
            .find_map(|r| r.contains(chunk_coord.1).then_some(r.uniformity))
    }

    //pack xz
    //search column ranges for either a containing range or the two ranges who could potentially combine
    //check if the range neighbors the new chunk and has the same uniformity, if so combine them
    //assume impossible for a chunk to be in the middle of an existing range but have different uniformity
    pub fn insert(&mut self, chunk_coord: (i16, i16, i16), uniformity: Uniformity) {
        let xz = pack_xz(chunk_coord.0, chunk_coord.2);
        let column_ranges = self.map.entry(xz).or_default();
        let mut new_low = chunk_coord.1;
        let mut new_high = chunk_coord.1;
        let mut i = 0;
        while i < column_ranges.len() {
            let range = &column_ranges[i];
            if range.contains(chunk_coord.1) {
                return;
            }
            if range.uniformity == uniformity && range.is_adjacent(chunk_coord.1) {
                let range = column_ranges.swap_remove(i);
                new_low = new_low.min(range.low);
                new_high = new_high.max(range.high);
            } else {
                i += 1;
            }
        }
        column_ranges.push(ColumnRange {
            low: new_low,
            high: new_high,
            uniformity,
        });
    }

    //remove a chunk coordinate from the map, splitting ranges if necessary
    //panics if the chunk does not exist or has a different uniformity
    pub fn remove(&mut self, chunk_coord: (i16, i16, i16), uniformity: Uniformity) {
        let xz = pack_xz(chunk_coord.0, chunk_coord.2);
        let column_ranges = self.map.get_mut(&xz).unwrap();
        let y = chunk_coord.1;
        let mut i = 0;
        while i < column_ranges.len() {
            let range = &column_ranges[i];
            if range.contains(y) {
                assert_eq!(range.uniformity, uniformity, "uniformity mismatch");
                let low = range.low;
                let high = range.high;
                let uniformity = range.uniformity;
                column_ranges.swap_remove(i);
                if low < y {
                    column_ranges.push(ColumnRange {
                        low,
                        high: y - 1,
                        uniformity,
                    });
                }
                if y < high {
                    column_ranges.push(ColumnRange {
                        low: y + 1,
                        high,
                        uniformity,
                    });
                }
                if column_ranges.is_empty() {
                    self.map.remove(&xz);
                }
                return;
            }
            i += 1;
        }
        panic!("chunk coordinate not found in map");
    }

    pub fn size_in_bytes(&self) -> usize {
        let mut total = size_of::<Self>();
        for (_, ranges) in &self.map {
            total += size_of::<u32>();
            total += size_of::<Vec<ColumnRange>>();
            total += ranges.capacity() * size_of::<ColumnRange>();
        }
        total += self.map.capacity() * (size_of::<u32>() + size_of::<Vec<ColumnRange>>());
        total
    }
}

#[derive(Clone, Copy)]
struct ColumnRange {
    low: i16,
    high: i16,
    uniformity: Uniformity,
}

impl ColumnRange {
    #[inline(always)]
    fn contains(&self, y: i16) -> bool {
        y >= self.low && y <= self.high
    }

    #[inline(always)]
    fn is_adjacent(&self, y: i16) -> bool {
        y == self.high + 1 || y == self.low - 1
    }
}

#[inline(always)]
fn pack_xz(x: i16, z: i16) -> u32 {
    let ux = (x as u16) as u32;
    let uz = (z as u16) as u32;
    ux | (uz << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_contiguous_into_one_range() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 1, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        for y in 0..=2 {
            assert_eq!(m.contains((0, y, 0)), Some(Uniformity::Air));
        }
        assert_eq!(m.contains((0, 3, 0)), None);
    }

    #[test]
    fn keeps_gaps_separate() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        assert_eq!(m.contains((0, 0, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 1, 0)), None);
        assert_eq!(m.contains((0, 2, 0)), Some(Uniformity::Air));
    }

    #[test]
    fn bridging_insert_merges_two_ranges() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        m.insert((0, 1, 0), Uniformity::Air);
        for y in 0..=2 {
            assert_eq!(m.contains((0, y, 0)), Some(Uniformity::Air));
        }
    }

    #[test]
    fn mixed_uniformity_stays_disjoint() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, -1, 0), Uniformity::Dirt);
        m.insert((0, 0, 0), Uniformity::Dirt);
        m.insert((0, 1, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        assert_eq!(m.contains((0, -1, 0)), Some(Uniformity::Dirt));
        assert_eq!(m.contains((0, 0, 0)), Some(Uniformity::Dirt));
        assert_eq!(m.contains((0, 1, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 2, 0)), Some(Uniformity::Air));
    }

    #[test]
    fn many_columns_work_independently() {
        let mut m = ColumnRangeMap::new();
        for x in -4..=4 {
            for z in -4..=4 {
                for y in -2..=2 {
                    m.insert((x, y, z), Uniformity::Air);
                }
            }
        }
        for x in -4..=4 {
            for z in -4..=4 {
                for y in -2..=2 {
                    assert_eq!(m.contains((x, y, z)), Some(Uniformity::Air));
                }
            }
        }
    }

    #[test]
    fn pathological_insertion_orders_still_merge_correctly() {
        let mut m = ColumnRangeMap::new();
        let ys: [i16; 25] = [
            10, -10, 0, 5, -5, 9, -9, 1, -1, 8, -8, 2, -2, 7, -7, 3, -3, 6, -6, 4, -4, 11, -11, 12,
            -12,
        ];
        for &y in &ys {
            m.insert((0, y, 0), Uniformity::Air);
        }
        for y in -12..=12 {
            assert_eq!(
                m.contains((0, y, 0)),
                Some(Uniformity::Air),
                "missing y={}",
                y
            );
        }
        assert_eq!(m.contains((0, -13, 0)), None);
        assert_eq!(m.contains((0, 13, 0)), None);
    }

    #[test]
    fn pathological_with_two_uniformities_interleaved() {
        let mut m = ColumnRangeMap::new();
        let dirt_ys: [i16; 10] = [-10, -1, -9, -2, -8, -3, -7, -4, -6, -5];
        let air_ys: [i16; 10] = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5];
        for i in 0..10 {
            m.insert((0, dirt_ys[i], 0), Uniformity::Dirt);
            m.insert((0, air_ys[i], 0), Uniformity::Air);
        }
        for y in -10..=-1 {
            assert_eq!(
                m.contains((0, y, 0)),
                Some(Uniformity::Dirt),
                "missing dirt y={}",
                y
            );
        }
        for y in 1..=10 {
            assert_eq!(
                m.contains((0, y, 0)),
                Some(Uniformity::Air),
                "missing air y={}",
                y
            );
        }
        assert_eq!(m.contains((0, 0, 0)), None);
    }

    #[test]
    fn many_disjoint_ranges_then_bridged_in_randomish_order() {
        let mut m = ColumnRangeMap::new();
        for y in (-40..=40).step_by(2) {
            m.insert((0, y, 0), Uniformity::Air);
        }
        for y in (-39..=39).rev().step_by(2) {
            m.insert((0, y, 0), Uniformity::Air);
        }
        for y in -40..=40 {
            assert_eq!(
                m.contains((0, y, 0)),
                Some(Uniformity::Air),
                "missing y={}",
                y
            );
        }
    }

    #[test]
    fn repro_missing_air_1_smaller() {
        let mut m = ColumnRangeMap::new();
        for &y in &[-6, -1, -5, -2, -4, -3] {
            m.insert((0, y, 0), Uniformity::Dirt);
        }
        for &y in &[6, 1, 5, 2, 4, 3] {
            m.insert((0, y, 0), Uniformity::Air);
        }
        assert_eq!(m.contains((0, 1, 0)), Some(Uniformity::Air));
    }

    #[test]
    fn remove_single_chunk_range() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.remove((0, 0, 0), Uniformity::Air);
        assert_eq!(m.contains((0, 0, 0)), None);
    }

    #[test]
    fn remove_from_middle_splits_range() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 1, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        m.remove((0, 1, 0), Uniformity::Air);
        assert_eq!(m.contains((0, 0, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 1, 0)), None);
        assert_eq!(m.contains((0, 2, 0)), Some(Uniformity::Air));
    }

    #[test]
    fn remove_from_start_of_range() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 1, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        m.remove((0, 0, 0), Uniformity::Air);
        assert_eq!(m.contains((0, 0, 0)), None);
        assert_eq!(m.contains((0, 1, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 2, 0)), Some(Uniformity::Air));
    }

    #[test]
    fn remove_from_end_of_range() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.insert((0, 1, 0), Uniformity::Air);
        m.insert((0, 2, 0), Uniformity::Air);
        m.remove((0, 2, 0), Uniformity::Air);
        assert_eq!(m.contains((0, 0, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 1, 0)), Some(Uniformity::Air));
        assert_eq!(m.contains((0, 2, 0)), None);
    }

    #[test]
    #[should_panic(expected = "chunk coordinate not found in map")]
    fn remove_nonexistent_panics() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.remove((0, 5, 0), Uniformity::Air);
    }

    #[test]
    #[should_panic(expected = "uniformity mismatch")]
    fn remove_wrong_uniformity_panics() {
        let mut m = ColumnRangeMap::new();
        m.insert((0, 0, 0), Uniformity::Air);
        m.remove((0, 0, 0), Uniformity::Dirt);
    }
}
