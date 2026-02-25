use crate::{
    marching_cubes::tables::{CORNER_OFFSETS, EDGE_VERTICES, TRIANGLE_TABLE},
    terrain::{chunk_generator::MaterialCode, terrain_material::TriData},
};

pub fn mc_mesh_generation(
    densities: &[i16],
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    half_extent: f32,
) -> (
    Vec<[f32; 3]>,
    Vec<[f32; 3]>,
    Vec<TriData>,
    Vec<u32>,
    Vec<u32>,
) {
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = (half_extent * 2.0) / (samples_per_chunk_dim - 1) as f32;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut tri_data: Vec<TriData> = Vec::new();
    let mut material_ids = Vec::new();
    let mut material_weights = Vec::new();
    let mut indices = Vec::new();
    let stride = samples_per_chunk_dim * samples_per_chunk_dim;
    for z in 0..cubes_per_chunk_dim {
        let base_z = -half_extent + z as f32 * voxel_size;
        let z_base = z * stride;
        for y in 0..cubes_per_chunk_dim {
            let base_y = -half_extent + y as f32 * voxel_size;
            let yz_base = z_base + y * samples_per_chunk_dim;
            for x in 0..cubes_per_chunk_dim {
                let base_x = -half_extent + x as f32 * voxel_size;
                let base = yz_base + x;
                process_cube_with_cache(
                    base_x,
                    base_y,
                    base_z,
                    base,
                    &mut vertices,
                    &mut normals,
                    &mut material_ids,
                    &mut material_weights,
                    &mut indices,
                    densities,
                    materials,
                    samples_per_chunk_dim,
                    voxel_size,
                    stride,
                    &mut tri_data,
                );
            }
        }
    }
    for normal in normals.iter_mut() {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 0.0001 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }
    let tri_ids: Vec<u32> = (0..vertices.len() as u32 / 3)
        .flat_map(|i| [i, i, i])
        .collect();
    (vertices, normals, tri_data, indices, tri_ids)
}

#[inline(always)]
fn calculate_cube_index(values: &[f32; 8]) -> u8 {
    let mut mask = 0u8;
    for i in 0..8 {
        mask |= (((values[i].to_bits() >> 31) ^ 1) as u8) << i;
    }
    mask
}

fn process_cube_with_cache(
    base_x: f32,
    base_y: f32,
    base_z: f32,
    base: usize,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    material_ids: &mut Vec<[u32; 3]>,
    material_weights: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    densities: &[i16],
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    voxel_size: f32,
    stride: usize,
    tri_data: &mut Vec<TriData>,
) {
    let cube_values = sample_cube_values_from_sdf(densities, samples_per_chunk_dim, stride, base);
    let cube_index = calculate_cube_index(&cube_values);
    if cube_index == 0 || cube_index == 255 {
        return;
    }
    triangulate_cube_with_cache(
        cube_index,
        &cube_values,
        base_x,
        base_y,
        base_z,
        voxel_size,
        vertices,
        normals,
        material_ids,
        material_weights,
        materials,
        samples_per_chunk_dim,
        indices,
        stride,
        base,
        tri_data,
    );
}

fn triangulate_cube_with_cache(
    cube_index: u8,
    values: &[f32; 8],
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    material_ids: &mut Vec<[u32; 3]>,
    material_weights: &mut Vec<[f32; 3]>,
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    indices: &mut Vec<u32>,
    stride: usize,
    base: usize,
    tri_data: &mut Vec<TriData>,
) {
    let edge_table = &TRIANGLE_TABLE[cube_index as usize];
    let mut i = 0;
    while edge_table[i] != -1 {
        let v1 = create_edge_vertex(
            edge_table[i] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            vertices,
            normals,
            material_ids,
            material_weights,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        let v2 = create_edge_vertex(
            edge_table[i + 1] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            vertices,
            normals,
            material_ids,
            material_weights,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        let v3 = create_edge_vertex(
            edge_table[i + 2] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            vertices,
            normals,
            material_ids,
            material_weights,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        let p1 = vertices[v1 as usize];
        let p2 = vertices[v2 as usize];
        let p3 = vertices[v3 as usize];
        let ux = p2[0] - p1[0];
        let uy = p2[1] - p1[1];
        let uz = p2[2] - p1[2];
        let vx = p3[0] - p1[0];
        let vy = p3[1] - p1[1];
        let vz = p3[2] - p1[2];
        let nx = uy * vz - uz * vy;
        let ny = uz * vx - ux * vz;
        let nz = ux * vy - uy * vx;
        let i1 = v1 as usize;
        let i2 = v2 as usize;
        let i3 = v3 as usize;
        let n = &mut normals[i1];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        let n = &mut normals[i2];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        let n = &mut normals[i3];
        n[0] += nx;
        n[1] += ny;
        n[2] += nz;
        let i1 = v1 as usize;
        let i2 = v2 as usize;
        let i3 = v3 as usize;
        let t = TriData {
            ids0: material_ids[i1],
            ids1: material_ids[i2],
            ids2: material_ids[i3],
            w0: material_weights[i1],
            w1: material_weights[i2],
            w2: material_weights[i3],
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        tri_data.push(t);
        indices.push(v1);
        indices.push(v2);
        indices.push(v3);
        i += 3;
    }
}

#[inline(always)]
fn create_edge_vertex(
    edge_index: usize,
    values: &[f32; 8],
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    material_ids: &mut Vec<[u32; 3]>,
    material_weights: &mut Vec<[f32; 3]>,
    materials: &[MaterialCode],
    samples_per_chunk_dim: usize,
    stride: usize,
    base: usize,
) -> u32 {
    let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
    let (position, t) =
        interpolate_edge_from_base(v1_idx, v2_idx, base_x, base_y, base_z, voxel_size, values);
    let (dx1, dy1, dz1) = CORNER_OFFSETS[v1_idx];
    let (dx2, dy2, dz2) = CORNER_OFFSETS[v2_idx];
    let c1 = base + dz1 * stride + dy1 * samples_per_chunk_dim + dx1;
    let c2 = base + dz2 * stride + dy2 * samples_per_chunk_dim + dx2;
    let m1 = materials[c1];
    let m2 = materials[c2];
    let (m1, m2) = match (m1, m2) {
        (MaterialCode::Air, _) => (m2, m2),
        (_, MaterialCode::Air) => (m1, m1),
        _ => (m1, m2),
    };
    let (layer_ids, layer_weights) = edge_vertex_materials(m1, m2, t);
    let idx = vertices.len() as u32;
    vertices.push(position);
    normals.push([0.0, 0.0, 0.0]);
    material_ids.push(layer_ids);
    material_weights.push(layer_weights);
    idx
}

fn edge_vertex_materials(m1: MaterialCode, m2: MaterialCode, t: f32) -> ([u32; 3], [f32; 3]) {
    // one or both corners are solid, air is already skipped by caller
    let l1 = material_to_layer(m1);
    let l2 = material_to_layer(m2);

    if l1 == l2 {
        return ([l1 as u32, 0, 0], [1.0, 0.0, 0.0]);
    }

    // slot 0 = dominant, slot 1 = secondary, paired correctly
    if (1.0 - t) >= t {
        ([l1 as u32, l2 as u32, 0], [1.0 - t, t, 0.0])
    } else {
        ([l2 as u32, l1 as u32, 0], [t, 1.0 - t, 0.0])
    }
}

fn material_to_layer(m: MaterialCode) -> usize {
    match m {
        MaterialCode::Air => unreachable!(),
        MaterialCode::Dirt => 0,
        MaterialCode::Grass => 1,
        MaterialCode::Sand => 2,
    }
}

/// Returns the interpolated position AND the `t` factor used, so the caller
/// can derive material weights without recomputing.
#[inline(always)]
fn interpolate_edge_from_base(
    v1_idx: usize,
    v2_idx: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    values: &[f32; 8],
) -> ([f32; 3], f32) {
    let (dx1, dy1, dz1) = CORNER_OFFSETS[v1_idx];
    let (dx2, dy2, dz2) = CORNER_OFFSETS[v2_idx];
    let v1x = base_x + dx1 as f32 * voxel_size;
    let v1y = base_y + dy1 as f32 * voxel_size;
    let v1z = base_z + dz1 as f32 * voxel_size;
    let v2x = base_x + dx2 as f32 * voxel_size;
    let v2y = base_y + dy2 as f32 * voxel_size;
    let v2z = base_z + dz2 as f32 * voxel_size;
    let val1 = values[v1_idx];
    let val2 = values[v2_idx];
    let denom = val2 - val1;
    let t = if denom.abs() < 0.0001 {
        0.5
    } else {
        (-val1 / denom).clamp(0.0, 1.0)
    };
    (
        [
            v1x + t * (v2x - v1x),
            v1y + t * (v2y - v1y),
            v1z + t * (v2z - v1z),
        ],
        t,
    )
}

#[inline(always)]
fn sample_cube_values_from_sdf(
    densities: &[i16],
    samples_per_chunk_dim: usize,
    stride: usize,
    base: usize,
) -> [f32; 8] {
    [
        densities[base] as f32,
        densities[base + 1] as f32,
        densities[base + samples_per_chunk_dim + 1] as f32,
        densities[base + samples_per_chunk_dim] as f32,
        densities[base + stride] as f32,
        densities[base + stride + 1] as f32,
        densities[base + stride + samples_per_chunk_dim + 1] as f32,
        densities[base + stride + samples_per_chunk_dim] as f32,
    ]
}
