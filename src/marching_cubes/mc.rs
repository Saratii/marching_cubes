use rustc_hash::{FxBuildHasher, FxHashMap as HashMap};

use crate::marching_cubes::tables::{
    CORNER_OFFSETS, EDGE_ID_OFFSETS, EDGE_VERTICES, TRIANGLE_TABLE,
};

pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
}

impl MeshBuffers {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: Vec::new(),
        }
    }
}

type EdgeKey = u64;

#[inline(always)]
fn make_edge_key(x: u16, y: u16, z: u16, dir: u8) -> u64 {
    (x as u64) | ((y as u64) << 16) | ((z as u64) << 32) | ((dir as u64) << 48)
}

struct VertexCache {
    edge_to_vertex: HashMap<EdgeKey, u32>,
    vertices: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
}

impl VertexCache {
    fn new() -> Self {
        Self {
            edge_to_vertex: HashMap::with_hasher(FxBuildHasher::default()),
            vertices: Vec::new(),
            uvs: Vec::new(),
        }
    }

    fn get_or_create_vertex(&mut self, edge_id: EdgeKey, position: [f32; 3], material: u8) -> u32 {
        *self.edge_to_vertex.entry(edge_id).or_insert_with(|| {
            let vertex_index = self.vertices.len() as u32;
            let uv = encode_material_to_uv(material);
            self.vertices.push(position);
            self.uvs.push(uv);
            vertex_index
        })
    }
}

pub fn mc_mesh_generation(
    mesh_buffers: &mut MeshBuffers,
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    half_extent: f32,
) {
    let cubes_per_chunk_dim = samples_per_chunk_dim - 1;
    let voxel_size = (half_extent * 2.0) / (samples_per_chunk_dim - 1) as f32;
    let mut vertex_cache = VertexCache::new();
    let mut indices = Vec::new();
    let stride = samples_per_chunk_dim * samples_per_chunk_dim;
    for x in 0..cubes_per_chunk_dim {
        let base_x = -half_extent + x as f32 * voxel_size;
        for y in 0..cubes_per_chunk_dim {
            let base_y = -half_extent + y as f32 * voxel_size;
            let other_y_base = y * samples_per_chunk_dim + x;
            for z in 0..cubes_per_chunk_dim {
                let base_z = -half_extent + z as f32 * voxel_size;
                let base = z * stride + other_y_base;
                process_cube_with_cache(
                    x,
                    y,
                    z,
                    base_x,
                    base_y,
                    base_z,
                    base,
                    &mut vertex_cache,
                    &mut indices,
                    densities,
                    materials,
                    samples_per_chunk_dim,
                    voxel_size,
                    stride,
                );
            }
        }
    }
    build_mesh_buffers_from_cache_and_indices(
        mesh_buffers,
        vertex_cache,
        indices,
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    );
}

#[inline]
fn calculate_cube_index(values: &[f32; 8]) -> u8 {
    ((values[0] > 0.0) as u8)
        | (((values[1] > 0.0) as u8) << 1)
        | (((values[2] > 0.0) as u8) << 2)
        | (((values[3] > 0.0) as u8) << 3)
        | (((values[4] > 0.0) as u8) << 4)
        | (((values[5] > 0.0) as u8) << 5)
        | (((values[6] > 0.0) as u8) << 6)
        | (((values[7] > 0.0) as u8) << 7)
}

fn process_cube_with_cache(
    x: usize,
    y: usize,
    z: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    base: usize,
    vertex_cache: &mut VertexCache,
    indices: &mut Vec<u32>,
    densities: &[i16],
    materials: &[u8],
    samples_per_chunk_dim: usize,
    voxel_size: f32,
    stride: usize,
) {
    let cube_values = sample_cube_values_from_sdf(densities, samples_per_chunk_dim, stride, base);
    let cube_index = calculate_cube_index(&cube_values);
    if cube_index == 0 || cube_index == 255 {
        return;
    }
    triangulate_cube_with_cache(
        cube_index,
        &cube_values,
        x,
        y,
        z,
        base_x,
        base_y,
        base_z,
        voxel_size,
        vertex_cache,
        materials,
        samples_per_chunk_dim,
        indices,
        stride,
        base,
    );
}

fn triangulate_cube_with_cache(
    cube_index: u8,
    values: &[f32; 8],
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    vertex_cache: &mut VertexCache,
    materials: &[u8],
    samples_per_chunk_dim: usize,
    indices: &mut Vec<u32>,
    stride: usize,
    base: usize,
) {
    let edge_table = &TRIANGLE_TABLE[cube_index as usize];
    let mut i = 0;
    while edge_table[i] != -1 {
        let v1 = get_or_create_edge_vertex(
            edge_table[i] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertex_cache,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        let v2 = get_or_create_edge_vertex(
            edge_table[i + 1] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertex_cache,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        let v3 = get_or_create_edge_vertex(
            edge_table[i + 2] as usize,
            values,
            base_x,
            base_y,
            base_z,
            voxel_size,
            cube_x,
            cube_y,
            cube_z,
            vertex_cache,
            materials,
            samples_per_chunk_dim,
            stride,
            base,
        );
        indices.extend_from_slice(&[v1, v2, v3]);
        i += 3;
    }
}

#[inline(always)]
fn get_or_create_edge_vertex(
    edge_index: usize,
    values: &[f32; 8],
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    cube_x: usize,
    cube_y: usize,
    cube_z: usize,
    vertex_cache: &mut VertexCache,
    materials: &[u8],
    samples_per_chunk_dim: usize,
    stride: usize,
    base: usize,
) -> u32 {
    let edge_id = get_canonical_edge_id(edge_index, cube_x as u16, cube_y as u16, cube_z as u16);
    let (v1_idx, v2_idx) = EDGE_VERTICES[edge_index];
    if let Some(&idx) = vertex_cache.edge_to_vertex.get(&edge_id) {
        return idx;
    }
    let position =
        interpolate_edge_from_base(v1_idx, v2_idx, base_x, base_y, base_z, voxel_size, values);
    let (dx1, dy1, dz1) = CORNER_OFFSETS[v1_idx];
    let (dx2, dy2, dz2) = CORNER_OFFSETS[v2_idx];
    let material1 = materials[base + dz1 * stride + dy1 * samples_per_chunk_dim + dx1];
    let material2 = materials[base + dz2 * stride + dy2 * samples_per_chunk_dim + dx2];
    let material = if material1 == 2 || material2 == 2 {
        2
    } else if material1 != 0 {
        material1
    } else {
        material2
    };
    vertex_cache.get_or_create_vertex(edge_id, position, material)
}

#[inline(always)]
fn get_canonical_edge_id(edge_index: usize, cube_x: u16, cube_y: u16, cube_z: u16) -> EdgeKey {
    let (dx, dy, dz, dir) = EDGE_ID_OFFSETS[edge_index];
    make_edge_key(cube_x + dx, cube_y + dy, cube_z + dz, dir)
}

#[inline(always)]
fn interpolate_edge_from_base(
    v1_idx: usize,
    v2_idx: usize,
    base_x: f32,
    base_y: f32,
    base_z: f32,
    voxel_size: f32,
    values: &[f32; 8],
) -> [f32; 3] {
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
    let t = if (val2 - val1).abs() < 0.0001 {
        0.5
    } else {
        (-val1 / (val2 - val1)).clamp(0.0, 1.0)
    };
    [
        v1x + t * (v2x - v1x),
        v1y + t * (v2y - v1y),
        v1z + t * (v2z - v1z),
    ]
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

fn calculate_vertex_normal(
    point: [f32; 3],
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
    stride: usize,
) -> [f32; 3] {
    let epsilon = voxel_size;
    let grad_x = sample_sdf_at_point_with_interpolation(
        point[0] + epsilon,
        point[1],
        point[2],
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    ) - sample_sdf_at_point_with_interpolation(
        point[0] - epsilon,
        point[1],
        point[2],
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    );
    let grad_y = sample_sdf_at_point_with_interpolation(
        point[0],
        point[1] + epsilon,
        point[2],
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    ) - sample_sdf_at_point_with_interpolation(
        point[0],
        point[1] - epsilon,
        point[2],
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    );
    let grad_z = sample_sdf_at_point_with_interpolation(
        point[0],
        point[1],
        point[2] + epsilon,
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    ) - sample_sdf_at_point_with_interpolation(
        point[0],
        point[1],
        point[2] - epsilon,
        densities,
        samples_per_chunk_dim,
        half_extent,
        voxel_size,
        stride,
    );
    let len_sq = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
    if len_sq > 0.0 {
        let inv_len = len_sq.sqrt().recip();
        [grad_x * inv_len, grad_y * inv_len, grad_z * inv_len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn sample_sdf_at_point_with_interpolation(
    x: f32,
    y: f32,
    z: f32,
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
    stride: usize,
) -> f32 {
    let voxel_x = (x + half_extent) / voxel_size;
    let voxel_y = (y + half_extent) / voxel_size;
    let voxel_z = (z + half_extent) / voxel_size;
    let max_idx = samples_per_chunk_dim - 1;
    let voxel_x = voxel_x.clamp(0.0, max_idx as f32);
    let voxel_y = voxel_y.clamp(0.0, max_idx as f32);
    let voxel_z = voxel_z.clamp(0.0, max_idx as f32);
    let x0 = voxel_x.floor() as usize;
    let y0 = voxel_y.floor() as usize;
    let z0 = voxel_z.floor() as usize;
    let x1 = (x0 + 1).min(max_idx);
    let y1 = (y0 + 1).min(max_idx);
    let z1 = (z0 + 1).min(max_idx);
    let fx = voxel_x - x0 as f32;
    let fy = voxel_y - y0 as f32;
    let fz = voxel_z - z0 as f32;
    let c000 = densities[z0 * stride + y0 * samples_per_chunk_dim + x0] as f32;
    let c100 = densities[z0 * stride + y0 * samples_per_chunk_dim + x1] as f32;
    let c010 = densities[z0 * stride + y1 * samples_per_chunk_dim + x0] as f32;
    let c110 = densities[z0 * stride + y1 * samples_per_chunk_dim + x1] as f32;
    let c001 = densities[z1 * stride + y0 * samples_per_chunk_dim + x0] as f32;
    let c101 = densities[z1 * stride + y0 * samples_per_chunk_dim + x1] as f32;
    let c011 = densities[z1 * stride + y1 * samples_per_chunk_dim + x0] as f32;
    let c111 = densities[z1 * stride + y1 * samples_per_chunk_dim + x1] as f32;
    let c00 = c000.mul_add(1.0 - fx, c100 * fx);
    let c10 = c010.mul_add(1.0 - fx, c110 * fx);
    let c01 = c001.mul_add(1.0 - fx, c101 * fx);
    let c11 = c011.mul_add(1.0 - fx, c111 * fx);
    let c0 = c00.mul_add(1.0 - fy, c10 * fy);
    let c1 = c01.mul_add(1.0 - fy, c11 * fy);
    c0.mul_add(1.0 - fz, c1 * fz)
}

fn build_mesh_buffers_from_cache_and_indices(
    mesh_buffers: &mut MeshBuffers,
    vertex_cache: VertexCache,
    indices: Vec<u32>,
    densities: &[i16],
    samples_per_chunk_dim: usize,
    half_extent: f32,
    voxel_size: f32,
    stride: usize,
) {
    if vertex_cache.vertices.is_empty() {
        return; //this is sussy
    }
    let mut normals = Vec::with_capacity(vertex_cache.vertices.len());
    for &v in &vertex_cache.vertices {
        normals.push(calculate_vertex_normal(
            v,
            densities,
            samples_per_chunk_dim,
            half_extent,
            voxel_size,
            stride,
        ));
    }
    mesh_buffers.positions = vertex_cache.vertices;
    mesh_buffers.normals = normals;
    mesh_buffers.indices = indices;
    mesh_buffers.uvs = vertex_cache.uvs;
}

#[inline(always)]
fn encode_material_to_uv(material: u8) -> [f32; 2] {
    [material as f32, 0.0]
}
