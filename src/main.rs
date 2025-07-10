pub mod triangle_table;

use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::window::PresentMode;
use bevy_flycam::PlayerPlugin;
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::prelude::*;

use crate::triangle_table::TRIANGLE_TABLE;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin,
            RenderDiagnosticsPlugin,
            SystemInformationDiagnosticsPlugin,
            PerfUiPlugin,
            PlayerPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(PerfUiDefaultEntries::default());
    let noise_params = TerrainParams::default();
    let chunk_mesh = create_marching_cubes_chunk(128, 128, &noise_params);
    commands.spawn((
        Mesh3d(meshes.add(chunk_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    commands.spawn((
        DirectionalLight { ..default() },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            1.0,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
}

fn create_marching_cubes_chunk(width: u32, depth: u32, terrain_params: &TerrainParams) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    let mut rng = rand::rng();
    let fbm = Fbm::<Perlin>::new(terrain_params.seed)
        .set_frequency(terrain_params.frequency)
        .set_octaves(terrain_params.octaves)
        .set_lacunarity(terrain_params.lacunarity)
        .set_persistence(terrain_params.persistence);
    let base_resolution = 128;
    let resolution_x = (width * base_resolution / 64).max(base_resolution);
    let resolution_y = base_resolution;
    let resolution_z = (depth * base_resolution / 64).max(base_resolution);
    let step_x = (width as f32 + 4.0) / resolution_x as f32;
    let step_z = (depth as f32 + 4.0) / resolution_z as f32;
    let chunk_height = 100.0;
    let step_y = chunk_height / resolution_y as f32;
    let start_y = terrain_params.base_height - (chunk_height / 2.0);
    for x in 0..resolution_x {
        for y in 0..resolution_y {
            for z in 0..resolution_z {
                let pos = Vec3::new(
                    (x as f32 * step_x) - 2.0,
                    start_y + (y as f32 * step_y),
                    (z as f32 * step_z) - 2.0,
                );
                let cube_index = get_cube_index_terrain(
                    pos,
                    Vec3::new(step_x, step_y, step_z),
                    width as f32,
                    depth as f32,
                    &terrain_params,
                    &fbm,
                );
                let triangles = TRIANGLE_TABLE[cube_index as usize];
                let mut i = 0;
                while triangles[i] != -1 {
                    let triangle_color = [
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        rng.random_range(0.2..1.0),
                        1.0,
                    ];

                    let edge_indices = [
                        triangles[i] as usize,
                        triangles[i + 1] as usize,
                        triangles[i + 2] as usize,
                    ];
                    let mut triangle_indices = [0u32; 3];
                    for (j, &edge_idx) in edge_indices.iter().enumerate() {
                        let edge_vertex = get_edge_vertex_terrain(
                            pos,
                            Vec3::new(step_x, step_y, step_z),
                            edge_idx,
                            width as f32,
                            depth as f32,
                            &terrain_params,
                            &fbm,
                        );
                        let vertex_idx = vertices.len() as u32;
                        vertices.push([edge_vertex.x, edge_vertex.y, edge_vertex.z]);
                        colors.push(triangle_color);
                        triangle_indices[j] = vertex_idx;
                    }
                    indices.extend_from_slice(&[
                        triangle_indices[0],
                        triangle_indices[1],
                        triangle_indices[2],
                    ]);
                    i += 3;
                }
            }
        }
    }
    let mut normals = vec![[0.0; 3]; vertices.len()];
    for triangle in indices.chunks(3) {
        let v0 = Vec3::from(vertices[triangle[0] as usize]);
        let v1 = Vec3::from(vertices[triangle[1] as usize]);
        let v2 = Vec3::from(vertices[triangle[2] as usize]);
        let cross = (v1 - v0).cross(v2 - v0);
        if cross.length_squared() > 0.000001 {
            let normal = cross.normalize();
            for &idx in triangle {
                let n = &mut normals[idx as usize];
                n[0] += normal.x;
                n[1] += normal.y;
                n[2] += normal.z;
            }
        }
    }
    for normal in &mut normals {
        let len_squared = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len_squared > 0.0 {
            let len = len_squared.sqrt();
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }
    let uvs: Vec<[f32; 2]> = vertices
        .iter()
        .map(|v| [v[0] / width as f32, v[2] / depth as f32])
        .collect();
    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn terrain_sdf(
    pos: Vec3,
    width: f32,
    depth: f32,
    params: &TerrainParams,
    fbm: &Fbm<Perlin>,
) -> f32 {
    let noise_val = fbm.get([pos.x as f64, pos.z as f64]) as f32;
    let terrain_height = params.base_height + noise_val * params.amplitude;
    let distance_to_surface = pos.y - terrain_height;
    let p_xz = Vec2::new(pos.x, pos.z);
    let closest_point_in_bounds = Vec2::new(pos.x.clamp(0.0, width), pos.z.clamp(0.0, depth));
    let boundary_distance = p_xz.distance(closest_point_in_bounds);
    if boundary_distance > 0.0 {
        (distance_to_surface.powi(2) + boundary_distance.powi(2)).sqrt()
    } else {
        distance_to_surface
    }
}

fn get_cube_index_terrain(
    pos: Vec3,
    step: Vec3,
    width: f32,
    depth: f32,
    terrain_params: &TerrainParams,
    fbm: &Fbm<Perlin>,
) -> u8 {
    let mut cube_index = 0u8;
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(step.x, 0.0, 0.0),
        Vec3::new(step.x, step.y, 0.0),
        Vec3::new(0.0, step.y, 0.0),
        Vec3::new(0.0, 0.0, step.z),
        Vec3::new(step.x, 0.0, step.z),
        Vec3::new(step.x, step.y, step.z),
        Vec3::new(0.0, step.y, step.z),
    ];
    for (i, corner) in corners.iter().enumerate() {
        let world_pos = pos + *corner;
        if terrain_sdf(world_pos, width, depth, terrain_params, fbm) > 0.0 {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_edge_vertex_terrain(
    pos: Vec3,
    step: Vec3,
    edge_idx: usize,
    width: f32,
    depth: f32,
    terrain_params: &TerrainParams,
    fbm: &Fbm<Perlin>,
) -> Vec3 {
    let edge_vertices = [
        (Vec3::ZERO, Vec3::new(step.x, 0.0, 0.0)),
        (Vec3::new(step.x, 0.0, 0.0), Vec3::new(step.x, step.y, 0.0)),
        (Vec3::new(step.x, step.y, 0.0), Vec3::new(0.0, step.y, 0.0)),
        (Vec3::new(0.0, step.y, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 0.0, step.z), Vec3::new(step.x, 0.0, step.z)),
        (
            Vec3::new(step.x, 0.0, step.z),
            Vec3::new(step.x, step.y, step.z),
        ),
        (
            Vec3::new(step.x, step.y, step.z),
            Vec3::new(0.0, step.y, step.z),
        ),
        (Vec3::new(0.0, step.y, step.z), Vec3::new(0.0, 0.0, step.z)),
        (Vec3::ZERO, Vec3::new(0.0, 0.0, step.z)),
        (Vec3::new(step.x, 0.0, 0.0), Vec3::new(step.x, 0.0, step.z)),
        (
            Vec3::new(step.x, step.y, 0.0),
            Vec3::new(step.x, step.y, step.z),
        ),
        (Vec3::new(0.0, step.y, 0.0), Vec3::new(0.0, step.y, step.z)),
    ];
    let (v1, v2) = edge_vertices[edge_idx];
    let p1 = pos + v1;
    let p2 = pos + v2;
    let val1 = terrain_sdf(p1, width, depth, terrain_params, fbm);
    let val2 = terrain_sdf(p2, width, depth, terrain_params, fbm);
    if (val1 * val2) >= 0.0 {
        return p1;
    }
    let t = val1 / (val1 - val2);
    p1.lerp(p2, t)
}

pub struct TerrainParams {
    pub seed: u32,
    pub frequency: f64,
    pub octaves: usize,
    pub lacunarity: f64,
    pub persistence: f64,
    pub base_height: f32,
    pub amplitude: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 100,
            frequency: 0.02,
            octaves: 3,
            lacunarity: 2.1,
            persistence: 0.4,
            base_height: 0.0,
            amplitude: 5.0,
        }
    }
}
