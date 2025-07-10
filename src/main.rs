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
use std::collections::HashMap;

use crate::triangle_table::TRIANGLE_TABLE;

const CHUNK_SIZE: u32 = 16;

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
        .add_systems(Startup, (setup, setup_crosshair))
        .add_systems(Update, handle_digging_input)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(PerfUiDefaultEntries::default());
    let noise_params = TerrainParams::default();
    let chunk_mesh = create_marching_cubes_chunk(&noise_params);
    commands.spawn((
        Mesh3d(meshes.add(chunk_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        TerrainChunk {
            width: 128,
            depth: 128,
            params: noise_params,
            dug_voxels: HashMap::new(),
        },
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

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut terrain_query: Query<(&mut TerrainChunk, &mut Mesh3d)>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<(&Camera, &GlobalTransform), (With<Camera>, Without<TerrainChunk>)>,
    windows: Query<&Window>,
) {
    if mouse_input.just_pressed(MouseButton::Left) {
        if let Ok(window) = windows.single() {
            if let Some(cursor_pos) = window.cursor_position() {
                if let Ok((camera, camera_transform)) = camera_query.single() {
                    if let Ok((mut terrain, mut mesh_handle)) = terrain_query.single_mut() {
                        if let Some(world_pos) =
                            screen_to_world_ray(cursor_pos, camera, camera_transform, &terrain)
                        {
                            dig_at_position(&mut terrain, world_pos);
                            let new_mesh = create_marching_cubes_chunk_with_modifications(
                                &terrain.params,
                                &terrain.dug_voxels,
                            );
                            *mesh_handle = Mesh3d(meshes.add(new_mesh));
                        }
                    }
                }
            }
        }
    }
}

fn setup_crosshair(mut commands: Commands) {
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            position_type: PositionType::Absolute,
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(Node {
                    width: Val::Px(20.0),
                    height: Val::Px(20.0),
                    position_type: PositionType::Relative,
                    ..default()
                })
                .with_children(|crosshair| {
                    crosshair.spawn((
                        Node {
                            width: Val::Px(20.0),
                            height: Val::Px(2.0),
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(9.0),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                    crosshair.spawn((
                        Node {
                            width: Val::Px(2.0),
                            height: Val::Px(20.0),
                            position_type: PositionType::Absolute,
                            left: Val::Px(9.0),
                            top: Val::Px(0.0),
                            ..default()
                        },
                        BackgroundColor(Color::WHITE),
                    ));
                });
        });
}

fn screen_to_world_ray(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    terrain: &TerrainChunk,
) -> Option<Vec3> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let fbm = Fbm::<Perlin>::new(terrain.params.seed)
        .set_frequency(terrain.params.frequency)
        .set_octaves(terrain.params.octaves)
        .set_lacunarity(terrain.params.lacunarity)
        .set_persistence(terrain.params.persistence);
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 500.0;
    let step_size = 0.1;
    let mut current_pos;
    let mut distance_traveled = 0.0;
    while distance_traveled < max_distance {
        current_pos = ray_origin + ray_direction * distance_traveled;
        if current_pos.x >= 0.0
            && current_pos.x <= terrain.width as f32
            && current_pos.z >= 0.0
            && current_pos.z <= terrain.depth as f32
        {
            let sdf_value = terrain_sdf_modified(
                current_pos,
                terrain.width as f32,
                terrain.depth as f32,
                &terrain.params,
                &fbm,
                &terrain.dug_voxels,
            );
            if sdf_value <= 0.0 {
                return Some(current_pos);
            }
        }
        distance_traveled += step_size;
    }
    None
}

fn dig_at_position(terrain: &mut TerrainChunk, dig_pos: Vec3) {
    let dig_depth = 0.1;
    let grid_size = 0.5;
    let grid_x = (dig_pos.x / grid_size).round() as i32;
    let grid_z = (dig_pos.z / grid_size).round() as i32;
    let voxel_pos = VoxelPos {
        x: grid_x,
        z: grid_z,
    };
    let current_depth = terrain.dug_voxels.get(&voxel_pos).unwrap_or(&0.0);
    terrain
        .dug_voxels
        .insert(voxel_pos, current_depth + dig_depth);
}

fn create_marching_cubes_chunk(terrain_params: &TerrainParams) -> Mesh {
    create_marching_cubes_chunk_with_modifications(terrain_params, &HashMap::new())
}

fn create_marching_cubes_chunk_with_modifications(
    terrain_params: &TerrainParams,
    dug_voxels: &HashMap<VoxelPos, f32>,
) -> Mesh {
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
    let resolution_x = (CHUNK_SIZE * base_resolution / 64).max(base_resolution);
    let resolution_y = base_resolution;
    let resolution_z = (CHUNK_SIZE * base_resolution / 64).max(base_resolution);
    let step_x = (CHUNK_SIZE as f32 + 4.0) / resolution_x as f32;
    let step_z = (CHUNK_SIZE as f32 + 4.0) / resolution_z as f32;
    let chunk_height = 10.0;
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
                let cube_index = get_cube_index_terrain_modified(
                    pos,
                    Vec3::new(step_x, step_y, step_z),
                    CHUNK_SIZE as f32,
                    CHUNK_SIZE as f32,
                    &terrain_params,
                    &fbm,
                    dug_voxels,
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
                        let edge_vertex = get_edge_vertex_terrain_modified(
                            pos,
                            Vec3::new(step_x, step_y, step_z),
                            edge_idx,
                            CHUNK_SIZE as f32,
                            CHUNK_SIZE as f32,
                            &terrain_params,
                            &fbm,
                            dug_voxels,
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
        .map(|v| [v[0] / CHUNK_SIZE as f32, v[2] / CHUNK_SIZE as f32])
        .collect();
    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn terrain_sdf_modified(
    pos: Vec3,
    width: f32,
    depth: f32,
    params: &TerrainParams,
    fbm: &Fbm<Perlin>,
    dug_voxels: &HashMap<VoxelPos, f32>,
) -> f32 {
    let noise_val = fbm.get([pos.x as f64, pos.z as f64]) as f32;
    let mut terrain_height = params.base_height + noise_val * params.amplitude;
    let grid_size = 0.5;
    let grid_x = (pos.x / grid_size).round() as i32;
    let grid_z = (pos.z / grid_size).round() as i32;
    let voxel_pos = VoxelPos {
        x: grid_x,
        z: grid_z,
    };
    if let Some(dig_depth) = dug_voxels.get(&voxel_pos) {
        terrain_height -= dig_depth;
    }
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

fn get_cube_index_terrain_modified(
    pos: Vec3,
    step: Vec3,
    width: f32,
    depth: f32,
    terrain_params: &TerrainParams,
    fbm: &Fbm<Perlin>,
    dug_voxels: &HashMap<VoxelPos, f32>,
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
        if terrain_sdf_modified(world_pos, width, depth, terrain_params, fbm, dug_voxels) > 0.0 {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_edge_vertex_terrain_modified(
    pos: Vec3,
    step: Vec3,
    edge_idx: usize,
    width: f32,
    depth: f32,
    terrain_params: &TerrainParams,
    fbm: &Fbm<Perlin>,
    dug_voxels: &HashMap<VoxelPos, f32>,
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
    let val1 = terrain_sdf_modified(p1, width, depth, terrain_params, fbm, dug_voxels);
    let val2 = terrain_sdf_modified(p2, width, depth, terrain_params, fbm, dug_voxels);
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

#[derive(Component)]
pub struct TerrainChunk {
    pub width: u32,
    pub depth: u32,
    pub params: TerrainParams,
    pub dug_voxels: HashMap<VoxelPos, f32>,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub struct VoxelPos {
    pub x: i32,
    pub z: i32,
}
