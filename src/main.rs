pub mod triangle_table;

use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::window::PresentMode;
use bevy_flycam::{FlyCam, NoCameraPlayerPlugin};
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rand::prelude::*;

use crate::triangle_table::TRIANGLE_TABLE;

const CHUNK_SIZE: u32 = 8; // World size in units (8×8×8 world units)
const VOXEL_RESOLUTION: u32 = 128; // Voxels per dimension (32×32×32 voxels)
const VOXEL_SIZE: f32 = CHUNK_SIZE as f32 / VOXEL_RESOLUTION as f32;

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
            NoCameraPlayerPlugin,
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

    let terrain_params = TerrainParams::default();
    let mut terrain_chunk = TerrainChunk::new(terrain_params.clone());
    terrain_chunk.generate_terrain();
    let chunk_mesh = create_marching_cubes_chunk(&terrain_chunk);
    commands.spawn((
        Mesh3d(meshes.add(chunk_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            cull_mode: None,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        terrain_chunk,
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0., 10., 0.).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam,
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
                            let new_mesh = create_marching_cubes_chunk(&terrain);
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
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 500.0;
    let step_size = 0.1;
    let mut distance_traveled = 0.0;

    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray_direction * distance_traveled;
        if current_pos.x >= 0.0
            && current_pos.x <= VOXEL_RESOLUTION as f32 * VOXEL_SIZE
            && current_pos.z >= 0.0
            && current_pos.z <= VOXEL_RESOLUTION as f32 * VOXEL_SIZE
            && current_pos.y >= 0.0
            && current_pos.y <= VOXEL_RESOLUTION as f32 * VOXEL_SIZE
        {
            let voxel_x = (current_pos.x / VOXEL_SIZE).floor() as u32;
            let voxel_y = (current_pos.y / VOXEL_SIZE).floor() as u32;
            let voxel_z = (current_pos.z / VOXEL_SIZE).floor() as u32;
            if terrain.is_solid(voxel_x, voxel_y, voxel_z) {
                return Some(current_pos);
            }
        }
        distance_traveled += step_size;
    }
    None
}

fn dig_at_position(terrain: &mut TerrainChunk, dig_pos: Vec3) {
    let voxel_x = (dig_pos.x / VOXEL_SIZE).floor() as u32;
    let voxel_y = (dig_pos.y / VOXEL_SIZE).floor() as u32;
    let voxel_z = (dig_pos.z / VOXEL_SIZE).floor() as u32;
    terrain.set_voxel(voxel_x, voxel_y, voxel_z, false);
}

fn create_marching_cubes_chunk(terrain: &TerrainChunk) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    let mut rng = rand::rng();

    // Use VOXEL_RESOLUTION instead of CHUNK_SIZE
    for x in 0..VOXEL_RESOLUTION {
        for y in 0..VOXEL_RESOLUTION {
            for z in 0..VOXEL_RESOLUTION {
                let pos = Vec3::new(
                    x as f32 * VOXEL_SIZE,
                    y as f32 * VOXEL_SIZE,
                    z as f32 * VOXEL_SIZE,
                );
                let cube_index = get_cube_index_voxel(terrain, x, y, z);
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
                    let mut triangle_vertices = Vec::new();
                    for &edge_idx in edge_indices.iter() {
                        let edge_vertex = get_edge_vertex_voxel(terrain, pos, edge_idx);
                        triangle_vertices.push(edge_vertex);
                    }
                    let base_vertex_index = vertices.len() as u32;
                    for vertex in triangle_vertices {
                        vertices.push([vertex.x, vertex.y, vertex.z]);
                        colors.push(triangle_color);
                    }
                    indices.extend_from_slice(&[
                        base_vertex_index,
                        base_vertex_index + 1,
                        base_vertex_index + 2,
                    ]);
                    i += 3;
                }
            }
        }
    }

    if vertices.is_empty() {
        return Mesh::new(PrimitiveTopology::TriangleList, default());
    }

    // Rest of the function remains the same...
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
        .map(|v| {
            [
                v[0] / (VOXEL_RESOLUTION as f32 * VOXEL_SIZE),
                v[2] / (VOXEL_RESOLUTION as f32 * VOXEL_SIZE),
            ]
        })
        .collect();

    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

fn get_cube_index_voxel(terrain: &TerrainChunk, x: u32, y: u32, z: u32) -> u8 {
    let mut cube_index = 0u8;
    let corners = [
        (x, y, z),
        (x + 1, y, z),
        (x + 1, y + 1, z),
        (x, y + 1, z),
        (x, y, z + 1),
        (x + 1, y, z + 1),
        (x + 1, y + 1, z + 1),
        (x, y + 1, z + 1),
    ];
    for (i, (cx, cy, cz)) in corners.iter().enumerate() {
        if !terrain.is_solid(*cx, *cy, *cz) {
            cube_index |= 1 << i;
        }
    }
    cube_index
}

fn get_edge_vertex_voxel(_terrain: &TerrainChunk, pos: Vec3, edge_idx: usize) -> Vec3 {
    let edge_vertices = [
        (Vec3::ZERO, Vec3::new(VOXEL_SIZE, 0.0, 0.0)),
        (
            Vec3::new(VOXEL_SIZE, 0.0, 0.0),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
            Vec3::new(0.0, VOXEL_SIZE, 0.0),
        ),
        (Vec3::new(0.0, VOXEL_SIZE, 0.0), Vec3::ZERO),
        (
            Vec3::new(0.0, 0.0, VOXEL_SIZE),
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
            Vec3::new(0.0, 0.0, VOXEL_SIZE),
        ),
        (Vec3::ZERO, Vec3::new(0.0, 0.0, VOXEL_SIZE)),
        (
            Vec3::new(VOXEL_SIZE, 0.0, 0.0),
            Vec3::new(VOXEL_SIZE, 0.0, VOXEL_SIZE),
        ),
        (
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, 0.0),
            Vec3::new(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
        ),
        (
            Vec3::new(0.0, VOXEL_SIZE, 0.0),
            Vec3::new(0.0, VOXEL_SIZE, VOXEL_SIZE),
        ),
    ];
    let (v1, v2) = edge_vertices[edge_idx];
    let p1 = pos + v1;
    let p2 = pos + v2;

    // Simple midpoint interpolation for binary voxels
    p1.lerp(p2, 0.5)
}

#[derive(Component, Clone, Debug)]
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
            base_height: 16.0,
            amplitude: 8.0,
        }
    }
}

#[derive(Component)]
pub struct TerrainChunk {
    pub params: TerrainParams,
    pub voxels: Vec<bool>,
}

impl TerrainChunk {
    pub fn new(params: TerrainParams) -> Self {
        let total_voxels = (VOXEL_RESOLUTION * VOXEL_RESOLUTION * VOXEL_RESOLUTION) as usize;
        Self {
            params,
            voxels: vec![false; total_voxels],
        }
    }

    pub fn generate_terrain(&mut self) {
        let fbm = Fbm::<Perlin>::new(self.params.seed)
            .set_frequency(self.params.frequency)
            .set_octaves(self.params.octaves)
            .set_lacunarity(self.params.lacunarity)
            .set_persistence(self.params.persistence);

        // Use VOXEL_RESOLUTION instead of CHUNK_SIZE
        for x in 0..VOXEL_RESOLUTION {
            for z in 0..VOXEL_RESOLUTION {
                let world_x = x as f64 * VOXEL_SIZE as f64;
                let world_z = z as f64 * VOXEL_SIZE as f64;
                let noise_val = fbm.get([world_x, world_z]) as f32;
                let terrain_height =
                    (self.params.base_height + noise_val * self.params.amplitude).max(0.0) as u32;

                for y in 0..VOXEL_RESOLUTION {
                    let is_solid = y < terrain_height;
                    self.set_voxel(x, y, z, is_solid);
                }
            }
        }
    }

    fn get_voxel_index(&self, x: u32, y: u32, z: u32) -> usize {
        (z * VOXEL_RESOLUTION * VOXEL_RESOLUTION + y * VOXEL_RESOLUTION + x) as usize
    }

    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        if x >= VOXEL_RESOLUTION || y >= VOXEL_RESOLUTION || z >= VOXEL_RESOLUTION {
            return false;
        }
        let index = self.get_voxel_index(x, y, z);
        self.voxels[index]
    }

    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, solid: bool) {
        if x >= VOXEL_RESOLUTION || y >= VOXEL_RESOLUTION || z >= VOXEL_RESOLUTION {
            return;
        }
        let index = self.get_voxel_index(x, y, z);
        self.voxels[index] = solid;
    }
}
