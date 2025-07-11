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
use std::collections::HashMap;

use crate::triangle_table::TRIANGLE_TABLE;

const CHUNK_SIZE: f32 = 8.0; // World size in units (8×8×8 world units)
const VOXEL_RESOLUTION: u32 = 64; // Voxels per dimension per chunk (32×32×32 voxels)
const VOXEL_SIZE: f32 = CHUNK_SIZE / VOXEL_RESOLUTION as f32;
const CHUNK_CREATION_RADIUS: i32 = 6; // Create chunks within this radius

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
        .insert_resource(ChunkManager::new())
        .add_systems(Startup, (setup, setup_crosshair))
        .add_systems(Update, (handle_digging_input, update_chunks))
        .run();
}

#[derive(Resource)]
pub struct ChunkManager {
    chunks: HashMap<(i32, i32, i32), Entity>,
    terrain_params: TerrainParams,
}

impl ChunkManager {
    fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            terrain_params: TerrainParams::default(),
        }
    }

    fn get_chunk_coord_from_world_pos(world_pos: Vec3) -> (i32, i32, i32) {
        let chunk_x = (world_pos.x / CHUNK_SIZE).floor() as i32;
        let chunk_y = (world_pos.y / CHUNK_SIZE).floor() as i32;
        let chunk_z = (world_pos.z / CHUNK_SIZE).floor() as i32;
        (chunk_x, chunk_y, chunk_z)
    }

    fn get_chunk_center_from_coord(chunk_coord: (i32, i32, i32)) -> Vec3 {
        Vec3::new(
            chunk_coord.0 as f32 * CHUNK_SIZE,
            chunk_coord.1 as f32 * CHUNK_SIZE,
            chunk_coord.2 as f32 * CHUNK_SIZE,
        )
    }

    fn create_chunk(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        chunk_coord: (i32, i32, i32),
    ) {
        if self.chunks.contains_key(&chunk_coord) {
            return;
        }

        let chunk_center = Self::get_chunk_center_from_coord(chunk_coord);
        let mut terrain_chunk = TerrainChunk::new(self.terrain_params.clone(), chunk_coord);
        terrain_chunk.generate_terrain();
        let chunk_mesh = create_marching_cubes_chunk(&terrain_chunk);

        let entity = commands
            .spawn((
                Mesh3d(meshes.add(chunk_mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    cull_mode: None,
                    ..default()
                })),
                Transform::from_translation(chunk_center),
                terrain_chunk,
            ))
            .id();

        self.chunks.insert(chunk_coord, entity);
    }

    fn should_create_chunk(&self, player_pos: Vec3, chunk_coord: (i32, i32, i32)) -> bool {
        let player_chunk = Self::get_chunk_coord_from_world_pos(player_pos);
        let dx = (chunk_coord.0 - player_chunk.0).abs();
        let dy = (chunk_coord.1 - player_chunk.1).abs();
        let dz = (chunk_coord.2 - player_chunk.2).abs();

        dx <= CHUNK_CREATION_RADIUS && dy <= CHUNK_CREATION_RADIUS && dz <= CHUNK_CREATION_RADIUS
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_manager: ResMut<ChunkManager>,
) {
    commands.spawn(PerfUiDefaultEntries::default());

    // Create initial chunk at (0,0,0)
    chunk_manager.create_chunk(&mut commands, &mut meshes, &mut materials, (0, 0, 0));

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

fn update_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_manager: ResMut<ChunkManager>,
    camera_query: Query<&Transform, (With<Camera>, With<FlyCam>)>,
) {
    if let Ok(camera_transform) = camera_query.single() {
        let player_pos = camera_transform.translation;
        let player_chunk = ChunkManager::get_chunk_coord_from_world_pos(player_pos);

        // Check all chunks within radius
        for dx in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
            for dy in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
                for dz in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
                    let chunk_coord = (
                        player_chunk.0 + dx,
                        player_chunk.1 + dy,
                        player_chunk.2 + dz,
                    );

                    if chunk_manager.should_create_chunk(player_pos, chunk_coord) {
                        chunk_manager.create_chunk(
                            &mut commands,
                            &mut meshes,
                            &mut materials,
                            chunk_coord,
                        );
                    }
                }
            }
        }
    }
}

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut terrain_query: Query<(Entity, &mut TerrainChunk, &mut Mesh3d, &Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<(&Camera, &GlobalTransform), (With<Camera>, Without<TerrainChunk>)>,
    windows: Query<&Window>,
) {
    if mouse_input.just_pressed(MouseButton::Left) {
        if let Ok(window) = windows.single() {
            if let Some(cursor_pos) = window.cursor_position() {
                if let Ok((camera, camera_transform)) = camera_query.single() {
                    if let Some((hit_entity, world_pos)) =
                        screen_to_world_ray(cursor_pos, camera, camera_transform, &terrain_query)
                    {
                        // Find the specific chunk that was hit and modify it
                        if let Ok((_, mut terrain, mut mesh_handle, chunk_transform)) =
                            terrain_query.get_mut(hit_entity)
                        {
                            let local_pos = world_pos - chunk_transform.translation;
                            dig_at_position(&mut terrain, local_pos);
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
    terrain_query: &Query<(Entity, &mut TerrainChunk, &mut Mesh3d, &Transform)>,
) -> Option<(Entity, Vec3)> {
    let ray = camera
        .viewport_to_world(camera_transform, cursor_pos)
        .unwrap();
    let ray_origin = ray.origin;
    let ray_direction = ray.direction;
    let max_distance = 100.0;
    let step_size = 0.1;
    let mut distance_traveled = 0.0;

    while distance_traveled < max_distance {
        let current_pos = ray_origin + ray_direction * distance_traveled;

        // Check each chunk to see if the ray hits it
        for (entity, terrain, _, chunk_transform) in terrain_query.iter() {
            let chunk_min = chunk_transform.translation - Vec3::splat(CHUNK_SIZE / 2.0);
            let chunk_max = chunk_transform.translation + Vec3::splat(CHUNK_SIZE / 2.0);

            // Check if current position is within this chunk
            if current_pos.x >= chunk_min.x
                && current_pos.x <= chunk_max.x
                && current_pos.y >= chunk_min.y
                && current_pos.y <= chunk_max.y
                && current_pos.z >= chunk_min.z
                && current_pos.z <= chunk_max.z
            {
                // Convert to local chunk coordinates
                let local_pos = current_pos - chunk_transform.translation;
                let half_chunk = CHUNK_SIZE / 2.0;

                // Convert to voxel coordinates (centered around chunk center)
                let voxel_x = ((local_pos.x + half_chunk) / VOXEL_SIZE).floor() as u32;
                let voxel_y = ((local_pos.y + half_chunk) / VOXEL_SIZE).floor() as u32;
                let voxel_z = ((local_pos.z + half_chunk) / VOXEL_SIZE).floor() as u32;

                if terrain.is_solid(voxel_x, voxel_y, voxel_z) {
                    return Some((entity, current_pos));
                }
            }
        }

        distance_traveled += step_size;
    }
    None
}

fn dig_at_position(terrain: &mut TerrainChunk, local_pos: Vec3) {
    let half_chunk = CHUNK_SIZE / 2.0;
    let voxel_x = ((local_pos.x + half_chunk) / VOXEL_SIZE).floor() as u32;
    let voxel_y = ((local_pos.y + half_chunk) / VOXEL_SIZE).floor() as u32;
    let voxel_z = ((local_pos.z + half_chunk) / VOXEL_SIZE).floor() as u32;
    terrain.set_voxel(voxel_x, voxel_y, voxel_z, false);
}

fn create_marching_cubes_chunk(terrain: &TerrainChunk) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colors = Vec::new();
    let mut rng = rand::rng();

    for x in 0..VOXEL_RESOLUTION {
        for y in 0..VOXEL_RESOLUTION {
            for z in 0..VOXEL_RESOLUTION {
                // Center the voxel positions around the chunk center
                let half_chunk = CHUNK_SIZE / 2.0;
                let pos = Vec3::new(
                    x as f32 * VOXEL_SIZE - half_chunk,
                    y as f32 * VOXEL_SIZE - half_chunk,
                    z as f32 * VOXEL_SIZE - half_chunk,
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

    // If no vertices were generated, return a minimal valid mesh
    if vertices.is_empty() {
        let mut empty_mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        // Add empty attributes to prevent issues with material systems that expect them
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
        empty_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new());
        return empty_mesh;
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
        .map(|v| {
            [
                (v[0] + CHUNK_SIZE / 2.0) / CHUNK_SIZE,
                (v[2] + CHUNK_SIZE / 2.0) / CHUNK_SIZE,
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
    pub chunk_coord: (i32, i32, i32),
    pub voxels: Vec<bool>,
}

impl TerrainChunk {
    pub fn new(params: TerrainParams, chunk_coord: (i32, i32, i32)) -> Self {
        let total_voxels = (VOXEL_RESOLUTION * VOXEL_RESOLUTION * VOXEL_RESOLUTION) as usize;
        Self {
            params,
            chunk_coord,
            voxels: vec![false; total_voxels],
        }
    }

    pub fn generate_terrain(&mut self) {
        let fbm = Fbm::<Perlin>::new(self.params.seed)
            .set_frequency(self.params.frequency)
            .set_octaves(self.params.octaves)
            .set_lacunarity(self.params.lacunarity)
            .set_persistence(self.params.persistence);

        let chunk_world_pos = Vec3::new(
            self.chunk_coord.0 as f32 * CHUNK_SIZE,
            self.chunk_coord.1 as f32 * CHUNK_SIZE,
            self.chunk_coord.2 as f32 * CHUNK_SIZE,
        );

        for x in 0..VOXEL_RESOLUTION {
            for z in 0..VOXEL_RESOLUTION {
                // Calculate world position for this voxel
                let local_x = (x as f32 * VOXEL_SIZE) - (CHUNK_SIZE / 2.0);
                let local_z = (z as f32 * VOXEL_SIZE) - (CHUNK_SIZE / 2.0);
                let world_x = chunk_world_pos.x + local_x;
                let world_z = chunk_world_pos.z + local_z;

                let noise_val = fbm.get([world_x as f64, world_z as f64]) as f32;
                let terrain_height =
                    (self.params.base_height + noise_val * self.params.amplitude).max(0.0);

                for y in 0..VOXEL_RESOLUTION {
                    let local_y = (y as f32 * VOXEL_SIZE) - (CHUNK_SIZE / 2.0);
                    let world_y = chunk_world_pos.y + local_y;
                    let is_solid = world_y < terrain_height;
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
