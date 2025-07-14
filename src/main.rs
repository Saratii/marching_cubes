use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::pbr::PbrPlugin;
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::window::PresentMode;
use bevy_flycam::{FlyCam, NoCameraPlayerPlugin};
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;
use marching_cubes::marching_cubes::{HALF_CHUNK, add_triangle_colors, march_cubes};
use marching_cubes::terrain_generation::{
    CHUNK_SIZE, ChunkMap, ChunkTag, NoiseFunction, VOXEL_SIZE, setup_map,
};

pub const CHUNK_CREATION_RADIUS: i32 = 10; //in chunks
pub const CHUNK_GENERATION_CIRCULAR_RADIUS_SQUARED: f32 =
    (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE) * (CHUNK_CREATION_RADIUS as f32 * CHUNK_SIZE);

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(PbrPlugin { ..default() }),
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin,
            RenderDiagnosticsPlugin,
            SystemInformationDiagnosticsPlugin,
            PerfUiPlugin,
            NoCameraPlayerPlugin,
        ))
        .insert_resource(ClearColor(Color::srgb(0.0, 1.0, 1.0)))
        .add_systems(Startup, (setup, setup_map, setup_crosshair))
        .add_systems(Update, (handle_digging_input, update_chunks))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(PerfUiDefaultEntries::default());
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

//this should ideally only trigger when the player moves across chunk borders
fn update_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_map: ResMut<ChunkMap>,
    camera_query: Query<&Transform, (With<Camera>, With<FlyCam>)>,
    perlin: Res<NoiseFunction>,
) {
    if let Ok(camera_transform) = camera_query.single() {
        let player_chunk = ChunkMap::get_chunk_coord_from_world_pos(camera_transform.translation);
        let radius = CHUNK_CREATION_RADIUS as f32;
        let radius_squared = radius * radius;
        for dx in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
            for dz in -CHUNK_CREATION_RADIUS..=CHUNK_CREATION_RADIUS {
                let xz_dist_sq = (dx * dx + dz * dz) as f32;
                if xz_dist_sq <= radius_squared {
                    let max_dy = (radius_squared - xz_dist_sq).sqrt() as i32;
                    for dy in -max_dy..=max_dy {
                        let chunk_coord = (
                            player_chunk.0 + dx,
                            player_chunk.1 + dy,
                            player_chunk.2 + dz,
                        );
                        if !chunk_map.0.contains_key(&chunk_coord) {
                            let entity = chunk_map.spawn_chunk(
                                &mut commands,
                                &mut meshes,
                                &mut materials,
                                chunk_coord,
                                &perlin.0,
                            );
                            chunk_map.0.insert(chunk_coord, entity);
                        }
                    }
                }
            }
        }
    }
}

fn handle_digging_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut chunk_mesh_query: Query<&mut Mesh3d, With<ChunkTag>>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera>>,
    windows: Query<&Window>,
    mut chunk_map: ResMut<ChunkMap>,
) {
    if mouse_input.just_pressed(MouseButton::Left) {
        if let Ok(window) = windows.single() {
            if let Some(cursor_pos) = window.cursor_position() {
                if let Ok((camera, camera_transform)) = camera_query.single() {
                    if let Some((hit_entity, world_pos, world_position_of_chunk, chunk_coord)) =
                        screen_to_world_ray(cursor_pos, camera, camera_transform, &chunk_map)
                    {
                        let mut mesh_handle = chunk_mesh_query.get_mut(hit_entity).unwrap();
                        let local_pos = world_pos - world_position_of_chunk;
                        let chunk = &mut chunk_map.0.get_mut(&chunk_coord).unwrap().1;
                        chunk.dig_sphere(local_pos, 1.0, 5.0);
                        let new_mesh = add_triangle_colors(march_cubes(&chunk.densities));
                        *mesh_handle = Mesh3d(meshes.add(new_mesh));
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
    chunk_map: &ResMut<ChunkMap>,
) -> Option<(Entity, Vec3, Vec3, (i32, i32, i32))> {
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
        let chunk_coord = ChunkMap::get_chunk_coord_from_world_pos(current_pos);
        let chunk = chunk_map.0.get(&chunk_coord).unwrap();
        let local_pos = current_pos - chunk.1.world_position;
        let voxel_x = ((local_pos.x + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
        let voxel_y = ((local_pos.y + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
        let voxel_z = ((local_pos.z + HALF_CHUNK) / VOXEL_SIZE).floor() as i32;
        if chunk.1.is_solid(voxel_x, voxel_y, voxel_z) {
            return Some((chunk.0, current_pos, chunk.1.world_position, chunk_coord));
        }
        distance_traveled += step_size;
    }
    None
}
