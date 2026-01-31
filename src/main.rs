use std::sync::atomic::Ordering;
use std::time::Duration;

use bevy::asset::UnapprovedPathMode;
use bevy::diagnostic::{
    EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::image::ImageSamplerDescriptor;
use bevy::pbr::{ExtendedMaterial, PbrPlugin};
use bevy::prelude::*;
use bevy::window::{PresentMode, WindowMode};
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_rapier3d::plugin::{NoUserData, PhysicsSet, RapierPhysicsPlugin};
// use bevy_rapier3d::render::RapierDebugRenderPlugin;
use fastnoise2::SafeNode;
use fastnoise2::generator::simplex::opensimplex2;
use fastnoise2::generator::{Generator, GeneratorWrapper};
use iyes_perf_ui::PerfUiPlugin;
use iyes_perf_ui::prelude::PerfUiDefaultEntries;

use marching_cubes::data_loader::driver::{
    INITIAL_CHUNKS_LOADED, chunk_spawn_reciever, project_downward, setup_chunk_driver,
};
use marching_cubes::data_loader::file_loader::setup_chunk_loading;
use marching_cubes::lighting::lighting_main::{setup_camera, setup_lighting};
use marching_cubes::player::digging::handle_digging_input;
use marching_cubes::player::player::{
    CameraController, KeyBindings, camera_look, camera_zoom, grab_on_click, handle_focus_change,
    initial_grab_cursor, player_movement, spawn_player, sync_player_mutex, toggle_camera,
};
use marching_cubes::settings::settings_driver::{load_settings, save_monitor_on_move};
use marching_cubes::terrain::terrain::{NoiseFunction, setup_map};
use marching_cubes::terrain::terrain_material::TerrainMaterialExtension;
use marching_cubes::ui::configurable_settings::{FpsLimit, load_configurable_settings};
use marching_cubes::ui::crosshair::spawn_crosshair;
#[cfg(feature = "debug_lines")]
use marching_cubes::ui::debug_lines::{
    draw_cluster_debug, draw_collider_debug, spawn_debug_spheres, update_debug_sphere_positions,
};
use marching_cubes::ui::menu::{menu_toggle, menu_update};
use marching_cubes::ui::minimap::spawn_minimap;

fn main() {
    let settings = load_settings(); //automatically saved state
    let configurable_settings = load_configurable_settings(); //user saved state
    let window_centered_position = settings.window_centered_position;
    let update_mode = match configurable_settings.fps_limit {
        FpsLimit::Fps60 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 60.0)),
        FpsLimit::Fps120 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 120.0)),
        FpsLimit::Unlimited => UpdateMode::Continuous,
    };
    App::new()
        .insert_resource(settings)
        .insert_resource(configurable_settings)
        .insert_resource(KeyBindings::default())
        .insert_resource(CameraController::default())
        .insert_resource(WinitSettings {
            focused_mode: update_mode,
            unfocused_mode: update_mode,
        })
        .insert_resource(NoiseFunction(|| -> GeneratorWrapper<SafeNode> {
            (opensimplex2().fbm(1.0, 0.65, 3, 2.2)).build()
        }()))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                        position: window_centered_position
                            .map(WindowPosition::At)
                            .unwrap_or(WindowPosition::Automatic),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin {
                    default_sampler: ImageSamplerDescriptor {
                        anisotropy_clamp: 16,
                        ..ImageSamplerDescriptor::linear()
                    },
                })
                .set(PbrPlugin { ..default() })
                .set(AssetPlugin {
                    unapproved_path_mode: UnapprovedPathMode::Allow,
                    ..default()
                }),
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
            PerfUiPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>>::default(
            ),
            // RapierDebugRenderPlugin::default(),
        ))
        .add_systems(
            Startup,
            (
                setup_chunk_driver.after(spawn_player),
                setup,
                setup_chunk_loading,
                // generate_large_map_utility.after(setup_chunk_loading),
                setup_map,
                spawn_crosshair,
                spawn_player.after(setup_chunk_loading).after(setup_camera),
                spawn_minimap.after(spawn_player),
                initial_grab_cursor,
                #[cfg(feature = "debug_lines")]
                spawn_debug_spheres.after(spawn_player),
                setup_lighting,
                setup_camera,
            ),
        )
        .add_systems(
            Update,
            (
                handle_digging_input,
                toggle_camera,
                camera_zoom,
                camera_look,
                player_movement,
                sync_player_mutex.after(player_movement),
                chunk_spawn_reciever,
                project_downward
                    .after(PhysicsSet::SyncBackend)
                    .run_if(|| !INITIAL_CHUNKS_LOADED.load(Ordering::Relaxed)),
                save_monitor_on_move,
                #[cfg(feature = "debug_lines")]
                update_debug_sphere_positions,
                #[cfg(feature = "debug_lines")]
                draw_cluster_debug,
                #[cfg(feature = "debug_lines")]
                draw_collider_debug,
                menu_toggle,
                menu_update,
                handle_focus_change,
                grab_on_click,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(PerfUiDefaultEntries::default());
}
