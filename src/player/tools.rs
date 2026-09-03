use std::fs::{create_dir_all, read_to_string, write};

use bevy::prelude::*;
use bevy_rapier3d::prelude::{Collider, ColliderMassProperties, Damping, RigidBody, Velocity};

use crate::{
    deformable_terrain::{
        digging::screen_to_world_ray,
        driver::TerrainChunkMap,
        file_loader::get_project_root,
        ore_debris::{OreDebris, debris_physics_bundle, debris_radius},
    },
    player::{
        player::{CameraController, KeyBindings, MainCameraTag},
        player_visual::PlayerHandTag,
    },
    ui::menu::MenuRoot,
};

/// How far the hand can reach to grab a loose rock, in world units.
const HAND_REACH: f32 = 5.0;

/// Extra pick radius around a rock so small ones are not fiddly to click.
const PICK_SLACK: f32 = 0.25;

/// How far ahead of the hand a dropped rock reappears, so it does not spawn
/// inside the player's own collider.
const DROP_OFFSET: f32 = 0.6;

/// Speed a dropped rock is tossed forward at, in world units per second.
const DROP_SPEED: f32 = 2.5;

const TOOL_SAVE_PATH: &str = "data/selected_tool.txt";

/// What left click does, picked with the tool slot keys or cycled with
/// `cycle_tool`.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Deform,
    Chip,
    Smooth,
    Hand,
}

pub const TOOLS: [Tool; 4] = [Tool::Deform, Tool::Chip, Tool::Smooth, Tool::Hand];

impl Tool {
    fn next(self) -> Self {
        TOOLS[(self.slot() + 1) % TOOLS.len()]
    }

    pub fn slot(self) -> usize {
        match self {
            Tool::Deform => 0,
            Tool::Chip => 1,
            Tool::Smooth => 2,
            Tool::Hand => 3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tool::Deform => "Shovel",
            Tool::Chip => "Pickaxe",
            Tool::Smooth => "Trowel",
            Tool::Hand => "Hand",
        }
    }

    pub fn digs(self) -> bool {
        self != Tool::Hand
    }
}

/// On the rock currently in the player's hand. Held rocks keep their
/// `OreDebris` but give up their physics components, so they ride the hand's
/// transform instead of the simulation.
#[derive(Component)]
pub struct HeldOre;

/// The tool the last session ended holding, by slot. An unreadable or stale
/// file just falls back to the default tool.
pub fn load_tool() -> Tool {
    read_to_string(get_project_root().join(TOOL_SAVE_PATH))
        .ok()
        .and_then(|text| text.trim().parse::<usize>().ok())
        .and_then(|slot| TOOLS.get(slot).copied())
        .unwrap_or_default()
}

pub fn save_tool(tool: Res<Tool>) {
    if tool.is_added() || !tool.is_changed() {
        return;
    }
    let path = get_project_root().join(TOOL_SAVE_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    let _ = write(path, tool.slot().to_string());
}

pub fn select_tool(
    keyboard: Res<ButtonInput<KeyCode>>,
    key_bindings: Res<KeyBindings>,
    menu_root_query: Query<&MenuRoot>,
    mut tool: ResMut<Tool>,
) {
    if !menu_root_query.is_empty() {
        return;
    }
    if keyboard.just_pressed(key_bindings.cycle_tool) {
        *tool = tool.next();
    }
    for (slot, key) in key_bindings.tool_slots.iter().enumerate() {
        if keyboard.just_pressed(*key) {
            *tool = TOOLS[slot];
        }
    }
}

pub fn handle_hand_input(
    mouse_input: Res<ButtonInput<MouseButton>>,
    tool: Res<Tool>,
    camera_query: Query<(&Camera, &GlobalTransform), With<MainCameraTag>>,
    window_query: Query<&Window>,
    menu_root_query: Query<&MenuRoot>,
    camera_controller: Res<CameraController>,
    terrain_chunk_map: Res<TerrainChunkMap>,
    hand_query: Query<(Entity, &GlobalTransform), With<PlayerHandTag>>,
    loose_ore: Query<(Entity, &GlobalTransform, &OreDebris), Without<HeldOre>>,
    held_ore: Query<(Entity, &OreDebris), With<HeldOre>>,
    mut was_grabbed: Local<bool>,
    mut commands: Commands,
) {
    if !menu_root_query.is_empty() {
        return;
    }
    // the click that grabs the cursor must not also grab a rock
    let was_already_grabbed = *was_grabbed;
    *was_grabbed = camera_controller.is_cursor_grabbed;
    if *tool != Tool::Hand
        || !camera_controller.is_cursor_grabbed
        || !was_already_grabbed
        || !mouse_input.just_pressed(MouseButton::Left)
    {
        return;
    }
    let Ok((hand, hand_transform)) = hand_query.single() else {
        return;
    };
    let Some((camera, camera_transform)) = camera_query.iter().next() else {
        return;
    };
    let forward = camera_transform.forward().as_vec3();
    if let Ok((rock, debris)) = held_ore.single() {
        let hand_world = hand_transform.compute_transform();
        commands
            .entity(rock)
            .remove::<ChildOf>()
            .remove::<HeldOre>()
            .insert((
                Transform {
                    translation: hand_world.translation + forward * DROP_OFFSET,
                    rotation: hand_world.rotation,
                    scale: Vec3::ONE,
                },
                debris_physics_bundle(
                    debris.volume,
                    Velocity {
                        linear: forward * DROP_SPEED,
                        angular: Vec3::ZERO,
                    },
                ),
            ));
        return;
    }
    let Some(window) = window_query.iter().next() else {
        return;
    };
    let Some(rock) = pick_ore(
        window.size() / 2.0,
        camera,
        camera_transform,
        &terrain_chunk_map,
        &loose_ore,
    ) else {
        return;
    };
    commands
        .entity(rock)
        .remove::<(RigidBody, Collider, ColliderMassProperties, Damping, Velocity)>()
        .insert((
            HeldOre,
            ChildOf(hand),
            Transform::default(),
            // the player mesh is hidden in first person; the rock is not
            Visibility::Visible,
        ));
}

/// Nearest rock the crosshair ray enters within reach, ignoring any that the
/// terrain hides.
fn pick_ore(
    cursor_pos: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    terrain_chunk_map: &TerrainChunkMap,
    loose_ore: &Query<(Entity, &GlobalTransform, &OreDebris), Without<HeldOre>>,
) -> Option<Entity> {
    let ray = camera.viewport_to_world(camera_transform, cursor_pos).ok()?;
    let terrain_limit =
        screen_to_world_ray(cursor_pos, camera, camera_transform, terrain_chunk_map)
            .map(|hit| hit.distance(ray.origin))
            .unwrap_or(f32::INFINITY);
    let mut best: Option<(f32, Entity)> = None;
    for (entity, transform, debris) in loose_ore.iter() {
        let to_center = transform.translation() - ray.origin;
        let along = to_center.dot(*ray.direction);
        if along < 0.0 {
            continue;
        }
        let radius = debris_radius(debris.volume) + PICK_SLACK;
        let perpendicular_squared = (to_center - *ray.direction * along).length_squared();
        if perpendicular_squared > radius * radius {
            continue;
        }
        let distance = (along - (radius * radius - perpendicular_squared).sqrt()).max(0.0);
        if distance > HAND_REACH || distance > terrain_limit {
            continue;
        }
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, entity));
        }
    }
    best.map(|(_, entity)| entity)
}
