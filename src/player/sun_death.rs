use bevy::prelude::*;

use crate::{
    build_initial_area::{ROOM_DEPTH, SHAFT_RADIUS},
    constants::PLAYER_CUBOID_SIZE,
    deformable_terrain::plugin::TerrainHeightSource,
    elevator::{COLLAR_HEIGHT, platform_climb_speed},
    player::player::{PlayerTag, VerticalVelocity, cave_spawn_position},
};

/// How far off the shaft's axis the sun counts as having caught him. Only the
/// elevator reaches the stretch of shaft this covers, so nothing else trips it.
const DAYLIGHT_RADIUS: f32 = SHAFT_RADIUS + 0.5;
/// Seconds the sun takes to swell from a glare to a total whiteout.
const BLIND_SECONDS: f32 = 1.8;
/// Seconds the whiteout takes to lift once he is back underground.
const FADE_SECONDS: f32 = 1.4;
/// How wide the sun's disc grows, as a multiple of the screen height.
const SUN_MAX_SCREEN_HEIGHTS: f32 = 3.0;
/// Bleached daylight: the glare over the whole screen.
const GLARE_COLOR: Color = Color::srgb(1.0, 0.97, 0.88);
/// The sun itself, hotter and whiter than the sky around it.
const SUN_COLOR: Color = Color::srgb(1.0, 1.0, 0.97);
// above every other HUD layer
const OVERLAY_Z_INDEX: i32 = 3_000_000;

/// Present only while the sun is killing the player. Systems the dwarf should
/// have no say in are gated on its absence.
#[derive(Resource)]
pub struct SunDeath {
    elapsed: f32,
    respawned: bool,
}

#[derive(Component)]
pub struct SunFlashOverlay;

#[derive(Component)]
pub struct SunFlashDisc;

pub fn spawn_sun_flash(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(GLARE_COLOR.with_alpha(0.0)),
            GlobalZIndex(OVERLAY_Z_INDEX),
            Visibility::Hidden,
            SunFlashOverlay,
        ))
        .with_children(|overlay| {
            overlay.spawn((
                Node {
                    width: Val::Vh(0.0),
                    height: Val::Vh(0.0),
                    border_radius: BorderRadius::all(Val::Percent(50.0)),
                    ..default()
                },
                BackgroundColor(SUN_COLOR.with_alpha(0.0)),
                SunFlashDisc,
            ));
        });
}

/// A dwarf who has never seen the sky rides the elevator out into open
/// daylight, and that is the end of him. The glare starts on him while he is
/// still down the shaft — far enough down that the ride takes exactly
/// `BLIND_SECONDS` to lift his head clear of the collar, so the whiteout is
/// total before he ever sees the sun that is killing him.
pub fn trigger_sun_death(
    mut commands: Commands,
    death: Option<Res<SunDeath>>,
    height_source: Res<TerrainHeightSource>,
    player: Query<&Transform, With<PlayerTag>>,
) {
    if death.is_some() {
        return;
    }
    let Ok(transform) = player.single() else {
        return;
    };
    let surface_y = height_source.0.height_at(0.0, 0.0);
    let daylight_y = surface_y + COLLAR_HEIGHT;
    let blind_climb = platform_climb_speed(surface_y - ROOM_DEPTH) * BLIND_SECONDS;
    let head_y = transform.translation.y + PLAYER_CUBOID_SIZE.y / 2.0;
    if head_y >= daylight_y - blind_climb
        && transform.translation.xz().length() <= DAYLIGHT_RADIUS
    {
        commands.insert_resource(SunDeath {
            elapsed: 0.0,
            respawned: false,
        });
    }
}

pub fn update_sun_death(
    mut commands: Commands,
    time: Res<Time>,
    death: Option<ResMut<SunDeath>>,
    height_source: Res<TerrainHeightSource>,
    mut player: Query<(&mut Transform, &mut VerticalVelocity), With<PlayerTag>>,
    mut overlay: Query<
        (&mut BackgroundColor, &mut Visibility),
        (With<SunFlashOverlay>, Without<SunFlashDisc>),
    >,
    mut disc: Query<(&mut Node, &mut BackgroundColor), With<SunFlashDisc>>,
) {
    let Some(mut death) = death else {
        return;
    };
    death.elapsed += time.delta_secs();
    let blinding = (death.elapsed / BLIND_SECONDS).min(1.0);
    if blinding >= 1.0 && !death.respawned {
        death.respawned = true;
        if let Ok((mut transform, mut vertical_velocity)) = player.single_mut() {
            transform.translation = cave_spawn_position(&height_source);
            vertical_velocity.y = 0.0;
        }
    }
    let alpha = if death.respawned {
        let fading = (death.elapsed - BLIND_SECONDS) / FADE_SECONDS;
        if fading >= 1.0 {
            commands.remove_resource::<SunDeath>();
        }
        1.0 - fading.clamp(0.0, 1.0)
    } else {
        //cubed so the glare holds off and then floods all at once
        blinding * blinding * blinding
    };
    if let Ok((mut background, mut visibility)) = overlay.single_mut() {
        *background = BackgroundColor(GLARE_COLOR.with_alpha(alpha));
        *visibility = if alpha > 0.0 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if let Ok((mut node, mut background)) = disc.single_mut() {
        //the sun swells toward him until it is all there is
        let size = blinding * blinding * SUN_MAX_SCREEN_HEIGHTS * 100.0;
        node.width = Val::Vh(size);
        node.height = Val::Vh(size);
        *background =
            BackgroundColor(SUN_COLOR.with_alpha(if death.respawned { 0.0 } else { blinding }));
    }
}
