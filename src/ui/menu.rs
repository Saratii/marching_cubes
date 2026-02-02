use std::time::Duration;

use bevy::{
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};

use crate::ui::configurable_settings::{
    ConfigurableSettings, FpsLimit, save_configurable_settings,
};

#[derive(Component)]
pub struct MenuRoot;

#[derive(Component)]
pub struct FpsText;

pub fn menu_toggle(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu_query: Query<Entity, With<MenuRoot>>,
    mut commands: Commands,
    settings: Res<ConfigurableSettings>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        if menu_query.is_empty() {
            spawn_menu(&mut commands, &settings);
        } else {
            for entity in menu_query.iter() {
                commands.entity(entity).despawn();
            }
        }
    }
}

pub fn menu_update(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu_query: Query<&MenuRoot>,
    mut settings: ResMut<ConfigurableSettings>,
    mut text_query: Query<&mut Text, With<FpsText>>,
    mut winit_settings: ResMut<WinitSettings>,
) {
    if menu_query.is_empty() {
        return;
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) || keyboard.just_pressed(KeyCode::KeyD) {
        settings.fps_limit = settings.fps_limit.next();
        save_configurable_settings(&settings);
        apply_fps_limit(&settings.fps_limit, &mut winit_settings);
        for mut text in text_query.iter_mut() {
            text.0 = format!("FPS: {}", settings.fps_limit.to_display_string());
        }
    } else if keyboard.just_pressed(KeyCode::ArrowLeft) || keyboard.just_pressed(KeyCode::KeyA) {
        settings.fps_limit = settings.fps_limit.previous();
        save_configurable_settings(&settings);
        apply_fps_limit(&settings.fps_limit, &mut winit_settings);
        for mut text in text_query.iter_mut() {
            text.0 = format!("FPS: {}", settings.fps_limit.to_display_string());
        }
    }
}

fn spawn_menu(commands: &mut Commands, settings: &ConfigurableSettings) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.0)),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        width: Val::Px(400.0),
                        height: Val::Px(300.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.3, 0.3, 0.5, 0.6)),
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text(format!("FPS: {}", settings.fps_limit.to_display_string())),
                        TextFont {
                            font_size: 40.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        FpsText,
                    ));
                });
        });
}

fn apply_fps_limit(fps_limit: &FpsLimit, winit_settings: &mut WinitSettings) {
    let update_mode = match fps_limit {
        FpsLimit::Fps60 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 60.0)),
        FpsLimit::Fps120 => UpdateMode::reactive_low_power(Duration::from_secs_f64(1.0 / 120.0)),
        FpsLimit::Unlimited => UpdateMode::Continuous,
    };
    winit_settings.focused_mode = update_mode;
    winit_settings.unfocused_mode = update_mode;
}
