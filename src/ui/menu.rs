use std::time::Duration;

use bevy::{
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};

use crate::ui::configurable_settings::{
    ConfigurableSettings, FpsLimit, MenuFocus, MenuTab, save_configurable_settings,
};

const BACKGROUND_COLOR: Color = Color::srgba(0.2, 0.2, 0.3, 0.8); // Dark for backgrounds
const HIGHLIGHT_COLOR: Color = Color::srgba(0.4, 0.4, 0.6, 1.0); // Lighter purple for active tab

#[derive(Component)]
pub struct MenuRoot;

#[derive(Component)]
pub struct FpsText;

#[derive(Component)]
pub struct TabButton(MenuTab);

#[derive(Component)]
pub struct TabContent(MenuTab);

#[derive(Component)]
pub struct SettingRow(usize);

#[derive(Component)]
pub struct TabContainer;

pub fn menu_toggle(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu_query: Query<Entity, With<MenuRoot>>,
    mut commands: Commands,
    mut settings: ResMut<ConfigurableSettings>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        if menu_query.is_empty() {
            settings.current_focus = MenuFocus::Tabs;
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
    mut tab_button_query: Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    mut tab_content_query: Query<(&TabContent, &mut Visibility)>,
    mut setting_row_query: Query<
        (&SettingRow, &mut BorderColor),
        (Without<TabButton>, Without<TabContainer>),
    >,
    mut tab_container_query: Query<&mut BorderColor, With<TabContainer>>,
) {
    if menu_query.is_empty() {
        return;
    }
    let mut tab_changed = false;
    let mut focus_changed = false;
    let mut setting_changed = false;
    if keyboard.just_pressed(KeyCode::ArrowDown) || keyboard.just_pressed(KeyCode::KeyS) {
        match settings.current_focus {
            MenuFocus::Tabs => {
                settings.current_focus = MenuFocus::Setting(0);
                focus_changed = true;
            }
            MenuFocus::Setting(index) => {
                let max_settings = get_max_settings_for_tab(settings.current_tab);
                if index + 1 < max_settings {
                    settings.current_focus = MenuFocus::Setting(index + 1);
                    focus_changed = true;
                }
            }
        }
    } else if keyboard.just_pressed(KeyCode::ArrowUp) || keyboard.just_pressed(KeyCode::KeyW) {
        match settings.current_focus {
            MenuFocus::Tabs => {}
            MenuFocus::Setting(index) => {
                if index == 0 {
                    settings.current_focus = MenuFocus::Tabs;
                    focus_changed = true;
                } else {
                    settings.current_focus = MenuFocus::Setting(index - 1);
                    focus_changed = true;
                }
            }
        }
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) || keyboard.just_pressed(KeyCode::KeyD) {
        match settings.current_focus {
            MenuFocus::Tabs => {
                settings.current_tab = match settings.current_tab {
                    MenuTab::General => {
                        #[cfg(feature = "debug")]
                        {
                            MenuTab::Debug
                        }
                        #[cfg(not(feature = "debug"))]
                        {
                            MenuTab::General
                        }
                    }
                    MenuTab::Debug => MenuTab::General,
                };
                tab_changed = true;
            }
            MenuFocus::Setting(index) => match settings.current_tab {
                MenuTab::General => {
                    if index == 0 {
                        settings.fps_limit = settings.fps_limit.next();
                        save_configurable_settings(&settings);
                        apply_fps_limit(&settings.fps_limit, &mut winit_settings);
                        setting_changed = true;
                    }
                }
                MenuTab::Debug => {}
            },
        }
    } else if keyboard.just_pressed(KeyCode::ArrowLeft) || keyboard.just_pressed(KeyCode::KeyA) {
        match settings.current_focus {
            MenuFocus::Tabs => {
                settings.current_tab = match settings.current_tab {
                    MenuTab::General => {
                        #[cfg(feature = "debug")]
                        {
                            MenuTab::Debug
                        }
                        #[cfg(not(feature = "debug"))]
                        {
                            MenuTab::General
                        }
                    }
                    MenuTab::Debug => MenuTab::General,
                };
                tab_changed = true;
            }
            MenuFocus::Setting(index) => match settings.current_tab {
                MenuTab::General => {
                    if index == 0 {
                        settings.fps_limit = settings.fps_limit.previous();
                        save_configurable_settings(&settings);
                        apply_fps_limit(&settings.fps_limit, &mut winit_settings);
                        setting_changed = true;
                    }
                }
                MenuTab::Debug => {}
            },
        }
    }
    if tab_changed {
        update_tab_visuals(&settings, &mut tab_button_query, &mut tab_content_query);
        settings.current_focus = MenuFocus::Tabs;
        focus_changed = true;
    }
    if focus_changed {
        update_focus_visuals(
            &settings,
            &mut tab_button_query,
            &mut setting_row_query,
            &mut tab_container_query,
        );
    }
    if setting_changed {
        for mut text in text_query.iter_mut() {
            text.0 = format!("FPS: {}", settings.fps_limit.to_display_string());
        }
    }
}

fn get_max_settings_for_tab(tab: MenuTab) -> usize {
    match tab {
        MenuTab::General => 1,
        MenuTab::Debug => 0,
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
                        height: Val::Px(350.0),
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    BackgroundColor(BACKGROUND_COLOR),
                ))
                .with_children(|parent| {
                    parent
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                height: Val::Px(50.0),
                                flex_direction: FlexDirection::Row,
                                border: UiRect::all(Val::Px(3.0)),
                                ..default()
                            },
                            BorderColor::all(Color::srgba(1.0, 0.8, 1.0, 1.0)),
                            TabContainer,
                        ))
                        .with_children(|parent| {
                            let is_active = MenuTab::General == settings.current_tab;
                            let bg_color = if is_active {
                                Color::srgba(0.4, 0.4, 0.6, 1.0)
                            } else {
                                Color::srgba(0.25, 0.25, 0.4, 0.8)
                            };
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(50.0),
                                        height: Val::Percent(100.0),
                                        justify_content: JustifyContent::Center,
                                        align_items: AlignItems::Center,
                                        border: UiRect::all(Val::Px(2.0)),
                                        ..default()
                                    },
                                    BackgroundColor(bg_color),
                                    BorderColor::all(Color::srgba(0.5, 0.5, 0.7, 1.0)),
                                    TabButton(MenuTab::General),
                                ))
                                .with_children(|parent| {
                                    parent.spawn((
                                        Text::new("General"),
                                        TextFont {
                                            font_size: 24.0,
                                            ..default()
                                        },
                                        TextColor(Color::WHITE),
                                    ));
                                });
                            #[cfg(feature = "debug")]
                            {
                                let is_active = MenuTab::Debug == settings.current_tab;
                                let bg_color = if is_active {
                                    Color::srgba(0.4, 0.4, 0.6, 1.0)
                                } else {
                                    Color::srgba(0.25, 0.25, 0.4, 0.8)
                                };
                                parent
                                    .spawn((
                                        Node {
                                            width: Val::Percent(50.0),
                                            height: Val::Percent(100.0),
                                            justify_content: JustifyContent::Center,
                                            align_items: AlignItems::Center,
                                            border: UiRect::all(Val::Px(2.0)),
                                            ..default()
                                        },
                                        BackgroundColor(bg_color),
                                        BorderColor::all(Color::srgba(0.5, 0.5, 0.7, 1.0)),
                                        TabButton(MenuTab::Debug),
                                    ))
                                    .with_children(|parent| {
                                        parent.spawn((
                                            Text::new("Debug"),
                                            TextFont {
                                                font_size: 24.0,
                                                ..default()
                                            },
                                            TextColor(Color::WHITE),
                                        ));
                                    });
                            }
                            #[cfg(not(feature = "debug"))]
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(50.0),
                                        height: Val::Percent(100.0),
                                        justify_content: JustifyContent::Center,
                                        align_items: AlignItems::Center,
                                        border: UiRect::all(Val::Px(2.0)),
                                        ..default()
                                    },
                                    BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.8)),
                                    BorderColor::all(Color::srgba(0.4, 0.4, 0.4, 1.0)),
                                ))
                                .with_children(|parent| {
                                    parent.spawn((
                                        Text::new("Debug 🔒"),
                                        TextFont {
                                            font_size: 24.0,
                                            ..default()
                                        },
                                        TextColor(Color::srgba(0.5, 0.5, 0.5, 1.0)),
                                    ));
                                });
                        });
                    parent
                        .spawn(Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(300.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        })
                        .with_children(|parent| {
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(100.0),
                                        height: Val::Percent(100.0),
                                        flex_direction: FlexDirection::Column,
                                        justify_content: JustifyContent::Center,
                                        align_items: AlignItems::Center,
                                        ..default()
                                    },
                                    Visibility::Visible,
                                    TabContent(MenuTab::General),
                                ))
                                .with_children(|parent| {
                                    parent
                                        .spawn((
                                            Node {
                                                width: Val::Percent(90.0),
                                                height: Val::Px(60.0),
                                                justify_content: JustifyContent::Center,
                                                align_items: AlignItems::Center,
                                                border: UiRect::all(Val::Px(3.0)),
                                                margin: UiRect::all(Val::Px(10.0)),
                                                ..default()
                                            },
                                            BackgroundColor(BACKGROUND_COLOR),
                                            BorderColor::all(Color::srgba(0.5, 0.5, 0.5, 0.5)),
                                            SettingRow(0),
                                        ))
                                        .with_children(|parent| {
                                            parent.spawn((
                                                Text(format!(
                                                    "FPS: {}",
                                                    settings.fps_limit.to_display_string()
                                                )),
                                                TextFont {
                                                    font_size: 32.0,
                                                    ..default()
                                                },
                                                TextColor(Color::WHITE),
                                                FpsText,
                                            ));
                                        });
                                });
                            #[cfg(feature = "debug")]
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(100.0),
                                        height: Val::Percent(100.0),
                                        justify_content: JustifyContent::Center,
                                        align_items: AlignItems::Center,
                                        ..default()
                                    },
                                    Visibility::Hidden,
                                    TabContent(MenuTab::Debug),
                                ))
                                .with_children(|parent| {
                                    parent.spawn((
                                        Text::new("Debug Settings"),
                                        TextFont {
                                            font_size: 40.0,
                                            ..default()
                                        },
                                        TextColor(Color::WHITE),
                                    ));
                                });
                        });
                });
        });
}

fn update_tab_visuals(
    settings: &ConfigurableSettings,
    tab_button_query: &mut Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    tab_content_query: &mut Query<(&TabContent, &mut Visibility)>,
) {
    for (tab_button, mut bg_color, _) in tab_button_query.iter_mut() {
        *bg_color = if tab_button.0 == settings.current_tab {
            BackgroundColor(HIGHLIGHT_COLOR)
        } else {
            BackgroundColor(Color::srgba(0.25, 0.25, 0.4, 0.8))
        };
    }
    for (tab_content, mut visibility) in tab_content_query.iter_mut() {
        *visibility = if tab_content.0 == settings.current_tab {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn update_focus_visuals(
    settings: &ConfigurableSettings,
    tab_button_query: &mut Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    setting_row_query: &mut Query<
        (&SettingRow, &mut BorderColor),
        (Without<TabButton>, Without<TabContainer>),
    >,
    tab_container_query: &mut Query<&mut BorderColor, With<TabContainer>>,
) {
    if let Ok(mut border_color) = tab_container_query.single_mut() {
        *border_color = if matches!(settings.current_focus, MenuFocus::Tabs) {
            BorderColor::all(HIGHLIGHT_COLOR)
        } else {
            BorderColor::all(Color::srgba(0.5, 0.5, 0.7, 0.5))
        };
    }
    for (_, _, mut border_color) in tab_button_query.iter_mut() {
        *border_color = BorderColor::all(Color::srgba(0.5, 0.5, 0.7, 1.0));
    }
    for (setting_row, mut border_color) in setting_row_query.iter_mut() {
        let is_focused = if let MenuFocus::Setting(index) = settings.current_focus {
            setting_row.0 == index
        } else {
            false
        };
        *border_color = if is_focused {
            BorderColor::all(HIGHLIGHT_COLOR)
        } else {
            BorderColor::all(Color::srgba(0.5, 0.5, 0.5, 0.5))
        };
    }
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
