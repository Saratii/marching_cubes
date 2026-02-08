use std::time::Duration;

use bevy::{
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};

use crate::ui::configurable_settings::{
    ConfigurableSettings, FpsLimit, MenuFocus, MenuTab, SettingsType, save_configurable_settings,
};

const BACKGROUND_COLOR: Color = Color::srgba(0.2, 0.2, 0.3, 0.8);
const HIGHLIGHT_COLOR: Color = Color::srgba(0.8, 0.4, 0.8, 1.0); // Brighter pink for focus
const ACTIVE_TAB_COLOR: Color = Color::srgba(0.4, 0.4, 0.6, 1.0); // Purple for active tab background
const INACTIVE_TAB_COLOR: Color = Color::srgba(0.25, 0.25, 0.4, 1.0); // Darker for inactive
const INACTIVE_BORDER_COLOR: Color = Color::srgba(0.5, 0.5, 0.7, 1.0);
const FONT_SIZE: f32 = 24.0;
const SETTINGS_ROW_HEIGHT: f32 = 40.0;
const SETTINGS_ROW_BORDER_SIZE: f32 = 3.0;
const GENERAL_SETTINGS: [SettingsType; 1] = [SettingsType::FpsChange];
#[cfg(feature = "debug")]
const DEBUG_SETTINGS: [SettingsType; 6] = [
    SettingsType::Lod1Toggle,
    SettingsType::Lod2Toggle,
    SettingsType::Lod3Toggle,
    SettingsType::Lod4Toggle,
    SettingsType::Lod5Toggle,
    SettingsType::ShowChunksToggle,
];

#[derive(Component)]
pub struct SettingLabel(pub SettingsType);

#[derive(Component)]
pub struct SettingRow(pub SettingsType);

#[derive(Resource)]
pub struct SettingsState {
    pub current_tab: MenuTab,
    pub current_focus: MenuFocus,
}

#[derive(Component)]
pub struct MenuRoot;

#[derive(Component)]
pub struct TabButton(MenuTab);

#[derive(Component)]
pub struct TabContent(MenuTab);

#[derive(Component)]
pub struct TabContainer;

pub fn menu_toggle(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu_root_query: Query<Entity, With<MenuRoot>>,
    mut commands: Commands,
    settings: Res<ConfigurableSettings>,
    mut settings_state: ResMut<SettingsState>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        match menu_root_query.iter().next() {
            Some(menu_entity) => {
                commands.entity(menu_entity).despawn();
            }
            None => {
                settings_state.current_focus = MenuFocus::Tabs;
                settings_state.current_tab = MenuTab::General;
                spawn_menu(&mut commands, &settings);
            }
        }
    }
}

pub fn menu_update(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu_query: Query<&MenuRoot>,
    mut settings: ResMut<ConfigurableSettings>,
    mut winit_settings: ResMut<WinitSettings>,
    mut tab_button_query: Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    mut tab_content_query: Query<(&TabContent, &mut Node)>,
    mut setting_row_query: Query<
        (&SettingRow, &mut BorderColor),
        (Without<TabButton>, Without<TabContainer>),
    >,
    mut settings_state: ResMut<SettingsState>,
    mut text_query: Query<(&SettingLabel, &mut Text)>,
) {
    if menu_query.is_empty() {
        return;
    }
    let settings_list: &[SettingsType] = match settings_state.current_tab {
        MenuTab::General => &GENERAL_SETTINGS,
        #[cfg(feature = "debug")]
        MenuTab::Debug => &DEBUG_SETTINGS,
    };
    #[allow(unused_mut)] //wont be unused when a second settings tab other than debug is added
    let mut tab_changed = false;
    let mut focus_changed = false;
    if keyboard.just_pressed(KeyCode::ArrowDown) || keyboard.just_pressed(KeyCode::KeyS) {
        match settings_state.current_focus {
            MenuFocus::Tabs => {
                settings_state.current_focus = MenuFocus::Setting(0);
                focus_changed = true;
            }
            MenuFocus::Setting(index) => {
                if index + 1 < settings_list.len() {
                    settings_state.current_focus = MenuFocus::Setting(index + 1);
                    focus_changed = true;
                }
            }
        }
    } else if keyboard.just_pressed(KeyCode::ArrowUp) || keyboard.just_pressed(KeyCode::KeyW) {
        match settings_state.current_focus {
            MenuFocus::Tabs => {}
            MenuFocus::Setting(index) => {
                if index == 0 {
                    settings_state.current_focus = MenuFocus::Tabs;
                    focus_changed = true;
                } else {
                    settings_state.current_focus = MenuFocus::Setting(index - 1);
                    focus_changed = true;
                }
            }
        }
    }
    let right = keyboard.just_pressed(KeyCode::ArrowRight) || keyboard.just_pressed(KeyCode::KeyD);
    let left = keyboard.just_pressed(KeyCode::ArrowLeft) || keyboard.just_pressed(KeyCode::KeyA);
    if right || left {
        let dir_next = right;
        match settings_state.current_focus {
            MenuFocus::Tabs => {
                #[cfg(feature = "debug")]
                {
                    settings_state.current_tab = match settings_state.current_tab {
                        MenuTab::General => MenuTab::Debug,
                        MenuTab::Debug => MenuTab::General,
                    };
                    tab_changed = true;
                }
            }
            MenuFocus::Setting(index) => {
                let setting = settings_list[index];
                setting.cycle(&mut settings, dir_next);
                save_configurable_settings(&settings);
                if setting == SettingsType::FpsChange {
                    apply_fps_limit(&settings.fps_limit, &mut winit_settings);
                }
                for (SettingLabel(setting_type), mut text) in text_query.iter_mut() {
                    if *setting_type == setting {
                        text.0 = setting_type.text(&settings);
                        break;
                    }
                }
            }
        }
    }
    if tab_changed {
        update_tab_visuals(
            &mut tab_button_query,
            &mut tab_content_query,
            &settings_state,
        );
        settings_state.current_focus = MenuFocus::Tabs;
        focus_changed = true;
    }
    if focus_changed {
        update_focus_visuals(
            &mut tab_button_query,
            &mut setting_row_query,
            &settings_state,
        );
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
                            BorderColor::all(INACTIVE_BORDER_COLOR),
                            TabContainer,
                        ))
                        .with_children(|parent| {
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
                                    BackgroundColor(ACTIVE_TAB_COLOR),
                                    BorderColor::all(HIGHLIGHT_COLOR),
                                    TabButton(MenuTab::General),
                                ))
                                .with_children(|parent| {
                                    parent.spawn((
                                        Text::new("General"),
                                        TextFont {
                                            font_size: FONT_SIZE,
                                            ..default()
                                        },
                                        TextColor(Color::WHITE),
                                    ));
                                });
                            #[cfg(feature = "debug")]
                            {
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
                                        BackgroundColor(INACTIVE_TAB_COLOR),
                                        BorderColor::all(INACTIVE_BORDER_COLOR),
                                        TabButton(MenuTab::Debug),
                                    ))
                                    .with_children(|parent| {
                                        parent.spawn((
                                            Text::new("Debug"),
                                            TextFont {
                                                font_size: FONT_SIZE,
                                                ..default()
                                            },
                                            TextColor(Color::WHITE),
                                        ));
                                    });
                            }
                        });
                    parent
                        .spawn(Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(300.0),
                            padding: UiRect::all(Val::Px(5.0)),
                            flex_direction: FlexDirection::Column,
                            justify_content: JustifyContent::Start,
                            align_items: AlignItems::Start,
                            ..default()
                        })
                        .with_children(|parent| {
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(100.0),
                                        flex_direction: FlexDirection::Column,
                                        justify_content: JustifyContent::Start,
                                        row_gap: Val::Px(5.0),
                                        align_items: AlignItems::Start,
                                        ..default()
                                    },
                                    TabContent(MenuTab::General),
                                ))
                                .with_children(|parent| {
                                    parent
                                        .spawn((
                                            Node {
                                                width: Val::Percent(100.0),
                                                height: Val::Px(SETTINGS_ROW_HEIGHT),
                                                justify_content: JustifyContent::Center,
                                                align_items: AlignItems::Center,
                                                border: UiRect::all(Val::Px(
                                                    SETTINGS_ROW_BORDER_SIZE,
                                                )),
                                                ..default()
                                            },
                                            BorderColor::all(INACTIVE_BORDER_COLOR),
                                            SettingRow(SettingsType::FpsChange),
                                        ))
                                        .with_children(|parent| {
                                            parent.spawn((
                                                SettingLabel(SettingsType::FpsChange),
                                                Text(SettingsType::FpsChange.text(settings)),
                                                TextFont {
                                                    font_size: FONT_SIZE,
                                                    ..default()
                                                },
                                                TextColor(Color::WHITE),
                                            ));
                                        });
                                });
                            #[cfg(feature = "debug")]
                            parent
                                .spawn((
                                    Node {
                                        width: Val::Percent(100.0),
                                        flex_direction: FlexDirection::Column,
                                        justify_content: JustifyContent::Start,
                                        align_items: AlignItems::Start,
                                        display: Display::None,
                                        row_gap: Val::Px(5.0),
                                        ..default()
                                    },
                                    TabContent(MenuTab::Debug),
                                ))
                                .with_children(|parent| {
                                    for &setting_type in DEBUG_SETTINGS.iter() {
                                        let settings_text = setting_type.text(settings);
                                        parent
                                            .spawn((
                                                Node {
                                                    width: Val::Percent(100.0),
                                                    height: Val::Px(SETTINGS_ROW_HEIGHT),
                                                    justify_content: JustifyContent::Center,
                                                    align_items: AlignItems::Center,
                                                    border: UiRect::all(Val::Px(
                                                        SETTINGS_ROW_BORDER_SIZE,
                                                    )),
                                                    ..default()
                                                },
                                                BorderColor::all(INACTIVE_BORDER_COLOR),
                                                SettingRow(setting_type),
                                            ))
                                            .with_children(|parent| {
                                                parent.spawn((
                                                    SettingLabel(setting_type),
                                                    Text(settings_text),
                                                    TextFont {
                                                        font_size: FONT_SIZE,
                                                        ..default()
                                                    },
                                                    TextColor(Color::WHITE),
                                                ));
                                            });
                                    }
                                });
                        });
                });
        });
}

fn update_tab_visuals(
    tab_button_query: &mut Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    tab_content_query: &mut Query<(&TabContent, &mut Node)>,
    settings_state: &SettingsState,
) {
    for (tab_button, mut bg_color, _) in tab_button_query.iter_mut() {
        *bg_color = if tab_button.0 == settings_state.current_tab {
            BackgroundColor(ACTIVE_TAB_COLOR)
        } else {
            BackgroundColor(INACTIVE_TAB_COLOR)
        };
    }
    for (tab_content, mut node) in tab_content_query.iter_mut() {
        node.display = if tab_content.0 == settings_state.current_tab {
            Display::Flex
        } else {
            Display::None
        };
    }
}

fn update_focus_visuals(
    tab_button_query: &mut Query<
        (&TabButton, &mut BackgroundColor, &mut BorderColor),
        (Without<SettingRow>, Without<TabContainer>),
    >,
    setting_row_query: &mut Query<
        (&SettingRow, &mut BorderColor),
        (Without<TabButton>, Without<TabContainer>),
    >,
    settings_state: &SettingsState,
) {
    for (tab_button, _, mut border_color) in tab_button_query.iter_mut() {
        let is_focused = matches!(settings_state.current_focus, MenuFocus::Tabs)
            && tab_button.0 == settings_state.current_tab;
        *border_color = if is_focused {
            BorderColor::all(HIGHLIGHT_COLOR)
        } else {
            BorderColor::all(INACTIVE_BORDER_COLOR)
        };
    }
    let settings_list: &[SettingsType] = match settings_state.current_tab {
        MenuTab::General => &GENERAL_SETTINGS,
        #[cfg(feature = "debug")]
        MenuTab::Debug => &DEBUG_SETTINGS,
    };
    for (setting_row, mut border_color) in setting_row_query.iter_mut() {
        let is_focused = if let MenuFocus::Setting(index) = settings_state.current_focus {
            settings_list[index] == setting_row.0
        } else {
            false
        };
        *border_color = if is_focused {
            BorderColor::all(HIGHLIGHT_COLOR)
        } else {
            BorderColor::all(INACTIVE_BORDER_COLOR)
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
