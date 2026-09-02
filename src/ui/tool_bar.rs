use bevy::{
    ecs::relationship::RelatedSpawnerCommands,
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
};

use crate::{
    deformable_terrain::{file_loader::get_project_root, ore_debris::OreDebris},
    player::tools::{HeldOre, TOOLS, Tool},
};

const ICON_DIR: &str = "assets/icons";
const SLOT_SIZE: f32 = 52.0;
const SLOT_GAP: f32 = 8.0;
const ICON_INSET: f32 = 8.0;
const BAR_MARGIN: f32 = 16.0;
// above the dev diagnostics overlay plane
const BAR_Z_INDEX: i32 = 2_000_000;

const LABEL_COLOR: Color = Color::srgb(1.0, 0.95, 0.75);
const SELECTED_SLOT_COLOR: Color = Color::srgba(0.20, 0.17, 0.09, 0.9);
const SELECTED_BORDER_COLOR: Color = Color::srgb(1.0, 0.82, 0.35);
const SLOT_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const BORDER_COLOR: Color = Color::srgba(1.0, 1.0, 1.0, 0.18);
const SELECTED_ICON_TINT: Color = Color::WHITE;
const ICON_TINT: Color = Color::srgba(0.72, 0.72, 0.72, 0.75);
const SLOT_KEY_COLOR: Color = Color::srgba(1.0, 1.0, 1.0, 0.55);

#[derive(Resource)]
pub struct ToolIcons {
    slots: [Handle<Image>; TOOLS.len()],
    hand_holding: Handle<Image>,
}

#[derive(Component)]
pub struct ToolSlot(Tool);

#[derive(Component)]
pub struct ToolSlotIcon(Tool);

#[derive(Component)]
pub struct ToolBarLabel;

/// Icons are pixel art, so they are point sampled: blowing a 16px tile up to
/// slot size with the default linear filter turns it to mush.
fn load_icon(asset_server: &AssetServer, file_name: &str) -> Handle<Image> {
    asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::nearest();
        })
        .load(get_project_root().join(ICON_DIR).join(file_name))
}

pub fn spawn_tool_bar(mut commands: Commands, asset_server: Res<AssetServer>, tool: Res<Tool>) {
    let icons = ToolIcons {
        slots: [
            load_icon(&asset_server, "shovel.png"),
            load_icon(&asset_server, "pickaxe.png"),
            load_icon(&asset_server, "trowel.png"),
            load_icon(&asset_server, "hand.png"),
        ],
        hand_holding: load_icon(&asset_server, "hand_holding.png"),
    };
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(BAR_MARGIN),
                bottom: Val::Px(BAR_MARGIN),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::FlexEnd,
                row_gap: Val::Px(6.0),
                ..default()
            },
            GlobalZIndex(BAR_Z_INDEX),
        ))
        .with_children(|bar| {
            bar.spawn((
                Text::new(tool.label()),
                TextFont {
                    font_size: FontSize::Px(18.0),
                    ..default()
                },
                TextColor(LABEL_COLOR),
                ToolBarLabel,
            ));
            bar.spawn(Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(SLOT_GAP),
                ..default()
            })
            .with_children(|row| {
                for slot_tool in TOOLS {
                    spawn_slot(row, slot_tool, &icons, *tool);
                }
            });
        });
    commands.insert_resource(icons);
}

fn spawn_slot(
    row: &mut RelatedSpawnerCommands<ChildOf>,
    slot_tool: Tool,
    icons: &ToolIcons,
    selected: Tool,
) {
    let is_selected = slot_tool == selected;
    row.spawn((
        Node {
            width: Val::Px(SLOT_SIZE),
            height: Val::Px(SLOT_SIZE),
            border: UiRect::all(Val::Px(2.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(if is_selected {
            SELECTED_SLOT_COLOR
        } else {
            SLOT_COLOR
        }),
        BorderColor::all(if is_selected {
            SELECTED_BORDER_COLOR
        } else {
            BORDER_COLOR
        }),
        ToolSlot(slot_tool),
    ))
    .with_children(|slot| {
        slot.spawn((
            ImageNode {
                image: icons.slots[slot_tool.slot()].clone(),
                color: if is_selected {
                    SELECTED_ICON_TINT
                } else {
                    ICON_TINT
                },
                image_mode: NodeImageMode::Stretch,
                ..default()
            },
            Node {
                width: Val::Px(SLOT_SIZE - ICON_INSET * 2.0),
                height: Val::Px(SLOT_SIZE - ICON_INSET * 2.0),
                ..default()
            },
            ToolSlotIcon(slot_tool),
        ));
        slot.spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(3.0),
                bottom: Val::Px(0.0),
                ..default()
            },
            Text::new((slot_tool.slot() + 1).to_string()),
            TextFont {
                font_size: FontSize::Px(12.0),
                ..default()
            },
            TextColor(SLOT_KEY_COLOR),
        ));
    });
}

pub fn update_tool_bar(
    tool: Res<Tool>,
    icons: Res<ToolIcons>,
    held_ore: Query<&OreDebris, With<HeldOre>>,
    mut slot_query: Query<(&ToolSlot, &mut BackgroundColor, &mut BorderColor)>,
    mut icon_query: Query<(&ToolSlotIcon, &mut ImageNode)>,
    mut label_query: Query<&mut Text, With<ToolBarLabel>>,
    mut was_holding: Local<Option<bool>>,
) {
    let holding = !held_ore.is_empty();
    if !tool.is_changed() && *was_holding == Some(holding) {
        return;
    }
    *was_holding = Some(holding);
    for (slot, mut background, mut border) in slot_query.iter_mut() {
        let is_selected = slot.0 == *tool;
        *background = BackgroundColor(if is_selected {
            SELECTED_SLOT_COLOR
        } else {
            SLOT_COLOR
        });
        *border = BorderColor::all(if is_selected {
            SELECTED_BORDER_COLOR
        } else {
            BORDER_COLOR
        });
    }
    for (icon, mut image_node) in icon_query.iter_mut() {
        image_node.color = if icon.0 == *tool {
            SELECTED_ICON_TINT
        } else {
            ICON_TINT
        };
        if icon.0 == Tool::Hand {
            image_node.image = if holding {
                icons.hand_holding.clone()
            } else {
                icons.slots[Tool::Hand.slot()].clone()
            };
        }
    }
    if let Ok(mut text) = label_query.single_mut() {
        **text = if *tool == Tool::Hand && holding {
            "Hand (holding ore)".to_string()
        } else {
            tool.label().to_string()
        };
    }
}
