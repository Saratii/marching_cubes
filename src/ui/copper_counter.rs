use bevy::{
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
};

use crate::{deformable_terrain::file_loader::get_project_root, ore_bank::DeliveredCopper};

const ICON_PATH: &str = "assets/icons/copper.png";
const ICON_SIZE: f32 = 34.0;
const TOP_MARGIN: f32 = 14.0;
const PANEL_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const PANEL_BORDER_COLOR: Color = Color::srgba(1.0, 1.0, 1.0, 0.18);
const LABEL_COLOR: Color = Color::srgb(1.0, 0.72, 0.45);
// above the dev diagnostics overlay plane
const COUNTER_Z_INDEX: i32 = 2_000_000;

#[derive(Component)]
pub struct CopperCounterLabel;

fn kilogram_label(kilograms: f32) -> String {
    format!("{kilograms:.1} kg")
}

pub fn spawn_copper_counter(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    delivered: Res<DeliveredCopper>,
) {
    //pixel art, so point sampled: linear filtering turns the 16px tile to mush
    let icon = asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::nearest();
        })
        .load(get_project_root().join(ICON_PATH));
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(TOP_MARGIN),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            GlobalZIndex(COUNTER_Z_INDEX),
        ))
        .with_children(|bar| {
            bar.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(PANEL_COLOR),
                BorderColor::all(PANEL_BORDER_COLOR),
            ))
            .with_children(|panel| {
                panel.spawn((
                    ImageNode {
                        image: icon,
                        image_mode: NodeImageMode::Stretch,
                        ..default()
                    },
                    Node {
                        width: Val::Px(ICON_SIZE),
                        height: Val::Px(ICON_SIZE),
                        ..default()
                    },
                ));
                panel.spawn((
                    Text::new(kilogram_label(delivered.kilograms)),
                    TextFont {
                        font_size: FontSize::Px(22.0),
                        ..default()
                    },
                    TextColor(LABEL_COLOR),
                    CopperCounterLabel,
                ));
            });
        });
}

pub fn update_copper_counter(
    delivered: Res<DeliveredCopper>,
    mut label: Query<&mut Text, With<CopperCounterLabel>>,
) {
    if !delivered.is_changed() {
        return;
    }
    if let Ok(mut text) = label.single_mut() {
        **text = kilogram_label(delivered.kilograms);
    }
}
