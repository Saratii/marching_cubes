use bevy::prelude::*;

use crate::deformable_terrain::digging::DigMode;

const DIG_MODE_TEXT_COLOR: Color = Color::srgb(1.0, 0.95, 0.75);
const DIG_MODE_PANEL_COLOR: Color = Color::srgba(0.0, 0.0, 0.0, 0.55);
const DIG_MODE_FONT_SIZE: FontSize = FontSize::Px(22.0);
const DIG_MODE_MARGIN: f32 = 16.0;
// above the dev diagnostics overlay plane
const DIG_MODE_Z_INDEX: i32 = 2_000_000;

#[derive(Component)]
pub struct DigModeTextTag;

pub fn spawn_dig_mode_text(mut commands: Commands, dig_mode: Res<DigMode>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(DIG_MODE_MARGIN),
                bottom: Val::Px(DIG_MODE_MARGIN),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                ..default()
            },
            BackgroundColor(DIG_MODE_PANEL_COLOR),
            GlobalZIndex(DIG_MODE_Z_INDEX),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new(dig_mode_label(*dig_mode)),
                TextFont {
                    font_size: DIG_MODE_FONT_SIZE,
                    ..default()
                },
                TextColor(DIG_MODE_TEXT_COLOR),
                DigModeTextTag,
            ));
        });
}

pub fn update_dig_mode_text(
    dig_mode: Res<DigMode>,
    mut text_query: Query<&mut Text, With<DigModeTextTag>>,
) {
    if !dig_mode.is_changed() {
        return;
    }
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    **text = dig_mode_label(*dig_mode);
}

fn dig_mode_label(dig_mode: DigMode) -> String {
    format!("Dig Mode: {}", dig_mode.label())
}
