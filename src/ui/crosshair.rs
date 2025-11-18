use bevy::prelude::*;

const CROSSHAIR_COLOR: Color = Color::srgb(0.4, 0.4, 0.45);

pub fn spawn_crosshair(mut commands: Commands) {
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
                        BackgroundColor(CROSSHAIR_COLOR),
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
                        BackgroundColor(CROSSHAIR_COLOR),
                    ));
                });
        });
}
