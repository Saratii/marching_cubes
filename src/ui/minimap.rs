use bevy::{
    camera::{ImageRenderTarget, RenderTarget},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
};

use crate::player::player::PlayerTag;

const MINIMAP_RADIUS_VW: f32 = 8.0; // 8% of viewport width
const BORDER_WIDTH_VW: f32 = 0.3; // 0.3% of viewport width
const BORDER_COLOR: Color = Color::srgb(0.4, 0.4, 0.45);

//depends on player existing from spawn_player
pub fn spawn_minimap(
    player_query: Query<Entity, With<PlayerTag>>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);
    let total_size = MINIMAP_RADIUS_VW * 2.0 + BORDER_WIDTH_VW * 2.0;
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Vw(1.0),
            top: Val::Vw(1.0),
            width: Val::Vw(total_size),
            height: Val::Vw(total_size),
            border: UiRect::all(Val::Vw(BORDER_WIDTH_VW)),
            overflow: Overflow::clip(),
            border_radius: BorderRadius::all(Val::Percent(50.0)),
            ..default()
        })
        .insert(BorderColor::all(BORDER_COLOR))
        .insert(BackgroundColor(Color::NONE))
        .insert(BackgroundColor(BORDER_COLOR))
        .with_children(|parent| {
            parent.spawn((
                ImageNode::new(image_handle.clone()),
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    border_radius: BorderRadius::all(Val::Percent(50.0)),
                    ..default()
                },
            ));
        });
    let child = commands
        .spawn((
            Camera3d { ..default() },
            Transform::from_translation(Vec3::new(0., 150., 0.)).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                order: 1,
                ..default()
            },
            RenderTarget::Image(ImageRenderTarget {
                handle: image_handle.clone(),
                scale_factor: 1.0,
            }),
        ))
        .id();
    commands
        .entity(player_query.iter().next().unwrap())
        .add_child(child);
}
