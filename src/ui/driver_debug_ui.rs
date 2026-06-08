use bevy::prelude::*;

use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

pub static QUEUE_SIZE: AtomicUsize = AtomicUsize::new(0);
pub static CHUNK_SPAWN_RECEIVER_QUEUE_SIZE: AtomicUsize = AtomicUsize::new(0);
pub static INTERNAL_QUEUE_SIZES: OnceLock<Box<[AtomicUsize]>> = OnceLock::new();

#[derive(Component)]
pub struct PriorityQueueSizeText;

#[derive(Component)]
pub struct ChunkSpawnReceiverText;

#[derive(Component)]
pub struct InternalQueueSizeText;

pub fn spawn_debug_texts(mut commands: Commands) {
    commands.spawn((
        PriorityQueueSizeText,
        Text::new("Compute Priority Queue Size: 0"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
    commands.spawn((
        ChunkSpawnReceiverText,
        Text::new("Spawn Receiver Queue: 0"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(40.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
    commands.spawn((
        InternalQueueSizeText,
        Text::new("Internal Queues: []"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(70.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

pub fn update_debug_texts(
    mut queue_text: Query<
        &mut Text,
        (
            With<PriorityQueueSizeText>,
            Without<ChunkSpawnReceiverText>,
            Without<InternalQueueSizeText>,
        ),
    >,
    mut spawn_receiver_text: Query<
        &mut Text,
        (
            With<ChunkSpawnReceiverText>,
            Without<PriorityQueueSizeText>,
            Without<InternalQueueSizeText>,
        ),
    >,
    mut internal_queue_text: Query<
        &mut Text,
        (
            With<InternalQueueSizeText>,
            Without<PriorityQueueSizeText>,
            Without<ChunkSpawnReceiverText>,
        ),
    >,
) {
    if let Ok(mut text) = queue_text.single_mut() {
        text.0 = format!(
            "Pending Request Queue: {}",
            QUEUE_SIZE.load(Ordering::Relaxed)
        );
    }
    if let Ok(mut text) = spawn_receiver_text.single_mut() {
        text.0 = format!(
            "Spawn Receiver Queue: {}",
            CHUNK_SPAWN_RECEIVER_QUEUE_SIZE.load(Ordering::Relaxed)
        );
    }
    if let Ok(mut text) = internal_queue_text.single_mut() {
        if let Some(sizes) = INTERNAL_QUEUE_SIZES.get() {
            text.0 = format!(
                "Compute Internal Queues: [{}]",
                sizes
                    .iter()
                    .map(|a| format!("{:>2}", a.load(Ordering::Relaxed)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}
