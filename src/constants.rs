use bevy::math::Vec3;

pub const SIMULATION_RADIUS: f32 = 800.0; //in world units. Distance where everything is loaded at all times and physically simulated.
pub const RENDER_RADIUS: f32 = 0.0; //in world units. Distance where chunks are rendered, only meta data should exist at this point besides vertex data for rendering. 
pub const CHUNK_WORLD_SIZE: f32 = 12.0; //in world units, required by noise to be an integer and even
pub const SAMPLES_PER_CHUNK_DIM: usize = 64; // Number of voxel sample points
pub const CHUNKS_PER_CLUSTER_DIM: usize = 3; //number of chunks along one edge of a cluster
pub const REDUCED_LOD_1_RADIUS: f32 = 1000.0; //red
pub const REDUCED_LOD_2_RADIUS: f32 = 2000.0; //cyan
pub const REDUCED_LOD_3_RADIUS: f32 = 5050.0; //blue
pub const REDUCED_LOD_4_RADIUS: f32 = 1150.0; //yellow
pub const REDUCED_LOD_5_RADIUS: f32 = 2600.0; //purple
pub const NOISE_SEED: i32 = 111; // Seed for noise generation
pub const NOISE_FREQUENCY: f32 = 0.0005; // Frequency of the noise
pub const NOISE_AMPLITUDE: f32 = 30.0; // Amplitude of the noise
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0., 0., 0.);
pub const PLAYER_CUBOID_SIZE: Vec3 = Vec3::new(0.5, 1.5, 0.5);
pub const CAMERA_FIRST_PERSON_OFFSET: Vec3 = Vec3::new(0., 0.75 * PLAYER_CUBOID_SIZE.y, 0.);

pub const CLUSTER_WORLD_LENGTH: f32 = CHUNK_WORLD_SIZE * CHUNKS_PER_CLUSTER_DIM as f32;
pub const SAMPLES_PER_CHUNK: usize = SAMPLES_PER_CHUNK_DIM.pow(3);
pub const SAMPLES_PER_CHUNK_2D: usize = SAMPLES_PER_CHUNK_DIM.pow(2);
pub const HALF_CHUNK: f32 = CHUNK_WORLD_SIZE / 2.0;
pub const MAX_RENDER_RADIUS: f32 = SIMULATION_RADIUS.max(RENDER_RADIUS);
pub const VOXEL_WORLD_SIZE: f32 = CHUNK_WORLD_SIZE / (SAMPLES_PER_CHUNK_DIM - 1) as f32;
pub const CHUNKS_PER_CLUSTER: usize = CHUNKS_PER_CLUSTER_DIM.pow(3);
pub const CHUNKS_PER_CLUSTER_2D: usize = CHUNKS_PER_CLUSTER_DIM.pow(2);
pub const MAX_RENDER_RADIUS_SQUARED: f32 = MAX_RENDER_RADIUS * MAX_RENDER_RADIUS;
pub const SIMULATION_RADIUS_SQUARED: f32 = SIMULATION_RADIUS * SIMULATION_RADIUS;
pub const RENDER_RADIUS_SQUARED: f32 = RENDER_RADIUS * RENDER_RADIUS;
pub const REDUCED_LOD_1_RADIUS_SQUARED: f32 = REDUCED_LOD_1_RADIUS * REDUCED_LOD_1_RADIUS;
pub const REDUCED_LOD_2_RADIUS_SQUARED: f32 = REDUCED_LOD_2_RADIUS * REDUCED_LOD_2_RADIUS;
pub const REDUCED_LOD_3_RADIUS_SQUARED: f32 = REDUCED_LOD_3_RADIUS * REDUCED_LOD_3_RADIUS;
pub const REDUCED_LOD_4_RADIUS_SQUARED: f32 = REDUCED_LOD_4_RADIUS * REDUCED_LOD_4_RADIUS;
pub const REDUCED_LOD_5_RADIUS_SQUARED: f32 = REDUCED_LOD_5_RADIUS * REDUCED_LOD_5_RADIUS;
