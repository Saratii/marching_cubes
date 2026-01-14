use crate::terrain::{
    chunk_generator::calculate_chunk_start,
    terrain::{CLUSTER_SIZE, SAMPLES_PER_CHUNK},
};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use pollster::block_on;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::{collections::HashMap, num::NonZeroU64};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor, Limits,
    MapMode, MemoryHints, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType,
    PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Trace,
};

const WORKGROUP_SIZE: u32 = SAMPLES_PER_CHUNK.div_ceil(64) as u32;
const NOISE_SAMPLES_DIM: usize = 9;
const NOISE_SAMPLES_PER_CHUNK: usize = NOISE_SAMPLES_DIM * NOISE_SAMPLES_DIM;
const HEIGHT_BYTES: u64 = (NOISE_SAMPLES_PER_CHUNK * size_of::<f32>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    chunk_start: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
    cluster_lower_chunk: [i32; 2],
    _pad: [i32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchClusterParamsHeader {
    cluster_count: u32,
    _pad: [u32; 3],
}

pub struct GpuHeightmapGenerator {
    params_buffer: Buffer,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    queue: Queue,
    device: Device,
    output_buffer: Buffer,
    download_buffer: Buffer,
    cluster_params_buffer: Buffer,
    cluster_bind_group: BindGroup,
    cluster_pipeline: ComputePipeline,
    cluster_output_buffer: Buffer,
    cluster_download_buffer: Buffer,
    batch_cluster_pipeline: ComputePipeline,
}

impl GpuHeightmapGenerator {
    pub fn new() -> Self {
        let instance = Instance::new(&InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to create adapter");
        let info = adapter.get_info();
        println!("Using GPU: {} ({:?})", info.name, info.device_type);
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
            experimental_features: ExperimentalFeatures::default(),
        }))
        .expect("Failed to create device");
        let shader_source = include_str!("../../assets/shaders/heightmap_compute.wgsl");
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Heightmap Generator"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Params Buffer"),
            size: size_of::<Params>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output Buffer"),
            size: HEIGHT_BYTES,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Download Buffer"),
            size: HEIGHT_BYTES,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(NonZeroU64::new(HEIGHT_BYTES).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("generate_heightmap"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let total_chunks = (CLUSTER_SIZE * CLUSTER_SIZE) as usize;
        let cluster_params_size = size_of::<ClusterParams>() as u64;
        let cluster_output_size = HEIGHT_BYTES * total_chunks as u64;
        let cluster_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Cluster Params Buffer"),
            size: cluster_params_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cluster_output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Cluster Output Buffer"),
            size: cluster_output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cluster_download_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Cluster Download Buffer"),
            size: cluster_output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cluster_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            min_binding_size: Some(NonZeroU64::new(cluster_params_size).unwrap()),
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            min_binding_size: Some(NonZeroU64::new(cluster_output_size).unwrap()),
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });
        let cluster_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &cluster_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: cluster_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cluster_output_buffer.as_entire_binding(),
                },
            ],
        });
        let cluster_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout, &cluster_bind_group_layout],
            push_constant_ranges: &[],
        });
        let cluster_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&cluster_pipeline_layout),
            module: &module,
            entry_point: Some("generate_cluster_heightmap"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let batch_cluster_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Batch Cluster Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });
        let batch_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Batch Cluster Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout,
                &cluster_bind_group_layout,
                &batch_cluster_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let batch_cluster_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Batch Cluster Pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &module,
            entry_point: Some("generate_batch_clusters"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            params_buffer,
            bind_group,
            pipeline,
            queue,
            device,
            output_buffer,
            download_buffer,
            cluster_params_buffer,
            cluster_bind_group,
            cluster_pipeline,
            cluster_output_buffer,
            cluster_download_buffer,
            batch_cluster_pipeline,
        }
    }

    pub fn generate_heightmap(&self, chunk_coord: &(i16, i16, i16)) -> Box<[f32]> {
        let chunk_start = calculate_chunk_start(chunk_coord);
        let params = Params {
            chunk_start: [chunk_start.x, chunk_start.y],
            _pad: [0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytes_of(&params));
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(WORKGROUP_SIZE, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.download_buffer,
            0,
            HEIGHT_BYTES,
        );
        self.queue.submit([encoder.finish()]);
        let slice = self.download_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range();
        let heights = cast_slice(&mapped).to_vec();
        drop(mapped);
        self.download_buffer.unmap();
        heights.into_boxed_slice()
    }

    pub fn generate_cluster(
        &self,
        cluster_x: i32,
        cluster_z: i32,
    ) -> HashMap<(i16, i16), Box<[f32]>> {
        let total_chunks = (CLUSTER_SIZE * CLUSTER_SIZE) as usize;
        let cluster_params = ClusterParams {
            cluster_lower_chunk: [cluster_x, cluster_z],
            _pad: [0, 0],
        };
        self.queue
            .write_buffer(&self.cluster_params_buffer, 0, bytes_of(&cluster_params));
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.cluster_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.cluster_bind_group, &[]);
            compute_pass.dispatch_workgroups(WORKGROUP_SIZE, total_chunks as u32, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.cluster_output_buffer,
            0,
            &self.cluster_download_buffer,
            0,
            HEIGHT_BYTES * total_chunks as u64,
        );
        self.queue.submit([encoder.finish()]);
        let slice = self.cluster_download_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range();
        let all_heights: &[f32] = cast_slice(&mapped);
        let mut results = HashMap::with_capacity(total_chunks);
        for local_z in 0..CLUSTER_SIZE {
            for local_x in 0..CLUSTER_SIZE {
                let chunk_coord = (
                    cluster_x as i16 + local_x as i16,
                    cluster_z as i16 + local_z as i16,
                );
                let chunk_index = (local_z * CLUSTER_SIZE + local_x) as usize;
                let start = chunk_index * NOISE_SAMPLES_PER_CHUNK;
                let end = start + NOISE_SAMPLES_PER_CHUNK;
                let heightmap = all_heights[start..end].to_vec().into_boxed_slice();
                results.insert(chunk_coord, heightmap);
            }
        }
        drop(mapped);
        self.cluster_download_buffer.unmap();
        results
    }

    pub fn generate_batch_clusters(
        &self,
        cluster_coords: &[(i32, i32)],
    ) -> FxHashMap<(i16, i16), Box<[f32]>> {
        let cluster_count = cluster_coords.len();
        const CHUNKS_PER_CLUSTER: usize = (CLUSTER_SIZE * CLUSTER_SIZE) as usize;
        let total_chunks = cluster_count * CHUNKS_PER_CLUSTER;
        let header = BatchClusterParamsHeader {
            cluster_count: cluster_count as u32,
            _pad: [0; 3],
        };
        let mut params_data = Vec::new();
        params_data.extend_from_slice(bytes_of(&header));
        for &(x, z) in cluster_coords {
            params_data.extend_from_slice(bytes_of(&[x, z]));
        }
        let params_size = params_data.len() as u64;
        let output_size = HEIGHT_BYTES * total_chunks as u64;
        let batch_params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Batch Cluster Params"),
            size: params_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let batch_output_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Batch Cluster Output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let batch_download_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Batch Cluster Download"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let batch_bind_group_layout =
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Batch Cluster Bind Group Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                min_binding_size: None,
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                    ],
                });
        let batch_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Batch Cluster Bind Group"),
            layout: &batch_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: batch_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: batch_output_buffer.as_entire_binding(),
                },
            ],
        });
        self.queue
            .write_buffer(&batch_params_buffer, 0, &params_data);
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Batch Cluster Generation"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.batch_cluster_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.cluster_bind_group, &[]);
            compute_pass.set_bind_group(2, &batch_bind_group, &[]);
            compute_pass.dispatch_workgroups(
                WORKGROUP_SIZE,
                CHUNKS_PER_CLUSTER as u32,
                cluster_count as u32,
            );
        }
        encoder.copy_buffer_to_buffer(
            &batch_output_buffer,
            0,
            &batch_download_buffer,
            0,
            output_size,
        );
        self.queue.submit([encoder.finish()]);
        let slice = batch_download_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range();
        let all_heights: &[f32] = cast_slice(&mapped);
        let mut results =
            FxHashMap::with_capacity_and_hasher(total_chunks, FxBuildHasher::default());
        for (cluster_idx, &(cluster_x, cluster_z)) in cluster_coords.iter().enumerate() {
            for local_z in 0..CLUSTER_SIZE {
                for local_x in 0..CLUSTER_SIZE {
                    let chunk_in_cluster_idx = (local_z * CLUSTER_SIZE + local_x) as usize;
                    let global_chunk_idx = cluster_idx * CHUNKS_PER_CLUSTER + chunk_in_cluster_idx;
                    let start = global_chunk_idx * NOISE_SAMPLES_PER_CHUNK;
                    let end = start + NOISE_SAMPLES_PER_CHUNK;
                    let heightmap = all_heights[start..end].to_vec().into_boxed_slice();
                    let chunk_x = cluster_x as i16 + local_x as i16;
                    let chunk_z = cluster_z as i16 + local_z as i16;
                    results.insert((chunk_x, chunk_z), heightmap);
                }
            }
        }
        drop(mapped);
        batch_download_buffer.unmap();
        results
    }
}
