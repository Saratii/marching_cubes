use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroup, Buffer, ComputePipeline, Device, Queue};

use crate::terrain::{chunk_generator::calculate_chunk_start, terrain::SAMPLES_PER_CHUNK};

const WORKGROUP_SIZE: u32 = SAMPLES_PER_CHUNK.div_ceil(64) as u32;
const HEIGHT_BYTES: u64 = (SAMPLES_PER_CHUNK * std::mem::size_of::<f32>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    chunk_start: [f32; 2],
    _pad: [f32; 2],
}

pub struct GpuHeightmapGenerator {
    params_buffer: Buffer,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    queue: Queue,
    device: Device,
    output_buffer: Buffer,
    download_buffer: Buffer,
    shader_module: wgpu::ShaderModule,
}

impl GpuHeightmapGenerator {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to create adapter");
        let info = adapter.get_info();
        println!("Using GPU: {} ({:?})", info.name, info.device_type);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("Failed to create device");
        let shader_source = include_str!("../../assets/shaders/heightmap_compute.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightmap Generator"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: HEIGHT_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download Buffer"),
            size: HEIGHT_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: Some(NonZeroU64::new(HEIGHT_BYTES).unwrap()),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("generate_heightmap"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
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
            shader_module: module,
        }
    }

    pub fn generate_heightmap(&self, chunk_coord: &(i16, i16, i16)) -> Box<[f32]> {
        let chunk_start = calculate_chunk_start(chunk_coord);
        let params = Params {
            chunk_start: [chunk_start.x, chunk_start.y],
            _pad: [0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).unwrap();
        let mapped = slice.get_mapped_range();
        let heights: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.download_buffer.unmap();
        heights.into_boxed_slice()
    }

    pub fn generate_region(
        &self,
        lower: (i16, i16, i16),
        upper: (i16, i16, i16),
    ) -> std::collections::HashMap<(i16, i16, i16), Box<[f32]>> {
        let x_chunks = (upper.0 - lower.0 + 1) as usize;
        let z_chunks = (upper.2 - lower.2 + 1) as usize;
        let total_chunks = x_chunks * z_chunks;
        if total_chunks == 0 {
            return std::collections::HashMap::new();
        }
        let batch_params_size = (std::mem::size_of::<Params>() * total_chunks) as u64;
        let batch_output_size = HEIGHT_BYTES * total_chunks as u64;
        let batch_params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Region Params Buffer"),
            size: batch_params_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let batch_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Region Output Buffer"),
            size: batch_output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let batch_download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Region Download Buffer"),
            size: batch_output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut chunk_coords = Vec::with_capacity(total_chunks);
        let mut params = Vec::with_capacity(total_chunks);
        for x in lower.0..=upper.0 {
            for z in lower.2..=upper.2 {
                let coord = (x, lower.1, z);
                chunk_coords.push(coord);
                let chunk_start = calculate_chunk_start(&coord);
                params.push(Params {
                    chunk_start: [chunk_start.x, chunk_start.y],
                    _pad: [0.0, 0.0],
                });
            }
        }
        self.queue
            .write_buffer(&batch_params_buffer, 0, bytemuck::cast_slice(&params));
        let batch_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: Some(NonZeroU64::new(batch_output_size).unwrap()),
                                has_dynamic_offset: false,
                            },
                            count: None,
                        },
                    ],
                });
        let batch_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_output_buffer.as_entire_binding(),
                },
            ],
        });
        let batch_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&batch_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let batch_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&batch_pipeline_layout),
                    module: &self.shader_module,
                    entry_point: Some("generate_heightmap"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&batch_pipeline);
            compute_pass.set_bind_group(0, &batch_bind_group, &[]);
            compute_pass.dispatch_workgroups(WORKGROUP_SIZE, total_chunks as u32, 1);
        }
        encoder.copy_buffer_to_buffer(
            &batch_output_buffer,
            0,
            &batch_download_buffer,
            0,
            batch_output_size,
        );
        self.queue.submit([encoder.finish()]);
        let slice = batch_download_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).unwrap();
        let mapped = slice.get_mapped_range();
        let all_heights: &[f32] = bytemuck::cast_slice(&mapped);
        let mut results = std::collections::HashMap::with_capacity(total_chunks);
        for (i, coord) in chunk_coords.iter().enumerate() {
            let start = i * SAMPLES_PER_CHUNK;
            let end = start + SAMPLES_PER_CHUNK;
            let heightmap = all_heights[start..end].to_vec().into_boxed_slice();
            results.insert(*coord, heightmap);
        }
        drop(mapped);
        batch_download_buffer.unmap();
        results
    }
}
