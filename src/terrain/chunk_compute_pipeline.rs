//Note: The compute shader is currently unused because upload and download to GPU is slower than just blasting 16 CPU cores. However the compute pipeline does function.

use std::num::NonZeroU64;

use bevy::math::Vec3;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use wgpu::{
    BindGroup, Buffer, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, ExperimentalFeatures, MapMode,
    PipelineCompilationOptions, PollType, Queue,
};

use crate::terrain::terrain::{CHUNK_SIZE, HALF_CHUNK, SAMPLES_PER_CHUNK};

const WORKGROUP_SIZE: u32 = SAMPLES_PER_CHUNK.div_ceil(64) as u32;
const DENSITY_BYTES: u64 = (SAMPLES_PER_CHUNK * std::mem::size_of::<i32>()) as u64;
const MATERIAL_BYTES: u64 = (SAMPLES_PER_CHUNK * std::mem::size_of::<u32>()) as u64;
const OUTPUT_BYTES: u64 = DENSITY_BYTES + MATERIAL_BYTES;
const MATERIAL_OFFSET: u64 = DENSITY_BYTES;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    chunk_start: [f32; 3],
    _pad: f32,
}

pub struct GpuTerrainGenerator {
    params_buffer: Buffer,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    queue: Queue,
    device: Device,
    output_buffer: Buffer,
    download_buffer: Buffer,
}

impl GpuTerrainGenerator {
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
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: ExperimentalFeatures::default(),
        }))
        .expect("Failed to create device");
        let shader_source = include_str!("../../assets/shaders/chunk_gen_compute.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Generator"),
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
            size: OUTPUT_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download Buffer"),
            size: OUTPUT_BYTES,
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
                        min_binding_size: Some(NonZeroU64::new(OUTPUT_BYTES).unwrap()),
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
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("generate_terrain"),
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
        }
    }

    pub fn generate_densities(
        &self,
        chunk_coord: &(i16, i16, i16),
    ) -> (Box<[i16]>, Box<[u8]>, bool) {
        let chunk_start = calculate_chunk_start(chunk_coord);
        let params = Params {
            chunk_start: [chunk_start.x, chunk_start.y, chunk_start.z],
            _pad: 0.0,
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
            OUTPUT_BYTES,
        );
        self.queue.submit([encoder.finish()]);
        let slice = self.download_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range();
        let dens_bytes = &mapped[0..DENSITY_BYTES as usize];
        let mat_bytes =
            &mapped[MATERIAL_OFFSET as usize..(MATERIAL_OFFSET + MATERIAL_BYTES) as usize];
        let dens_i32: Vec<i16> = cast_slice(dens_bytes)
            .iter()
            .map(|&d: &i32| d as i16)
            .collect();
        let mats_u32: Vec<u8> = cast_slice(mat_bytes)
            .iter()
            .map(|&m: &u32| m as u8)
            .collect();
        let mut is_uniform = true;
        let first_d = dens_i32[0];
        let first_m = mats_u32[0];
        for i in 1..SAMPLES_PER_CHUNK {
            if dens_i32[i] != first_d || mats_u32[i] != first_m {
                is_uniform = false;
                break;
            }
        }
        drop(mapped);
        self.download_buffer.unmap();
        (
            dens_i32.into_boxed_slice(),
            mats_u32.into_boxed_slice(),
            is_uniform,
        )
    }
}

pub fn calculate_chunk_start(chunk_coord: &(i16, i16, i16)) -> Vec3 {
    Vec3::new(
        chunk_coord.0 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.1 as f32 * CHUNK_SIZE - HALF_CHUNK,
        chunk_coord.2 as f32 * CHUNK_SIZE - HALF_CHUNK,
    )
}
