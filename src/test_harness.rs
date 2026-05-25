use wgpu::{self, util::DeviceExt, Buffer, BufferUsages, Device, Maintain, MapMode, Queue};

use crate::ComputeStepTrait;

pub struct GpuTestHarness {
    pub device: Device,
    pub queue: Queue,
}

impl GpuTestHarness {
    pub fn new() -> Self {
        pollster::block_on(Self::init())
    }

    async fn init() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no wgpu adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 12,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("failed to create wgpu device");

        Self { device, queue }
    }

    /// Storage buffer initialized with `data`, readable by GPU and copyable to staging.
    pub fn storage_buf(&self, data: &[u32]) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
    }

    /// Zero-filled storage buffer of `count` u32s.
    pub fn zeroed_buf(&self, count: usize) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vec![0u32; count]),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
    }

    /// Single-u32 storage buffer — convenient for uniforms passed as storage bindings.
    pub fn scalar_buf(&self, value: u32) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&value),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
    }

    /// Dispatch a compute step and wait for the GPU to finish.
    pub fn run_step<T: ComputeStepTrait>(&self, step: &T) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            step.dispatch(&mut pass);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(Maintain::Wait);
    }

    /// Dispatch, then read back the step's result buffer. Consumes the step's result.
    pub fn run_and_readback<T: ComputeStepTrait>(&self, step: &mut T) -> Vec<u32> {
        self.run_step(step);
        self.readback(&step.take_result())
    }

    /// Copy a storage buffer back to the CPU and return its contents as `Vec<u32>`.
    pub fn readback(&self, buf: &Buffer) -> Vec<u32> {
        let size = buf.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(Maintain::Wait);

        let mapped = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }
}
