use wgpu::{util::DeviceExt, BindGroupEntry, Buffer, Device};

pub fn buf_entry(binding: u32, buf: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

pub fn zeroed_storage_buf(device: &Device, label: &str, count: usize) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&vec![0u32; count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

pub fn create_u32_buf(device: &Device, label: &str, value: u32) -> Buffer {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    buffer
}
