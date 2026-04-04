// taken from https://github.com/YohYamasaki/wgpu-prefix-sum-demo/tree/main

use wgpu::{BindGroup, Buffer, ComputePass, ComputePipeline, Device};

fn split_dispatch_3d(workgroups_needed: u32, max_dim: u32) -> [u32; 3] {
    let x = workgroups_needed.min(max_dim);
    let remaining_after_x = (workgroups_needed + x - 1) / x;
    let y = remaining_after_x.min(max_dim);

    let xy = (x as u64) * (y as u64);
    let z = ((workgroups_needed as u64) + xy - 1) / xy;
    assert!(z <= max_dim as u64, "dispatch exceeds max_dim^3");

    [x, y, z as u32]
}
pub struct PrefixScan {
    pipeline_write_sum: ComputePipeline,
    pipeline_no_sum: ComputePipeline,
    pipeline_add_carry: ComputePipeline,
    bind_groups_write_sum: Vec<BindGroup>,
    bind_group_no_sum: BindGroup,
    bind_groups_add_carry: Vec<BindGroup>,
    data_buffers: Vec<Buffer>,
    elements_per_level: Vec<u32>,
    max_dimensions: u32,
}

impl PrefixScan {
    pub fn new(device: &Device, input_buf: Buffer) -> PrefixScan {
        let numbers_to_sum = input_buf.size() as usize / 4;

        let block_scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block-scan shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("step3_2_block_scan.wgsl").into()),
        });

        let add_carry_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add-carry shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("step3_2_add_carry.wgsl").into()),
        });

        let pipeline_write_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block_scan_write_sum pipeline"),
            layout: None,
            module: &block_scan_shader,
            entry_point: "block_scan_write_sum",
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let pipeline_no_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block_scan_no_sum pipeline"),
            layout: None,
            module: &block_scan_shader,
            entry_point: "block_scan_no_sum",
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let pipeline_add_carry = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_carry pipeline"),
            layout: None,
            module: &add_carry_shader,
            entry_point: "add_carry",
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        // Build all required buffers + block scan bind groups for each level
        const TILE_SIZE: usize = 64;
        let mut data_buffers: Vec<wgpu::Buffer> = vec![];
        let mut bind_groups_write_sum: Vec<wgpu::BindGroup> = vec![];
        let mut elements_per_level: Vec<u32> = vec![];
        // For original data
        data_buffers.push(input_buf);
        // Create buffers for blocks
        let mut level_elms = numbers_to_sum;
        let mut i = 1;
        while level_elms > TILE_SIZE {
            elements_per_level.push(level_elms as u32);
            let num_blocks = level_elms.div_ceil(TILE_SIZE).max(1);
            let sum_bytes = (num_blocks * size_of::<u32>()) as u64;
            data_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block-sum"),
                size: sum_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // bind group: (prev_level -> this_level)
            let src = &data_buffers[i - 1];
            let dst = &data_buffers[i];
            bind_groups_write_sum.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("block-scan bind group"),
                layout: &pipeline_write_sum.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            }));

            level_elms = num_blocks;
            i += 1;
        }
        // The last buffer's elements number is for `block_scan_no_sum`
        elements_per_level.push(level_elms as u32);

        let last_buffer = &data_buffers[data_buffers.len() - 1];
        let bind_group_no_sum = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block-scan bind group"),
            layout: &pipeline_no_sum.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: last_buffer.as_entire_binding(),
            }],
        });

        // Build Add-carry bind groups
        let mut bind_groups_add_carry: Vec<wgpu::BindGroup> = vec![];
        for i in (1..data_buffers.len()).rev() {
            bind_groups_add_carry.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add-carry bind group"),
                layout: &pipeline_add_carry.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffers[i - 1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buffers[i].as_entire_binding(),
                    },
                ],
            }));
        }

        Self {
            pipeline_add_carry,
            pipeline_no_sum,
            pipeline_write_sum,
            bind_group_no_sum,
            bind_groups_add_carry,
            bind_groups_write_sum,
            data_buffers,
            elements_per_level,
            max_dimensions: device.limits().max_compute_workgroups_per_dimension,
        }
    }
    pub fn dispatch(&self, pass: &mut ComputePass) {
        const WG_SIZE: u32 = 64;

        pass.set_pipeline(&self.pipeline_write_sum);

        // apply the scan for block sums recursively until the size of the block sums array becomes smaller than one block size
        self.bind_groups_write_sum
            .iter()
            .enumerate()
            .for_each(|(i, bind_group)| {
                let workgroups_needed = self.elements_per_level[i].div_ceil(WG_SIZE).max(1);
                pass.set_bind_group(0, bind_group, &[]);
                let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dimensions);
                pass.dispatch_workgroups(x, y, z);
            });

        // The last sums also requires scan but no need to write the new block sums since it is already fitting in one block
        let last_idx = self.elements_per_level.len() - 1;
        let workgroups_needed = self.elements_per_level[last_idx].div_ceil(WG_SIZE).max(1);
        pass.set_pipeline(&self.pipeline_no_sum);
        pass.set_bind_group(0, &self.bind_group_no_sum, &[]);
        let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dimensions);
        pass.dispatch_workgroups(x, y, z);

        // add carry to the previous data
        pass.set_pipeline(&self.pipeline_add_carry);
        for level in (1..self.data_buffers.len()).rev() {
            let bind_group = &self.bind_groups_add_carry[self.data_buffers.len() - 1 - level];
            let block_len = self.elements_per_level[level - 1];
            let workgroups_needed = block_len.div_ceil(WG_SIZE).max(1);

            pass.set_bind_group(0, bind_group, &[]);
            let [x, y, z] = split_dispatch_3d(workgroups_needed, self.max_dimensions);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn result_buf(&self) -> &Buffer {
        &self.data_buffers[0]
    }
}
