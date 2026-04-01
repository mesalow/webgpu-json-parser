use wgpu::util::DeviceExt;

const STEP1: &str = include_str!("step1.wgsl");
const STEP2: &str = include_str!("step2.wgsl");

fn main() {
    env_logger::init();
    let json_string = "{'a':null,'b':123,'c':24562472.12346757,'d':'a string','e':[1,2,3],'f':['a','b','c'],'g':{'a':{'b':1},'c':[{'x':1},{'y':2}],'d':[[1,2],[3,4]]}}";

    let expected: Vec<u32> = json_string
        .bytes()
        .enumerate()
        .filter(|(_, b)| matches!(b, b'{' | b'}' | b'[' | b']' | b':' | b','))
        .map(|(i, _)| i as u32)
        .collect();

    match pollster::block_on(run(json_string)) {
        Ok(output) => {
            let gpu: Vec<u32> = output
                .iter()
                .enumerate()
                .flat_map(|(word_idx, word)| {
                    (0..32u32).filter_map(move |bit| {
                        if (word >> bit) & 1 == 1 {
                            Some(word_idx as u32 * 32 + bit)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if gpu == expected {
                println!("OK — {} structural chars match", gpu.len());
                println!("{:?}", gpu)
            } else {
                println!("MISMATCH");
                println!("  GPU:      {:?}", gpu);
                println!("  Expected: {:?}", expected);
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}

async fn run(json_string: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // ── 1. Adapter & device ──────────────────────────────────────────────────
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No adapter found")?;

    println!(
        "🖥  Adapter: {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    // ── 2. Shader & pipeline ─────────────────────────────────────────────────
    let shader1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step1"),
        source: wgpu::ShaderSource::Wgsl(STEP1.into()),
    });

    let pipeline1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline1"),
        layout: None,
        module: &shader1,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // ── 3. Buffers ───────────────────────────────────────────────────────────
    // WGSL has no u8 type — pack bytes into u32 words (little-endian, zero-padded).
    // Padding bytes are 0 so they don't affect the sum.
    let mut bytes = json_string.as_bytes().to_vec();
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let word_count = bytes.len() / 4;
    let output_word_count = (word_count + 7) / 8;
    let output_size = (output_word_count * std::mem::size_of::<u32>()) as u64;

    let bitmap_structural = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitmap_structural"),
        contents: bytemuck::cast_slice(&vec![0u32; output_word_count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bitmap_backslash = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitmap_backslash"),
        contents: bytemuck::cast_slice(&vec![0u32; output_word_count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let bitmap_quote = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitmap_quote"),
        contents: bytemuck::cast_slice(&vec![0u32; output_word_count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bitmap_quote_final = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bitmap_quote_final"),
        contents: bytemuck::cast_slice(&vec![0u32; output_word_count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ── 4. Bind group ────────────────────────────────────────────────────────
    let bind_group_layout = pipeline1.get_bind_group_layout(0);
    let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step1_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bitmap_structural.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bitmap_backslash.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bitmap_quote.as_entire_binding(),
            },
        ],
    });

    let shader2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step2"),
        source: wgpu::ShaderSource::Wgsl(STEP2.into()),
    });

    let pipeline2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline2"),
        layout: None,
        module: &shader2,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step2_bg"),
        layout: &pipeline2.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bitmap_backslash.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bitmap_quote.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bitmap_quote_final.as_entire_binding(),
            },
        ],
    });

    // ── 5. Encode & submit ───────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parser_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step1_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline1);
        pass.set_bind_group(0, &bind_group1, &[]);
        let workgroup_size = 64u32;
        pass.dispatch_workgroups(
            (word_count as u32 + workgroup_size - 1) / workgroup_size,
            1,
            1,
        );
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step2_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline2);
        pass.set_bind_group(0, &bind_group2, &[]);
        let workgroup_size = 64u32;
        pass.dispatch_workgroups(
            (output_word_count as u32 + workgroup_size - 1) / workgroup_size,
            1,
            1,
        );
    }

    encoder.copy_buffer_to_buffer(&bitmap_structural, 0, &staging_buf, 0, output_size);

    queue.submit(std::iter::once(encoder.finish()));

    // ── 6. Read back result ──────────────────────────────────────────────────
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());

    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;

    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    Ok(result)
}
