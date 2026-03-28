use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("parser.wgsl");

fn main() {
    env_logger::init();
    let json_string = "{'a':null,'b':123,'c':24562472.12346757,'d':'a string','e':[1,2,3],'f':['a','b','c'],'g':{'a':{'b':1},'c':[{'x':1},{'y':2}],'d':[[1,2],[3,4]]}}";
    let expected: u32 = json_string.bytes().map(|b| b as u32).sum();
    match pollster::block_on(run(json_string)) {
        Ok(sum) => {
            println!("GPU byte sum : {sum}");
            println!("CPU byte sum : {expected}");
            assert_eq!(sum, expected, "GPU/CPU mismatch!");
            println!("✅ Match confirmed.");
        }
        Err(e) => eprintln!("❌ Error: {e}"),
    }
}

async fn run(json_string: &str) -> Result<u32, Box<dyn std::error::Error>> {
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
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("parser_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("parser_pipeline"),
        layout: None,
        module: &shader,
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

    let output_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("output"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ── 4. Bind group ────────────────────────────────────────────────────────
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("parser_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // ── 5. Encode & submit ───────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parser_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("parser_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &output_buf,
        0,
        &staging_buf,
        0,
        std::mem::size_of::<u32>() as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    // ── 6. Read back result ──────────────────────────────────────────────────
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());

    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;

    let data = slice.get_mapped_range();
    let result: u32 = bytemuck::cast_slice(&data)[0];
    drop(data);
    staging_buf.unmap();

    Ok(result)
}
