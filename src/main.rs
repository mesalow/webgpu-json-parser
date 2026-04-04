mod compute_step;
mod utils;

use compute_step::ComputeStep;
use utils::{buf_entry, zeroed_storage_buf};
use wgpu::util::DeviceExt;

const STEP1: &str = include_str!("step1.wgsl");
const STEP2: &str = include_str!("step2.wgsl");

fn buf_entry(binding: u32, buf: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn zeroed_storage_buf(device: &Device, label: &str, count: usize) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&vec![0u32; count]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn main() {
    env_logger::init();
    let json_string = r#"{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a","b","c"],"g":{"a":{"b":1},"c":[{"x":1},{"y":2}],"d":[[1,2],[3,4]]}}"#;

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

    // ── 2. Buffers ───────────────────────────────────────────────────────────
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

    // step 1
    let bitmap_structural = zeroed_storage_buf(&device, "bitmap_structural", output_word_count);
    let bitmap_backslash = zeroed_storage_buf(&device, "bitmap_backslash", output_word_count);
    let bitmap_quote = zeroed_storage_buf(&device, "bitmap_quote", output_word_count);

    // step2
    let bitmap_quote_final = zeroed_storage_buf(&device, "bitmap_quote_final", output_word_count);

    // step3

    // final output
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ── 3. Create steps  ───────────────────────────────────────────────────
    let workgroup_size = 64u32;

    let step1 = ComputeStep::new(
        &device,
        STEP1,
        "step1",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &bitmap_structural),
            buf_entry(2, &bitmap_backslash),
            buf_entry(3, &bitmap_quote),
        ],
        (word_count as u32 + workgroup_size - 1) / workgroup_size,
    );

    let step2 = ComputeStep::new(
        &device,
        STEP2,
        "step2",
        &[
            buf_entry(0, &bitmap_backslash),
            buf_entry(1, &bitmap_quote),
            buf_entry(2, &bitmap_quote_final),
        ],
        (word_count as u32 + workgroup_size - 1) / workgroup_size,
    );

    let steps = vec![step1, step2];

    // ── 4. Encode & submit ───────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parser_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("main_pass"),
            timestamp_writes: None,
        });
        for step in &steps {
            step.dispatch(&mut pass);
        }
    }

    encoder.copy_buffer_to_buffer(&bitmap_quote_final, 0, &staging_buf, 0, output_size);

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
