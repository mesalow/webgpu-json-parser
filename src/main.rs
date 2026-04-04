mod compute_step;
mod prefix_scan;
mod utils;

use compute_step::ComputeStep;
use utils::{buf_entry, zeroed_storage_buf};
use wgpu::util::DeviceExt;

use crate::prefix_scan::PrefixScan;

const STEP1: &str = include_str!("step1.wgsl");
const STEP2: &str = include_str!("step2.wgsl");
const STEP3_1: &str = include_str!("step3_1.wgsl");
// STEP3_2 will be handled by prefix scan
const STEP3_3: &str = include_str!("step3_3.wgsl");

fn main() {
    env_logger::init();
    let json_string = r#"{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a","b","c"],"g":{"a":{"b":1},"c":[{"x":1},{"y":2}],"d":[[1,2],[3,4]]}}"#;

    match pollster::block_on(run(json_string)) {
        Ok(output) => {
            println!("quotes per word: {:?}", &output[..output.len().min(20)]);
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
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP,
                ..Default::default()
            },
            None,
        )
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

    // step3_1 --> get bitmap_quote_final and return per word quote count
    let per_word_quote_count =
        zeroed_storage_buf(&device, "per_word_quote_count", output_word_count);

    let acc_quote_count = zeroed_storage_buf(&device, "acc_quote_count", output_word_count);

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
    let prefix_scan = PrefixScan::new(&device, per_word_quote_count);

    let step3_1 = ComputeStep::new(
        &device,
        STEP3_1,
        "step3_1",
        &[
            buf_entry(0, &bitmap_quote_final),
            buf_entry(1, &prefix_scan.result_buf()),
        ],
        (word_count as u32 + workgroup_size - 1) / workgroup_size,
    );

    let steps_1to3 = vec![step1, step2, step3_1];

    // ── 4. Encode & submit ───────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parser_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("main_pass"),
            timestamp_writes: None,
        });
        for step in &steps_1to3 {
            step.dispatch(&mut pass);
        }
        prefix_scan.dispatch(&mut pass);
    } // pass dropped here, encoder unlocked

    encoder.copy_buffer_to_buffer(&prefix_scan.result_buf(), 0, &staging_buf, 0, output_size);

    queue.submit(std::iter::once(encoder.finish()));

    // ── 5. Read back result ──────────────────────────────────────────────────
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
