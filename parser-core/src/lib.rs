mod compute_step;
mod prefix_scan;
mod radix_sort_by_key;
mod utils;

#[cfg(test)]
mod shader_tests;
#[cfg(test)]
mod test_harness;

use compute_step::ComputeStep;
use utils::{buf_entry, zeroed_storage_buf};
use wgpu::{util::DeviceExt, Buffer, ComputePass};

use crate::{prefix_scan::PrefixScan, radix_sort_by_key::RadixSortByKey, utils::create_u32_buf};

const WORKGROUP_SIZE: u32 = 64;

const STEP1: &str = include_str!("step1.wgsl");
const STEP2: &str = include_str!("step2.wgsl");
const STEP3_1: &str = include_str!("step3_1.wgsl");
// STEP3_2 will be handled by prefix scan
const STEP3_3: &str = include_str!("step3_3.wgsl");
const STEP4_1: &str = include_str!("step4_1.wgsl");
const STEP4_3: &str = include_str!("step4_3.wgsl");

const PARSE_STEP1_1: &str = include_str!("parse_step1_1.wgsl");
const EXPAND: &str = include_str!("expand.wgsl");

// tweak this empirically or based on input size even
const RADIX_SORT_TARGET_SORT_WORKGROUPS: u32 = 512;

pub async fn run(json_string: &str) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
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
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    ..wgpu::Limits::default()
                },
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
    let output_size = (bytes.len() * std::mem::size_of::<u32>()) as u64;
    let output_size_bitmap = (output_word_count * std::mem::size_of::<u32>()) as u64;

    let number_of_workgroups = (word_count as u32).div_ceil(WORKGROUP_SIZE);

    let radix_sort_elements_per_thread = (bytes.len() as u32)
        .div_ceil(RADIX_SORT_TARGET_SORT_WORKGROUPS * WORKGROUP_SIZE)
        .max(1);

    let number_of_radix_sort_workgroups = (bytes.len() as u32)
        .div_ceil(radix_sort_elements_per_thread * WORKGROUP_SIZE)
        .max(1);

    // step 1
    let bitmap_structural = zeroed_storage_buf(&device, "bitmap_structural", output_word_count);
    let bitmap_backslash = zeroed_storage_buf(&device, "bitmap_backslash", output_word_count);
    let bitmap_quote = zeroed_storage_buf(&device, "bitmap_quote", output_word_count);
    let bitmap_open_close = zeroed_storage_buf(&device, "bitmap_open_close", output_word_count);

    // step2
    let bitmap_quote_final = zeroed_storage_buf(&device, "bitmap_quote_final", output_word_count);

    // step3_1 --> get bitmap_quote_final and return per word quote count
    let per_word_quote_count =
        zeroed_storage_buf(&device, "per_word_quote_count", output_word_count);

    //step 3_3 --> string mask to mask out struct chars in strings
    let string_mask = zeroed_storage_buf(&device, "string_mask", output_word_count);

    // step 4_1 --> count of structural + count of oc
    let count_structural = zeroed_storage_buf(&device, "count_structural", output_word_count);
    let count_open_close = zeroed_storage_buf(&device, "count_open_close", output_word_count);
    let total_count_open_close = create_u32_buf(&device, "total_count_open_close", 0);

    // step 4_3 --> structural index, open-close and open-close-index

    let structural_index = zeroed_storage_buf(&device, "structural_index", bytes.len());
    let open_close_chars = zeroed_storage_buf(&device, "open_close_chars", bytes.len());
    let open_close_chars_mapped = zeroed_storage_buf(&device, "open_close_chars", bytes.len());
    let open_close_chars_mapped_for_parser =
        zeroed_storage_buf(&device, "open_close_chars", bytes.len());
    let open_close_index = zeroed_storage_buf(&device, "open_close_index", bytes.len());

    // parsing
    let depth_array = zeroed_storage_buf(&device, "depth_array", bytes.len());

    // final output
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let staging_buf_bitmap = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_size_bitmap,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // structural_index holds `bytes.len()` u32s, same size as `output_size`.
    let staging_buf_structural = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_structural"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ── 3. Create steps  ───────────────────────────────────────────────────

    let step1 = ComputeStep::new(
        &device,
        STEP1,
        "step1",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &bitmap_structural),
            buf_entry(2, &bitmap_backslash),
            buf_entry(3, &bitmap_quote),
            buf_entry(4, &bitmap_open_close),
        ],
        number_of_workgroups,
        None,
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
        number_of_workgroups,
        None,
    );
    let mut prefix_scan_quotes = PrefixScan::new(&device, per_word_quote_count);
    let quote_prefix_buf = prefix_scan_quotes.take_result();

    let step3_1 = ComputeStep::new(
        &device,
        STEP3_1,
        "step3_1",
        &[
            buf_entry(0, &bitmap_quote_final),
            buf_entry(1, &quote_prefix_buf),
        ],
        number_of_workgroups,
        None,
    );

    let steps_1to3 = vec![step1, step2, step3_1];

    // step3_2 is prefix scan, extra step

    let step3_3 = ComputeStep::new(
        &device,
        STEP3_3,
        "step3_3",
        &[
            buf_entry(0, &bitmap_quote_final),
            buf_entry(1, &quote_prefix_buf),
            buf_entry(2, &string_mask),
        ],
        number_of_workgroups,
        None,
    );

    // get total count of structural and open close (braces / brackets)
    let step4_1 = ComputeStep::new(
        &device,
        STEP4_1,
        "step4_1",
        &[
            buf_entry(0, &bitmap_structural),
            buf_entry(1, &bitmap_open_close),
            buf_entry(2, &string_mask),
            buf_entry(3, &count_structural),
            buf_entry(4, &count_open_close),
            buf_entry(5, &total_count_open_close),
        ],
        number_of_workgroups,
        None,
    );
    let steps3_to_4 = vec![step3_3, step4_1];

    let mut prefix_scan_structural = PrefixScan::new(&device, count_structural);
    let mut prefix_scan_open_close = PrefixScan::new(&device, count_open_close);
    let structural_prefix_buf = prefix_scan_structural.take_result();
    let open_close_prefix_buf = prefix_scan_open_close.take_result();

    let step4_3 = ComputeStep::new(
        &device,
        STEP4_3,
        "step4_3",
        &[
            buf_entry(0, &open_close_prefix_buf),
            buf_entry(1, &structural_prefix_buf),
            buf_entry(2, &bitmap_open_close),
            buf_entry(3, &input_buf),
            buf_entry(4, &bitmap_structural),
            buf_entry(5, &structural_index),
            buf_entry(6, &open_close_chars),
            buf_entry(7, &open_close_index),
            buf_entry(8, &open_close_chars_mapped),
            buf_entry(9, &open_close_chars_mapped_for_parser),
        ],
        number_of_workgroups,
        None,
    );

    // parser
    let mut prefix_scan_depth = PrefixScan::new(&device, open_close_chars_mapped);
    let depth_prefix_buf = prefix_scan_depth.take_result();

    let parser_step1_1 = ComputeStep::new(
        &device,
        PARSE_STEP1_1,
        "parser_step1_1",
        &[
            buf_entry(0, &depth_prefix_buf),
            buf_entry(1, &open_close_chars_mapped_for_parser),
            buf_entry(2, &depth_array),
        ],
        number_of_workgroups,
        None,
    );

    let mut radix_sort_scan = RadixSortByKey::new(
        &device,
        depth_array,
        open_close_index,
        total_count_open_close,
        number_of_radix_sort_workgroups as usize,
        bytes.len(),
    );

    let sorted_oc_indexes = radix_sort_scan.take_result();
    let output_buf = zeroed_storage_buf(&device, "expanded_output", bytes.len());
    let expand = ComputeStep::new(
        &device,
        EXPAND,
        "expand_step",
        &[buf_entry(0, &sorted_oc_indexes), buf_entry(1, &output_buf)],
        number_of_workgroups,
        None,
    );

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
        prefix_scan_quotes.dispatch(&mut pass);

        for step in &steps3_to_4 {
            step.dispatch(&mut pass);
        }
        prefix_scan_structural.dispatch(&mut pass);
        prefix_scan_open_close.dispatch(&mut pass);

        step4_3.dispatch(&mut pass);
        prefix_scan_depth.dispatch(&mut pass);
        parser_step1_1.dispatch(&mut pass);

        radix_sort_scan.dispatch(&mut pass);
        expand.dispatch(&mut pass);
    } // pass dropped here, encoder unlocked

    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    encoder.copy_buffer_to_buffer(
        &structural_index,
        0,
        &staging_buf_structural,
        0,
        output_size,
    );

    queue.submit(std::iter::once(encoder.finish()));

    // ── Debug: `DUMP=<name> cargo run` prints one intermediate buffer ─────────
    // Everything ran in one submit, so every buffer now holds its final value.
    if let Ok(want) = std::env::var("DUMP") {
        let mut registry: Vec<(String, &Buffer)> = vec![
            ("input".into(), &input_buf),
            ("bitmap_structural".into(), &bitmap_structural),
            ("bitmap_backslash".into(), &bitmap_backslash),
            ("bitmap_quote".into(), &bitmap_quote),
            ("bitmap_open_close".into(), &bitmap_open_close),
            ("bitmap_quote_final".into(), &bitmap_quote_final),
            ("quote_prefix".into(), &quote_prefix_buf),
            ("string_mask".into(), &string_mask),
            ("structural_prefix".into(), &structural_prefix_buf),
            ("open_close_prefix".into(), &open_close_prefix_buf),
            ("structural_index".into(), &structural_index),
            ("open_close_chars".into(), &open_close_chars),
            ("depth_prefix".into(), &depth_prefix_buf),
            ("sorted_oc_indexes".into(), &sorted_oc_indexes),
            ("output".into(), &output_buf),
        ];
        // pull buffers buried inside the wrapper steps, under sub-namespaces
        for (prefix, step) in [
            ("quote_scan", &prefix_scan_quotes as &dyn ComputeStepTrait),
            ("structural_scan", &prefix_scan_structural),
            ("open_close_scan", &prefix_scan_open_close),
            ("depth_scan", &prefix_scan_depth),
            ("radix", &radix_sort_scan),
        ] {
            for (n, b) in step.debug_buffers() {
                registry.push((format!("{prefix}.{n}"), b));
            }
        }

        match registry.iter().find(|(n, _)| *n == want) {
            Some((n, b)) => dump(&device, &queue, b, n),
            None => {
                eprintln!("DUMP: no buffer named '{want}'. Available:");
                for (n, _) in &registry {
                    eprintln!("  {n}");
                }
            }
        }
    }

    // ── 5. Read back result ──────────────────────────────────────────────────
    let slice = staging_buf.slice(..);
    let structural_slice = staging_buf_structural.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    let (tx_s, rx_s) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    structural_slice.map_async(wgpu::MapMode::Read, move |r| tx_s.send(r).unwrap());

    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    rx_s.recv()??;

    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    let structural_data = structural_slice.get_mapped_range();
    let structural_indexes: Vec<u32> = bytemuck::cast_slice(&structural_data).to_vec();
    drop(structural_data);
    staging_buf_structural.unmap();

    Ok((result, structural_indexes))
}

/// Copy a storage buffer back to the CPU and print it. Self-contained: uses its
/// own encoder + staging buffer, so it can be called any time after the buffer's
/// producing submit has been queued. Bitmap buffers are printed as set-bit
/// positions; everything else as a (capped) list of u32s.
fn dump(device: &wgpu::Device, queue: &wgpu::Queue, buf: &Buffer, label: &str) {
    let size = buf.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dump_staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let values: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("── dump: {label} ({} u32) ──", values.len());
    if label.contains("bitmap") {
        let set: Vec<u32> = values
            .iter()
            .enumerate()
            .flat_map(|(word_idx, word)| {
                (0..32u32).filter_map(move |bit| {
                    ((word >> bit) & 1 == 1).then_some(word_idx as u32 * 32 + bit)
                })
            })
            .collect();
        println!("set bits: {set:?}");
    } else {
        const CAP: usize = 512;
        if values.len() > CAP {
            println!("{:?} … (+{} more)", &values[..CAP], values.len() - CAP);
        } else {
            println!("{values:?}");
        }
    }
}

pub trait ComputeStepTrait {
    fn dispatch(&self, pass: &mut ComputePass);
    /// Move the step's output buffer out. Single-shot: panics if called twice.
    /// The step's bind groups internally retain the buffer for GPU use, so
    /// dispatch is still valid after taking the result.
    fn take_result(&mut self) -> Buffer;

    /// Intermediate buffers a step wants to expose for debugging, by name.
    /// Default: none. Wrappers that bury buffers internally override this so
    /// they can still be inspected from the outside.
    fn debug_buffers(&self) -> Vec<(String, &Buffer)> {
        vec![]
    }
}
