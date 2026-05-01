use std::iter::repeat;

use rand::seq::SliceRandom;

use crate::{compute_step::ComputeStep, test_harness::GpuTestHarness, utils::buf_entry};

/// Each of the 64 distinct values 0..63 appears exactly once → hist[i] == 1 for i < 64.
#[test]
fn radix_histogram_pass0_counts_each_digit_once() {
    let h = GpuTestHarness::new();
    let n_wg = 1u32;

    let input: Vec<u32> = (0u32..64).collect();
    let input_buf = h.storage_buf(&input);
    let pass_buf = h.scalar_buf(0); // extract lowest byte
    let ept_buf = h.scalar_buf(1); // 1 element per thread
    let hist_buf = h.zeroed_buf(256 * n_wg as usize);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_histogram.wgsl"),
        "radix_histogram",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &pass_buf),
            buf_entry(2, &ept_buf),
            buf_entry(3, &hist_buf),
        ],
        n_wg,
    );

    h.run_step(&step);
    let result = h.readback(&hist_buf);

    for i in 0u32..64 {
        assert_eq!(result[i as usize], 1, "digit {i} should appear once");
    }
    for i in 64..256 {
        assert_eq!(result[i], 0, "digit {i} should be absent");
    }
}

/// All 64 elements share the same second byte (0xAB) → only bucket 0xAB is non-zero.
#[test]
fn radix_histogram_pass1_second_byte() {
    let h = GpuTestHarness::new();
    let n_wg = 1u32;

    let input = vec![0x0000_AB00u32; 64];
    let input_buf = h.storage_buf(&input);
    let pass_buf = h.scalar_buf(1); // extract second byte
    let ept_buf = h.scalar_buf(1);
    let hist_buf = h.zeroed_buf(256 * n_wg as usize);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_histogram.wgsl"),
        "radix_histogram",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &pass_buf),
            buf_entry(2, &ept_buf),
            buf_entry(3, &hist_buf),
        ],
        n_wg,
    );

    h.run_step(&step);
    let result = h.readback(&hist_buf);

    assert_eq!(
        result[0xAB], 64,
        "all 64 elements should land in bucket 0xAB"
    );
    let total: u32 = result.iter().sum();
    assert_eq!(total, 64, "total count must equal input length");
}

/// Two workgroups: wg0 processes all-zeros, wg1 processes all-ones.
/// Column-major layout: global_hist[n_wg * digit + wgid].
#[test]
fn radix_histogram_column_major_two_workgroups() {
    let h = GpuTestHarness::new();
    let n_wg = 2u32;

    let input: Vec<u32> = std::iter::repeat(0u32)
        .take(64)
        .chain(std::iter::repeat(1u32).take(64))
        .collect();
    let input_buf = h.storage_buf(&input);
    let pass_buf = h.scalar_buf(0);
    let ept_buf = h.scalar_buf(1);
    let hist_buf = h.zeroed_buf(256 * n_wg as usize);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_histogram.wgsl"),
        "radix_histogram",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &pass_buf),
            buf_entry(2, &ept_buf),
            buf_entry(3, &hist_buf),
        ],
        n_wg,
    );

    h.run_step(&step);
    let result = h.readback(&hist_buf);

    // Column-major: index = n_wg * digit + wgid
    assert_eq!(result[n_wg as usize * 0 + 0], 64, "wg0 digit0");
    assert_eq!(result[n_wg as usize * 0 + 1], 0, "wg1 digit0");
    assert_eq!(result[n_wg as usize * 1 + 0], 0, "wg0 digit1");
    assert_eq!(result[n_wg as usize * 1 + 1], 64, "wg1 digit1");
}

/// elements_per_thread > 1: two threads each process 2 elements → 128 total inputs.
#[test]
fn radix_histogram_elements_per_thread() {
    let h = GpuTestHarness::new();
    let n_wg = 1u32;
    let elements_per_thread = 2u32; // each thread processes 2 elements → 64 * 2 = 128 total

    // All 128 elements == 7 → bucket 7 should have count 128
    let input = vec![7u32; 64 * elements_per_thread as usize];
    let input_buf = h.storage_buf(&input);
    let pass_buf = h.scalar_buf(0);
    let ept_buf = h.scalar_buf(elements_per_thread);
    let hist_buf = h.zeroed_buf(256 * n_wg as usize);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_histogram.wgsl"),
        "radix_histogram",
        &[
            buf_entry(0, &input_buf),
            buf_entry(1, &pass_buf),
            buf_entry(2, &ept_buf),
            buf_entry(3, &hist_buf),
        ],
        n_wg,
    );

    h.run_step(&step);
    let result = h.readback(&hist_buf);

    assert_eq!(result[7], 128, "bucket 7 should hold all 128 elements");
    let total: u32 = result.iter().sum();
    assert_eq!(total, 128);
}

fn compute_prefix_sums(
    input: &[u32],
    num_workgroups: usize,
    elements_per_thread: u32,
    pass_index: u32,
) -> Vec<u32> {
    const NUM_DIGITS: usize = 256;
    let elements_per_wg = (64 * elements_per_thread) as usize;

    let mut histogram = vec![[0u32; NUM_DIGITS]; num_workgroups];
    for wg in 0..num_workgroups {
        let start = wg * elements_per_wg;
        let end = (start + elements_per_wg).min(input.len());
        for &val in &input[start..end] {
            let digit = ((val >> (pass_index * 8)) & 0xFF) as usize;
            histogram[wg][digit] += 1;
        }
    }
    println!("histogram {:?}", histogram);

    let mut result = vec![0u32; NUM_DIGITS * num_workgroups];
    let mut acc = 0u32;
    for digit in 0..NUM_DIGITS {
        for wg in 0..num_workgroups {
            result[digit * num_workgroups + wg] = acc;
            acc += histogram[wg][digit];
        }
    }
    result
}
#[test]
fn radix_reorder_normal() {
    let h = GpuTestHarness::new();

    // lets start with 16 digit prefix sums and 4 workgroups

    // digit 0: 3,5,6,4
    // digit 1: 0,4,4,3
    // digit 2: 6,2,1,2
    // digit 3: 3,1,1,0
    // digit 4: 1,0,0,1
    // rest 0s

    // the original values
    let mut original_input_vec = vec![];
    original_input_vec.extend(vec![1u32; 112]);
    original_input_vec.extend(vec![2u32; 101]);
    original_input_vec.extend(vec![3u32; 94]);
    original_input_vec.extend(vec![4u32; 81]);
    original_input_vec.extend(vec![5u32; 75]);
    original_input_vec.extend(vec![6u32; 49]);

    let mut rng = rand::rng();
    original_input_vec.shuffle(&mut rng);

    let workgroup_size = 64;
    let elements_per_thread_value = 4u32;
    let number_of_wgs = original_input_vec
        .len()
        .div_ceil(workgroup_size * elements_per_thread_value as usize);

    let prefix_sums_vec = compute_prefix_sums(
        &original_input_vec,
        number_of_wgs,
        elements_per_thread_value,
        0,
    );
    let prefix_sums = &h.storage_buf(&prefix_sums_vec);

    let original_input = &h.storage_buf(&original_input_vec);
    let elements_per_thread = &h.scalar_buf(elements_per_thread_value);
    let output_vec = vec![0; prefix_sums_vec.len()];
    let output = &h.storage_buf(&output_vec);
    println!("input before step {:?}", original_input_vec);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_reorder.wgsl"),
        "radix_reorder",
        &[
            buf_entry(0, prefix_sums),
            buf_entry(1, original_input),
            buf_entry(2, elements_per_thread),
            buf_entry(3, output),
        ],
        number_of_wgs as u32,
    );
    h.run_step(&step);
    let result = h.readback(&output);
    println!("input {:?}", original_input_vec);
    println!("prefix {:?}", prefix_sums_vec);
    println!("result {:?}", result);
    assert_eq!(result[0], 1, "first number is 1");
    assert_eq!(result[111], 1, "112 1s");
    assert_eq!(result[112], 2, "113 = 2");
    assert_eq!(result[212], 2, "101 2s");
    assert_eq!(result[213], 3, "214 = 3");
}

#[test]
fn radix_reorder_with_bigger_than_16() {
    let h = GpuTestHarness::new();

    // lets start with 16 digit prefix sums and 4 workgroups

    // digit 0: 3,5,6,4
    // digit 1: 0,4,4,3
    // digit 2: 6,2,1,2
    // digit 3: 3,1,1,0
    // digit 4: 1,0,0,1
    // rest 0s

    // the original values
    let mut original_input_vec = vec![];
    original_input_vec.extend(vec![1u32; 112]);
    original_input_vec.extend(vec![2u32; 101]);
    original_input_vec.extend(vec![3u32; 94]);
    original_input_vec.extend(vec![4u32; 81]);
    original_input_vec.extend(vec![5u32; 75]);
    original_input_vec.extend(vec![6u32; 9]);
    original_input_vec.extend(vec![10u32; 10]);
    original_input_vec.extend(vec![16u32; 10]);
    original_input_vec.extend(vec![26u32; 10]);
    original_input_vec.extend(vec![36u32; 10]);

    let mut rng = rand::rng();
    original_input_vec.shuffle(&mut rng);

    let workgroup_size = 64;
    let elements_per_thread_value = 4u32;
    let number_of_wgs = original_input_vec
        .len()
        .div_ceil(workgroup_size * elements_per_thread_value as usize);

    let prefix_sums_vec = compute_prefix_sums(
        &original_input_vec,
        number_of_wgs,
        elements_per_thread_value,
        0,
    );
    let prefix_sums = &h.storage_buf(&prefix_sums_vec);

    let original_input = &h.storage_buf(&original_input_vec);
    let elements_per_thread = &h.scalar_buf(elements_per_thread_value);
    let output_vec = vec![0; prefix_sums_vec.len()];
    let output = &h.storage_buf(&output_vec);
    println!("input before step {:?}", original_input_vec);

    let step = ComputeStep::new(
        &h.device,
        include_str!("radix_reorder.wgsl"),
        "radix_reorder",
        &[
            buf_entry(0, prefix_sums),
            buf_entry(1, original_input),
            buf_entry(2, elements_per_thread),
            buf_entry(3, output),
        ],
        number_of_wgs as u32,
    );
    h.run_step(&step);
    let result = h.readback(&output);
    println!("input {:?}", original_input_vec);
    println!("prefix {:?}", prefix_sums_vec);
    println!("result {:?}", result);
    assert_eq!(result[0], 1, "first number is 1");
    assert_eq!(result[111], 1, "112 1s");
    assert_eq!(result[112], 2, "113 = 2");
    assert_eq!(result[212], 2, "101 2s");
    assert_eq!(result[213], 3, "214 = 3");
    assert_eq!(result[501], 26, "last - 10 = 26");
    assert_eq!(result[511], 36, "last = 36");
}
