use rand::seq::SliceRandom;

use crate::{
    compute_step::ComputeStep, prefix_scan::PrefixScan, radix_sort_by_key::RadixSortByKey,
    test_harness::GpuTestHarness, utils::buf_entry,
};

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

    let mut step = ComputeStep::new(
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
        None,
    );
    step.set_result(hist_buf);

    let result = h.run_and_readback(&mut step);

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

    let mut step = ComputeStep::new(
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
        None,
    );
    step.set_result(hist_buf);

    let result = h.run_and_readback(&mut step);

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

    let mut step = ComputeStep::new(
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
        None,
    );
    step.set_result(hist_buf);

    let result = h.run_and_readback(&mut step);

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

    let mut step = ComputeStep::new(
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
        None,
    );
    step.set_result(hist_buf);

    let result = h.run_and_readback(&mut step);

    assert_eq!(result[7], 128, "bucket 7 should hold all 128 elements");
    let total: u32 = result.iter().sum();
    assert_eq!(total, 128);
}

const NUM_DIGITS: usize = 256;
fn compute_prefix_from_hist(histogram: Vec<u32>, num_workgroups: usize) -> Vec<u32> {
    let mut result = vec![0u32; NUM_DIGITS * num_workgroups];
    let mut acc = 0u32;
    for digit in 0..NUM_DIGITS {
        for wg in 0..num_workgroups {
            result[digit * num_workgroups + wg] = acc;
            acc += histogram[digit * num_workgroups + wg];
        }
    }
    result
}

fn compute_prefix_sums(
    input: &[u32],
    num_workgroups: usize,
    elements_per_thread: u32,
    pass_index: u32,
) -> Vec<u32> {
    let elements_per_wg = (64 * elements_per_thread) as usize;

    let mut histogram = vec![0u32; NUM_DIGITS * num_workgroups];
    for wg in 0..num_workgroups {
        let start = (wg * elements_per_wg).min(input.len());
        let end = (start + elements_per_wg).min(input.len());
        for &val in &input[start..end] {
            let digit = ((val >> (pass_index * 8)) & 0xFF) as usize;
            histogram[digit * num_workgroups + wg] += 1;
        }
    }
    println!("histogram {:?}", histogram);

    compute_prefix_from_hist(histogram, num_workgroups)
}
#[test]
fn radix_scatter_normal() {
    let h = GpuTestHarness::new();

    // lets start with 16 digit prefix sums and 4 workgroups

    // wg size = 64, per thread is hardcoded to 16 now = 4096 entries needed

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
    original_input_vec.extend(vec![7; 3584]);

    let mut rng = rand::rng();
    original_input_vec.shuffle(&mut rng);

    let workgroup_size = 64;
    let elements_per_thread_value = 16u32;
    let number_of_wgs = original_input_vec
        .len()
        .div_ceil(workgroup_size * elements_per_thread_value as usize);

    let prefix_sums_vec = compute_prefix_sums(
        &original_input_vec,
        number_of_wgs,
        elements_per_thread_value,
        0,
    );
    let prefix_sums = h.storage_buf(&prefix_sums_vec);

    let input_values_vec: Vec<u32> = original_input_vec.iter().map(|&v| v * 1000).collect();
    let original_input = h.storage_buf(&original_input_vec);
    let input_values = h.storage_buf(&input_values_vec);
    let output_vec = vec![0; original_input_vec.len()];
    let output = h.storage_buf(&output_vec);
    let scratch_size = 64 * elements_per_thread_value as usize;
    let debug_scratch_buf = h.zeroed_buf(scratch_size * number_of_wgs);

    let mut step = ComputeStep::new(
        &h.device,
        include_str!("radix_scatter.wgsl"),
        "radix_scatter",
        &[
            buf_entry(0, &prefix_sums),
            buf_entry(1, &original_input),
            buf_entry(2, &input_values),
            buf_entry(3, &output),
            buf_entry(4, &debug_scratch_buf),
        ],
        number_of_wgs as u32,
        None,
    );
    step.set_result(output);
    let result = h.run_and_readback(&mut step);

    original_input_vec.sort();
    let expected: Vec<u32> = original_input_vec.iter().map(|&v| v * 1000).collect();
    assert_eq!(result, expected, "values sorted by key order");
}

#[test]
fn radix_scatter_with_bigger_than_16() {
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
    original_input_vec.extend(vec![7; 3584]);

    let mut rng = rand::rng();
    original_input_vec.shuffle(&mut rng);

    let workgroup_size = 64;
    let elements_per_thread_value = 16u32;
    let number_of_wgs = original_input_vec
        .len()
        .div_ceil(workgroup_size * elements_per_thread_value as usize);

    let prefix_sums_vec = compute_prefix_sums(
        &original_input_vec,
        number_of_wgs,
        elements_per_thread_value,
        0,
    );
    let prefix_sums = h.storage_buf(&prefix_sums_vec);

    let input_values_vec: Vec<u32> = original_input_vec.iter().map(|&v| v * 1000).collect();
    let original_input = h.storage_buf(&original_input_vec);
    let input_values = h.storage_buf(&input_values_vec);
    let output_vec = vec![0; original_input_vec.len()];
    let output = h.storage_buf(&output_vec);
    let scratch_size = 64 * elements_per_thread_value as usize;
    let debug_scratch_buf = h.zeroed_buf(scratch_size * number_of_wgs);

    let mut step = ComputeStep::new(
        &h.device,
        include_str!("radix_scatter.wgsl"),
        "radix_scatter",
        &[
            buf_entry(0, &prefix_sums),
            buf_entry(1, &original_input),
            buf_entry(2, &input_values),
            buf_entry(3, &output),
            buf_entry(4, &debug_scratch_buf),
        ],
        number_of_wgs as u32,
        None,
    );
    step.set_result(output);
    let result = h.run_and_readback(&mut step);

    original_input_vec.sort();
    let expected: Vec<u32> = original_input_vec.iter().map(|&v| v * 1000).collect();
    assert_eq!(result, expected, "values sorted by key order");
}

#[test]
fn radix_step_by_step_hist() {
    let h = GpuTestHarness::new();
    let input_keys_vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let n_wg = 4;
    let hist_buf = h.zeroed_buf(256 * n_wg as usize);

    let input_keys = h.storage_buf(&input_keys_vec);
    let mut step = ComputeStep::new(
        &h.device,
        include_str!("radix_histogram.wgsl"),
        "radix_histogram",
        &[
            buf_entry(0, &input_keys),
            buf_entry(1, &h.scalar_buf(0)),
            buf_entry(2, &h.scalar_buf(4)),
            buf_entry(3, &hist_buf),
        ],
        n_wg,
        None,
    );
    step.set_result(hist_buf);
    let result = h.run_and_readback(&mut step);

    // global_hist is column-major: slot for digit value v in workgroup wg = n_wg * v + wg.
    // With 10 elements and elements_per_thread=4, all elements fall in workgroup 0,
    // so only the wg=0 slot (offset 0) per digit group is non-zero.
    let mut expected = vec![0u32; 256 * n_wg as usize];
    for v in 1u32..=10 {
        let index = (n_wg * v) as usize;
        expected[index] = 1u32;
    }
    assert_eq!(result, expected);
}

#[test]
fn radix_step_by_step_prefix() {
    let h = GpuTestHarness::new();
    let n_wg = 4;

    let mut input = vec![0u32; 256 * n_wg as usize]; // histogram
    for v in 1u32..=10 {
        let index = (n_wg * v) as usize;
        input[index] = 1u32;
    }
    let input_buf = h.storage_buf(&input);
    let mut prefix_scan = PrefixScan::new(&h.device, input_buf);

    let expected = compute_prefix_from_hist(input, 4);

    let result = h.run_and_readback(&mut prefix_scan);
    assert_eq!(result, expected);
}

#[test]
fn radix_scatter_step() {
    let h = GpuTestHarness::new();

    let input_keys_vec: Vec<u32> = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let input_values_vec: Vec<u32> = vec![34, 22, 20, 18, 16, 14, 12, 9, 7, 5];

    let input_keys = h.storage_buf(&input_keys_vec);
    let input_values = h.storage_buf(&input_values_vec);
    let input_values_len = input_values_vec.len();

    let output = h.zeroed_buf(input_values_len);
    let debug = h.zeroed_buf(input_values_len);

    let prefix_result_vec = compute_prefix_sums(&input_keys_vec, 4, 16, 0);
    let prefix_result = h.storage_buf(&prefix_result_vec);

    let mut scatter_step = ComputeStep::new(
        &h.device,
        include_str!("radix_scatter.wgsl").into(),
        "radix_scatter",
        &[
            buf_entry(0, &prefix_result),
            buf_entry(1, &input_keys),
            buf_entry(2, &input_values),
            buf_entry(3, &output),
            buf_entry(4, &debug),
        ],
        4u32,
        None,
    );
    scatter_step.set_result(output);

    let result = h.run_and_readback(&mut scatter_step);
    assert_eq!(
        result,
        vec![5, 7, 9, 12, 14, 16, 18, 20, 22, 34],
        "correct sort"
    );
}

#[test]
fn complete_radix_step() {
    let h = GpuTestHarness::new();
    let input_keys_vec: Vec<u32> = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let input_values_vec: Vec<u32> = vec![34, 22, 20, 18, 16, 14, 12, 9, 7, 5];

    let input_keys = h.storage_buf(&input_keys_vec);
    let input_values_len = input_values_vec.len();
    let input_values = h.storage_buf(&input_values_vec);
    let mut radix_sort =
        RadixSortByKey::new(&h.device, input_keys, input_values, 4, input_values_len); // TODO does not work for wg > 4 and with == 4 it fails sometimes
    let result = h.run_and_readback(&mut radix_sort);
    assert_eq!(
        result,
        vec![5, 7, 9, 12, 14, 16, 18, 20, 22, 34],
        "correct sort"
    );
}

#[test]
fn expand_step() {
    let h = GpuTestHarness::new();
    let input: Vec<u32> = vec![1, 17, 19, 22, 26, 27, 32, 60, 71, 100];

    let mut expected: Vec<u32> = vec![0u32; 100];
    for i in 0..input.len() {
        if i % 2 == 0 {
            expected[input[i] as usize] = input[i + 1];
        }
    }
    let input_buf = h.storage_buf(&input);
    let output_buf = h.zeroed_buf(100);
    let mut expand = ComputeStep::new(
        &h.device,
        include_str!("expand.wgsl").into(),
        "expand",
        &[buf_entry(0, &input_buf), buf_entry(1, &output_buf)],
        4,
        None,
    );
    expand.set_result(output_buf);

    let result = h.run_and_readback(&mut expand);
    assert_eq!(result, expected);
}
