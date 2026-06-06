use rand::seq::SliceRandom;

use crate::{
    compute_step::ComputeStep, prefix_scan::PrefixScan, radix_sort_by_key::RadixSortByKey,
    test_harness::GpuTestHarness, utils::buf_entry,
};

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
