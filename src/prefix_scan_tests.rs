//! Tests for `PrefixScan` (exclusive Blelloch scan over a storage buffer).
//!
//! These tests reuse `GpuTestHarness` from `shader_tests`. The scan operates
//! in-place: the input buffer handed to `PrefixScan::new` is returned via
//! `take_result()` after the dispatch, holding the exclusive prefix sum.

use crate::{prefix_scan::PrefixScan, test_harness::GpuTestHarness};

/// CPU reference: exclusive prefix sum.
fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0u32;
    for &v in input {
        out.push(acc);
        acc = acc.wrapping_add(v);
    }
    out
}

fn run_scan(h: &GpuTestHarness, input: &[u32]) -> Vec<u32> {
    let buf = h.storage_buf(input);
    let mut scan = PrefixScan::new(&h.device, buf);
    h.run_and_readback(&mut scan)
}

/// All ones, single block (n == TILE_SIZE) → result is 0,1,2,…,63.
#[test]
fn prefix_scan_single_block_all_ones() {
    let h = GpuTestHarness::new();
    let input = vec![1u32; 64];
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// Small input below the tile size — exercises the `n_blocks==1` `no_sum` path.
#[test]
fn prefix_scan_small_input_under_tile() {
    let h = GpuTestHarness::new();
    let input: Vec<u32> = (1u32..=10).collect();
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// Two-block input: needs a write_sum level + add_carry.
#[test]
fn prefix_scan_two_blocks() {
    let h = GpuTestHarness::new();
    let input: Vec<u32> = (0u32..128).collect();
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// Non-power-of-two size spanning multiple blocks but not aligned.
#[test]
fn prefix_scan_unaligned_size() {
    let h = GpuTestHarness::new();
    let input: Vec<u32> = (0u32..200).map(|i| i % 7).collect();
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// Input large enough to require two recursive levels (n > 64*64 = 4096).
#[test]
fn prefix_scan_two_levels() {
    let h = GpuTestHarness::new();
    let input: Vec<u32> = (0u32..5000).map(|i| (i % 13) + 1).collect();
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// Input large enough to require three recursive levels (n > 64^3 = 262144).
#[test]
fn prefix_scan_three_levels() {
    let h = GpuTestHarness::new();
    let n = 300_000usize;
    let input: Vec<u32> = (0..n as u32).map(|i| i % 5).collect();
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

/// All zeros — result must be all zeros regardless of size.
#[test]
fn prefix_scan_all_zeros() {
    let h = GpuTestHarness::new();
    let input = vec![0u32; 1000];
    let result = run_scan(&h, &input);
    assert!(result.iter().all(|&v| v == 0));
}

/// Single sentinel value at the end of a multi-block input: every prior
/// slot is 0, the final slot equals the running sum just before it.
#[test]
fn prefix_scan_sparse_end_sentinel() {
    let h = GpuTestHarness::new();
    let mut input = vec![0u32; 500];
    input[499] = 42;
    let result = run_scan(&h, &input);
    let expected = cpu_exclusive_scan(&input);
    assert_eq!(result, expected);
}

#[test]
fn prefix_scan_with_negatives() {
    let h = GpuTestHarness::new();
    let input = vec![
        1, 1, 4294967295, 1, 4294967295, 1, 1, 4294967295, 4294967295,
    ];
    let result = run_scan(&h, &input);
    let expected = vec![0, 1, 2, 1, 2, 1, 2, 3, 2];
    assert_eq!(result, expected);
}
