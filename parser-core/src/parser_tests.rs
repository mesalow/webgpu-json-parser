//! End-to-end tests for the full parsing pipeline (`crate::run`).
//!
//! Each case feeds a JSON string through the GPU pipeline and checks the two
//! returned vectors:
//!
//! * `structural_index` — the byte offset of every *structural* character
//!   (`,` `:` `[` `]` `{` `}`) that is **not** inside a string, in document
//!   order. The returned buffer is zero-padded; we only compare the leading
//!   `expected.len()` entries.
//!
//! * `paired_index` — bracket matching. `paired_index[i] == j` means the
//!   structural char at rank `i` (an opening `{` / `[`) is matched by the
//!   closing char at rank `j`, where both ranks index into `structural_index`.
//!   Every non-opening slot is `0`.

/// Run `json` through the pipeline and assert the (trimmed) structural and
/// paired index vectors match the expectations.
fn check(json: &str, expected_structural: &[u32], expected_paired: &[u32]) {
    let (paired_index, structural_index) =
        pollster::block_on(crate::run(json)).expect("pipeline run failed");

    assert_eq!(
        &structural_index[..expected_structural.len()],
        expected_structural,
        "structural_index mismatch for input {json:?}"
    );
    assert_eq!(
        &paired_index[..expected_paired.len()],
        expected_paired,
        "paired_index mismatch for input {json:?}"
    );
}

#[test]
fn flat_object() {
    // {"a":1}
    //  0123456
    // structural: { @0, : @4, } @6
    check(r#"{"a":1}"#, &[0, 4, 6], &[2, 0, 0]);
}

#[test]
fn flat_array() {
    // [1,2,3]
    //  0123456
    // structural: [ @0, , @2, , @4, ] @6
    check(r#"[1,2,3]"#, &[0, 2, 4, 6], &[3, 0, 0, 0]);
}

#[test]
fn nested_objects() {
    // {"a":{"b":2}}
    //  0         1
    //  0123456789012
    // structural: { @0, : @4, { @5, : @9, } @11, } @12
    // outer {@0 ↔ }@12 (rank 0 ↔ 5), inner {@5 ↔ }@11 (rank 2 ↔ 4)
    check(
        r#"{"a":{"b":2}}"#,
        &[0, 4, 5, 9, 11, 12],
        &[5, 0, 4, 0, 0, 0],
    );
}

#[test]
fn array_of_objects() {
    // [{"a":1},{"b":2}]
    //  0         1
    //  0123456789012345 6
    // structural: [ @0, { @1, : @5, } @7, , @8, { @9, : @13, } @15, ] @16
    // [@0 ↔ ]@16 (0↔8), {@1 ↔ }@7 (1↔3), {@9 ↔ }@15 (5↔7)
    check(
        r#"[{"a":1},{"b":2}]"#,
        &[0, 1, 5, 7, 8, 9, 13, 15, 16],
        &[8, 3, 0, 0, 0, 7, 0, 0, 0],
    );
}

#[test]
fn structural_chars_inside_string_are_ignored() {
    // {"a,b:c":1}
    //  0         1
    //  01234567890
    // The , @3 and : @5 sit inside the string and must NOT be structural.
    // structural: { @0, : @8, } @10
    check(r#"{"a,b:c":1}"#, &[0, 8, 10], &[2, 0, 0]);
}

#[test]
fn escaped_quote_does_not_close_string() {
    // {"a\"b":[1]}   (the \" is an escaped quote, string runs "..b")
    //  0         1
    //  012345678901
    // bytes: { " a \ " b " : [ 1 ] }
    // string spans quote@1 .. quote@6; : @7, [ @8, ] @10, } @11
    // {@0 ↔ }@11 (0↔4), [@8 ↔ ]@10 (2↔3)
    check(r#"{"a\"b":[1]}"#, &[0, 7, 8, 10, 11], &[4, 0, 3, 0, 0]);
}

#[test]
fn escaped_backslash_then_closing_quote() {
    // {"a\\":1}   (the \\ is an escaped backslash, so the next " really closes)
    //  0       1
    //  012345678
    // bytes: { " a \ \ " : 1 }
    // string spans quote@1 .. quote@5; : @6, } @8
    check(r#"{"a\\":1}"#, &[0, 6, 8], &[2, 0, 0]);
}

#[test]
fn complex_input() {
    check(
        r#"{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a","b","c"],"g":{"a":{"b":1},"c":[{"x":1},{"y":2}],"d":[[1,2],[3,4]]}}"#,
        &[
            0, 5, 12, 18, 67, 72, 77, 81, 85, 89, 107, 111, 122, 126, 127, 129, 131, 133, 134, 138,
            139, 143, 147, 151, 152, 156, 157, 161, 162, 166, 168, 169, 173, 174, 175, 179, 181,
            182, 183, 187, 189, 190, 191, 195, 196, 197, 199, 201, 202, 203, 205, 207, 208, 209,
            210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        &[
            // TODO first element should be 55
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 53, 0,
            30, 0, 0, 0, 0, 41, 36, 0, 0, 0, 40, 0, 0, 0, 0, 0, 52, 47, 0, 0, 0, 51, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
}
