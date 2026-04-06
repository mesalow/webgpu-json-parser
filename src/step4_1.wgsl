@group(0) @binding(0) var<storage, read_write> bitmap_structural: array<u32>;
@group(0) @binding(1) var<storage, read_write> bitmap_open_close: array<u32>;
@group(0) @binding(2) var<storage, read> string_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> count_structural: array<u32>;
@group(0) @binding(4) var<storage, read_write> count_open_close: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_structural) { return ;}

    let word_structural = bitmap_structural[index];
    let not_in_string_mask = ~string_mask[index];
    let not_in_string_structural = word_structural & not_in_string_mask;

    count_structural[index] = countOneBits(not_in_string_structural);
    bitmap_structural[index] = not_in_string_structural; // update bitmap_structural so that it shows only the stuff we need

    let word_open_close = bitmap_open_close[index];
    let not_in_string_open_close = word_open_close & not_in_string_mask;

    count_open_close[index] = countOneBits(not_in_string_open_close);
    bitmap_open_close[index] = not_in_string_open_close; // update bitmap_open_close so that it shows only the stuff we need

}   