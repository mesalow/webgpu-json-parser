@group(0) @binding(0) var<storage, read>  exclusive_scan_depth: array<u32>;
@group(0) @binding(1) var<storage, read>  open_close_chars_mapped: array<u32>;
@group(0) @binding(2) var<storage, read_write>  depth_array: array<u32>;

@compute  @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&exclusive_scan_depth) { return ;}

    let inclusive_depth= exclusive_scan_depth[index] + open_close_chars_mapped[index]; 
    depth_array[index] = select(inclusive_depth, inclusive_depth - 1u ,open_close_chars_mapped[index] == 1u);
}