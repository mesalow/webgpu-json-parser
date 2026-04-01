@group(0) @binding(0) var<storage, read> bitmap_backslash:   array<u32>;
@group(0) @binding(1) var<storage, read> bitmap_quote:       array<u32>;
@group(0) @binding(2) var<storage, read_write> bitmap_quote_final:      array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_quote) { return; }
    atomicOr(&bitmap_quote_final[index],bitmap_quote[index] | bitmap_backslash[index]);
   
}