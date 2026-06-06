@group(0) @binding(0) var<storage, read> bitmap_quote_final: array<u32>;
@group(0) @binding(1) var<storage, read_write> per_word_quote_count: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_quote_final) { return ;}
    per_word_quote_count[index] = countOneBits(bitmap_quote_final[index]);
}