@group(0) @binding(0) var<storage, read> bitmap_quote: array<u32>;
@group(0) @binding(1) var<storage, read> quote_sums_per_word: array<u32>;
@group(0) @binding(2) var<storage, read_write> string_mask: array<u32>;


fn prefix_xor(x:u32) -> u32 {
    var v = x;
    
    v ^= v << 1u;
    v ^= v << 2u;
    v ^= v << 4u;
    v ^= v << 8u;
    v ^= v << 16u;

    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_quote) { return ;}

    let parity = quote_sums_per_word[index] & 1; // were there even or odd number of quotes before current word?

    let stringMaskWithoutParity = prefix_xor(bitmap_quote[index]); 
    string_mask[index] = select( ~stringMaskWithoutParity, stringMaskWithoutParity, parity == 0); // if odd, negate the string mask
}