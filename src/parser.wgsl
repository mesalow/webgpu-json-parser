@group(0) @binding(0) var<storage, read>       input:  array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main() {
    var sum: u32 = 0u;
    let word_count = arrayLength(&input);
    for (var i: u32 = 0u; i < word_count; i++) {
        let word = input[i];
        sum += (word         & 0xFFu);
        sum += ((word >> 8u)  & 0xFFu);
        sum += ((word >> 16u) & 0xFFu);
        sum += ((word >> 24u) & 0xFFu);
    }
    output[0] = sum;
}
