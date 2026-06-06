@group(0) @binding(0) var<storage, read> acc_open_close_count: array<u32>;
@group(0) @binding(1) var<storage, read> acc_structural_count: array<u32>;
@group(0) @binding(2) var<storage, read> bitmap_open_close: array<u32>;
@group(0) @binding(3) var<storage, read> original_json: array<u32>;
@group(0) @binding(4) var<storage, read> bitmap_structural: array<u32>;
@group(0) @binding(5) var<storage, read_write> structural_index: array<u32>; // save for each structural element the origin index in json
@group(0) @binding(6) var<storage, read_write> open_close_chars: array<u32>; // save the open-close chars in order of appearance
@group(0) @binding(7) var<storage, read_write> open_close_index: array<u32>; // save for each open close char the index in structural_index array
@group(0) @binding(8) var<storage, read_write> open_close_chars_mapped: array<u32>; // save the open-close chars in order of appearance
@group(0) @binding(9) var<storage, read_write> open_close_chars_mapped_for_parser: array<u32>; // save the open-close chars in order of appearance


const BRACKET_OPEN = 0x5Bu;
const BRACKET_CLOSE = 0x5Du;
const BRACE_OPEN = 0x7Bu;
const BRACE_CLOSE = 0x7Du;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_open_close) { return ;}
    
    var structural_count_until_now = acc_structural_count[index];
    let word_structural = bitmap_structural[index];

    let word_open_close = bitmap_open_close[index];

    if word_open_close > 0 { // no open_close characters in word, just continue
        
        var open_close_count_until_now = acc_open_close_count[index];
        
        let idx_first_one_bit = countTrailingZeros(word_open_close); // trailing zeros: number of 0s starting from bit0 = the index of the first 1
        let idx_last_one_bit = 32 - countLeadingZeros(word_open_close) -1; // leading zeros: number of 0s starting from bit31 
        
        for (var i=idx_first_one_bit; i<=idx_last_one_bit; i++ ) {
            let current_bit = (word_open_close >> i) & 1; // shift word to current idx and reduce to the LSB
            if current_bit == 1 {
                let set_bits_in_structural_before_current = word_structural & ((1u << i) - 1u); 

                let structural_index_of_char = structural_count_until_now + countOneBits(set_bits_in_structural_before_current);
                open_close_index[open_close_count_until_now] = structural_index_of_char;
                
                let index_in_json = index*32+i;
                let word_in_json = original_json[index_in_json / 4];
                let lowest_byte = (word_in_json >> ((index_in_json % 4u) * 8u)) & 0xFFu;
                open_close_chars[open_close_count_until_now] = lowest_byte;


                let mapped = select(u32(-1i), 1u, lowest_byte == BRACKET_OPEN || lowest_byte == BRACE_OPEN); // 1 for open, -1 for closed;
                open_close_chars_mapped[open_close_count_until_now] = mapped;
                open_close_chars_mapped_for_parser[open_close_count_until_now] = mapped; // 1 for open, -1 for closed
                
                open_close_count_until_now += 1u; // for each bit found in the word we have to update the count_until_now
            }
        }
    }


    if word_structural > 0 { // dont need to do anything if no structural in word

        let idx_first_one_bit = countTrailingZeros(word_structural); // trailing zeros: number of 0s starting from bit0 = the index of the first 1
        let idx_last_one_bit = 32 - countLeadingZeros(word_structural) -1; // leading zeros: number of 0s starting from bit31 

        for (var i=idx_first_one_bit; i<=idx_last_one_bit; i++ ) {
            let current_bit = (word_structural >> i) & 1; // shift word to current idx and reduce to the LSB
            if current_bit == 1 {
                let index_in_json = index*32+i;
                structural_index[structural_count_until_now] = index_in_json;
                structural_count_until_now += 1u; // for each bit found in the word we have to update the count_until_now
            }
        }
    }
}   

/**
 * 5D = 93 ->   0101 1101
 * 7D = 125 ->  0111 1101
 * 
 * 5B = 91 ->   0101 1011
 * 7B = 123 ->  0111 1011
 */