@group(0) @binding(0) var<storage, read>       input:              array<u32>;
@group(0) @binding(1) var<storage, read_write> bitmap_structural:  array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bitmap_backslash:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bitmap_quote:       array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bitmap_open_close:       array<atomic<u32>>;

const BACKSLASH = 0x5Cu;

const DOUBLE_QUOTE = 0x22u;

// structural
const COMMA = 0x2Cu;
const DOUBLE_COLON = 0x3Au;
const BRACKET_OPEN = 0x5Bu;
const BRACKET_CLOSE = 0x5Du;
const BRACE_OPEN = 0x7Bu;
const BRACE_CLOSE = 0x7Du;

fn extractNibbleFromWordOC(word: u32) -> u32 {
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u)  & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;
    let is0 = u32(b0 == BRACE_OPEN || b0 == BRACE_CLOSE || b0 == BRACKET_OPEN || b0 == BRACKET_CLOSE);
    let is1 = u32(b1 == BRACE_OPEN || b1 == BRACE_CLOSE || b1 == BRACKET_OPEN || b1 == BRACKET_CLOSE);
    let is2 = u32(b2 == BRACE_OPEN || b2 == BRACE_CLOSE || b2 == BRACKET_OPEN || b2 == BRACKET_CLOSE);
    let is3 = u32(b3 == BRACE_OPEN || b3 == BRACE_CLOSE || b3 == BRACKET_OPEN || b3 == BRACKET_CLOSE);
    return is0 | (is1 << 1u) | (is2 << 2u) | (is3 << 3u);
}

fn extractNibbleFromWordCC(word: u32) -> u32 {
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u)  & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;
    let is0 = u32(b0 == COMMA || b0 == DOUBLE_COLON);
    let is1 = u32(b1 == COMMA || b1 == DOUBLE_COLON);
    let is2 = u32(b2 == COMMA || b2 == DOUBLE_COLON);
    let is3 = u32(b3 == COMMA || b3 == DOUBLE_COLON);
    return is0 | (is1 << 1u) | (is2 << 2u) | (is3 << 3u);
}



fn extractNibbleFromWordBackslash(word: u32) -> u32 {
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u)  & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;
    let is0 = u32(b0 == BACKSLASH);
    let is1 = u32(b1 == BACKSLASH);
    let is2 = u32(b2 == BACKSLASH);
    let is3 = u32(b3 == BACKSLASH);

    return is0 | (is1 << 1u) | (is2 << 2u) | (is3 << 3u);
}

fn extractNibbleFromWordDoubleQuote(word: u32) -> u32 {
    let b0 = word & 0xFFu;
    let b1 = (word >> 8u)  & 0xFFu;
    let b2 = (word >> 16u) & 0xFFu;
    let b3 = (word >> 24u) & 0xFFu;
    let is0 = u32(b0 == DOUBLE_QUOTE);
    let is1 = u32(b1 == DOUBLE_QUOTE);
    let is2 = u32(b2 == DOUBLE_QUOTE);
    let is3 = u32(b3 == DOUBLE_QUOTE);

    return is0 | (is1 << 1u) | (is2 << 2u) | (is3 << 3u);
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&input) { return ;}
    let word    = input[index];
    let out_idx = index / 8u;
    let shift   = (index % 8u) * 4u;
    let oc_nibble = extractNibbleFromWordOC(word);
    let cc_nibble = extractNibbleFromWordCC(word);
    atomicOr(&bitmap_structural[out_idx],  (oc_nibble | cc_nibble) << shift);
    atomicOr(&bitmap_open_close[out_idx],  oc_nibble << shift);
    atomicOr(&bitmap_backslash[out_idx],   extractNibbleFromWordBackslash(word)   << shift);
    atomicOr(&bitmap_quote[out_idx],       extractNibbleFromWordDoubleQuote(word) << shift);
   
}
