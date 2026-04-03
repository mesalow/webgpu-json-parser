@group(0) @binding(0) var<storage, read> bitmap_backslash:   array<u32>;
@group(0) @binding(1) var<storage, read> bitmap_quote:       array<u32>;
@group(0) @binding(2) var<storage, read_write> bitmap_quote_final:      array<u32>;



@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= arrayLength(&bitmap_quote) { return; }
    let currentQuotesWord = bitmap_quote[index];
    var currentBackslashWord = bitmap_backslash[index];

    let maybeEscapedPositions = currentBackslashWord << 1; // left shift by 1 (to the higher bit) -> a 1 at bit1 means = bit0 had a backslash = escaped
    let pessimisticEscapedPositions = maybeEscapedPositions | 1; // for the LSB we take 1 = assume carry for early return check
    let possibleEscapedQuotes = currentQuotesWord & pessimisticEscapedPositions;

    if possibleEscapedQuotes == 0 {
        bitmap_quote_final[index] = currentQuotesWord;
    }

    let evenBits = 0x55555555u; // 5 = 0101 in bits
    let oddBits = ~evenBits;

    // we want to find out the escaped quotes, for those we need to know how many backslashes are before it
    // since we are looking at words in parallel, we could run into the edge case that a word starts with a backslash that is escaped in the previous word
    // for that we calculate the "carry" escape
    // - look into previous word
    // - if word is completely full of backslashes, need to look into the word before that (= while loop)
    // - otherwise, just count the backslashes starting from the high bit ("countLeading")
    // - we invert the bitmap so that we can use inbuilt function countLeadingZeros
    var preceding_index = index - 1;
    var carry = select(2u, 0u, index == 0); // if index = 0 there can be no carry
    while carry == 2 {
        let preceding_word_inverted = ~bitmap_backslash[preceding_index];
        let leadingZeros = countLeadingZeros(preceding_word_inverted);
        carry = select(leadingZeros & 1u, 2u, leadingZeros == 32u); // if leadingZeros is < 32, we just return if its even (= no escape) or odd (= escape)
        if carry == 2u {
            if preceding_index == 0u { carry = 0u; break; }  // edge case
            preceding_index -= 1u;
        }
    }
    // -> now we have the carry of 1 or 0 for the current word


    currentBackslashWord = currentBackslashWord & (~carry); // if carry = remove backlash from bit0
    let applyEscapeChar = (currentBackslashWord << 1) | carry; // left shift again and add carry back
    /**
     * backslash at bit0 | carry | output bit0 | output bit1
                    0    |  0    |         0   |    0
                    1    |  0    |         0   |    1
                    0    |  1    |         1   |    0
                    1    |  1    |         0   |    0 
     */
    // if all is 0 = bit0 and bit1 zero, makes sense
    // if bit0 and carry is 1 = bit0 can be treated as escaped and ignored. if bit1 had a backslash, than that will be handled appropriately
    // bit bit0 is 1 and carry is 0 = backslash should be applied -> move it to bit1 -> bit1=1
    // it bit0 is 0 and carry is 1 = bit0 may have to be escaped = bit0=1

    let startingBackslashesAtOdd = currentBackslashWord & (~applyEscapeChar) & oddBits; // ~applyEscapeChar removes every backslash that could be escaped by the one before it, leaving only those backslashes that do not have one before it
    let sequenceStartAtEven = startingBackslashesAtOdd + currentBackslashWord; 
    /**
     * explanation: adding any one bit to a byte where there are consecutive 1s will flip all those 1 to zero and produce a 1 at the first non-1 position
     *    0 0 0 1 0 0 0   ← single 1 added (startingBackslashesAtOdd bit inside a run)
        + 0 0 1 1 1 0 0   ← run of backslashes at positions 2,3,4
        = 0 1 0 0 1 0 0   ← carry exits at position 5 (past run end)
                  ^
          original start of run, before the added bit
          The 1 at bit2 doesn't matter because bit3 cannot be a quote. 
          We "start" at the odd bit (bit3 in startingBackslashesAtOdd here) and land either at an oddbit (bit5) or an even bit. 

          If the sequence would start at bit3 and only have tow 1s, we would end at bit4 = even
          If the sequence would start at bit2 and end at bit3, we would also end at bit4
          
          startingBackslashesAtOdd only has the first backslash of any such run (all others are treated as "escaped")

          If there is no odd bit at all, the even bits are reconstituted here
          0 0 0 0 0 0 0
        + 0 0 0 0 1 0 0
        = 0 0 0 0 1 0 0 = even bit is there 

        If there is only one odd bit at all, we move it by one
          0 0 0 0 0 1 0
        + 0 0 0 0 0 1 0
        = 0 0 0 0 1 0 0 = again an even bit 
     */

     let invertMask = sequenceStartAtEven << 1; // shift one to the left = the bits that will actually be escaped
     
     
     let toggled = evenBits ^ invertMask; // toggle even 1s to 0, odd stays as it was
     // what this does: 
     
     let escaped = toggled & applyEscapeChar; 
     bitmap_quote_final[index] = bitmap_quote[index] & (~escaped);
   
}