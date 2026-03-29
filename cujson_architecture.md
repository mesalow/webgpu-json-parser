# cuJSON Architecture Reference

**Paper:** cuJSON: A Highly Parallel JSON Parser for GPUs
**Authors:** Ashkan Vedadi Gargary, Soroosh Safari Loaliyan, Zhijia Zhao (UC Riverside)
**Published:** ASPLOS '26 (March 22–26, 2026, Pittsburgh, PA)
**Source:** https://github.com/AutomataLab/cuJSON

---

## Core Philosophy

Conventional GPU JSON parsers (cuDF, MetaJSON, GPJSON) fail because they rely on **branch-heavy, sequential parsing logic** — bad for GPU path divergence. cuJSON redesigns around:

- **Minimal branching** — uses bitwise logic and SIMD intrinsics instead of conditionals
- **Scan & sort primitives** — replaces stack-based nesting recognition with sort-based pairing
- **Bitmap-centric design** — a single "all-in-one" bitmap for all six structural characters rather than one per type, reducing global memory writes
- **Kernel fusion** — reduces kernel calls from 10 to 6 in the tokenization stage

JSONL input is converted to standard JSON (wrap all lines in `[]`, replace newlines with `,`) before processing.

---

## Three-Phase GPU Pipeline

```
Input JSON bytes
      │
      ▼
┌─────────────┐
│  Phase 1:   │  UTF-8 Validation  (GPU, branchless)
│  Validate   │
└──────┬──────┘
       │ pass / fail
       ▼
┌─────────────┐
│  Phase 2:   │  Tokenization  (GPU, bitmap-based)
│  Tokenize   │──► structural_index[]  (byte pos of each structural char)
│             │──► open_close[]        (the '{','[','}',']' chars only)
│             │──► oc_idx[]            (their indices in structural_index)
└──────┬──────┘
       ▼
┌─────────────┐
│  Phase 3:   │  Structure Recognition  (GPU, scan & sort)
│  Struct.    │──► pairing_index[]      (for each structural char: index of its matching closer, or 0)
│  Recog.     │
└──────┬──────┘
       │
       ▼  (to CPU host memory)
┌─────────────┐
│  Querying   │  CPU-side iterator over structural_index + pairing_index
│  Engine     │
└─────────────┘
```

---

## Phase 1: UTF-8 Validation

**Goal:** Reject malformed byte sequences before parsing begins.

UTF-8 encodes characters as 1–4 bytes. The header byte tells you the length; continuation bytes all start with `10`. Three error categories:

- **Malformed bytes:** wrong continuation byte count, dangling bytes ("Too Long", "Too Short")
- **Invalid characters:** surrogates (U+D800–U+DFFF), values above U+10FFFF ("Too Large", "Overlong")
- **Overlong sequences:** encoding an ASCII char in more bytes than necessary

**Algorithm (branchless):**

Each GPU thread processes two consecutive 4-byte words (`prev` and `curr`). For a non-ASCII UTF-8 character spanning 2–4 bytes, its header may be anywhere in `prev` while its continuation bytes land in `curr`.

- `hasUTF8()` — checks if a `u32` word contains any non-ASCII byte (`word & 0x80808080 != 0`); uses `atomicOr` to set a flag. If no non-ASCII bytes exist, validation is skipped entirely (pure ASCII fast path).
- `getHeaders()` — extracts possible 2-byte, 3-byte, 4-byte header positions (head2B, head3B, head4B)
- `check2Bytes()` — uses `__vcmpltu4` / `__vcmpeq4` SIMD intrinsics to check 4 bytes simultaneously against error pattern tables. Error codes are 1-bit flags OR'd together.
- `checkContBytes()` — validates that continuation bytes match the expected pattern for 3- and 4-byte characters

All error bits across threads are accumulated via `atomicOr`. No branches — pure bitwise logic.

---

## Phase 2: Tokenization

**Goal:** Find the byte positions of all structural characters (`[`, `]`, `{`, `}`, `:`, `,`) that are **not inside strings**.

Two dependencies make this hard to parallelize:

1. **Escape dependency** — `\"` inside a string doesn't close it; depends on number of preceding `\` (odd = escaped)
2. **Parity dependency** — which `"` opens vs closes a string depends on counting all preceding unescaped quotes

### Step 1: Build Initial Character Bitmaps

One "all-in-one" bitmap for all 6 structural chars + separate bitmaps for `\` and `"`. Uses `__vcmpeq4` to compare 4 bytes at once. The least significant bit of each byte's result is combined into a single bitmap word (8 bytes → 1 byte of bitmap).

### Step 2: Build Structural Quote Bitmap (resolve escape dependency)

**Bitwise Backward Counting** — resolves whether each `"` is escaped:

For each word, count **trailing ones** in the backslash bitmap using `__clz()` on the negated word (counts leading zeros = trailing ones of original). This gives the "escape carry" — 0, 1, or 2 (possible carry from two words back). Algorithm 3:

```
repeat:
    bs_cnt = __clz(~backslash_bitmap[i])
    esc_carry[i] = (bs_cnt == 32) ? 2 : bs_cnt & 1
    i--
until esc_carry != 2
```

A `"` is escaped if the total trailing backslash count before it is odd. This backward pass resolves cross-word carry dependencies without forward communication.

### Step 3: Build String Mask Bitmap (resolve parity dependency)

**Emulated Bitwise Prefix-XOR** — determines which bytes are inside strings:

No native prefix-XOR exists in CUDA. cuJSON emulates it with 3 kernel calls:

1. `countQuotePerWord()` — count unescaped quotes per word using `__popc()`
2. `thrust::exclusive_scan()` — compute accumulated quote count up to each word
3. `buildStringMask()` — parity = `acc_quote_cnt[i] & 1`; apply intra-word prefix-XOR (5 left-shifts + 5 XORs); fix boundary using carry

Result: a bitmask where 1 = "this byte is inside a string literal."

### Step 4: Generate Tokenization Output

`removePseudo()` — AND the structural bitmap with `~str_mask` to zero out pseudo-structural characters inside strings.

`extractStructural()` — uses `__popc()` to count structural chars per word, `thrust::exclusive_scan()` for start indices, then scans each word's set bits to emit their positions into `structural_index[]`.

**Outputs:**
- `structural_index[]` — byte position in original JSON of each structural character
- `open_close[]` — the actual characters `{`, `[`, `}`, `]` (in order of appearance)
- `oc_idx[]` — index into `structural_index` for each open/close character

---

## Phase 3: Structure Recognition

**Goal:** For each `{`/`[`, find its matching `}`/`]`. Output: `pairing_index[]`.

**Algorithm: Scan & Sort (not a stack)**

Takes `open_close[]` and `oc_idx[]` as input.

**Step 1: Find Depths**

`map()` — convert `{`/`[` → +1, `}`/`]` → -1 using `__vcmpeq4` SIMD.
`inclusive_scan()` — prefix sum gives the depth of each close character.
`transform_if()` — decrement depth of each open character by 1 (so open and its matching close share the same depth value).

**Step 2: Sort by Depth**

`zip_iterator` pairs `(open_close, oc_idx)` together, then `stable_sort_by_key()` sorts by depth. After sorting, opens and closes at the same depth are adjacent and in order — they pair up sequentially.

**Step 3: Validate and Expand**

`validate()` — checks each adjacent pair using branchless XOR: `{` XOR `}` = `0x06`, `[` XOR `]` = `0x06`. Any mismatch = unmatched bracket error. Uses `atomicOr` to accumulate errors.

`expand()` — writes the pairing: `index_pair[pair_idx[i]] = pair_idx[i+1]` and vice versa. Converts sorted pairs back into the `pairing_index[]` array indexed by position in `structural_index`.

**Output:**
```
pairing_index[i] = j    // structural[j] is the closing char for structural[i]
pairing_index[i] = 0    // structural[i] is not an open bracket
```

---

## Output Data Structures (sent to CPU)

```
structural_index[]   // u32[] — byte positions of all structural chars
pairing_index[]      // u32[] — for structural[i] = '{' or '[': index of matching closer
```

Values are extracted lazily: slice `inputJSON[structural_index[i]+1 .. structural_index[pairing_index[i]]]`.

---

## CPU-Side Query Iterator

Walks `structural_index` tracking current index. To navigate:

- **Descend into object/array:** `i + 1` (next structural char is first child)
- **Skip subtree:** `pairing_index[i] + 1` (jump past the matching close bracket)
- **Next sibling:** scan forward past the next `,` within current container

Query APIs (Table 4):

| API | Description |
|---|---|
| `getKey()` | get the next key of current object |
| `getValue()` | get value at current position |
| `gotoKey(string)` | move to value of given key |
| `gotoArrayIdx(int)` | move to array element by index |
| `gotoNextSibling(int)` | move index forward |
| `checkKeyValue(string, string)` | check if key=value at current pos |

---

## Optimizations

**Kernel Fusion:** Steps 2+3 fused into `fusedStep2_3()`, Steps 3+4 fused into `fusedStep3_4()`. Reduces global memory round-trips — intermediate values (e.g., `overflow`) stay in registers.

**Multi-Streaming:** Host→Device and Device→Host transfers overlapped with kernel execution using CUDA streams + pinned memory. Hides transfer latency.

**ASCII Fast Path:** If `hasUTF8()` finds no non-ASCII bytes, skip UTF-8 validation entirely.

---

## Performance

- Outperforms simdjson and Pison (CPU) by **1.3×–2.8×** on standard JSON datasets
- Outperforms cuDF, GPJSON, MetaJSON (GPU) on JSONL datasets
- **Breakeven point vs simdjson: ~8 MB** — below this, GPU transfer overhead dominates
- CPU-GPU transfer cost is included in all measurements

---

## Tradeoffs vs Arena/Typed-Node Approach

| | cuJSON (structural arrays) | Arena (typed nodes) |
|---|---|---|
| GPU output | ~8 bytes/entry (2× u32 arrays) | ~20 bytes/node |
| Value decoding | lazy, at query time | eager, during parse |
| Subtree skip | O(1) via pairing_index | must walk linked list |
| Full deserialization | no faster than arena | marginally more ergonomic |
| Best for | selective path queries | full materialization |

**For full deserialization (our use case):** structural arrays use 2–3× less GPU memory. The arena's pre-typed nodes don't help since the CPU visits every node regardless. Adopt the structural arrays approach; do materialization CPU-side.
