@group(0) @binding(0) var<storage, read> prefix_sums: array<u32>;
@group(0) @binding(1) var<storage, read> original_input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>; // TODO: pass this based on target workgroups from rust side
@group(0) @binding(3) var<storage, read_write> debug_scratch: array<u32>; // scratch_size * num_workgroups

const ELEMENTS_PER_THREAD: u32 = 16;
const WORKGROUP_SIZE: u32 = 64;

var<workgroup> per_thread_local_hist: array<array<u32, 64>, 16>; // we want to save our 256 bit counter for all 64 threads, but if we save 4bytes (u32) than we will have 64kb shared memory, which is too much
// again from paper https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf we employ the idea to split the sort up into two passes
// we do not save the full 256 bit counter but two 16 bit counters -> 16bit * 64 threads * 4 bytes = 4kb

var<workgroup> scratch: array<u32, 1024>; 
var<workgroup> scratch2: array<u32, 1024>; 
var<workgroup> workgroup_byte_hist: array<atomic<u32>, 256>; 

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) number_of_workgroups: vec3<u32>) {
    let workgroup_id = linearize_workgroup_id(wid, number_of_workgroups);
    let total_number_of_workgroups = number_of_workgroups.x * number_of_workgroups.y * number_of_workgroups.z; 

    // prefix_sums are the prefixes per digit + workgroup
    // but in each workgroup we still have to sort the elements per thread
    // e.g. lets say we have 10k elements. We are in wg 44. For digit0 we get a prefix sum of say 2300. Now what. we still need to bring the number of elements in the correct order
    // so we need a) the original input and b) a way to sort it quickly inside the workgroup
    // we have some target number of workgroups and the workgroup size, say 512 * 64 = 30k. So elements per thread is number of depth changes / 30k. This should be on the order of 1e3-1e4 mostly. Should we do it in one thread?
    // we could also for the scatter phase increase the number of workgroups again right? So that we the size of the blocks to look at is smaller and we can create use an "inner" radix sort (or even some other sort). Hm no, we can't because then these blocks would again depend on each other. 

    // correct we have the base offset per digit based on workgroup - digit * number_of_workgroups + wgid - and now we also need the local offset for that

    // local offset means: this workgroup scanned n elements, digit 0 happend 11 times, digit 1 happend 9 times, etc, so prefix scan is 0, 11, 20 etc. 

    // local histogram
    let global_thread_index = (workgroup_id * WORKGROUP_SIZE + lid.x) * ELEMENTS_PER_THREAD;
    var inner_pass_index = 0u;

    // build per-thread histogram: per_thread_local_hist[digit][thread]
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = global_thread_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let full_byte_index = input & 0xFF;
            atomicAdd(&workgroup_byte_hist[full_byte_index],1u);
            let low_nibble_index = input & 0x0F;
            per_thread_local_hist[low_nibble_index][lid.x]++;
        }
    }
    workgroupBarrier();

    if lid.x == 0 {
        var acc = 0u;
        for (var d = 0u; d < 16u; d++) {
            for (var t = 0u; t < WORKGROUP_SIZE; t++) {
                let current = per_thread_local_hist[d][t];
                per_thread_local_hist[d][t] = acc;
                acc += current;
            }
        }
    }

    // use another thread to accumulate the byte prefix scan
    if lid.x == 1 {
        var acc = 0u;
        for (var d = 0u; d < 256u; d++) {
            let current = workgroup_byte_hist[d];
            workgroup_byte_hist[d] = acc;
            acc += current;
        }
    }

    workgroupBarrier();

    var local_nibble_counter = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = global_thread_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let low_nibble_index = input & 0x0F;
            let local_offset = per_thread_local_hist[low_nibble_index][lid.x] + local_nibble_counter[low_nibble_index];
            local_nibble_counter[low_nibble_index]++;
            scratch[local_offset] = input;
        }
    }

    workgroupBarrier();
    // copy this workgroup's scratch into debug_scratch
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let scratch_idx = lid.x * ELEMENTS_PER_THREAD + i;
        debug_scratch[workgroup_id * WORKGROUP_SIZE * ELEMENTS_PER_THREAD + scratch_idx] = scratch[scratch_idx];
    }
    workgroupBarrier();
 
    //this is the second 4bit pass which we now replace with a per byte histogram and then scatter in sequence 
   // might want to revisit this approach from the Introduction paper here

    // reset local hist
    for (var d = 0u; d < 16u; d++) {
        per_thread_local_hist[d][lid.x] = 0u;
    }

    workgroupBarrier();

    let scratch_size = WORKGROUP_SIZE * ELEMENTS_PER_THREAD;

   // build per-thread histogram: per_thread_local_hist[digit][thread]
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = lid.x * ELEMENTS_PER_THREAD + i;
        if index < scratch_size {
            let input = scratch[index];
            let high_nibble_index = (input >> 4u) & 0x0F;
            per_thread_local_hist[high_nibble_index][lid.x]++;
        }
    }
    workgroupBarrier();

    if lid.x == 0 {
        var acc = 0u;
        for (var d = 0u; d < 16u; d++) {
            for (var t = 0u; t < WORKGROUP_SIZE; t++) {
                let current = per_thread_local_hist[d][t];
                per_thread_local_hist[d][t] = acc;
                acc += current;
            }
        }
    }
    workgroupBarrier();

    var local_nibble_counter_2 = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);

    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = lid.x * ELEMENTS_PER_THREAD + i; // here we want to index into threads which is 0..ELEMENTS_PER*THREAD * WORKGROUP_SIZE
        if index < scratch_size {
            let input = scratch[index];
            let high_nibble_index = (input >> 4u) & 0x0F;
            let local_offset = per_thread_local_hist[high_nibble_index][lid.x] + local_nibble_counter_2[high_nibble_index];
            local_nibble_counter_2[high_nibble_index]++;
            scratch2[local_offset] = input;
        }
    }
    workgroupBarrier(); 

    // global scatter: for now, only do it in one thread to avoid concurrency issues
    // can we do it in parallel actually? go back to the introduction paper to find out
    if lid.x == 0 {

        for (var i: u32 = 0u; i < WORKGROUP_SIZE * ELEMENTS_PER_THREAD; i++) {
            let input = scratch2[i];
            let full_byte_index = input & 0xFF;
            let global_offset = prefix_sums[full_byte_index * total_number_of_workgroups + workgroup_id];
            let end_offset = global_offset + i - workgroup_byte_hist[full_byte_index]; // just adding i is wrong as it goes just through the scratch. It needs to reset after each full_byte jump, e.g. if index is 40 and the input is 2 it would need to know the prefix for 2 in this workgroup (let's say 35), and then it would need to add 5 (40-35) instead of 40 to the global offset
            // problem is how to get that prefix = we get it from the workgroup_byte_hist
            output[end_offset] = input;
        }
    }
        

}