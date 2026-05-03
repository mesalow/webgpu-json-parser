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
    workgroupBarrier();

    var local_counter = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = global_thread_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let low_nibble_index = input & 0x0F;
            let local_offset = per_thread_local_hist[low_nibble_index][lid.x] + local_counter[low_nibble_index];
            local_counter[low_nibble_index]++;
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

    var local_counter_2 = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let index = lid.x * ELEMENTS_PER_THREAD + i;
        if index < scratch_size {
            let input = scratch[index]; // scratch contains per-workgroup-sorted original inputs
            let global_offset = prefix_sums[(input & 0xFF) * total_number_of_workgroups + workgroup_id]; // number of times digits < current have occured in all workgroups and number of times the digit in question occured in workgroups before this one
            let high_nibble_index = (input >> 4u) & 0x0F;
            let local_offset = per_thread_local_hist[high_nibble_index][lid.x] + local_counter_2[high_nibble_index]; // number of times lesser nibbles occured in all threads + number of times this nibble occured in threads before that one + local_counter for current thread (how often did we see this nibble already)
            local_counter_2[high_nibble_index]++;
            
            output[global_offset+local_offset] = input; 
        }
    }
}