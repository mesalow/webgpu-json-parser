@group(0) @binding(0) var<storage, read> prefix_sums: array<u32>;
@group(0) @binding(1) var<storage, read> original_input: array<u32>;
@group(0) @binding(2) var<storage, read> elements_per_thread: u32; // TODO: pass this based on target workgroups from rust side
@group(0) @binding(3) var<storage, read_write> output: array<u32>; // TODO: pass this based on target workgroups from rust side

var<workgroup> per_thread_local_hist: array<array<u32, 64>, 16>; // we want to save our 256 bit counter for all 64 threads, but if we save 4bytes (u32) than we will have 64kb shared memory, which is too much
//var<workgroup> scratch: array<u32, elements_per_thread * WORKGROUP_SIZE>; 
// again from paper https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf we employ the idea to split the sort up into two passes
// we do not save the full 256 bit counter but two 16 bit counters -> 16bit * 64 threads * 4 bytes = 4kb

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

const WORKGROUP_SIZE: u32 = 64;

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
    let base_index = (workgroup_id * WORKGROUP_SIZE + lid.x) * elements_per_thread;
    var inner_pass_index = 0u;

    // build per-thread histogram: per_thread_local_hist[digit][thread]
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let index = base_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let digit_index = (input >> (inner_pass_index * 4u)) & 0xFF;
            per_thread_local_hist[digit_index][lid.x]++;
        }
    }
    workgroupBarrier();

    // per-digit prefix scan across threads — one thread, reset acc for each digit
    if lid.x == 0 {
        for (var d = 0u; d < 16u; d++) {
            var acc = 0u;
            for (var t = 0u; t < WORKGROUP_SIZE; t++) {
                let current = per_thread_local_hist[d][t];
                per_thread_local_hist[d][t] = acc;
                acc += current;
            }
        }
    }
    workgroupBarrier();

    var local_counter = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let index = base_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let digit_index = (input >> (inner_pass_index * 4u)) & 0xFF;
            let global_offset = prefix_sums[total_number_of_workgroups * digit_index + workgroup_id];
            let local_offset = per_thread_local_hist[digit_index][lid.x] + local_counter[digit_index];
            local_counter[digit_index]++;
            output[global_offset + local_offset] = input;
        }
    }


    inner_pass_index = 1u;

     // build per-thread histogram: per_thread_local_hist[digit][thread]
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let index = base_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let digit_index = (input >> (inner_pass_index * 4u)) & 0xFF;
            per_thread_local_hist[digit_index][lid.x]++;
        }
    }
    workgroupBarrier();

    // per-digit prefix scan across threads — one thread, reset acc for each digit
    if lid.x == 0 {
        for (var d = 0u; d < 16u; d++) {
            var acc = 0u;
            for (var t = 0u; t < WORKGROUP_SIZE; t++) {
                let current = per_thread_local_hist[d][t];
                per_thread_local_hist[d][t] = acc;
                acc += current;
            }
        }
    }
    workgroupBarrier();

    var local_counter_2 = array<u32, 16>(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    for (var i: u32 = 0u; i < elements_per_thread; i++) {
        let index = base_index + i;
        if index < arrayLength(&original_input) {
            let input = original_input[index];
            let digit_index = (input >> (inner_pass_index * 4u)) & 0xFF;
            let global_offset = prefix_sums[total_number_of_workgroups * digit_index + workgroup_id];
            let local_offset = per_thread_local_hist[digit_index][lid.x] + local_counter_2[digit_index];
            local_counter_2[digit_index]++;
           // output[global_offset + local_offset] = input;
        }
    }
}