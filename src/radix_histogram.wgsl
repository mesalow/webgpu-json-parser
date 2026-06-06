@group(0) @binding(0) var<storage, read> input_array: array<u32>;
@group(0) @binding(1) var<storage, read> pass_index: u32;
@group(0) @binding(2) var<storage, read> elements_per_thread: u32; // TODO: pass this based on target workgroups from rust side
@group(0) @binding(3) var<storage, read> num_of_values: u32;
@group(0) @binding(4) var<storage, read_write> global_hist: array<atomic<u32>>;

var<workgroup> local_hist: array<atomic<u32>, 256>;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) number_of_workgroups: vec3<u32>) {
    let workgroup_id = linearize_workgroup_id(wid, number_of_workgroups);
    let total_number_of_workgroups = number_of_workgroups.x * number_of_workgroups.y * number_of_workgroups.z; 

    let base_index = (workgroup_id * WORKGROUP_SIZE + lid.x) * elements_per_thread;
    
    /**
    * we limit number of workgroups per https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf ->  thus we increase the number of elements scanned per thread so that each workgroup does more work 
    * the local histogram then gets "more values" per workgroup
    */
    for (var i = 0u; i< elements_per_thread; i++) {
        let index = base_index + i;
        if index < num_of_values {
            let input = input_array[index];
            
            let byte_for_pass = (input >> (pass_index * 8u)) & 0xFF;
            
            atomicAdd(&local_hist[byte_for_pass], 1u);
        }
    }
    workgroupBarrier();

    // lid.x = 0..63, add 64 3 times so that we get every index until 255 -> basically a trick to loop from 0..255 in all workgroups
    for (var idx = lid.x; idx < 256; idx = idx + 64)  { 
        // global_hist should have column major layout, so it should be digit0_wg0, digit0_wg1, digit0_wgN ... digit1_wg0, digit1_wg1, ...digitN_wgN 
        // idx will be running from 0..255 -> for each bit in the local_hist
        atomicAdd(&global_hist[total_number_of_workgroups * idx + workgroup_id], atomicLoad(&local_hist[idx]));
    }

}

