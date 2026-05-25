@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) number_of_workgroups: vec3<u32>) {
    let index = gid.x * 2;
    if index < arrayLength(&input) {
        output[input[index]] = input[index+1];
    }
}