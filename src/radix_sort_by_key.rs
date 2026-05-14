use core::num;

use wgpu::{
    util::BufferInitDescriptor, BindGroup, BindGroupDescriptor, Buffer, BufferDescriptor,
    ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
};

use crate::{
    compute_step::ComputeStep,
    prefix_scan::PrefixScan,
    utils::{buf_entry, create_u32_buf, zeroed_storage_buf},
    ComputeStepTrait,
};

pub struct RadixSortByKey {
    input_keys: Buffer,
    input_values: Buffer,
    histogram_step: ComputeStep,
    prefix_scan_step: PrefixScan,
    scatter_step: ComputeStep,
}

impl RadixSortByKey {
    pub fn new(
        device: &Device,
        input_keys: Buffer,
        input_values: Buffer,
        number_of_workgroups: usize,
        scatter_output_buf: &Buffer,
    ) -> RadixSortByKey {
        let elements_per_thread_value = 16u32;

        let histogram_output_buf =
            zeroed_storage_buf(device, "histogram_output", 256 * number_of_workgroups); // TODO: only until 256 digits

        let scratch_size = 64 * elements_per_thread_value as usize;

        let debug_buf =
            zeroed_storage_buf(device, "debug_buf", scratch_size * number_of_workgroups);

        let histogram_step = ComputeStep::new(
            &device,
            include_str!("radix_histogram.wgsl").into(),
            "radix_histogram",
            &[
                buf_entry(0, &input_keys),
                buf_entry(1, &create_u32_buf(&device, "pass-index", 0u32)),
                buf_entry(
                    2,
                    &create_u32_buf(&device, "elements_per_thread", elements_per_thread_value),
                ),
                buf_entry(3, &histogram_output_buf),
            ],
            number_of_workgroups as u32,
            None,
        );

        let prefix_scan_step = PrefixScan::new(device, histogram_output_buf);

        let scatter_step = ComputeStep::new(
            &device,
            include_str!("radix_scatter.wgsl").into(),
            "radix_scatter",
            &[
                buf_entry(0, &prefix_scan_step.result_buf()),
                buf_entry(1, &input_keys),
                buf_entry(2, &input_values),
                buf_entry(3, &scatter_output_buf),
                buf_entry(4, &debug_buf),
            ],
            number_of_workgroups as u32,
            None,
        );
        RadixSortByKey {
            input_keys,
            input_values,
            histogram_step,
            prefix_scan_step,
            scatter_step,
        }
    }
}
impl ComputeStepTrait for RadixSortByKey {
    fn dispatch(&self, pass: &mut ComputePass) {
        self.histogram_step.dispatch(pass);
        self.prefix_scan_step.dispatch(pass);
        self.scatter_step.dispatch(pass);
    }
}
