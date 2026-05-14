use std::{collections::HashMap, hash::RandomState};

use wgpu::{
    BindGroup, BindGroupEntry, ComputePass, ComputePipeline, Device, PipelineCompilationOptions,
};

use crate::ComputeStepTrait;

pub struct ComputeStep {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    workgroups: u32,
}

impl ComputeStep {
    pub fn new(
        device: &Device,
        shader_source: &str,
        label: &str,
        bg_entries: &[BindGroupEntry],
        workgroups: u32,
        pipeline_constants: Option<&HashMap<String, f64, RandomState>>,
    ) -> ComputeStep {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}_shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_options = match pipeline_constants {
            Some(constants) => PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            None => Default::default(),
        };

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}_pipeline")),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: pipeline_options,
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}_bg")),
            layout: &pipeline.get_bind_group_layout(0),
            entries: bg_entries,
        });

        ComputeStep {
            pipeline,
            bind_group,
            workgroups,
        }
    }
}

impl ComputeStepTrait for ComputeStep {
    fn dispatch(&self, pass: &mut ComputePass) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }
}
