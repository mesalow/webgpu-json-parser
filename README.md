# WebGPU Parser

This repo is an experimental re-implementation of the cuJSON paper: https://dl.acm.org/doi/epdf/10.1145/3760250.3762222 with WebGPU instead of CUDA. The main goal was for me to learn how to write (bad) compute shaders and use them in Rust. This is not intentend to be used in the current state anywhere important. 

Since the paper makes use of some CUDA / Thrust functions that are not available on WebGPU, I suspect performance is worse. 

I also had to re-implement some of these functions from scratch. The prefix scan implementation is taken from https://github.com/YohYamasaki/wgpu-prefix-sum-demo/tree/main more or less verbatim. The radix sort was written from abstract explanations like https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf and is therefore probably even buggier and unoptimized.

## Current short-comings compared to cuJSON 

- no UTF8 validation
- no early JSON validation
- only implements the getValue API, though some of the other functions are implemented internally in a way
- no performance measurements have been done yet
- There is still a minor race condition somewhere, probably more bugs.

## Next steps

Not sure what I want to do next, but some options could be

- add some of the missing features 
- profile and optimize performance
- port the whole thing to Typescript for use in the browser (my initial inspiration)
- I think the prefix scan could be done better with subgroup support - I didn't pursued this since subgroup support isn't really there in browserland, but for Rust it could be worth it
- since the whole thing works by lazily querying the json, it might be nice to write a kind of "JSON structure discovery tool" where you can click yourself through big json files without having to read the whole thing. Might be an actual use case there.

## AI use

Coding Agents were invaluable for research and understanding. I wrote most of the code myself, though some fixes and refactorings were authored by Claude or Codex.

## Key learnings

- you can unit test GPU code easily with some scaffolding - wish I had done this sooner when developing the first shaders
- radix sort on the GPU took me way longer to implement than I thought it would and it still does not work completely. Definitely tested my boundaries. Same for the prefix scan implementation
- initially I thought this whole idea could also be used for an eager JSON parser but I think the main performance benefit is when using it for lazy JSON querying