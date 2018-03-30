# Mandelbrot
A Mandelbrot renderer using my DX11 framework (https://github.com/RyanTorant/FrameDX), with both GPU and CPU implementations
It features x2 AA, either by using groupshared memory on the GPU implementation, or AVX for the CPU implementation.
The GPU implementation can use either single or double precision, the CPU one runs always at double precision

## Results on my system
Running on a Xeon E5-2683v3 and a GTX 980 Ti I'm getting the following FPS, for the starting configuration 

| Renderer        | FPS           | 
| ------------- |:-------------:| 
| CPU     | 179 | 
| GPU (SP)     | 1839      |  
| GPU (DP) | 218      |  

DP is so slow on consumer NVidia cards that my CPU (ok, it's a 14c/28t CPU, but still a CPU) catches up with it, and even passes it, depending on where you are on the fractal.

## Why do this?
For fun! Besides, while small, it's quite complete as a project, you get compute shaders with memory sharing, SIMD programming, and general DX11 stuff.