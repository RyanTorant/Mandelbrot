# Mandelbrot
A Mandelbrot renderer using my DX11 framework (https://github.com/RyanTorant/FrameDX) and with a CPU fallback
It features x2 AA, either by using groupshared memory on the GPU implementation, or AVX for the CPU implementation.
The GPU implementation can use either single or double precision, the CPU one runs always at double precision