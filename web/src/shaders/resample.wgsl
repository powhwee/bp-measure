// WebGPU Compute Shader: FFT-based Resampling
// Implements scipy.signal.resample logic using Cooley-Tukey FFT

// Uniforms for signal parameters
struct Params {
    input_len: u32,
    output_len: u32,
    direction: i32,  // 1 = forward FFT, -1 = inverse FFT
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_real: array<f32>;
@group(0) @binding(2) var<storage, read> input_imag: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_real: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_imag: array<f32>;

const PI: f32 = 3.14159265358979323846;

// Naive DFT kernel - each thread computes one frequency bin
// For small N (~1000), this is fast enough on GPU due to massive parallelism
@compute @workgroup_size(256)
fn fft_naive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let N = params.input_len;
    
    if (k >= N) {
        return;
    }
    
    var sum_real: f32 = 0.0;
    var sum_imag: f32 = 0.0;
    
    let sign = f32(params.direction);
    
    for (var n: u32 = 0u; n < N; n = n + 1u) {
        let angle = sign * 2.0 * PI * f32(k) * f32(n) / f32(N);
        let cos_val = cos(angle);
        let sin_val = sin(angle);
        
        // Complex multiplication: (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
        sum_real = sum_real + input_real[n] * cos_val - input_imag[n] * sin_val;
        sum_imag = sum_imag + input_real[n] * sin_val + input_imag[n] * cos_val;
    }
    
    // For inverse FFT, divide by N
    if (params.direction < 0) {
        output_real[k] = sum_real / f32(N);
        output_imag[k] = sum_imag / f32(N);
    } else {
        output_real[k] = sum_real;
        output_imag[k] = sum_imag;
    }
}

// Spectrum resize kernel - zero-pad or truncate spectrum for resampling
@compute @workgroup_size(256)
fn resize_spectrum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let N = params.input_len;   // Original length
    let M = params.output_len;  // Target length
    
    if (k >= M) {
        return;
    }
    
    let N_half = N / 2u;
    
    // DC and positive frequencies
    if (k <= N_half) {
        output_real[k] = input_real[k];
        output_imag[k] = input_imag[k];
    }
    // Negative frequencies (stored at end of array)
    else if (k >= M - N_half) {
        let src_idx = N - (M - k);
        output_real[k] = input_real[src_idx];
        output_imag[k] = input_imag[src_idx];
    }
    // Zero-padding region (new frequencies)
    else {
        output_real[k] = 0.0;
        output_imag[k] = 0.0;
    }
}

// Scale kernel - multiply by M/N for amplitude correction
@compute @workgroup_size(256)
fn scale_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let M = params.output_len;
    let N = params.input_len;
    
    if (k >= M) {
        return;
    }
    
    let scale = f32(M) / f32(N);
    output_real[k] = input_real[k] * scale;
}
