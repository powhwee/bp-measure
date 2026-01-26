// WebGPU Compute Shader: Butterworth Bandpass Filter
// Implements 2nd order IIR highpass + lowpass cascade

struct FilterParams {
    signal_len: u32,
    fs: f32,           // Sampling frequency
    lowcut: f32,       // Highpass cutoff
    highcut: f32,      // Lowpass cutoff
}

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read> input_signal: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_signal: array<f32>;

const PI: f32 = 3.14159265358979323846;
const SQRT2: f32 = 1.41421356237;

// Calculate 2nd order Butterworth highpass coefficients
fn calc_highpass_coeffs(fs: f32, fc: f32) -> array<f32, 5> {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = sin(w0) / SQRT2;
    let cosw0 = cos(w0);
    
    let a0 = 1.0 + alpha;
    let b0 = (1.0 + cosw0) / 2.0 / a0;
    let b1 = -(1.0 + cosw0) / a0;
    let b2 = (1.0 + cosw0) / 2.0 / a0;
    let a1 = -2.0 * cosw0 / a0;
    let a2 = (1.0 - alpha) / a0;
    
    return array<f32, 5>(b0, b1, b2, a1, a2);
}

// Calculate 2nd order Butterworth lowpass coefficients
fn calc_lowpass_coeffs(fs: f32, fc: f32) -> array<f32, 5> {
    let w0 = 2.0 * PI * fc / fs;
    let alpha = sin(w0) / SQRT2;
    let cosw0 = cos(w0);
    
    let a0 = 1.0 + alpha;
    let b0 = (1.0 - cosw0) / 2.0 / a0;
    let b1 = (1.0 - cosw0) / a0;
    let b2 = (1.0 - cosw0) / 2.0 / a0;
    let a1 = -2.0 * cosw0 / a0;
    let a2 = (1.0 - alpha) / a0;
    
    return array<f32, 5>(b0, b1, b2, a1, a2);
}

// Note: IIR filters are inherently sequential due to feedback.
// This shader applies the filter in a single thread per signal.
// For parallel processing, we'd need overlap-save or parallel prefix methods.
// Given typical signal lengths (~3000 samples), single-thread GPU is still fast.

@compute @workgroup_size(1)
fn bandpass_filter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let N = params.signal_len;
    
    // Get filter coefficients
    let hp = calc_highpass_coeffs(params.fs, params.lowcut);
    let lp = calc_lowpass_coeffs(params.fs, params.highcut);
    
    // Pass 1: Highpass filter
    // Uses Direct Form I with explicit state tracking
    var hp_x1: f32 = 0.0;  // x[n-1]
    var hp_x2: f32 = 0.0;  // x[n-2]
    var hp_y1: f32 = 0.0;  // y[n-1]
    var hp_y2: f32 = 0.0;  // y[n-2]
    
    for (var i: u32 = 0u; i < N; i = i + 1u) {
        let x0 = input_signal[i];
        
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        let y = hp[0] * x0 + hp[1] * hp_x1 + hp[2] * hp_x2 - hp[3] * hp_y1 - hp[4] * hp_y2;
        
        output_signal[i] = y;
        
        // Update state
        hp_x2 = hp_x1;
        hp_x1 = x0;
        hp_y2 = hp_y1;
        hp_y1 = y;
    }
    
    // Pass 2: Lowpass filter
    // Must read from output_signal (highpass result), write to output_signal
    // Use explicit state tracking to avoid reading modified values
    var lp_x1: f32 = 0.0;  // x[n-1] (from highpass output)
    var lp_x2: f32 = 0.0;  // x[n-2]
    var lp_y1: f32 = 0.0;  // y[n-1]
    var lp_y2: f32 = 0.0;  // y[n-2]
    
    for (var i: u32 = 0u; i < N; i = i + 1u) {
        let x0 = output_signal[i];  // Read highpass output before we overwrite it
        
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        let y = lp[0] * x0 + lp[1] * lp_x1 + lp[2] * lp_x2 - lp[3] * lp_y1 - lp[4] * lp_y2;
        
        output_signal[i] = y;
        
        // Update state
        lp_x2 = lp_x1;
        lp_x1 = x0;  // Store the original highpass value before it was overwritten
        lp_y2 = lp_y1;
        lp_y1 = y;
    }
}
