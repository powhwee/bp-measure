// Compute Shader: Extract RGB Channel Averages

@group(0) @binding(0) var myTexture : texture_external;
@group(0) @binding(1) var<storage, read_write> output : array<atomic<u32>>; // [Red, Green, Blue]
@group(0) @binding(2) var<uniform> params : vec2<f32>; // [width, height]

// workgroup_size(16, 16) = 256 threads per group
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width = u32(params.x);
    let height = u32(params.y);

    // Bounds check
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    // Load pixel from external texture (video)
    let color = textureLoad(myTexture, vec2<i32>(global_id.xy));

    // Convert Linear -> sRGB to match OpenCV "Raw" input
    // WebGPU linearizes automatically; we want the original non-linear values
    let gamma = 1.0 / 2.2;
    let r_srgb = pow(color.r, gamma);
    let g_srgb = pow(color.g, gamma);
    let b_srgb = pow(color.b, gamma);

    // Convert to fixed-point integer (0-2550)
    let r_int = u32(r_srgb * 2550.0);
    let g_int = u32(g_srgb * 2550.0);
    let b_int = u32(b_srgb * 2550.0);

    // Atomic Add to respective indices
    atomicAdd(&output[0], r_int);
    atomicAdd(&output[1], g_int);
    atomicAdd(&output[2], b_int);
}
