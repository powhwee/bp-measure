// Compute Shader: Extract Green Channel Average

@group(0) @binding(0) var myTexture : texture_external;
@group(0) @binding(1) var<storage, read_write> output : array<atomic<u32>>;
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
    // textureLoad(texture, coords) returns vec4<f32> in [0.0, 1.0]
    let color = textureLoad(myTexture, vec2<i32>(global_id.xy));

    // Extract Green channel
    let green = color.g;

    // Convert to fixed-point integer (0-25500) to allow atomic add
    // We multiply by 255.0 to get 8-bit equivalent, then by 100 for precision
    let green_int = u32(green * 2550.0);

    // Atomic Add to the first element of output buffer
    atomicAdd(&output[0], green_int);
}
