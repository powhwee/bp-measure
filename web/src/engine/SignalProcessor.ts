import shaderCode from '../shaders/extract_rgb.wgsl?raw';

export class SignalProcessor {
    device: GPUDevice | null = null;
    pipeline: GPUComputePipeline | null = null;
    outputBuffer: GPUBuffer | null = null;
    stagingBuffer: GPUBuffer | null = null;
    uniformBuffer: GPUBuffer | null = null;
    bindGroupLayout: GPUBindGroupLayout | null = null;

    async initialize(): Promise<void> {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported in this browser.');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No appropriate GPUAdapter found.');
        }

        this.device = await adapter.requestDevice();

        // Compile Shader
        const shaderModule = this.device.createShaderModule({
            code: shaderCode
        });

        // Create Bind Group Layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    externalTexture: {} // Binding 0: Video Texture
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' } // Binding 1: Output Buffer (Atomic)
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' } // Binding 2: Params (Width, Height)
                }
            ]
        });

        // Create Pipeline Layout
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });

        // Create Compute Pipeline
        this.pipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });

        // Create Output Buffer (Size: 12 bytes for 3x u32: R, G, B)
        this.outputBuffer = this.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Create Staging Buffer (for reading back to CPU)
        this.stagingBuffer = this.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Create Uniform Buffer
        this.uniformBuffer = this.device.createBuffer({
            size: 16, // vec2<f32> + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    async processFrame(video: HTMLVideoElement): Promise<{ r: number, g: number, b: number }> {
        if (!this.device || !this.pipeline || !this.outputBuffer || !this.stagingBuffer || !this.bindGroupLayout || !this.uniformBuffer) {
            throw new Error('SignalProcessor not initialized');
        }

        const width = video.videoWidth;
        const height = video.videoHeight;

        // Update uniforms
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            new Float32Array([width, height])
        );

        // Reset output buffer to 0
        this.device.queue.writeBuffer(this.outputBuffer, 0, new Uint32Array([0, 0, 0]));

        // Create source texture from video
        const externalTexture = this.device.importExternalTexture({
            source: video
        });

        // Create Bind Group for this specific frame
        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: externalTexture
                },
                {
                    binding: 1,
                    resource: { buffer: this.outputBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.uniformBuffer }
                }
            ]
        });

        // Encode commands
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // Dispatch workgroups (16x16 threads per group)
        const workgroupX = Math.ceil(width / 16);
        const workgroupY = Math.ceil(height / 16);
        passEncoder.dispatchWorkgroups(workgroupX, workgroupY);
        passEncoder.end();

        // Copy output to staging buffer
        commandEncoder.copyBufferToBuffer(
            this.outputBuffer, 0,
            this.stagingBuffer, 0,
            12
        );

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Map staging buffer to read
        await this.stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.stagingBuffer.getMappedRange();
        const data = new Uint32Array(arrayBuffer);

        const sumR = data[0];
        const sumG = data[1];
        const sumB = data[2];

        this.stagingBuffer.unmap();

        const count = width * height;
        if (count === 0) return { r: 0, g: 0, b: 0 };

        // Scale factor: 2550.0 (matches shader) / 10.0 = 255
        // Average = (Sum / Count) / 10.0
        return {
            r: (sumR / count) / 10.0,
            g: (sumG / count) / 10.0,
            b: (sumB / count) / 10.0
        };
    }
}
