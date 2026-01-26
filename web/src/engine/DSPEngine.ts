import resampleShaderCode from '../shaders/resample.wgsl?raw';
import bandpassShaderCode from '../shaders/bandpass.wgsl?raw';

/**
 * GPU-accelerated Digital Signal Processing Engine
 * 
 * Performs FFT-based resampling and bandpass filtering on the GPU
 * using WebGPU compute shaders.
 */
export class DSPEngine {
    private device: GPUDevice | null = null;

    // Resample pipeline resources
    private resampleModule: GPUShaderModule | null = null;
    private fftPipeline: GPUComputePipeline | null = null;
    private resizePipeline: GPUComputePipeline | null = null;
    private scalePipeline: GPUComputePipeline | null = null;

    // Bandpass pipeline resources
    private bandpassModule: GPUShaderModule | null = null;
    private bandpassPipeline: GPUComputePipeline | null = null;

    async initialize(): Promise<void> {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }

        this.device = await adapter.requestDevice();

        // Compile resample shader
        this.resampleModule = this.device.createShaderModule({
            code: resampleShaderCode
        });

        // Create resample pipelines
        this.fftPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.resampleModule,
                entryPoint: 'fft_naive'
            }
        });

        this.resizePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.resampleModule,
                entryPoint: 'resize_spectrum'
            }
        });

        this.scalePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.resampleModule,
                entryPoint: 'scale_output'
            }
        });

        // Compile and create bandpass pipeline
        this.bandpassModule = this.device.createShaderModule({
            code: bandpassShaderCode
        });

        this.bandpassPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.bandpassModule,
                entryPoint: 'bandpass_filter'
            }
        });

        console.log('DSPEngine initialized (WebGPU)');
    }

    /**
     * Resample signal to target length using FFT method (matches scipy.signal.resample)
     */
    async resample(signal: number[], targetLength: number): Promise<number[]> {
        if (!this.device || !this.fftPipeline || !this.resizePipeline || !this.scalePipeline) {
            throw new Error('DSPEngine not initialized');
        }

        const N = signal.length;
        const M = targetLength;

        if (N === M) return signal;
        if (N === 0) return [];

        // Create buffers
        const inputReal = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const inputImag = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const spectrumReal = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const spectrumImag = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const resizedReal = this.device.createBuffer({
            size: M * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const resizedImag = this.device.createBuffer({
            size: M * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const outputReal = this.device.createBuffer({
            size: M * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const outputImag = this.device.createBuffer({
            size: M * 4,
            usage: GPUBufferUsage.STORAGE
        });
        const stagingBuffer = this.device.createBuffer({
            size: M * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Upload input (real signal, imaginary = 0)
        this.device.queue.writeBuffer(inputReal, 0, new Float32Array(signal));
        this.device.queue.writeBuffer(inputImag, 0, new Float32Array(N).fill(0));

        // Create uniform buffers
        const fftParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const ifftParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const resizeParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const scaleParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Set parameters
        this.device.queue.writeBuffer(fftParams, 0, new Int32Array([N, M, 1, 0])); // direction = 1 (forward)
        this.device.queue.writeBuffer(ifftParams, 0, new Int32Array([M, N, -1, 0])); // direction = -1 (inverse)
        this.device.queue.writeBuffer(resizeParams, 0, new Int32Array([N, M, 0, 0]));
        this.device.queue.writeBuffer(scaleParams, 0, new Int32Array([N, M, 0, 0]));

        const commandEncoder = this.device.createCommandEncoder();

        // Step 1: Forward FFT
        const fftBindGroup = this.device.createBindGroup({
            layout: this.fftPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: fftParams } },
                { binding: 1, resource: { buffer: inputReal } },
                { binding: 2, resource: { buffer: inputImag } },
                { binding: 3, resource: { buffer: spectrumReal } },
                { binding: 4, resource: { buffer: spectrumImag } }
            ]
        });

        const fftPass = commandEncoder.beginComputePass();
        fftPass.setPipeline(this.fftPipeline);
        fftPass.setBindGroup(0, fftBindGroup);
        fftPass.dispatchWorkgroups(Math.ceil(N / 256));
        fftPass.end();

        // Step 2: Resize spectrum
        const resizeBindGroup = this.device.createBindGroup({
            layout: this.resizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: resizeParams } },
                { binding: 1, resource: { buffer: spectrumReal } },
                { binding: 2, resource: { buffer: spectrumImag } },
                { binding: 3, resource: { buffer: resizedReal } },
                { binding: 4, resource: { buffer: resizedImag } }
            ]
        });

        const resizePass = commandEncoder.beginComputePass();
        resizePass.setPipeline(this.resizePipeline);
        resizePass.setBindGroup(0, resizeBindGroup);
        resizePass.dispatchWorkgroups(Math.ceil(M / 256));
        resizePass.end();

        // Step 3: Inverse FFT
        const ifftBindGroup = this.device.createBindGroup({
            layout: this.fftPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: ifftParams } },
                { binding: 1, resource: { buffer: resizedReal } },
                { binding: 2, resource: { buffer: resizedImag } },
                { binding: 3, resource: { buffer: outputReal } },
                { binding: 4, resource: { buffer: outputImag } }
            ]
        });

        const ifftPass = commandEncoder.beginComputePass();
        ifftPass.setPipeline(this.fftPipeline);
        ifftPass.setBindGroup(0, ifftBindGroup);
        ifftPass.dispatchWorkgroups(Math.ceil(M / 256));
        ifftPass.end();

        // Step 4: Scale by M/N
        const scaleBindGroup = this.device.createBindGroup({
            layout: this.scalePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: scaleParams } },
                { binding: 1, resource: { buffer: outputReal } },
                { binding: 2, resource: { buffer: outputImag } }, // Not used but needed for layout
                { binding: 3, resource: { buffer: resizedReal } }, // Reuse as temp output
                { binding: 4, resource: { buffer: resizedImag } }
            ]
        });

        const scalePass = commandEncoder.beginComputePass();
        scalePass.setPipeline(this.scalePipeline);
        scalePass.setBindGroup(0, scaleBindGroup);
        scalePass.dispatchWorkgroups(Math.ceil(M / 256));
        scalePass.end();

        // Copy result to staging buffer
        commandEncoder.copyBufferToBuffer(resizedReal, 0, stagingBuffer, 0, M * 4);

        this.device.queue.submit([commandEncoder.finish()]);

        // Read back result
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(stagingBuffer.getMappedRange());
        const result = Array.from(resultData);
        stagingBuffer.unmap();

        // Cleanup
        inputReal.destroy();
        inputImag.destroy();
        spectrumReal.destroy();
        spectrumImag.destroy();
        resizedReal.destroy();
        resizedImag.destroy();
        outputReal.destroy();
        outputImag.destroy();
        stagingBuffer.destroy();
        fftParams.destroy();
        ifftParams.destroy();
        resizeParams.destroy();
        scaleParams.destroy();

        return result;
    }

    /**
     * Apply Butterworth bandpass filter
     */
    async bandpass(signal: number[], fs: number, lowcut: number, highcut: number): Promise<number[]> {
        if (!this.device || !this.bandpassPipeline) {
            throw new Error('DSPEngine not initialized');
        }

        const N = signal.length;

        // Create buffers
        const inputBuffer = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const outputBuffer = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const stagingBuffer = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Upload data
        this.device.queue.writeBuffer(inputBuffer, 0, new Float32Array(signal));

        // Pack params: signal_len (u32), fs (f32), lowcut (f32), highcut (f32)
        const paramsData = new ArrayBuffer(16);
        new Uint32Array(paramsData, 0, 1)[0] = N;
        new Float32Array(paramsData, 4, 3).set([fs, lowcut, highcut]);
        this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.bandpassPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: inputBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } }
            ]
        });

        // Execute
        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.bandpassPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1); // Single workgroup, sequential filter
        pass.end();

        commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, N * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read back
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(stagingBuffer.getMappedRange());
        const result = Array.from(resultData);
        stagingBuffer.unmap();

        // Cleanup
        inputBuffer.destroy();
        outputBuffer.destroy();
        paramsBuffer.destroy();
        stagingBuffer.destroy();

        return result;
    }

    /**
     * Z-score normalization (CPU - fast enough for small arrays)
     */
    normalize(signal: number[], scale: number = 1.0): number[] {
        if (signal.length === 0) return [];

        const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
        const variance = signal.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / signal.length;
        const std = Math.sqrt(variance);

        if (std < 0.0001) return new Array(signal.length).fill(0);

        return signal.map(x => ((x - mean) / std) * scale);
    }

    /**
     * Find peaks using Elgendi method (CPU - inherently sequential)
     */
    findPeaks(signal: number[], fs: number): number[] {
        // Half-wave rectification + squaring
        const squared = signal.map(x => x > 0 ? x * x : 0);

        // Moving average windows
        const w1Len = Math.floor(0.111 * fs) | 1;
        const w2Len = Math.floor(0.667 * fs) | 1;

        const maPeak = this.movingAverage(squared, w1Len);
        const maBeat = this.movingAverage(squared, w2Len);

        const maMean = squared.reduce((a, b) => a + b, 0) / squared.length;
        const threshold = maBeat.map(v => v + 0.02 * maMean);

        const peaks: number[] = [];
        const minDist = 0.3 * fs;
        let lastPeak = -minDist;

        for (let i = 1; i < signal.length - 1; i++) {
            if (maPeak[i] > threshold[i] && maPeak[i] > maPeak[i - 1] && maPeak[i] > maPeak[i + 1]) {
                if (i - lastPeak > minDist) {
                    peaks.push(i);
                    lastPeak = i;
                }
            }
        }

        return peaks;
    }

    private movingAverage(data: number[], windowSize: number): number[] {
        const result = new Array(data.length).fill(0);
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(data.length, i + Math.ceil(windowSize / 2));
            let sum = 0;
            for (let j = start; j < end; j++) sum += data[j];
            result[i] = sum / (end - start);
        }
        return result;
    }
}
