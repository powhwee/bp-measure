import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime
// In Dev: Load directly from node_modules (Vite allows this)
// In Prod: Load from public/onnx folder (copied during build)
ort.env.wasm.wasmPaths = import.meta.env.DEV
    ? '/node_modules/onnxruntime-web/dist/'
    : '/onnx/';

export class InferenceEngine {
    session: ort.InferenceSession | null = null;
    readonly WINDOW_SIZE = 625; // 5 seconds @ 125 Hz

    async initialize(modelPath: string = '/model.onnx'): Promise<void> {
        try {
            // Try to use WebGPU, fall back to WASM/CPU if needed
            const options: ort.InferenceSession.SessionOptions = {
                executionProviders: ['webgpu', 'wasm'],
                graphOptimizationLevel: 'all'
            };

            this.session = await ort.InferenceSession.create(modelPath, options);
            console.log('Inference Engine Initialized');
        } catch (e) {
            console.warn('WebGPU inference failed to initialize, falling back to CPU', e);
            this.session = await ort.InferenceSession.create(modelPath);
        }
    }

    async run(signal: Float32Array): Promise<{ sbp: number, dbp: number }> {
        if (!this.session) {
            throw new Error('Inference Engine not initialized');
        }

        if (signal.length !== this.WINDOW_SIZE) {
            throw new Error(`Invalid signal length: ${signal.length}. Expected ${this.WINDOW_SIZE}.`);
        }

        // Per-window Z-score normalization: (x - mean) / std
        // This is the correct approach for handling domain shift between:
        // - UCI clinical PPG data (used for training)
        // - Smartphone camera RGB data (used for inference)
        // Both get normalized to N(0,1) distribution, making them comparable.
        const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
        const variance = signal.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / signal.length;
        const std = Math.sqrt(variance) || 1e-6; // Avoid div by zero

        const normalizedSignal = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            let val = (signal[i] - mean) / std;
            // Robust Clamping: Limit to +/- 3 StdDev (widened from 2 per user request)
            // This prevents extreme outliers while allowing more signal peaks
            if (val > 3) val = 3;
            if (val < -3) val = -3;
            normalizedSignal[i] = val;
        }

        // Prepare Tensor
        // Input shape: (Batch=1, Channels=1, Length=625)
        const dims = [1, 1, this.WINDOW_SIZE];
        const tensor = new ort.Tensor('float32', normalizedSignal, dims);

        // Run Inference
        const feeds: Record<string, ort.Tensor> = {};
        feeds[this.session.inputNames[0]] = tensor;

        const results = await this.session.run(feeds);

        // Output shape: (Batch=1, 2) -> [SBP, DBP]
        const outputData = results[this.session.outputNames[0]].data as Float32Array;

        return {
            sbp: outputData[0],
            dbp: outputData[1]
        };
    }
}

