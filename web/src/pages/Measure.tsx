import { useState, useRef, useEffect } from 'react';
import { SignalProcessor } from '../engine/SignalProcessor';
import { InferenceEngine } from '../engine/InferenceEngine';
import { DSPEngine } from '../engine/DSPEngine';

import { useHistory } from '../hooks/useHistory';
import { useCalibration } from '../hooks/useCalibration';
import { useNavigate } from 'react-router-dom';

const processor = new SignalProcessor();
const engine = new InferenceEngine();
const dspEngine = new DSPEngine();

export function Measure() {
    const [status, setStatus] = useState<'idle' | 'ready' | 'recording' | 'processing' | 'complete' | 'error'>('idle');
    const [timeLeft, setTimeLeft] = useState(30);
    const [errorMsg, setErrorMsg] = useState('');
    const [results, setResults] = useState<{ sbp: number; dbp: number; hr: number; calibrated: boolean } | null>(null);

    const { saveMeasurement } = useHistory();
    const { applyCalibration } = useCalibration();
    const navigate = useNavigate();

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const signalRef = useRef<{ r: number[], g: number[], b: number[] }>({ r: [], g: [], b: [] });
    // Keep track of the currently displayed channel for the graph (default: Green)
    const displaySignalRef = useRef<number[]>([]);
    const frameIdRef = useRef<number>(0);

    // Initialize Engines
    useEffect(() => {
        async function init() {
            try {
                await processor.initialize();
                await engine.initialize();
                await dspEngine.initialize();
                setStatus('idle');
            } catch (e) {
                console.error(e);
                setErrorMsg('WebGPU is not supported or failed to initialize.');
                setStatus('error');
            }
        }
        init();
    }, []);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment', // Rear camera
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                    // Try torch
                    const track = stream.getVideoTracks()[0];
                    const capabilities = track.getCapabilities() as any; // Cast for torch support
                    if (capabilities.torch) {
                        track.applyConstraints({ advanced: [{ torch: true }] } as any).catch(console.warn);
                    }
                    setStatus('ready');
                };
            }
            streamRef.current = stream;
        } catch (e) {
            console.error(e);
            setErrorMsg('Could not access camera. Please allow permissions.');
            setStatus('error');
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    };

    const startTimeRef = useRef<number>(0);

    const startRecording = () => {
        if (status !== 'ready') return;
        setStatus('recording');
        signalRef.current = { r: [], g: [], b: [] };
        displaySignalRef.current = [];
        setTimeLeft(30);

        // Start loop
        startTimeRef.current = Date.now();
        const startTime = startTimeRef.current;

        const loop = async () => {
            if (!videoRef.current) return;

            try {
                // 1. Process Frame (WebGPU) - Now returns RGB object
                const { r, g, b } = await processor.processFrame(videoRef.current);

                signalRef.current.r.push(r);
                signalRef.current.g.push(g);
                signalRef.current.b.push(b);

                // For live graph, we just show Green for now as it's usually best
                displaySignalRef.current.push(g);
                drawGraph(displaySignalRef.current);

                // 3. Check Time
                const elapsed = (Date.now() - startTime) / 1000;
                const remaining = Math.max(0, 30 - elapsed);
                setTimeLeft(remaining);

                if (remaining <= 0) {
                    finishRecording();
                } else {
                    frameIdRef.current = requestAnimationFrame(loop);
                }
            } catch (e) {
                console.error(e);
            }
        };

        loop();
    };

    const calculateVariance = (arr: number[]) => {
        if (arr.length === 0) return 0;
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
    };

    const finishRecording = async () => {
        cancelAnimationFrame(frameIdRef.current);
        setStatus('processing');
        stopCamera();

        try {
            // Calculate Actual FPS
            const durationSec = (Date.now() - startTimeRef.current) / 1000;
            // Use Green channel length for FPS calc
            const actualFps = signalRef.current.g.length / durationSec;
            console.log(`Captured ${signalRef.current.g.length} frames in ${durationSec.toFixed(1)}s (${actualFps.toFixed(1)} FPS)`);

            // Channel Selection (Variance-based)
            const methods = ['r', 'g', 'b'] as const;
            let bestChannel: 'r' | 'g' | 'b' = 'g';
            let maxVariance = -1;

            methods.forEach(c => {
                const v = calculateVariance(signalRef.current[c]);
                console.log(`Variance [${c.toUpperCase()}]: ${v.toFixed(2)}`);
                if (v > maxVariance) {
                    maxVariance = v;
                    bestChannel = c;
                }
            });

            console.log(`Selected Channel: ${bestChannel.toUpperCase()}`);
            const rawSignal = signalRef.current[bestChannel];

            // 1. Resample ENTIRE signal to 125 Hz using GPU FFT (Matches scipy.signal.resample)
            const targetFs = 125;
            const targetTotalSamples = Math.floor(durationSec * targetFs);
            const resampledTotal = await dspEngine.resample(rawSignal, targetTotalSamples);

            // ===============================================
            // Path A: Heart Rate (NeuroKit2 Logic)
            // Needs: Invert -> Normalize(*100) -> Bandpass -> FindPeaks
            // ===============================================
            // 1. Invert Signal (Absorption: More blood = Darker = Lower Value)
            // We want Peaks to be Positive for Elgendi Detect
            const invertedTotal = resampledTotal.map(v => -v);

            let hrSignal = dspEngine.normalize(invertedTotal, 100);
            hrSignal = await dspEngine.bandpass(hrSignal, targetFs, 0.5, 8.0);

            const peaks = dspEngine.findPeaks(hrSignal, targetFs);
            const peakCount = peaks.length;
            const hr = peakCount * (60 / durationSec);
            console.log(`HR Calc (Elgendi): Peaks=${peakCount}, BPM=${hr.toFixed(1)}`);


            // ===============================================
            // Path B: AI Model (CNN Logic)
            // Needs: Raw Resampled -> Windowing -> Z-Score Normalize (Mean=0, Std=1)
            // Method: Robust Local Normalization (fixes Python Domain Mismatch bug)
            // ===============================================
            const WINDOW_SIZE = 625;
            const STRIDE = Math.floor(WINDOW_SIZE / 2); // 50% overlap

            let sbpSum = 0;
            let dbpSum = 0;
            let count = 0;

            console.log(`Running inference on ${resampledTotal.length} samples (Window=${WINDOW_SIZE}, Stride=${STRIDE})`);

            if (resampledTotal.length < WINDOW_SIZE) {
                console.warn('Signal too short for windowing. Padding single window.');

                // Pad/Resize logic matching Python
                const window = new Float32Array(WINDOW_SIZE);
                for (let i = 0; i < WINDOW_SIZE; i++) window[i] = resampledTotal[i % resampledTotal.length];

                // InferenceEngine applies UCI scaler internally
                const preds = await engine.run(window);
                sbpSum += preds.sbp;
                dbpSum += preds.dbp;
                count++;
            } else {
                for (let i = 0; i <= resampledTotal.length - WINDOW_SIZE; i += STRIDE) {
                    const windowSlice = resampledTotal.slice(i, i + WINDOW_SIZE);

                    // InferenceEngine applies UCI scaler internally
                    const window = new Float32Array(windowSlice);
                    const preds = await engine.run(window);
                    sbpSum += preds.sbp;
                    dbpSum += preds.dbp;
                    count++;
                }
            }

            if (count === 0) throw new Error('No valid windows found');

            const avgSbp = sbpSum / count;
            const avgDbp = dbpSum / count;
            console.log(`Averaged ${count} windows. SBP: ${avgSbp.toFixed(1)}, DBP: ${avgDbp.toFixed(1)}`);

            // HR is already calculated in Path A above


            // Apply Calibration
            const { sbp, dbp, isCalibrated } = applyCalibration(avgSbp, avgDbp);
            const hrRounded = Math.round(hr);

            const resultObj = {
                sbp: Math.round(sbp),
                dbp: Math.round(dbp),
                hr: hrRounded,
                calibrated: isCalibrated
            };

            setResults(resultObj);
            setStatus('complete');

            // Save to History
            saveMeasurement({
                sbp: Math.round(sbp),
                dbp: Math.round(dbp),
                hr: hrRounded,
                calibrated: isCalibrated
            });

        } catch (e) {
            console.error(e);
            setErrorMsg('Inference failed.');
            setStatus('error');
        }
    };

    const drawGraph = (data: number[]) => {
        const cvs = canvasRef.current;
        if (!cvs) return;
        const ctx = cvs.getContext('2d');
        if (!ctx) return;

        const width = cvs.width;
        const height = cvs.height;
        ctx.clearRect(0, 0, width, height);

        // Show last 300 points
        const slice = data.slice(-300);
        if (slice.length < 2) return;

        // Robust Scaling: Use Mean +/- 2*StdDev to ignore outliers
        const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
        const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / slice.length;
        const std = Math.sqrt(variance);

        // Determine view range (approx 95% of data)
        // Ensure a minimum range to prevent zooming in on pure sensor noise
        const rangePadding = 2.0 * std;
        const minVal = mean - rangePadding;
        const maxVal = mean + rangePadding;
        const range = maxVal - minVal || 1;

        ctx.beginPath();
        ctx.strokeStyle = '#4ade80'; // Green 400
        ctx.lineWidth = 2;

        slice.forEach((val, i) => {
            const x = (i / (slice.length - 1)) * width;

            // Normalize and Clamp to 0-1 range to handle outliers
            let normalized = (val - minVal) / range;
            normalized = Math.max(0, Math.min(1, normalized));

            const y = height - normalized * height;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    };

    // Start camera on mount if ready
    useEffect(() => {
        if (status === 'idle') {
            // Maybe wait for user click? 
            // Instructions say "Allow camera access".
        }
    }, [status]);


    return (
        <div className="flex flex-col gap-8">
            <h1 className="text-4xl font-bold text-center">Measure</h1>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Instructions */}
                <div className="card bg-base-100 shadow-xl lg:order-first order-last h-fit">
                    <div className="card-body">
                        <h3 className="card-title text-sm">Instructions</h3>
                        <ol className="list-decimal list-inside text-sm space-y-2">
                            <li>Allow camera access</li>
                            <li>Put your finger to cover the camera lens</li>
                            <li>Start recording for 30 seconds</li>
                            <li>Keep finger still</li>
                        </ol>
                    </div>
                </div>

                {/* Main Interface */}
                <div className="lg:col-span-2 space-y-6">

                    {status === 'error' && (
                        <div className="alert alert-error">
                            <span>{errorMsg}</span>
                            <button className="btn btn-sm" onClick={() => window.location.reload()}>Retry</button>
                        </div>
                    )}

                    {status === 'complete' && results && (
                        <div className="card bg-base-100 shadow-xl">
                            <div className="card-body text-center">
                                <h2 className="card-title justify-center text-2xl">Results</h2>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 my-6 w-full">
                                    <div className="flex flex-col items-center p-4 bg-base-200 rounded-box">
                                        <div className="text-xs opacity-70 uppercase font-black tracking-wider">Blood Pressure</div>
                                        <div className="text-3xl font-black text-primary mt-1">{results.sbp}/{results.dbp}</div>
                                        <div className="text-xs opacity-60 font-bold">mmHg</div>
                                    </div>
                                    <div className="flex flex-col items-center p-4 bg-base-200 rounded-box">
                                        <div className="text-xs opacity-70 uppercase font-black tracking-wider">Heart Rate</div>
                                        <div className="text-3xl font-black text-secondary mt-1">{results.hr}</div>
                                        <div className="text-xs opacity-60 font-bold">BPM</div>
                                    </div>
                                </div>

                                {results.calibrated && (
                                    <div className="badge badge-success gap-2 mx-auto">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="inline-block w-4 h-4 stroke-current"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                                        Calibrated
                                    </div>
                                )}

                                <div className="card-actions justify-center mt-6 gap-4">
                                    <button className="btn bg-gray-600 hover:bg-gray-500 text-white border-none" onClick={() => setStatus('idle')}>Measure Again</button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={() => navigate('/calibrate', { state: { ...results, timestamp: new Date().toISOString() } })}
                                    >
                                        Calibrate This Measurement
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {(status === 'idle' || status === 'ready' || status === 'recording' || status === 'processing') && (
                        <div className="card bg-base-100 shadow-xl overflow-hidden">
                            <div className="card-body p-0">
                                {/* Status Header */}
                                <div className="p-4 text-center bg-base-200">
                                    {status === 'idle' && <button className="btn bg-gray-600 hover:bg-gray-500 text-white border-none" onClick={startCamera}>Enable Camera</button>}
                                    {status === 'ready' && <div className="text-2xl font-bold">Ready</div>}
                                    {status === 'recording' && <div className="text-2xl font-bold animate-pulse">{Math.ceil(timeLeft)}s remaining</div>}
                                    {status === 'processing' && <div className="text-xl">Processing...</div>}
                                </div>

                                {/* Video Area */}
                                <div className="relative bg-black aspect-video">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className={`w-full h-full object-cover ${status === 'processing' ? 'opacity-50' : ''}`}
                                    />
                                    {/* Canvas Overlay for Graph */}
                                    <div className="absolute bottom-0 left-0 right-0 h-1/3 bg-black/60 backdrop-blur-sm">
                                        <canvas ref={canvasRef} width={640} height={150} className="w-full h-full" />
                                    </div>
                                </div>

                                {/* Controls */}
                                <div className="p-6 flex justify-center bg-base-100">
                                    {status === 'ready' && (
                                        <button className="btn bg-gray-600 hover:bg-gray-500 text-white border-none btn-lg w-full max-w-md" onClick={startRecording}>
                                            Start Recording
                                        </button>
                                    )}
                                    {status === 'recording' && (
                                        <button className="btn btn-secondary btn-lg w-full max-w-md" onClick={finishRecording}>
                                            Stop Recording
                                        </button>
                                    )}
                                    {status === 'processing' && (
                                        <span className="loading loading-spinner loading-lg text-primary"></span>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
