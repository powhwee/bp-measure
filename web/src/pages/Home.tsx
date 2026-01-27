import { Link } from 'react-router-dom';
import { Activity, Camera, ShieldCheck } from 'lucide-react';

export function Home() {
    return (
        <div className="hero min-h-[60vh] relative">
            <div className="hero-content text-center">
                <div className="max-w-md">
                    <h1 className="text-5xl font-bold mb-8">Measure Blood Pressure</h1>
                    <p className="py-6 text-xl">
                        Measure your blood pressure using your device's camera.<br></br>
                        Your device should have WebGPU support <br></br>
                        (most newer phones have it)
                    </p>

                    <Link to="/measure" className="btn bg-gray-600 hover:bg-gray-500 text-white border-none btn-lg gap-2 text-lg px-8">
                        <Activity className="w-6 h-6" />
                        Start Measurement
                    </Link>

                    <div className="grid grid-cols-2 gap-8 mt-16">
                        <div className="flex flex-col items-center text-center">
                            <div className="p-4 bg-primary/10 rounded-full mb-4">
                                <Camera className="w-8 h-8 text-primary" />
                            </div>
                            <h3 className="font-bold mb-2">Camera Based</h3>
                            <p className="text-sm opacity-70">Uses WebGPU to run AI model against video frames captured.</p>
                        </div>
                        <div className="flex flex-col items-center text-center">
                            <div className="p-4 bg-secondary/10 rounded-full mb-4">
                                <ShieldCheck className="w-8 h-8 text-secondary" />
                            </div>
                            <h3 className="font-bold mb-2">Private & Secure</h3>
                            <p className="text-sm opacity-70"> No video data is uploaded, inference is local</p>
                        </div>
                    </div>
                    <div className="mt-16 text-sm text-blue-900">
                        <p className="mb-2">This work is based on Sau-Sheong Chang's work:</p>
                        <div className="flex flex-col gap-1">
                            <a href="https://sausheong.com/monitoring-blood-pressure-with-machine-learning-c21ae044dd73" target="_blank" rel="noopener noreferrer" className="link link-hover">
                                Monitoring Blood Pressure with Machine Learning
                            </a>
                            <a href="https://bpmon.sausheong.com/" target="_blank" rel="noopener noreferrer" className="link link-hover">
                                bpmon.sausheong.com
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
