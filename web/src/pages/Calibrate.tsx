import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useCalibration } from '../hooks/useCalibration';
import { AlertCircle, RotateCcw, Save } from 'lucide-react';

export function Calibrate() {
    const location = useLocation();
    const navigate = useNavigate();
    const { calibration, addCalibration, clearCalibration, deleteCalibration } = useCalibration();

    // Expecting state to be passed from Measure page
    const measurement = location.state as { sbp: number; dbp: number; timestamp: string } | null;

    const [actualSbp, setActualSbp] = useState<string>('120');
    const [actualDbp, setActualDbp] = useState<string>('80');

    const handleSave = () => {
        if (!measurement) return;

        const sbp = parseInt(actualSbp);
        const dbp = parseInt(actualDbp);

        if (isNaN(sbp) || isNaN(dbp)) {
            alert("Please enter valid numbers");
            return;
        }

        addCalibration(measurement.sbp, measurement.dbp, sbp, dbp);
        navigate('/measure');
    };

    const handleReset = () => {
        if (confirm("Are you sure you want to clear all calibration data?")) {
            clearCalibration();
            alert("Calibration cleared.");
            navigate('/measure');
        }
    };

    return (
        <div className="space-y-6 max-w-2xl mx-auto">
            <div className="text-center mb-8">
                <h1 className="text-3xl font-bold">Calibration</h1>
                {!measurement && (
                    <p className="text-sm opacity-60 mt-2">
                        View your calibration history below. To add a new calibration, take a measurement first.
                    </p>
                )}
            </div>

            {/* Recent Measurement & Notification (Only if measurement exists) */}
            {measurement && (
                <>
                    <div className="alert alert-info shadow-sm">
                        <AlertCircle className="w-6 h-6" />
                        <span className="text-sm">To improve accuracy, calibrate your measurement with a reference blood pressure reading from a medical device.</span>
                    </div>

                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body">
                            <h3 className="card-title text-lg mb-4">Your Recent Measurement</h3>
                            <div className="grid grid-cols-3 gap-4 text-center">
                                <div>
                                    <div className="text-xs opacity-60 uppercase mb-1">Predicted SBP</div>
                                    <div className="text-2xl font-bold text-primary">{measurement.sbp}</div>
                                    <div className="text-xs opacity-60">mmHg</div>
                                </div>
                                <div>
                                    <div className="text-xs opacity-60 uppercase mb-1">Predicted DBP</div>
                                    <div className="text-2xl font-bold text-secondary">{measurement.dbp}</div>
                                    <div className="text-xs opacity-60">mmHg</div>
                                </div>
                                <div>
                                    <div className="text-xs opacity-60 uppercase mb-1">Timestamp</div>
                                    <div className="font-mono text-sm">
                                        {new Date().toLocaleTimeString()}
                                    </div>
                                    <div className="text-xs opacity-60">{new Date().toLocaleDateString()}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Input Form */}
                    <div className="card bg-base-100 shadow-lg">
                        <div className="card-body">
                            <h3 className="card-title text-lg mb-4">Enter Reference Values</h3>
                            <p className="text-sm opacity-60 mb-6">Enter the blood pressure values from your reference device:</p>

                            <div className="form-control w-full mb-4">
                                <label className="label">
                                    <span className="label-text font-bold">Systolic Blood Pressure (SBP)</span>
                                    <span className="label-text-alt opacity-60">Normal range: 90-140 mmHg</span>
                                </label>
                                <input
                                    type="number"
                                    value={actualSbp}
                                    onChange={e => setActualSbp(e.target.value)}
                                    className="input input-bordered w-full"
                                />
                            </div>

                            <div className="form-control w-full mb-8">
                                <label className="label">
                                    <span className="label-text font-bold">Diastolic Blood Pressure (DBP)</span>
                                    <span className="label-text-alt opacity-60">Normal range: 60-90 mmHg</span>
                                </label>
                                <input
                                    type="number"
                                    value={actualDbp}
                                    onChange={e => setActualDbp(e.target.value)}
                                    className="input input-bordered w-full"
                                />
                            </div>

                            <div className="flex justify-end gap-3">
                                <button className="btn btn-ghost" onClick={() => navigate('/measure')}>Cancel</button>
                                <button className="btn btn-primary" onClick={handleSave}>
                                    <Save className="w-4 h-4 mr-2" />
                                    Save Calibration
                                </button>
                            </div>
                        </div>
                    </div>
                </>
            )}

            {/* Calibration History */}
            {calibration.history.length > 0 && (
                <div className="card bg-base-100 shadow-sm">
                    <div className="card-body">
                        <h3 className="font-bold text-lg mb-2">Calibration History</h3>
                        <div className="overflow-x-auto">
                            <table className="table table-zebra w-full">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Ref (Actual)</th>
                                        <th>Measured</th>
                                        <th>Offset</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {calibration.history.map((entry, idx) => (
                                        <tr key={idx}>
                                            <td className="text-xs opacity-70">
                                                {new Date(entry.date).toLocaleDateString()}
                                                <br />
                                                {new Date(entry.date).toLocaleTimeString()}
                                            </td>
                                            <td className="font-bold">
                                                {entry.actualSbp}/{entry.actualDbp}
                                            </td>
                                            <td className="opacity-70 text-xs">
                                                {entry.predictedSbp}/{entry.predictedDbp}
                                            </td>
                                            <td className="font-mono text-xs">
                                                {entry.sbpOffset > 0 ? '+' : ''}{entry.sbpOffset} / {entry.dbpOffset > 0 ? '+' : ''}{entry.dbpOffset}
                                            </td>
                                            <td>
                                                <button
                                                    onClick={() => deleteCalibration(idx)}
                                                    className="btn btn-ghost btn-xs text-error"
                                                    title="Remove this calibration">
                                                    Remove
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Tips */}
            <div className="card bg-base-100 shadow-sm">
                <div className="card-body">
                    <h3 className="font-bold text-lg mb-2">Calibration Tips</h3>
                    <ul className="list-disc list-inside text-sm opacity-80 space-y-1">
                        <li>Use a clinically validated blood pressure monitor for reference</li>
                        <li>Take the reference measurement immediately after the app measurement</li>
                    </ul>
                </div>
            </div>

            {/* Reset */}
            <div className="card border border-error/20">
                <div className="card-body">
                    <h3 className="font-bold text-error">Reset Calibrations</h3>
                    <p className="text-sm opacity-70 mb-4">
                        Clear all saved calibration data. This will remove all calibration adjustments and return to using raw model predictions.
                    </p>
                    <div className="text-right">
                        <button onClick={handleReset} className="btn btn-outline btn-error btn-sm">
                            <RotateCcw className="w-4 h-4 mr-2" />
                            Reset All Calibrations
                        </button>
                    </div>
                </div>
            </div>

        </div>
    );
}
