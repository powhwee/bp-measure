import { Trash2 } from 'lucide-react';
import { useHistory } from '../hooks/useHistory';
import { useCalibration } from '../hooks/useCalibration';

export function History() {
    const { history, deleteMeasurement, clearHistory } = useHistory();
    const { clearCalibration } = useCalibration();

    const handleClearAll = () => {
        clearHistory();
        clearCalibration();
    };

    const formatDate = (isoString: string) => {
        return new Date(isoString).toLocaleString('en-GB', {
            day: '2-digit', month: '2-digit', year: 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    };

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center bg-base-100 p-6 rounded-box shadow-sm">
                <h1 className="text-3xl font-bold">Measurement History</h1>
            </div>

            <div className="card bg-base-100 shadow-xl">
                <div className="card-body">
                    <div className="flex justify-between items-center mb-4">
                        <span className="text-sm opacity-70">Showing {history.length} measurement(s)</span>
                        {history.length > 0 && (
                            <button onClick={handleClearAll} className="btn btn-outline btn-error btn-sm">
                                Clear History
                            </button>
                        )}
                    </div>

                    {history.length === 0 ? (
                        <div className="text-center py-10 opacity-50">
                            No measurements recorded yet.
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="table table-zebra w-full">
                                <thead>
                                    <tr>
                                        <th>Timestamp</th>
                                        <th>SBP</th>
                                        <th>DBP</th>
                                        <th>HR</th>
                                        <th>Calibrated</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {history.map((record) => (
                                        <tr key={record.id}>
                                            <td className="font-mono text-sm">{formatDate(record.timestamp)}</td>
                                            <td className="font-bold">{record.sbp} <span className="text-xs font-normal opacity-70">mmHg</span></td>
                                            <td className="font-bold">{record.dbp} <span className="text-xs font-normal opacity-70">mmHg</span></td>
                                            <td className="font-bold">{record.hr} <span className="text-xs font-normal opacity-70">BPM</span></td>
                                            <td>
                                                {record.calibrated ? (
                                                    <span className="badge badge-success badge-sm">Yes</span>
                                                ) : (
                                                    <span className="badge badge-ghost badge-sm">No</span>
                                                )}
                                            </td>
                                            <td>
                                                <button
                                                    onClick={() => deleteMeasurement(record.id)}
                                                    className="btn btn-ghost btn-xs text-error"
                                                    title="Delete"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>

            <div className="card bg-base-200 shadow-inner">
                <div className="card-body">
                    <h3 className="card-title text-base">About This History</h3>
                    <p className="text-sm opacity-80">
                        This history is stored in your browser's local storage and will persist across sessions.
                        The data stays on your device and is never sent to the server.
                    </p>
                </div>
            </div>
        </div>
    );
}
