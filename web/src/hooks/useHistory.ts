import { useState, useEffect } from 'react';

export interface MeasurementRecord {
    id: string;
    timestamp: string; // ISO string
    sbp: number;
    dbp: number;
    hr: number;
    calibrated: boolean;
}

const STORAGE_KEY = 'bpm_history_v1';

export function useHistory() {
    const [history, setHistory] = useState<MeasurementRecord[]>([]);

    // Load from local storage on mount
    useEffect(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            try {
                setHistory(JSON.parse(stored));
            } catch (e) {
                console.error('Failed to parse history', e);
            }
        }
    }, []);

    const saveMeasurement = (record: Omit<MeasurementRecord, 'id' | 'timestamp'>) => {
        const newRecord: MeasurementRecord = {
            ...record,
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
        };

        const newHistory = [newRecord, ...history];
        setHistory(newHistory);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
    };

    const deleteMeasurement = (id: string) => {
        const newHistory = history.filter(h => h.id !== id);
        setHistory(newHistory);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newHistory));
    };

    const clearHistory = () => {
        setHistory([]);
        localStorage.removeItem(STORAGE_KEY);
    };

    return { history, saveMeasurement, deleteMeasurement, clearHistory };
}
