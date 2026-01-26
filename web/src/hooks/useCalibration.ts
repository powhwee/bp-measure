import { useState, useEffect } from 'react';

const STORAGE_KEY = 'bpm_calibration_v1';

interface CalibrationData {
    sbpOffset: number;
    dbpOffset: number;
    lastCalibrated: string | null; // ISO Date
    history: Array<{
        date: string;
        predictedSbp: number;
        predictedDbp: number;
        actualSbp: number;
        actualDbp: number;
        sbpOffset: number;
        dbpOffset: number;
    }>;
}

export function useCalibration() {
    const [calibration, setCalibration] = useState<CalibrationData>({
        sbpOffset: 0,
        dbpOffset: 0,
        lastCalibrated: null,
        history: []
    });

    useEffect(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            try {
                setCalibration(JSON.parse(stored));
            } catch (e) {
                console.error('Failed to parse calibration', e);
            }
        }
    }, []);

    const addCalibration = (predictedSbp: number, predictedDbp: number, actualSbp: number, actualDbp: number) => {
        const sbpOffset = actualSbp - predictedSbp;
        const dbpOffset = actualDbp - predictedDbp;

        // In a real app, we might use a weighted average of history
        // For now, we take the latest calibration as the source of truth for simplicity, 
        // effectively "zeroing" the error on the most recent measurement.

        const newEntry = {
            date: new Date().toISOString(),
            predictedSbp,
            predictedDbp,
            actualSbp,
            actualDbp,
            sbpOffset,
            dbpOffset
        };

        const newHistory = [newEntry, ...calibration.history].slice(0, 5); // Keep last 5

        // Simple strategy: Average of last 3 calibrations to smooth out noise
        // Or just use the latest. Let's use latest for immediate feedback.

        const newData = {
            sbpOffset: sbpOffset,
            dbpOffset: dbpOffset,
            lastCalibrated: new Date().toISOString(),
            history: newHistory
        };

        setCalibration(newData);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newData));
    };

    const clearCalibration = () => {
        const newData = {
            sbpOffset: 0,
            dbpOffset: 0,
            lastCalibrated: null,
            history: []
        };
        setCalibration(newData);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newData));
    };

    const deleteCalibration = (index: number) => {
        const newHistory = [...calibration.history];
        newHistory.splice(index, 1);

        let sbpOffset = 0;
        let dbpOffset = 0;
        let lastCalibrated = null;

        if (newHistory.length > 0) {
            // Revert to the most recent remaining calibration
            sbpOffset = newHistory[0].sbpOffset;
            dbpOffset = newHistory[0].dbpOffset;
            lastCalibrated = newHistory[0].date;
        }

        const newData = {
            sbpOffset,
            dbpOffset,
            lastCalibrated,
            history: newHistory
        };

        setCalibration(newData);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newData));
    };

    const applyCalibration = (sbp: number, dbp: number) => {
        if (!calibration.lastCalibrated) return { sbp, dbp, isCalibrated: false };

        return {
            sbp: Math.round(sbp + calibration.sbpOffset),
            dbp: Math.round(dbp + calibration.dbpOffset),
            isCalibrated: true
        };
    };

    return { calibration, addCalibration, clearCalibration, deleteCalibration, applyCalibration };
}
