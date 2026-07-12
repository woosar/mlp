import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { calculate, train, RawData } from "@/utilities/invocations.ts";
import { createFlattenedGrid } from "@/utilities/mathy_stuff.ts";

// Mirrors the values hardcoded in src-tauri/src/lib.rs — keep in sync until
// the backend exposes a config/completion event.
export const TOTAL_EPOCHS = 10000;
export const EMIT_INTERVAL = 10;

export interface LossPoint {
    epoch: number;
    loss: number;
}

export type TrainingStatus = "idle" | "training" | "trained";

export interface TrainingSession {
    prediction: RawData | null;
    trainingData: RawData | null;
    epoch: number;
    loss: number | null;
    lossHistory: LossPoint[];
    status: TrainingStatus;
    progress: number;
    startTraining: () => void;
}

/**
 * Owns all Tauri event wiring: listens for training-data / epoch-completed /
 * loss events and re-runs grid inference whenever the network advances.
 */
export function useTrainingSession(): TrainingSession {
    const gridInput = useMemo(() => createFlattenedGrid(), []);
    const [prediction, setPrediction] = useState<RawData | null>(null);
    const [trainingData, setTrainingData] = useState<RawData | null>(null);
    const [epoch, setEpoch] = useState(0);
    const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
    const [status, setStatus] = useState<TrainingStatus>("idle");
    // The backend emits "epoch-completed" then "loss" back to back; the ref
    // pairs each loss value with the epoch it belongs to.
    const epochRef = useRef(0);

    useEffect(() => {
        let cancelled = false;
        let busy = false;

        const runInference = async () => {
            if (busy) return;
            busy = true;
            try {
                const result = await calculate(gridInput);
                if (!cancelled) setPrediction(result);
            } catch (error) {
                console.error("Tauri invocation failed:", error);
            } finally {
                busy = false;
            }
        };

        void runInference();

        const unlistenPromises = [
            listen<RawData>("training-data", (event) => {
                if (!cancelled) setTrainingData(event.payload);
            }),
            listen<number>("epoch-completed", (event) => {
                if (cancelled) return;
                epochRef.current = event.payload;
                setEpoch(event.payload);
                if (event.payload + EMIT_INTERVAL >= TOTAL_EPOCHS) {
                    setStatus("trained");
                }
                void runInference();
            }),
            listen<number>("loss", (event) => {
                if (cancelled) return;
                const point = { epoch: epochRef.current, loss: event.payload };
                setLossHistory((history) => [...history, point]);
            }),
        ];

        return () => {
            cancelled = true;
            unlistenPromises.forEach((promise) => {
                promise.then((unlisten) => unlisten());
            });
        };
    }, [gridInput]);

    const startTraining = useCallback(() => {
        setStatus("training");
        setEpoch(0);
        epochRef.current = 0;
        setLossHistory([]);
        void train();
    }, []);

    const loss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : null;
    const progress =
        status === "trained" ? 1 : status === "idle" ? 0 : Math.min(epoch / TOTAL_EPOCHS, 1);

    return {
        prediction,
        trainingData,
        epoch,
        loss,
        lossHistory,
        status,
        progress,
        startTraining,
    };
}
