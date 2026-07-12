import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import DecisionBoundary from "@/components/DecisionBoundary.tsx";
import LossChart from "@/components/LossChart.tsx";
import { chartTheme } from "@/lib/chart-theme.ts";
import { useIsDark } from "@/hooks/use-is-dark.ts";
import {
    EMIT_INTERVAL,
    TOTAL_EPOCHS,
    TrainingStatus,
    useTrainingSession,
} from "@/hooks/use-training-session.ts";

const formatLoss = (value: number | null): string => {
    if (value === null) return "—";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs >= 10000 || abs < 0.001)) return value.toExponential(2);
    return value.toFixed(4);
};

const STATUS_LABEL: Record<TrainingStatus, string> = {
    idle: "Idle",
    training: "Training",
    trained: "Trained",
};

const StatTile = ({
    label,
    value,
    detail,
}: {
    label: string;
    value: string;
    detail?: ReactNode;
}) => (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="mt-1 truncate text-2xl font-semibold tracking-tight">{value}</div>
        {detail && <div className="mt-0.5 text-xs text-muted-foreground">{detail}</div>}
    </div>
);

const LegendChip = ({ color, label }: { color: string; label: string }) => (
    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span
            className="size-2.5 rounded-full ring-2 ring-card"
            style={{ backgroundColor: color }}
        />
        {label}
    </span>
);

const Dashboard = () => {
    const isDark = useIsDark();
    const theme = chartTheme(isDark);
    const session = useTrainingSession();

    const isTraining = session.status === "training";
    const previousLoss =
        session.lossHistory.length > 1
            ? session.lossHistory[session.lossHistory.length - 2].loss
            : null;
    const lossDelta =
        session.loss !== null && previousLoss !== null && previousLoss !== 0
            ? (session.loss - previousLoss) / Math.abs(previousLoss)
            : null;

    const statusDotColor = isTraining
        ? theme.accent
        : session.status === "trained"
          ? theme.good
          : theme.muted;

    return (
        <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
            <header className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
                <div className="flex items-baseline gap-3">
                    <h1 className="text-sm font-semibold tracking-tight">MLP Playground</h1>
                    <span className="text-xs text-muted-foreground tabular-nums">
                        2 · 24 · 24 · 24 · 2 — ReLU, softmax
                    </span>
                </div>
                <div className="flex items-center gap-4">
                    <span className="hidden text-xs text-muted-foreground sm:block">
                        press <kbd className="rounded border border-border px-1 font-sans">D</kbd>{" "}
                        to toggle theme
                    </span>
                    <span className="flex items-center gap-2 rounded-full border border-border px-2.5 py-1 text-xs">
                        <span
                            className={`size-2 rounded-full ${isTraining ? "animate-pulse" : ""}`}
                            style={{ backgroundColor: statusDotColor }}
                        />
                        {STATUS_LABEL[session.status]}
                    </span>
                </div>
            </header>

            <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_320px] gap-3 p-3">
                <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card">
                    <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-2.5">
                        <div>
                            <h2 className="text-sm font-medium">Decision boundary</h2>
                            <p className="text-xs text-muted-foreground">
                                P(class A) over the input plane
                            </p>
                        </div>
                        <div className="flex items-center gap-3">
                            <LegendChip color={theme.classA} label="Class A" />
                            <LegendChip color={theme.classB} label="Class B" />
                        </div>
                    </div>
                    <div className="min-h-0 flex-1 p-1.5">
                        {session.prediction ? (
                            <DecisionBoundary
                                prediction={session.prediction}
                                trainingData={session.trainingData}
                                theme={theme}
                            />
                        ) : (
                            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                                Waiting for the network…
                            </div>
                        )}
                    </div>
                </section>

                <aside className="flex min-h-0 flex-col gap-3">
                    <div className="grid shrink-0 grid-cols-2 gap-3">
                        <StatTile
                            label="Epoch"
                            value={session.epoch.toLocaleString()}
                            detail={`of ${TOTAL_EPOCHS.toLocaleString()}`}
                        />
                        <StatTile
                            label="Loss"
                            value={formatLoss(session.loss)}
                            detail={
                                lossDelta !== null ? (
                                    <span
                                        style={{
                                            color: lossDelta <= 0 ? theme.good : theme.bad,
                                        }}
                                        className="tabular-nums"
                                    >
                                        {lossDelta <= 0 ? "▾" : "▴"}{" "}
                                        {Math.abs(lossDelta * 100).toFixed(1)}% vs last
                                    </span>
                                ) : (
                                    "last emitted batch"
                                )
                            }
                        />
                    </div>

                    <div className="shrink-0 rounded-xl border border-border bg-card px-4 py-3">
                        <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Progress</span>
                            <span className="font-medium tabular-nums">
                                {Math.round(session.progress * 100)}%
                            </span>
                        </div>
                        <div
                            className="mt-2 h-1.5 overflow-hidden rounded-full"
                            style={{ backgroundColor: theme.accentTrack }}
                        >
                            <div
                                className="h-full rounded-full transition-[width] duration-500 ease-out"
                                style={{
                                    width: `${session.progress * 100}%`,
                                    backgroundColor: theme.accent,
                                }}
                            />
                        </div>
                    </div>

                    <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card">
                        <div className="shrink-0 border-b border-border px-4 py-2.5">
                            <h2 className="text-sm font-medium">Training loss</h2>
                            <p className="text-xs text-muted-foreground">
                                log scale, one point per {EMIT_INTERVAL} epochs
                            </p>
                        </div>
                        <div className="min-h-0 flex-1 p-2">
                            <LossChart
                                history={session.lossHistory}
                                totalEpochs={TOTAL_EPOCHS}
                                theme={theme}
                            />
                        </div>
                    </section>

                    <Button
                        size="lg"
                        className="w-full shrink-0"
                        disabled={isTraining}
                        onClick={session.startTraining}
                    >
                        {isTraining ? (
                            <>
                                <span className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                                Training…
                            </>
                        ) : session.status === "trained" ? (
                            "Train again"
                        ) : (
                            "Start training"
                        )}
                    </Button>
                </aside>
            </main>
        </div>
    );
};

export default Dashboard;
