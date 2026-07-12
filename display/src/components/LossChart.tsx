import { useEffect, useMemo, useRef, useState } from "react";
import { LossPoint } from "@/hooks/use-training-session.ts";
import { ChartTheme } from "@/lib/chart-theme.ts";

interface LossChartProps {
    history: LossPoint[];
    totalEpochs: number;
    theme: ChartTheme;
}

const MARGIN = { top: 10, right: 12, bottom: 22, left: 44 };

const formatTick = (value: number): string => {
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (abs >= 10000 || abs < 0.001) return value.toExponential(0);
    if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`;
    if (abs >= 1) return value.toPrecision(3).replace(/\.?0+$/, "");
    return value.toPrecision(2);
};

const formatEpoch = (epoch: number): string =>
    epoch >= 1000 ? `${(epoch / 1000).toFixed(epoch % 1000 === 0 ? 0 : 1)}k` : `${epoch}`;

const useContainerSize = () => {
    const ref = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const element = ref.current;
        if (!element) return;
        const observer = new ResizeObserver((entries) => {
            const rect = entries[0].contentRect;
            setSize({ width: rect.width, height: rect.height });
        });
        observer.observe(element);
        return () => observer.disconnect();
    }, []);

    return { ref, size };
};

const LossChart = ({ history, totalEpochs, theme }: LossChartProps) => {
    const { ref, size } = useContainerSize();
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    const plot = useMemo(() => {
        const innerWidth = Math.max(size.width - MARGIN.left - MARGIN.right, 0);
        const innerHeight = Math.max(size.height - MARGIN.top - MARGIN.bottom, 0);
        if (history.length === 0 || innerWidth === 0 || innerHeight === 0) return null;

        const losses = history.map((p) => p.loss);
        // Log scale reads loss decay best; fall back to linear near/below zero.
        const useLog = losses.every((l) => l > 0);
        const transform = useLog ? Math.log10 : (v: number) => v;

        let lo = Math.min(...losses.map(transform));
        let hi = Math.max(...losses.map(transform));
        if (hi - lo < 1e-9) {
            lo -= useLog ? 0.5 : Math.abs(lo) * 0.5 + 1e-6;
            hi += useLog ? 0.5 : Math.abs(hi) * 0.5 + 1e-6;
        }
        const pad = (hi - lo) * 0.08;
        lo -= pad;
        hi += pad;

        const xFor = (epoch: number) => MARGIN.left + (epoch / totalEpochs) * innerWidth;
        const yFor = (loss: number) =>
            MARGIN.top + innerHeight - ((transform(loss) - lo) / (hi - lo)) * innerHeight;

        const points = history.map((p) => ({ ...p, x: xFor(p.epoch), y: yFor(p.loss) }));
        const linePath = points
            .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
            .join("");
        const baseline = MARGIN.top + innerHeight;
        const areaPath = `${linePath}L${points[points.length - 1].x.toFixed(1)},${baseline}L${points[0].x.toFixed(1)},${baseline}Z`;

        const tickCount = 3;
        const yTicks = Array.from({ length: tickCount }, (_, i) => {
            const t = lo + ((i + 0.5) / tickCount) * (hi - lo);
            const value = useLog ? Math.pow(10, t) : t;
            return { y: yFor(value), label: formatTick(value) };
        });

        const xTicks = [0.25, 0.5, 0.75, 1].map((f) => ({
            x: MARGIN.left + f * innerWidth,
            label: formatEpoch(Math.round(f * totalEpochs)),
        }));

        return { points, linePath, areaPath, yTicks, xTicks, baseline };
    }, [history, size, totalEpochs]);

    const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
        if (!plot) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - rect.left;
        let nearest = 0;
        let nearestDistance = Infinity;
        plot.points.forEach((p, i) => {
            const distance = Math.abs(p.x - x);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearest = i;
            }
        });
        setHoverIndex(nearest);
    };

    const hovered = plot && hoverIndex !== null ? plot.points[hoverIndex] : null;
    const lastPoint = plot ? plot.points[plot.points.length - 1] : null;
    const tooltipOnLeft = hovered !== null && hovered.x > size.width * 0.62;

    return (
        <div ref={ref} className="relative h-full w-full">
            {plot ? (
                <svg
                    width={size.width}
                    height={size.height}
                    className="absolute inset-0"
                    onPointerMove={handlePointerMove}
                    onPointerLeave={() => setHoverIndex(null)}
                >
                    {plot.yTicks.map((tick) => (
                        <g key={tick.y}>
                            <line
                                x1={MARGIN.left}
                                x2={size.width - MARGIN.right}
                                y1={tick.y}
                                y2={tick.y}
                                stroke={theme.grid}
                                strokeWidth={1}
                            />
                            <text
                                x={MARGIN.left - 8}
                                y={tick.y + 3.5}
                                textAnchor="end"
                                fontSize={10}
                                fill={theme.muted}
                                style={{ fontVariantNumeric: "tabular-nums" }}
                            >
                                {tick.label}
                            </text>
                        </g>
                    ))}
                    {plot.xTicks.map((tick) => (
                        <text
                            key={tick.x}
                            x={tick.x}
                            y={plot.baseline + 14}
                            textAnchor="middle"
                            fontSize={10}
                            fill={theme.muted}
                            style={{ fontVariantNumeric: "tabular-nums" }}
                        >
                            {tick.label}
                        </text>
                    ))}

                    <path d={plot.areaPath} fill={theme.accent} fillOpacity={0.1} />
                    <path
                        d={plot.linePath}
                        fill="none"
                        stroke={theme.accent}
                        strokeWidth={2}
                        strokeLinejoin="round"
                        strokeLinecap="round"
                    />

                    {lastPoint && (
                        <circle
                            cx={lastPoint.x}
                            cy={lastPoint.y}
                            r={4}
                            fill={theme.accent}
                            stroke={theme.surface}
                            strokeWidth={2}
                        />
                    )}

                    {hovered && (
                        <g>
                            <line
                                x1={hovered.x}
                                x2={hovered.x}
                                y1={MARGIN.top}
                                y2={plot.baseline}
                                stroke={theme.muted}
                                strokeWidth={1}
                                strokeOpacity={0.5}
                            />
                            <circle
                                cx={hovered.x}
                                cy={hovered.y}
                                r={4.5}
                                fill={theme.accent}
                                stroke={theme.surface}
                                strokeWidth={2}
                            />
                        </g>
                    )}
                </svg>
            ) : (
                <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                    Run training to record loss
                </div>
            )}

            {hovered && (
                <div
                    className="pointer-events-none absolute z-10 rounded-md border border-border bg-popover px-2.5 py-1.5 shadow-sm"
                    style={{
                        left: tooltipOnLeft ? undefined : hovered.x + 10,
                        right: tooltipOnLeft ? size.width - hovered.x + 10 : undefined,
                        top: Math.max(hovered.y - 34, 4),
                    }}
                >
                    <div className="text-sm font-semibold text-popover-foreground">
                        {formatTick(hovered.loss)}
                    </div>
                    <div className="text-[10px] text-muted-foreground tabular-nums">
                        epoch {hovered.epoch.toLocaleString()}
                    </div>
                </div>
            )}
        </div>
    );
};

export default LossChart;
