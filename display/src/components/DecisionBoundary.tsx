import { useMemo } from "react";
import Plot from "react-plotly.js";
import { RawData } from "@/utilities/invocations.ts";
import { ChartTheme } from "@/lib/chart-theme.ts";

interface DecisionBoundaryProps {
    prediction: RawData | null;
    trainingData: RawData | null;
    theme: ChartTheme;
}

interface ClassPoints {
    x: number[];
    y: number[];
}

const splitClasses = (trainingData: RawData | null): { a: ClassPoints; b: ClassPoints } => {
    const a: ClassPoints = { x: [], y: [] };
    const b: ClassPoints = { x: [], y: [] };

    if (trainingData?.input?.length === 2 && trainingData.output?.[0]) {
        const [xs, ys] = trainingData.input;
        trainingData.output[0].forEach((value, i) => {
            const target = value > 0.5 ? a : b;
            target.x.push(xs[i]);
            target.y.push(ys[i]);
        });
    }

    return { a, b };
};

const DecisionBoundary = ({ prediction, trainingData, theme }: DecisionBoundaryProps) => {
    const classes = useMemo(() => splitClasses(trainingData), [trainingData]);

    const field = prediction
        ? {
              x: prediction.input[0] ?? [],
              y: prediction.input[1] ?? [],
              z: prediction.prediction[0] ?? [],
          }
        : { x: [], y: [], z: [] };

    // Diverging probability field: class B pole -> transparent midpoint -> class A pole.
    // Muted alpha keeps the field a backdrop so the training points stay in front.
    const colorscale: Array<[number, string]> = [
        [0, `${theme.classB}73`],
        [0.5, `${theme.surface}00`],
        [1, `${theme.classA}73`],
    ];

    return (
        <Plot
            data={[
                {
                    type: "contour",
                    x: field.x,
                    y: field.y,
                    z: field.z,
                    zmin: 0,
                    zmax: 1,
                    colorscale,
                    contours: {
                        coloring: "heatmap",
                        showlines: true,
                        start: 0,
                        end: 1,
                        size: 0.1,
                    },
                    line: { width: 1, color: `${theme.ink}14`, smoothing: 1 },
                    colorbar: {
                        thickness: 8,
                        outlinewidth: 0,
                        tickvals: [0, 0.5, 1],
                        ticktext: ["B", "0.5", "A"],
                        tickfont: { color: theme.muted, size: 11, family: theme.fontFamily },
                        len: 0.6,
                        x: 1.02,
                    },
                    hovertemplate: "P(A) = %{z:.2f}<extra></extra>",
                    showlegend: false,
                },
                {
                    type: "scatter",
                    mode: "markers",
                    name: "Class A",
                    x: classes.a.x,
                    y: classes.a.y,
                    marker: {
                        color: theme.classA,
                        size: 9,
                        line: { color: theme.surface, width: 2 },
                    },
                    hovertemplate: "(%{x:.2f}, %{y:.2f})<extra>Class A</extra>",
                    showlegend: false,
                },
                {
                    type: "scatter",
                    mode: "markers",
                    name: "Class B",
                    x: classes.b.x,
                    y: classes.b.y,
                    marker: {
                        color: theme.classB,
                        size: 9,
                        line: { color: theme.surface, width: 2 },
                    },
                    hovertemplate: "(%{x:.2f}, %{y:.2f})<extra>Class B</extra>",
                    showlegend: false,
                },
            ]}
            layout={{
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                autosize: true,
                margin: { t: 8, b: 36, l: 40, r: 48 },
                font: { family: theme.fontFamily, color: theme.muted, size: 11 },
                hovermode: "closest",
                hoverlabel: {
                    bgcolor: theme.surface,
                    bordercolor: theme.grid,
                    font: { family: theme.fontFamily, color: theme.ink, size: 12 },
                },
                xaxis: {
                    range: [-1.05, 1.05],
                    constrain: "domain",
                    zeroline: false,
                    gridcolor: theme.grid,
                    linecolor: theme.grid,
                    tickfont: { color: theme.muted, size: 11 },
                    fixedrange: true,
                },
                yaxis: {
                    range: [-1.05, 1.05],
                    scaleanchor: "x",
                    scaleratio: 1,
                    constrain: "domain",
                    zeroline: false,
                    gridcolor: theme.grid,
                    linecolor: theme.grid,
                    tickfont: { color: theme.muted, size: 11 },
                    fixedrange: true,
                },
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
        />
    );
};

export default DecisionBoundary;
