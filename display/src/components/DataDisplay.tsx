import Plot from "react-plotly.js";
import { RawData } from "@/utilities/invocations.ts";

interface DataDisplayProps {
    data: RawData | null;
}

const DataDisplay = ({ data }: DataDisplayProps) => {
    let x: number[] = [];
    let y: number[] = [];
    let z: number[] = [];
    if (data) {
        x = data.input[0];
        y = data.input[1];
        z = data.prediction[0];
    }
    return (
        <div className={"w-full h-full"}>
            <Plot
                className={"h-full"}
                data={[
                    {
                        z: z,
                        x: x,
                        y: y,
                        type: "contour",
                        colorscale: "Greys",
                        contours: {
                            coloring: "heatmap",
                            showlines: true,
                            start: 0,
                            end: 1,
                            size: 0.05,
                        },
                        colorbar: {
                            len: 780,
                            lenmode: "pixels",
                            y: 0.5,
                            yanchor: "middle",
                            thickness: 22,
                        },
                        line: {
                            width: 1.5,
                            color: "rgba(255, 255, 255, 0.5)",
                        },
                    },
                ]}
                layout={{
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    autosize: true,
                    margin: { t: 5, b: 30, l: 35, r: 5 },
                    xaxis: {
                        range: [-1, 1],
                        constrain: "domain",
                        zeroline: false,
                        title: { text: "X" },
                    },
                    yaxis: {
                        range: [-1, 1],
                        scaleanchor: "x",
                        scaleratio: 1,
                        constrain: "domain",
                        zeroline: false,
                        title: { text: "Y" },
                    },
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
            />
        </div>
    );
};

export default DataDisplay;
