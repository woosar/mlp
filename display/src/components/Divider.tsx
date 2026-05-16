import DataDisplay from "@/components/DataDisplay.tsx";
import { calculate, RawData, train } from "@/utilities/invocations.ts";
import { useEffect, useMemo, useState } from "react";
import { createFlattenedGrid } from "@/utilities/mathy_stuff.ts";
import { Button } from "@/components/ui/button";
import { listen } from "@tauri-apps/api/event";

const Divider = () => {
    const input = useMemo(() => createFlattenedGrid(), []);
    const [data, setData] = useState<RawData | null>(null);
    const [loss, setLoss] = useState<number>(0);
    const [epoch, setEpoch] = useState<number>(0);

    useEffect(() => {
        let isCalculating = false;

        const runInference = async () => {
            if (isCalculating) return;

            isCalculating = true;
            try {
                const result = await calculate(input);
                setData(result);
            } catch (error) {
                console.error("Tauri invocation failed:", error);
            } finally {
                isCalculating = false;
            }
        };

        void runInference();

        const unlistenPromise = listen<number>("epoch-completed", (event) => {
            setEpoch(event.payload);
            void runInference();
        });

        return () => {
            unlistenPromise.then((unlisten) => unlisten());
        };
    }, [input]);

    useEffect(() => {
        const unlistenLossPromise = listen<number>("loss", (event) => {
            setLoss(event.payload);
        });

        return () => {
            unlistenLossPromise.then((unlisten) => unlisten());
        };
    }, []);
    return (
        <div className={"w-398 h-223 bg-primary-foreground m-1 rounded p-1 flex space-x-1"}>
            <div className={"h-221 w-221 bg-sidebar-border rounded"}>
                <DataDisplay data={data} />
            </div>
            <div className={"h-221 w-173 bg-sidebar-border rounded"}>
                <Button onClick={() => train()}>Train</Button>Loss:{loss}, Epoch: {epoch}
            </div>
        </div>
    );
};

export default Divider;
