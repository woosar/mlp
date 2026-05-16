import { invoke } from "@tauri-apps/api/core";

export async function greetel(name: string): Promise<string> {
    return await invoke("greet", { name });
}

export async function calculate(input: number[]): Promise<RawData> {
    return await invoke("calculate", { input });
}

export async function train() {
    return await invoke("train", {});
}

export interface RawData {
    //todo: move to different file
    input: number[][];
    output: number[][];
    prediction: number[][];
}
