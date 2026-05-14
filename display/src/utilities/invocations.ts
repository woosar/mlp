import { invoke } from "@tauri-apps/api/core";
export async function greetel(name:string):Promise<string>{
    return await invoke("greet", { name })
}