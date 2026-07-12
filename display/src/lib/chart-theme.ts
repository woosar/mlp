/**
 * Chart color roles, resolved per mode. Values follow a CVD-validated palette:
 * class A / class B are the diverging poles (blue <-> red) of the probability
 * field, so the scatter points and the contour fill share one color story.
 */
export interface ChartTheme {
    surface: string;
    ink: string;
    muted: string;
    grid: string;
    classA: string;
    classB: string;
    accent: string;
    accentTrack: string;
    good: string;
    bad: string;
    fontFamily: string;
}

const fontFamily = "'Geist Variable', sans-serif";

const light: ChartTheme = {
    surface: "#ffffff",
    ink: "#0b0b0b",
    muted: "#898781",
    grid: "#e1e0d9",
    classA: "#2a78d6",
    classB: "#e34948",
    accent: "#2a78d6",
    accentTrack: "#cde2fb",
    good: "#006300",
    bad: "#d03b3b",
    fontFamily,
};

const dark: ChartTheme = {
    surface: "#171717",
    ink: "#ffffff",
    muted: "#898781",
    grid: "#2c2c2a",
    classA: "#3987e5",
    classB: "#e66767",
    accent: "#3987e5",
    accentTrack: "#0d366b",
    good: "#0ca30c",
    bad: "#e66767",
    fontFamily,
};

export const chartTheme = (isDark: boolean): ChartTheme => (isDark ? dark : light);
