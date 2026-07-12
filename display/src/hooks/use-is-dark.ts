import { useEffect, useState } from "react";
import { useTheme } from "@/components/theme-provider.tsx";

const COLOR_SCHEME_QUERY = "(prefers-color-scheme: dark)";

/** Resolves the theme-provider value ("system" included) to a boolean. */
export function useIsDark(): boolean {
    const { theme } = useTheme();
    const [systemDark, setSystemDark] = useState(
        () => window.matchMedia(COLOR_SCHEME_QUERY).matches,
    );

    useEffect(() => {
        const mediaQuery = window.matchMedia(COLOR_SCHEME_QUERY);
        const handleChange = () => setSystemDark(mediaQuery.matches);
        mediaQuery.addEventListener("change", handleChange);
        return () => mediaQuery.removeEventListener("change", handleChange);
    }, []);

    return theme === "dark" || (theme === "system" && systemDark);
}
