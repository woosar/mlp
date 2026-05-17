export const linear_space = (from: number, to: number, number_of_elements: number = 50) => {
    const step_width = (to - from) / (number_of_elements - 1);
    return Array.from({ length: number_of_elements }).map((_, idx) => from + idx * step_width);
};

export const createFlattenedGrid = () => {
    const grid = linear_space(-1, 1, 51);
    return grid.flatMap((x) => grid.flatMap((y) => [x, y]));
};
