export default function arrayShape(arr: unknown[]): number[] {
    const shape: number[] = [];
    let current: unknown = arr;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[current.length - 1];
    }
    return shape;
}
