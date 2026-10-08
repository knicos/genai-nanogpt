export default class MaxHeap<T> {
    private readonly fn: (a: T) => number;
    private readonly heap: T[] = [];

    constructor(fn: (a: T) => number) {
        this.fn = fn;
    }

    public get length(): number {
        return this.heap.length;
    }

    public get size(): number {
        return this.heap.length;
    }

    public get values(): T[] {
        return [...this.heap];
    }

    public isEmpty(): boolean {
        return this.heap.length === 0;
    }

    public peek(): T | undefined {
        return this.heap[0];
    }

    public push(item: T): void {
        this.heap.push(item);
        this.siftUp(this.heap.length - 1);
    }

    public pop(): T | undefined {
        if (this.heap.length === 0) {
            return undefined;
        }

        const max = this.heap[0];
        const last = this.heap.pop();

        if (this.heap.length > 0 && last !== undefined) {
            this.heap[0] = last;
            this.siftDown(0);
        }

        return max;
    }

    public clear(): void {
        this.heap.length = 0;
    }

    public toArray(): T[] {
        return [...this.heap];
    }

    private siftUp(index: number): void {
        let currentIndex = index;

        while (currentIndex > 0) {
            const parentIndex = Math.floor((currentIndex - 1) / 2);
            if (this.fn(this.heap[currentIndex]) <= this.fn(this.heap[parentIndex])) {
                break;
            }

            [this.heap[currentIndex], this.heap[parentIndex]] = [this.heap[parentIndex], this.heap[currentIndex]];
            currentIndex = parentIndex;
        }
    }

    private siftDown(index: number): void {
        let currentIndex = index;

        while (true) {
            const leftIndex = currentIndex * 2 + 1;
            const rightIndex = currentIndex * 2 + 2;
            let largestIndex = currentIndex;

            if (leftIndex < this.heap.length && this.fn(this.heap[leftIndex]) > this.fn(this.heap[largestIndex])) {
                largestIndex = leftIndex;
            }

            if (rightIndex < this.heap.length && this.fn(this.heap[rightIndex]) > this.fn(this.heap[largestIndex])) {
                largestIndex = rightIndex;
            }

            if (largestIndex === currentIndex) {
                break;
            }

            [this.heap[currentIndex], this.heap[largestIndex]] = [this.heap[largestIndex], this.heap[currentIndex]];
            currentIndex = largestIndex;
        }
    }
}
