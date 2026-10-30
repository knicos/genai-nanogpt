export default function multinomialCPU(probs: number[], randomNumber?: number): number {
    let cdf = 0;
    const rand = randomNumber ?? Math.random();
    for (let i = 0; i < probs.length; i++) {
        cdf += probs[i];
        if (rand < cdf) {
            return i; //tensor2d([[i]], [1, 1], 'int32');
        }
    }

    return probs.length - 1; //tensor2d([[probs.length - 1]], [1, 1], 'int32');
}
