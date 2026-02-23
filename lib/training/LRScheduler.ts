import { LRSchedulerConfig } from './types';

export default class LRScheduler {
    private step = 0;
    private startLearningRate: number;

    constructor(
        protected learningRate: number,
        private config: LRSchedulerConfig
    ) {
        this.startLearningRate = learningRate;
    }

    updateConfig(newConfig: Partial<LRSchedulerConfig>, learningRate?: number) {
        this.config = { ...this.config, ...newConfig };
        if (learningRate !== undefined) {
            this.startLearningRate = learningRate;
        }
    }

    get lr(): number {
        return this.learningRate;
    }

    getNextLR(): number {
        const step = this.step;

        // Warmup (linear) for steps [0, warmupSteps - 1]
        if (this.config.warmupSteps > 0 && step < this.config.warmupSteps) {
            const warmupFactor = (step + 1) / this.config.warmupSteps;
            const adjustedLearningRate = this.startLearningRate * warmupFactor;
            this.learningRate = adjustedLearningRate;
            this.step++;
            return adjustedLearningRate;
        }

        // Clamp to min LR for steps >= decaySteps
        const decaySteps = this.config.epochSteps * this.config.decayEpochs;
        if (step >= decaySteps || decaySteps <= this.config.warmupSteps) {
            this.learningRate = this.config.minLearningRate;
            this.step++;
            return this.config.minLearningRate;
        }

        // Cosine decay for steps [warmupSteps, decaySteps - 1]
        const decayRatio = (step - this.config.warmupSteps) / (decaySteps - this.config.warmupSteps);
        const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decayRatio));
        const newLR = this.config.minLearningRate + coeff * (this.startLearningRate - this.config.minLearningRate);

        this.learningRate = newLR;
        this.step++;
        return newLR;
    }
}
