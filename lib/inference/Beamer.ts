import type { Conversation, ITokeniser } from '../tokeniser/type';
import Model, { ModelForwardAttributes } from '../models/model';
import { BeamerOptions, IBeam, IGenerateOptions, IGeneratorOutput } from './types';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor, concat, oneHot, scalar, stack, tensor1d, tidy, softmax } from '@tensorflow/tfjs-core';
import { CHARS, padArray } from './utilities';
import topP from '@base/utilities/topP';
import MaxHeap from '../utilities/maxHeap';

interface BeamNode {
    parent?: BeamNode;
    context: Tensor;
    contextLength: number;
    terminated: boolean;
    tokens: number[];
    score: number;
    probabilities: number[];
    text: string;
    sortedChildren?: { token: number; prob: number }[];
    nextChild: number;
}

interface BeamEntry {
    idx: number;
    tokenId: number;
    tokenText: string;
    probability: number;
    terminated: boolean;
    score: number;
    reason?: 'token' | 'whitespace' | 'length';
    parent: BeamNode;
}

function topTokenCandidates(probs: number[], maxCandidates?: number): { token: number; prob: number }[] {
    const sorted = probs
        .map((prob, token) => ({ token, prob }))
        .filter((candidate) => candidate.prob > 0)
        .sort((a, b) => b.prob - a.prob);

    if (maxCandidates !== undefined) {
        return sorted.slice(0, Math.max(1, maxCandidates));
    }
    return sorted;
}

export default class Beamer {
    private actualTokeniser: ITokeniser;
    private active = false;
    private options: BeamerOptions | null = null;
    private conversation: Conversation[] = [];
    private heap = new MaxHeap<BeamEntry>((node) => node.score);
    private hasRoot = false;
    private totalCandidates = 0;
    private seen = new Set<string>();

    constructor(
        private readonly model: Model<ModelForwardAttributes>,
        private readonly tokeniser: ITokeniser
    ) {
        this.actualTokeniser = tokeniser;
    }

    private shouldTerminate(allowSpecial: boolean, token: number): boolean {
        if (allowSpecial) return false;
        const assistantEndToken = this.tokeniser.getSpecialTokenIndex('<|assistant_end|>');
        return token === this.actualTokeniser.eosToken || token === assistantEndToken;
    }

    private initialise(options?: IGenerateOptions): void {
        const tokeniser = this.tokeniser.trained
            ? this.tokeniser
            : new CharTokeniser(padArray(CHARS, this.tokeniser.vocabSize));
        this.actualTokeniser = tokeniser;

        if (options?.loraName) {
            this.model.attachLoRA(options.loraName);
        } else if (this.model.hasLoRA()) {
            this.model.detachLoRA();
        }
    }

    private async tokeniseConversation(conversation: Conversation[]): Promise<number[]> {
        const tokens = this.actualTokeniser.encodeConversation(conversation, false);
        while (
            tokens.length > 0 &&
            (tokens[tokens.length - 1] === this.actualTokeniser.eosToken ||
                tokens[tokens.length - 1] === this.actualTokeniser.getSpecialTokenIndex('<|assistant_end|>'))
        ) {
            tokens.pop();
        }
        return tokens;
    }

    private createInitialContext(tokens: number[]): { context: Tensor; contextLength: number } {
        const blockSize = this.model.config.blockSize;
        const cropped = tokens.length > blockSize ? tokens.slice(-blockSize) : tokens.slice();
        const padded = cropped.concat(Array(blockSize - cropped.length).fill(0));
        return {
            context: tensor1d(padded, 'int32'),
            contextLength: cropped.length,
        };
    }

    private appendTokenToContext(
        context: Tensor,
        contextLength: number,
        token: number
    ): { context: Tensor; contextLength: number } {
        const blockSize = this.model.config.blockSize;
        if (contextLength < blockSize) {
            const nextContext = tidy(() => {
                const mask = oneHot([contextLength], blockSize).squeeze([0]).asType('int32');
                const inverseMask = scalar(1, 'int32').sub(mask);
                const tokenVector = mask.mul(scalar(token, 'int32'));
                return context.mul(inverseMask).add(tokenVector);
            });
            return { context: nextContext, contextLength: contextLength + 1 };
        }

        const nextContext = tidy(() => {
            const tail = context.slice([1], [blockSize - 1]);
            const nextToken = tensor1d([token], 'int32');
            return concat([tail, nextToken], 0);
        });

        return { context: nextContext, contextLength: blockSize };
    }

    public cancel(): void {
        this.active = false;
    }

    public begin(conversation: Conversation[], options: BeamerOptions): void {
        if (!options || options.beams < 1 || options.maxBeamLength < 1) {
            return;
        }
        if (this.options) {
            throw new Error('Beamer has already been initialized.');
        }
        this.conversation = conversation;
        this.options = options;
        this.initialise(options);
    }

    private async expand(nodes: BeamNode[], fixedBatchSize: number): Promise<void> {
        if (nodes.length === 0) {
            return;
        }

        if (nodes.length > fixedBatchSize) {
            throw new Error(`Number of nodes (${nodes.length}) exceeds fixed batch size (${fixedBatchSize}).`);
        }

        const paddedNodes = [...nodes];
        if (nodes.length < fixedBatchSize) {
            while (paddedNodes.length < fixedBatchSize) {
                paddedNodes.push(nodes[nodes.length - 1]);
            }
        }

        const idx = stack(paddedNodes.map((b) => b.context));
        const attrs: ModelForwardAttributes = { training: false, mixedPrecision: true };
        const logits = this.model.forward(attrs, idx);
        const lastSteps = paddedNodes.map((b) => Math.max(0, b.contextLength - 1));
        const lastProbs = tidy(() => {
            const selector = oneHot(lastSteps, this.model.config.blockSize).expandDims(2);
            return softmax(logits.mul(selector).sum(1));
        });

        const activeCount = nodes.length;
        const activeProbs =
            activeCount < fixedBatchSize
                ? lastProbs.slice([0, 0], [activeCount, this.model.config.vocabSize])
                : lastProbs;
        const probsArray = (await activeProbs.array()) as number[][];

        if (activeProbs !== lastProbs) {
            activeProbs.dispose();
        }
        lastProbs.dispose();
        logits.dispose();
        idx.dispose();

        const tP = this.options?.topP ?? 1;

        const beamProbs = probsArray.map((rowProbs) => {
            return topTokenCandidates(topP(rowProbs, tP), this.options?.topK);
        });

        for (let i = 0; i < nodes.length; i++) {
            nodes[i].sortedChildren = beamProbs[i];
            this.totalCandidates += beamProbs[i].length - 1;
        }
    }

    private pushNextChild(n: BeamNode): void {
        if (!n.sortedChildren || n.nextChild >= n.sortedChildren.length) {
            //if (n.sortedChildren) {
            //    n.sortedChildren = undefined;
            //}
            return;
        }
        const idx = n.nextChild++;
        const tokenId = n.sortedChildren[idx].token;
        const tokenProb = n.sortedChildren[idx].prob;

        const minBeamLength = this.options!.maxBeamLength;
        const terminatedByToken = this.shouldTerminate(this.options!.allowSpecial ?? false, tokenId);
        const tokenText = this.actualTokeniser.decode([tokenId]);
        const reachesMinLength = n.tokens.length + 1 > minBeamLength;
        const terminatedByWhitespace =
            this.options!.endOnWhiteSpace === true && reachesMinLength && /\s/.test(tokenText);
        const terminateByLength = !this.options!.endOnWhiteSpace
            ? n.tokens.length + 1 >= minBeamLength
            : n.tokens.length + 1 >= minBeamLength + 10; // sanity check

        this.heap.push({
            score: n.score * tokenProb,
            parent: n,
            idx,
            tokenId,
            tokenText,
            probability: tokenProb,
            terminated: terminatedByToken || terminatedByWhitespace || terminateByLength,
            reason: terminatedByToken
                ? 'token'
                : terminatedByWhitespace
                  ? 'whitespace'
                  : terminateByLength
                    ? 'length'
                    : undefined,
        });
    }

    private entryToBeam(e: BeamEntry): IBeam | null {
        const dropLeadingWhitespaceTerminator =
            e.reason === 'whitespace' && e.parent.tokens.length > 0 && /^\s/.test(e.tokenText);

        const output: IGeneratorOutput[] = e.parent.tokens.map((tokenId, idx) => ({
            token: tokenId,
            score: e.parent.probabilities[idx],
            text: this.actualTokeniser.decode([tokenId]),
            terminated: e.terminated,
            confidence: null,
            logits: null,
            scores: null,
            hiddenStates: null,
            attention: null,
            loss: null,
            multinomialRand: null,
        }));
        output.push({
            token: e.tokenId,
            score: e.probability,
            text: e.tokenText,
            terminated: e.terminated,
            confidence: null,
            logits: null,
            scores: null,
            hiddenStates: null,
            attention: null,
            loss: null,
            multinomialRand: null,
        });

        const beam: IBeam = {
            tokens: dropLeadingWhitespaceTerminator ? e.parent.tokens.slice() : [...e.parent.tokens, e.tokenId],
            score: e.score,
            text: dropLeadingWhitespaceTerminator ? e.parent.text : e.parent.text + e.tokenText,
            candidates: this.totalCandidates,
            _output: output,
        };

        if (!this.options?.keepDuplicates) {
            if (this.seen.has(beam.text.trim())) {
                this.totalCandidates--;
                return null;
            }
            this.seen.add(beam.text.trim());
        }

        return beam;
    }

    public async next(): Promise<IBeam | null> {
        this.active = true;
        const result = await this._next();
        this.active = false;
        return result;
    }

    private async _next(): Promise<IBeam | null> {
        if (!this.options || !this.conversation) {
            throw new Error('Beamer not properly initialized with options and conversation.');
        }
        const batchSize = 8;

        // Init the heap
        if (!this.hasRoot) {
            const tokens = await this.tokeniseConversation(this.conversation);
            const ctx = this.createInitialContext(tokens);
            const rootNode: BeamNode = {
                tokens: [],
                text: '',
                score: 1,
                context: ctx.context,
                contextLength: ctx.contextLength,
                terminated: false,
                nextChild: 0,
                sortedChildren: undefined,
                probabilities: [],
            };

            this.totalCandidates = 1;
            await this.expand([rootNode], batchSize);
            this.pushNextChild(rootNode);
            this.hasRoot = true;
        }

        while (this.heap.size && this.active) {
            // Case 1: top is a finished hypothesis -> exact next-best, yield it.
            if (this.heap.peek()!.terminated) {
                const e = this.heap.pop()!;
                this.pushNextChild(e.parent);

                const beam = this.entryToBeam(e);
                if (!beam) {
                    continue;
                }
                return beam;
            }

            // Case 2: gather expandable entries from the top, stopping at a complete one.
            const batch: BeamNode[] = [];
            while (batch.length < batchSize && this.heap.size && !this.heap.peek()!.terminated) {
                const e = this.heap.pop()!;
                this.pushNextChild(e.parent);

                batch.push({
                    text: e.parent.text + e.tokenText,
                    tokens: [...e.parent.tokens, e.tokenId],
                    score: e.score,
                    ...this.appendTokenToContext(e.parent.context, e.parent.contextLength, e.tokenId),
                    terminated: e.terminated,
                    nextChild: 0,
                    sortedChildren: undefined,
                    probabilities: [...e.parent.probabilities, e.probability],
                });
            }

            await this.expand(batch, batchSize);
            for (const n of batch) this.pushNextChild(n);
        }
        return null;
    }

    public async beam(onStep?: (beams: IBeam[]) => void): Promise<IBeam[]> {
        this.active = true;
        const beams: IBeam[] = [];
        while (beams.length < (this.options?.beams ?? 8)) {
            const nextBeam = await this._next();
            if (!nextBeam) break;
            beams.push(nextBeam);
            onStep?.(beams);
        }
        this.active = false;

        // Patch total candidates
        beams.forEach((beam) => {
            beam.candidates = this.totalCandidates;
        });

        return beams;
    }

    public dispose() {
        this.active = false;
        this.heap.values.forEach((entry) => {
            entry.parent.context.dispose();
        });
        this.heap.clear();
    }
}
