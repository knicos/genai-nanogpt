import type { Conversation, ITokeniser } from '../tokeniser/type';
import Model, { ModelForwardAttributes } from '../models/model';
import EE from 'eventemitter3';
import { BeamerOptions, IBeam, IGenerateOptions } from './types';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor, concat, oneHot, scalar, stack, tensor1d, tidy, softmax } from '@tensorflow/tfjs-core';
import { CHARS, padArray } from './utilities';
import topP from '@base/utilities/topP';

interface WorkingBeam extends IBeam {
    terminated: boolean;
    context: Tensor;
    contextLength: number;
}

interface BeamExpansion {
    parent: WorkingBeam;
    token: number;
    tokenText: string;
    score: number;
    terminatedByToken: boolean;
    terminatedByWhitespace: boolean;
    dropLeadingWhitespaceTerminator: boolean;
}

function topTokenCandidates(probs: number[], maxCandidates: number): { token: number; prob: number }[] {
    return probs
        .map((prob, token) => ({ token, prob }))
        .filter((candidate) => candidate.prob > 0)
        .sort((a, b) => b.prob - a.prob)
        .slice(0, Math.max(1, maxCandidates));
}

export default class Beamer extends EE {
    private actualTokeniser: ITokeniser;

    constructor(
        private readonly model: Model<ModelForwardAttributes>,
        private readonly tokeniser: ITokeniser
    ) {
        super();
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

    private async batchedNextProbabilities(
        beams: WorkingBeam[],
        fixedBatchSize: number,
        options: BeamerOptions
    ): Promise<number[][]> {
        if (beams.length === 0) {
            return [];
        }

        const filler = beams[0];
        const batchBeams: WorkingBeam[] = beams.slice();
        while (batchBeams.length < fixedBatchSize) {
            batchBeams.push(filler);
        }

        const idx = stack(batchBeams.map((b) => b.context));
        const attrs: ModelForwardAttributes = { training: false, mixedPrecision: true };
        const logits = this.model.forward(attrs, idx);
        const lastSteps = batchBeams.map((b) => Math.max(0, b.contextLength - 1));
        const lastProbs = tidy(() => {
            const selector = oneHot(lastSteps, this.model.config.blockSize).expandDims(2);
            return softmax(logits.mul(selector).sum(1));
        });

        const activeCount = beams.length;
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

        const tP = options.topP ?? 1;

        return probsArray.map((rowProbs) => {
            return topP(rowProbs, tP);
        });
    }

    public async beam(
        conversation: Conversation[],
        options: BeamerOptions,
        onStep?: (beams: IBeam[]) => void
    ): Promise<IBeam[]> {
        if (!options || options.beams < 1 || options.maxBeamLength < 1) {
            return [];
        }

        const beamCount = Math.max(8, options.beams);

        this.initialise(options);

        const promptTokens = await this.tokeniseConversation(conversation);
        const initialContext = this.createInitialContext(promptTokens);
        let beams: WorkingBeam[] = [
            {
                tokens: [],
                score: 0,
                text: '',
                terminated: false,
                context: initialContext.context,
                contextLength: initialContext.contextLength,
            },
        ];
        const minBeamLength = options.maxBeamLength;
        const loopLimit =
            options.endOnWhiteSpace === true
                ? Math.max(minBeamLength, options.maxLength ?? minBeamLength + this.model.config.blockSize)
                : minBeamLength;

        let lastStepTime = Date.now();

        for (let step = 0; step < loopLimit; step++) {
            const expandable = beams.filter((b) => !b.terminated);
            if (expandable.length === 0) {
                break;
            }

            const probsByBeam = await this.batchedNextProbabilities(expandable, beamCount, options);

            const expansions: BeamExpansion[] = [];
            for (let i = 0; i < expandable.length; i++) {
                const parent = expandable[i];
                const probs = probsByBeam[i];
                const perBeamCandidates = options.topK ? Math.min(options.topK, beamCount) : Math.max(1, beamCount);
                const candidates = topTokenCandidates(probs, perBeamCandidates);

                for (const candidate of candidates) {
                    const tokenText = this.actualTokeniser.decode([candidate.token]);
                    const terminatedByToken = this.shouldTerminate(options.allowSpecial ?? false, candidate.token);
                    const reachesMinLength = parent.tokens.length + 1 >= minBeamLength;
                    const terminatedByWhitespace =
                        options.endOnWhiteSpace === true && reachesMinLength && /\s/.test(tokenText);
                    const dropLeadingWhitespaceTerminator =
                        terminatedByWhitespace && parent.tokens.length > 0 && /^\s/.test(tokenText);

                    expansions.push({
                        parent,
                        token: candidate.token,
                        tokenText,
                        score: parent.score + Math.log(Math.max(candidate.prob, 1e-12)),
                        terminatedByToken,
                        terminatedByWhitespace,
                        dropLeadingWhitespaceTerminator,
                    });
                }
            }

            const completed = beams.filter((b) => b.terminated);
            const next: WorkingBeam[] = completed.slice();

            expansions.sort((a, b) => b.score - a.score);
            for (const expanded of expansions) {
                if (next.length >= beamCount) {
                    break;
                }

                const terminated = expanded.terminatedByToken || expanded.terminatedByWhitespace;
                const keepToken = !expanded.dropLeadingWhitespaceTerminator;
                const nextTokens = keepToken
                    ? expanded.parent.tokens.concat(expanded.token)
                    : expanded.parent.tokens.slice();
                const nextText =
                    expanded.terminatedByToken || expanded.dropLeadingWhitespaceTerminator
                        ? expanded.parent.text
                        : expanded.parent.text + expanded.tokenText;

                const nextContext = keepToken
                    ? this.appendTokenToContext(expanded.parent.context, expanded.parent.contextLength, expanded.token)
                    : { context: expanded.parent.context.clone(), contextLength: expanded.parent.contextLength };

                next.push({
                    tokens: nextTokens,
                    score: expanded.score,
                    text: nextText,
                    terminated,
                    context: nextContext.context,
                    contextLength: nextContext.contextLength,
                });
            }

            beams.forEach((beam) => beam.context.dispose());

            if (next.length === 0) {
                break;
            }
            beams = next;

            if (onStep && Date.now() - lastStepTime >= 40) {
                onStep(beams.slice(0, options.beams));
                lastStepTime = Date.now();
            }
        }

        beams.forEach((beam) => {
            beam.score /= Math.max(1, beam.text.length);
        });

        // Remove duplicates
        const sorted = beams.slice().sort((a, b) => b.score - a.score);
        const unique: IBeam[] = [];
        const seen = new Set<string>();
        for (const beam of sorted) {
            const key = beam.text.trim();
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(beam);
            }
            if (unique.length >= options.beams) {
                break;
            }
        }

        beams.forEach((beam) => beam.context.dispose());
        return unique;
    }
}
