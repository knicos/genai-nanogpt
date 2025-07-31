import { TokeniserMessage } from './messages';
import EE from 'eventemitter3';
import { ITokeniser } from './type';

const worker = new Worker(new URL('./worker.js', import.meta.url), {
    type: 'module',
});

let COUNTER = 0;

export default class WebTokeniser extends EE<'trainStatus'> implements ITokeniser {
    private id: number;
    public vocabSize: number = 0;
    private handler?: (event: MessageEvent<TokeniserMessage>) => void;

    constructor() {
        super();
        this.id = COUNTER++;

        this.handler = (event: MessageEvent<TokeniserMessage>) => {
            if (event.data.type === 'trainStatus' && event.data.id === this.id) {
                this.vocabSize = event.data.vocabSize;
                this.emit('trainStatus', event.data.progress, event.data.vocabSize);
            }
        };
        worker.addEventListener('message', this.handler);
    }

    public destroy() {
        if (this.handler) {
            worker.removeEventListener('message', this.handler);
            this.handler = undefined;
        }
    }

    private post(message: TokeniserMessage) {
        worker.postMessage(message);
    }

    public async train(text: string[], vocabSize: number): Promise<number> {
        return new Promise<number>((resolve) => {
            const h = (event: MessageEvent<TokeniserMessage>) => {
                if (event.data.type === 'trainResponse' && event.data.id === this.id) {
                    worker.removeEventListener('message', h);
                    this.vocabSize = event.data.vocabSize;
                    resolve(this.vocabSize);
                }
            };
            worker.addEventListener('message', h);
            this.post({
                type: 'train',
                id: this.id,
                text,
                vocabSize,
            });
        });
    }

    public async tokenise(text: string[], numeric: true): Promise<number[][]>;
    public async tokenise(text: string[]): Promise<string[][]>;
    public async tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]> {
        return new Promise<string[][] | number[][]>((resolve) => {
            const h = (event: MessageEvent<TokeniserMessage>) => {
                if (event.data.type === 'tokeniseResponse' && event.data.id === this.id) {
                    worker.removeEventListener('message', h);
                    resolve(event.data.tokens);
                }
            };
            worker.addEventListener('message', h);
            this.post({
                type: 'tokenise',
                id: this.id,
                text,
                numeric,
            });
        });
    }

    public async detokenise(tokens: number[][]): Promise<string[]> {
        return new Promise<string[]>((resolve) => {
            const h = (event: MessageEvent<TokeniserMessage>) => {
                if (event.data.type === 'detokeniseResponse' && event.data.id === this.id) {
                    worker.removeEventListener('message', h);
                    resolve(event.data.text);
                }
            };
            worker.addEventListener('message', h);
            this.post({
                type: 'detokenise',
                id: this.id,
                tokens,
            });
        });
    }

    public async encode(text: string): Promise<number[]> {
        return (await this.tokenise([text], true))[0];
    }

    public async decode(tokens: number[]): Promise<string> {
        return (await this.detokenise([tokens]))[0];
    }

    public async getVocab(): Promise<string[]> {
        return new Promise<string[]>((resolve) => {
            const h = (event: MessageEvent<TokeniserMessage>) => {
                if (event.data.type === 'tokensResponse' && event.data.id === this.id) {
                    worker.removeEventListener('message', h);
                    resolve(event.data.tokens);
                }
            };
            worker.addEventListener('message', h);
            this.post({
                type: 'tokens',
                id: this.id,
            });
        });
    }

    public async createTrainingData(text: string[], windowSize: number = 5): Promise<[number[], number[]]> {
        return new Promise<[number[], number[]]>((resolve) => {
            const h = (event: MessageEvent<TokeniserMessage>) => {
                if (event.data.type === 'buildTrainingDataResponse' && event.data.id === this.id) {
                    worker.removeEventListener('message', h);
                    resolve(event.data.trainingData);
                }
            };
            worker.addEventListener('message', h);
            this.post({
                type: 'buildTrainingData',
                id: this.id,
                text,
                windowSize,
            });
        });
    }
}
