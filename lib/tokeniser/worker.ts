import BPE from './bpe';
import { TokeniserMessage } from './messages';

let bpe = new BPE();

onmessage = async (event: MessageEvent<TokeniserMessage>) => {
    if (event.data.type === 'tokenise') {
        if (event.data.numeric) {
            const tokens = bpe.tokenise(event.data.text, true);
            const msg: TokeniserMessage = {
                type: 'tokeniseResponse',
                id: event.data.id,
                tokens,
                numeric: true,
            };
            postMessage(msg);
        } else {
            const tokens = bpe.tokenise(event.data.text);
            const msg: TokeniserMessage = {
                type: 'tokeniseResponse',
                id: event.data.id,
                tokens,
                numeric: false,
            };
            postMessage(msg);
        }
    } else if (event.data.type === 'detokenise') {
        const vocab = bpe.getVocab();
        const text = event.data.tokens.map((t) => t.map((tt) => vocab[tt]).join(''));
        const msg: TokeniserMessage = {
            type: 'detokeniseResponse',
            id: event.data.id,
            text,
        };
        postMessage(msg);
    } else if (event.data.type === 'train') {
        bpe = new BPE(); // Reset BPE instance to train a new model
        bpe.train(event.data.text, event.data.vocabSize ?? 100, (progress, vocabSize) => {
            const statusMsg: TokeniserMessage = {
                type: 'trainStatus',
                id: event.data.id,
                progress,
                vocabSize,
            };
            postMessage(statusMsg);
        });
        const msg: TokeniserMessage = {
            type: 'trainResponse',
            id: event.data.id,
            vocabSize: bpe.getVocab().length,
        };
        postMessage(msg);
    } else if (event.data.type === 'tokens') {
        const tokens = bpe.getVocab();
        const msg: TokeniserMessage = {
            type: 'tokensResponse',
            id: event.data.id,
            tokens,
        };
        postMessage(msg);
    }
};
