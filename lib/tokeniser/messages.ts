interface TrainMessage {
    type: 'train';
    id: number;
    text: string[];
    vocabSize: number;
}

interface TrainResponse {
    type: 'trainResponse';
    id: number;
    vocabSize: number;
}

interface TrainStatusMessage {
    type: 'trainStatus';
    id: number;
    progress: number;
    vocabSize: number;
}

interface TokeniseMessage {
    type: 'tokenise';
    id: number;
    numeric?: boolean;
    text: string[];
}

interface TokeniseResponse {
    type: 'tokeniseResponse';
    id: number;
    numeric: boolean;
    tokens: string[][] | number[][];
}

interface DetokeniseMessage {
    type: 'detokenise';
    id: number;
    tokens: number[][];
}

interface DetokeniseResponse {
    type: 'detokeniseResponse';
    id: number;
    text: string[];
}

interface TokensMessage {
    type: 'tokens';
    id: number;
}

interface TokensResponse {
    type: 'tokensResponse';
    id: number;
    tokens: string[];
}

interface BuildTrainingDataMessage {
    type: 'buildTrainingData';
    id: number;
    text: string[];
    windowSize: number;
}

interface BuildTrainingDataResponse {
    type: 'buildTrainingDataResponse';
    id: number;
    trainingData: [number[], number[]];
}

export type TokeniserMessage =
    | TrainMessage
    | TrainResponse
    | TrainStatusMessage
    | TokeniseMessage
    | DetokeniseMessage
    | TokeniseResponse
    | DetokeniseResponse
    | TokensMessage
    | TokensResponse
    | BuildTrainingDataMessage
    | BuildTrainingDataResponse;
