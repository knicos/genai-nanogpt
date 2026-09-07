import EE from 'eventemitter3';
import { Conversation, ITokeniser } from '@base/tokeniser/type';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { GPTConfig } from '@base/models/config';
import Beamer from '@base/inference/Beamer';
import type { BeamerOptions, IBeam } from '@base/inference/types';
import { v4 as uuidv4 } from 'uuid';

interface BeamEvents {
    error: (error: Error) => void;
    status: (status: 'busy' | 'ready') => void;
    progress: (id: string, beams: IBeam[]) => void;
    done: (id: string, beams: IBeam[]) => void;
}

export default class BeamAPI {
    private ee: EE;
    // private _config?: GPTConfig;
    private _model: Model<ModelForwardAttributes, GPTConfig>;
    private _tokeniser: ITokeniser;
    private _busyCount = 0;

    constructor(model: Model<ModelForwardAttributes, GPTConfig>, tokeniser: ITokeniser) {
        this._model = model;
        this._tokeniser = tokeniser;
        this.ee = new EE();
    }

    public on<E extends keyof BeamEvents>(event: E, listener: BeamEvents[E]): void {
        this.ee.on(event, listener);
    }

    public off<E extends keyof BeamEvents>(event: E, listener: BeamEvents[E]): void {
        this.ee.off(event, listener);
    }

    public create(conversation: Conversation[], options: BeamerOptions): string {
        const id = uuidv4();
        this._busyCount++;
        this.ee.emit('status', 'busy');
        const beamer = new Beamer(this._model, this._tokeniser);
        beamer
            .beam(conversation, options, (beams) => {
                this.ee.emit('progress', id, beams);
            })
            .then((result) => {
                this._busyCount--;
                if (this._busyCount === 0) {
                    this.ee.emit('status', 'ready');
                }
                this.ee.emit('done', id, result);
            })
            .catch((error) => {
                this._busyCount--;
                if (this._busyCount === 0) {
                    this.ee.emit('status', 'ready');
                }
                this.ee.emit('error', error);
            });

        return id;
    }
}
