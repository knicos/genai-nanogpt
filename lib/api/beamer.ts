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

interface BeamJob {
    id: string;
    beamer: Beamer;
    canceled: boolean;
    conversation: Conversation[];
    options: BeamerOptions;
}

export default class BeamAPI {
    private ee: EE;
    // private _config?: GPTConfig;
    private _model: Model<ModelForwardAttributes, GPTConfig>;
    private _tokeniser: ITokeniser;
    private _busyCount = 0;
    private _jobs = new Map<string, BeamJob>();
    private _queue: BeamJob[] = [];

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

    public cancel(id?: string): void {
        if (id) {
            const job = this._jobs.get(id);
            if (job) {
                job.canceled = true;
                job.beamer.cancel();
            }
        } else {
            for (const job of this._jobs.values()) {
                job.canceled = true;
                job.beamer.cancel();
            }
        }
    }

    private startJob(job: BeamJob): void {
        this._busyCount++;
        this.ee.emit('status', 'busy');

        job.beamer
            .beam(job.conversation, job.options, (beams) => {
                this.ee.emit('progress', job.id, beams);
            })
            .then((result) => {
                this._busyCount--;
                if (this._busyCount === 0) {
                    this.ee.emit('status', 'ready');
                }
                this.ee.emit('done', job.id, result);
                this._jobs.delete(job.id);

                const nextJob = this._queue.shift()!;
                if (nextJob) {
                    this.startJob(nextJob);
                }
            })
            .catch((error) => {
                this._busyCount--;
                if (this._busyCount === 0) {
                    this.ee.emit('status', 'ready');
                }
                this.ee.emit('error', error);
                this._jobs.delete(job.id);

                const nextJob = this._queue.shift()!;
                if (nextJob) {
                    this.startJob(nextJob);
                }
            });
    }

    public create(conversation: Conversation[], options: BeamerOptions): string {
        const id = uuidv4();
        const beamer = new Beamer(this._model, this._tokeniser);
        const job: BeamJob = { id, beamer, canceled: false, conversation, options };
        this._jobs.set(id, job);

        if (this._busyCount > 0) {
            this._queue.push(job);
            return id;
        }

        this.startJob(job);
        return id;
    }
}
