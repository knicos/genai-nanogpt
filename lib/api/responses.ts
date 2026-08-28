import { IGenerateOptions, IGeneratorResponse } from '@base/inference/types';
import Generator, { IGenerator } from '@base/inference/Generator';
import EE from 'eventemitter3';
import { ITokeniser } from '@base/tokeniser/type';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { GPTConfig } from '@base/models/config';
import { v4 as uuidv4 } from 'uuid';

interface ResponseEvents {
    status: (status: 'ready' | 'busy') => void;
    error: (error: Error) => void;
    done: (id: string) => void;
    hook: (id: string) => void;
    generating: (id: string) => void;
}

interface ResponseRecord {
    id: string;
    options: IGenerateOptions;
    generator: IGenerator;
    done: boolean;
    timestamp: number;
    callback?: (response: IGeneratorResponse) => void;
}

interface JobItem {
    id: string;
    options: IGenerateOptions;
    callback?: (response: IGeneratorResponse) => void;
    // Promise handlers for queued jobs
    resolve?: (value: IGeneratorResponse | PromiseLike<IGeneratorResponse>) => void;
    reject?: (reason?: unknown) => void;
}

export default class Responses {
    private ee: EE;
    // private _config?: GPTConfig;
    private _model: Model<ModelForwardAttributes, GPTConfig>;
    private _tokeniser: ITokeniser;
    private _busyCount = 0;
    private _responses = new Map<string, ResponseRecord>();
    private _jobQueue: JobItem[] = [];
    private _hookedResponses = new Set<string>();
    private _resumeWaiters = new Map<string, (() => void)[]>();

    constructor(model: Model<ModelForwardAttributes, GPTConfig>, tokeniser: ITokeniser) {
        this._model = model;
        this._tokeniser = tokeniser;
        this.ee = new EE();
    }

    /** Number of generation jobs currently queued. */
    public get queued(): number {
        return this._jobQueue.length;
    }

    private async _processNextJob() {
        if (this._jobQueue.length === 0) return;

        const job = this._jobQueue.shift()!;
        try {
            const resp = await this._runForRecord(job.id, job.options, job.callback);
            if (job.resolve) {
                job.resolve(resp);
            }
        } catch (err) {
            if (job.reject) {
                job.reject(err);
            }
            this.ee.emit('error', err as Error);
        }
    }

    private async _runForRecord(
        id: string,
        options: IGenerateOptions,
        callback?: (response: IGeneratorResponse) => void
    ): Promise<IGeneratorResponse> {
        const previousResponse = options.previous_response_id
            ? this._responses.get(options.previous_response_id)
            : undefined;

        if (previousResponse) {
            const oldConversation = previousResponse.generator.getConversation();

            if (options.input && Array.isArray(options.input)) {
                options.input = [...oldConversation, ...options.input];
            } else if (options.input && typeof options.input === 'string') {
                options.input = [
                    ...oldConversation,
                    { role: options.nonConversational ? 'text' : 'user', content: options.input },
                ];
            } else {
                options.input = oldConversation;
            }
        }

        const generator = previousResponse ? previousResponse.generator : this._responses.get(id)!.generator;

        const f = callback
            ? () =>
                  callback({
                      output: generator.getConversation(),
                      id,
                      done: false,
                  })
            : null;

        if (f) {
            generator.on('tokens', f);
        }

        const record = this._responses.get(id)!;

        const hookedOptions: IGenerateOptions = {
            ...options,
            _onChunk: async () => {
                if (!this._hookedResponses.has(id)) {
                    return;
                }
                await new Promise<void>((resolve) => {
                    const waiters = this._resumeWaiters.get(id) || [];
                    waiters.push(resolve);
                    this._resumeWaiters.set(id, waiters);
                    this.ee.emit('hook', id);
                });
            },
        };

        this.ee.emit('generating', options.previous_response_id || id);

        const outputPromise = options.input
            ? generator.generate(
                  Array.isArray(options.input)
                      ? options.input
                      : [{ role: options.nonConversational ? 'text' : 'user', content: options.input }],
                  hookedOptions
              )
            : generator.generate(hookedOptions);

        if (options.background) {
            outputPromise.then(() => {
                record.done = true;
                this._hookedResponses.delete(id);
                this._resumeWaiters.delete(id);

                if (f) {
                    generator.off('tokens', f);
                }

                this.ee.emit('done', id);
                // After finishing a background job, process next queued job
                this._processNextJob();
            });
            return {
                output: null,
                id,
                done: false,
            };
        }

        const output = await outputPromise;

        if (f) {
            generator.off('tokens', f);
        }

        record.done = true;
        this._hookedResponses.delete(id);
        this._resumeWaiters.delete(id);
        this.ee.emit('done', id);
        // After finishing, process next queued job
        this._processNextJob();

        return {
            output,
            id,
            done: true,
        };
    }

    /**
     * Subscribe to response lifecycle events.
     * @param event One of `status`, `error`, `done`, `hook`, or `generating`
     * @param listener Callback invoked when the event fires
     */
    public on<E extends keyof ResponseEvents>(event: E, listener: ResponseEvents[E]): void {
        this.ee.on(event, listener);
    }

    /**
     * Unsubscribe a previously registered event listener.
     * @param event Event name the listener was registered for
     * @param listener The listener to remove
     */
    public off<E extends keyof ResponseEvents>(event: E, listener: ResponseEvents[E]): void {
        this.ee.off(event, listener);
    }

    private generator(): IGenerator {
        if (!this._model || !this._tokeniser) {
            throw new Error('model_or_tokeniser_not_initialized.');
        }
        const generator = new Generator(this._model, this._tokeniser);
        generator.on('start', () => {
            if (this._busyCount === 0) {
                this.ee.emit('status', 'busy');
            }
            this._busyCount++;
        });
        generator.on('stop', () => {
            this._busyCount--;
            if (this._busyCount === 0) {
                this.ee.emit('status', 'ready');
            }
        });
        return generator;
    }

    private _cleanup() {
        const now = Date.now();
        this._responses.forEach((record, recordId) => {
            if (record.done && now - record.timestamp > 5 * 60 * 1000) {
                record.generator.dispose();
                this._responses.delete(recordId);
            }
        });
    }

    /** Start a new text generation or resume a previous one.
     *  @param options Response options object
     *  @param callback Intermediate responses per token or chunk.
     */
    public async create(
        options: IGenerateOptions,
        callback?: (response: IGeneratorResponse) => void
    ): Promise<IGeneratorResponse> {
        const id = uuidv4();
        const previousResponse = options.previous_response_id
            ? this._responses.get(options.previous_response_id)
            : undefined;
        if (previousResponse) {
            previousResponse.timestamp = Date.now();
        } else {
            this._cleanup();
        }
        const generator = previousResponse ? previousResponse.generator : this.generator();

        const record: ResponseRecord = {
            id,
            generator,
            done: false,
            timestamp: Date.now(),
            options,
            callback,
        };
        this._responses.set(id, record);

        // If we're currently busy with another generation, enqueue this job
        if (this._busyCount > 0 && !options.background) {
            if (this._jobQueue.length > 10) {
                throw new Error('queue_too_long');
            }
            return new Promise<IGeneratorResponse>((resolve, reject) => {
                this._jobQueue.push({ id, options, callback, resolve, reject });
            });
        }

        // Run generation now (or enqueue earlier). Delegate to runner to avoid duplication.
        return this._runForRecord(id, options, callback);
    }

    /**
     * Retry a previous response generation, optionally truncating the conversation at `index`.
     * @param id ID of the existing response to retry
     * @param index Optional conversation index to truncate before regenerating
     */
    public async retry(id: string, index?: number): Promise<IGeneratorResponse> {
        const record = this._responses.get(id);
        if (record) {
            const conversation = record.generator.getConversation();
            if (index !== undefined && index >= 0 && index < conversation.length) {
                conversation.splice(index);
            }
            record.done = false;
            record.timestamp = Date.now();

            // If we're currently busy with another generation, enqueue this job
            if (this._busyCount > 0 && !record.options.background) {
                if (this._jobQueue.length > 10) {
                    throw new Error('queue_too_long');
                }
                return new Promise<IGeneratorResponse>((resolve, reject) => {
                    this._jobQueue.push({ id, options: record.options, callback: record.callback, resolve, reject });
                });
            }

            // Run generation now (or enqueue earlier). Delegate to runner to avoid duplication.
            return this._runForRecord(id, record.options, record.callback);
        }
        throw new Error('response_not_found');
    }

    /**
     * Get the current generator response for a given ID.
     * Returns `null` if no response with that ID exists.
     * @param id Response ID
     */
    public getResponse(id: string): IGeneratorResponse | null {
        const record = this._responses.get(id);
        if (record) {
            return {
                output: record.generator.getConversation(),
                id: record.id,
                done: record.done,
            };
        }
        return null;
    }

    /**
     * Cancel an in-progress generation and resolve any waiting hooks.
     * @param id Response ID to cancel
     * @returns `true` if the response was found and cancelled, otherwise `false`
     */
    public cancel(id: string): boolean {
        const record = this._responses.get(id);
        if (record) {
            this._hookedResponses.delete(id);
            const waiters = this._resumeWaiters.get(id) || [];
            waiters.forEach((resolve) => resolve());
            this._resumeWaiters.delete(id);
            record.generator.stop();
            return true;
        }
        return false;
    }

    /**
     * Put a response into "hooked" mode so generation pauses at the next chunk.
     * @param id Response ID to hook
     * @returns `true` if the response exists and was hooked, otherwise `false`
     */
    public hook(id: string): boolean {
        if (!this._responses.has(id)) {
            return false;
        }
        this._hookedResponses.add(id);
        return true;
    }

    public unhook(id: string) {
        this._hookedResponses.delete(id);
        const waiters = this._resumeWaiters.get(id) || [];
        waiters.forEach((resolve) => resolve());
        this._resumeWaiters.delete(id);
    }

    /**
     * Resume a previously hooked response, releasing a single paused chunk.
     * @param id Response ID to resume
     * @returns `true` if the resume action was performed or the response is still hooked
     */
    public resume(id: string): boolean {
        const waiters = this._resumeWaiters.get(id);
        if (!waiters || waiters.length === 0) {
            return this._hookedResponses.has(id);
        }

        const next = waiters.shift()!;
        if (waiters.length === 0) {
            this._resumeWaiters.delete(id);
        }
        next();
        return true;
    }

    /**
     * Dispose all generators, clear queues and internal bookkeeping.
     * This releases resources held by this Responses manager.
     */
    dispose() {
        this._resumeWaiters.forEach((waiters) => {
            waiters.forEach((resolve) => resolve());
        });
        this._resumeWaiters.clear();
        this._hookedResponses.clear();
        this._responses.forEach((record) => {
            record.generator.dispose();
        });
        this._responses.clear();
        this.ee.removeAllListeners();
    }
}
