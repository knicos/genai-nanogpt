import { TeachableLLM } from '@base/main';

export default function waitForModel(model: TeachableLLM): Promise<void> {
    return new Promise((resolve, reject) => {
        if (model.ready) {
            resolve();
        } else {
            model.on('status', (status) => {
                if (status === 'ready') {
                    resolve();
                }
            });
            model.on('error', (err) => {
                reject(err);
            });
        }
    });
}
