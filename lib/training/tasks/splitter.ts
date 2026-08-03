import { MemoryConversationStream } from '@base/data/stream';
import { Conversation } from '../../tokeniser/type';
import ConversationTask from './ConversationTask';
import { Task } from './Task';

export default async function splitValidation(
    tasks: Task[],
    validationSplit: number
): Promise<{ training: Task; validation: Task }> {
    if (validationSplit <= 0 || validationSplit >= 1) {
        throw new Error('validationSplit must be between 0 and 1');
    }

    // tasks.forEach((task) => task.shuffle());

    const trainingConversations: Conversation[][] = [];
    const validationConversations: Conversation[][] = [];

    for (const task of tasks) {
        while (task.hasMoreConversations()) {
            const nextConvo = await task.nextConversation();
            if (!nextConvo) {
                break;
            }

            if (Math.random() < validationSplit) {
                validationConversations.push(nextConvo);
            } else {
                trainingConversations.push(nextConvo);
            }
        }
    }

    return {
        training: new ConversationTask([new MemoryConversationStream(trainingConversations)]),
        validation: new ConversationTask([new MemoryConversationStream(validationConversations)]),
    };
}
