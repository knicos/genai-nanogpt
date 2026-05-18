import { Conversation } from '../../tokeniser/type';
import ConversationTask from './ConversationTask';
import { Task } from './Task';

export default function splitValidation(tasks: Task[], validationSplit: number): { training: Task; validation: Task } {
    if (validationSplit <= 0 || validationSplit >= 1) {
        throw new Error('validationSplit must be between 0 and 1');
    }

    tasks.forEach((task) => task.shuffle());

    const trainingConversations: Conversation[][] = [];
    const validationConversations: Conversation[][] = [];

    for (const task of tasks) {
        while (task.hasMoreConversations()) {
            const nextConvo = task.nextConversation();
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
        training: new ConversationTask(trainingConversations),
        validation: new ConversationTask(validationConversations),
    };
}
