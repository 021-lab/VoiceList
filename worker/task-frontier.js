import { calculateFrontier } from '../src/list-frontier.js';
import { compareByDeadline, isDeadline } from '../src/task-deadline.js';

const ROOT_PARENT_TITLE = 'Мой список';

export function taskFrontierFromItems(items = [], now = new Date()) {
  const itemById = new Map(items.map((item) => [String(item?.id || ''), item]));
  const { frontier } = calculateFrontier(items);

  return [...frontier]
    .sort((left, right) => compareByDeadline(left, right, now))
    .map((task) => ({
      parentTitle: task.parentId == null
        ? ROOT_PARENT_TITLE
        : String(itemById.get(String(task.parentId))?.line1 || ''),
      taskId: String(task.id || ''),
      taskTitle: String(task.line1 || ''),
      status: String(task.status || 'Open'),
      deadline: isDeadline(task.deadline) ? task.deadline : null
    }));
}
