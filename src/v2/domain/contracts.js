import { z } from 'zod';

export const statuses = ['Open', 'Focus', 'Pause', 'Done', 'Archive', 'Info'];
export const views = ['list', 'frontier', 'search', 'log', 'action', 'edit', 'settings', 'dialogues'];
export const clone = (value) => structuredClone(value);
export function stable(value) {
  if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + stable(value[k])).join(',') + '}';
  return JSON.stringify(value);
}
export class ContractError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}
export const fail = (code, message) => { throw new ContractError(code, message); };
const id = z.string().min(1).max(160);
export const inputSchema = z.object({
  key: z.object({ clientKey: id, seq: z.number().int().positive().safe() }).strict(),
  context: z.object({
    elementId: id, view: z.enum(views), revision: z.number().int().nonnegative(),
    actionId: id.optional()
  }).strict(),
  text: z.string().trim().min(1).max(16000).optional(),
  command: z.object({
    command: id, actId: z.string().max(160).nullable().optional(),
    actType: z.string().max(30).optional(), payload: z.record(z.string(), z.unknown()).optional(),
    source: z.string().max(60).optional(), transcript: z.string().max(16000).optional()
  }).strict().optional()
}).strict().refine(v => Number(v.text !== undefined) + Number(v.command !== undefined) === 1, 'Provide text or command, not both');

export const graphCommands = new Set(['addItem', 'addChild', 'editItem', 'setStatus', 'setParent', 'setTags', 'setDeadline', 'toggleCollapse', 'deleteItem', 'reorderItems', 'importWorkflowyTree']);
export const uiCommands = {
  showList: 'list', showFrontier: 'frontier', showActionLog: 'log', showSearch: 'search',
  showAddModal: 'edit', showEditModal: 'edit', showNestModal: 'edit', viewItem: 'edit',
  showSettings: 'settings', showDialogues: 'dialogues'
};
export function safeError(error) {
  return { code: error.code || 'PROCESSING_FAILED', message: error.code ? error.message : 'Не удалось выполнить действие. Попробуйте ещё раз.' };
}
