// voice-session.js — сборка слоёв в один жест.
// Жест -> взвод -> речь -> кандидаты -> отпускание -> команда.
// DOM здесь нет: то же самое крутит и приём касаний, и автотест.

'use strict';

import * as R from './resolver.js';
import * as CR from './command-resolver.js';
import * as A from './snapshot-adapter.js';
import * as G from './gesture.js';

const MESSAGES = {
  denied: 'Нет доступа к микрофону',
  aborted: 'Распознавание прервано',
  timeout: 'Не расслышал, повторите',
  stale: 'Задача изменилась, повторите',
  notFound: 'Не нашёл',
  degraded: 'Выбор команды недоступен',
};

// Что показывает оверлей для текущего текста.
// finger — действие в точке пальца, stack — кандидаты вверх от пальца.
function buildOverlay(text, context, tasks) {
  const p = CR.parseCommand(text, context);
  if (p.kind !== 'one') return { finger: { kind: 'fallback' }, stack: [], parse: p };

  const h = p.hypothesis;
  const cmd = CR.toCommand(h, context);

  if (h.rule.command === 'setParent') {
    const rows = A.adaptForReparent(tasks, context);   // без поддерева — защита от циклов
    const r = R.resolve(h.tail, R.findCandidates(h.tail, rows));
    if (r.kind === 'empty') return { finger: { kind: 'blocked', why: 'notFound' }, stack: [], parse: p };
    const ids = r.kind === 'one' ? [r.id] : r.candidates.map((c) => c.id);
    const stack = ids.map((id) => ({
      label: labelOf(tasks, id),
      command: { ...cmd, payload: { parentId: id } },
    }));
    return { finger: stack[0], stack: stack.slice(1), parse: p };
  }

  if (h.rule.command === 'showSearch' || h.rule.command === 'addChild') {
    // для поиска — найденное; для добавления — дубли.
    // Выбор строки из стека отменяет добавление и открывает существующий элемент.
    const rows = A.adaptSnapshot(tasks);
    const r = R.resolve(h.tail, R.findCandidates(h.tail, rows));
    const ids = r.kind === 'one' ? [r.id] : r.kind === 'vector' ? r.candidates.map((c) => c.id) : [];
    return {
      finger: { kind: 'command', command: cmd },
      stack: ids.map((id) => ({ label: labelOf(tasks, id),
                               command: { command: 'viewItem', actId: id, payload: {} } })),
      parse: p,
    };
  }

  return { finger: { kind: 'command', command: cmd }, stack: [], parse: p };
}

function labelOf(tasks, id) {
  const t = tasks.find((x) => x.id === id);
  return t ? t.line1 : id;
}

class VoiceSession {
  constructor({ tasks, asr, constants = G.C, onUpdate = null }) {
    this.tasks = tasks;
    this.asr = asr;
    this.c = constants;
    this.onUpdate = onUpdate;
    this.messages = [];
    this.log = [];
    this.overlay = { finger: { kind: 'fallback' }, stack: [] };
    this.recomputes = 0;
    this.tokenCount = 0;
    this.text = '';
    this.finalText = null;
    this.context = null;
    this.armed = false;
    this.frozenStack = null;
  }

  emitUpdate() { this.onUpdate?.(this); }

  message(code) {
    this.messages.push(MESSAGES[code] ?? code);
    this.log.push({ event: code });
    this.emitUpdate();
  }

  arm(context) {
    const ok = this.asr.start({
      onInterim: (t) => this.onInterim(t),
      onFinal: (t) => { this.finalText = t; },
      onError: (code) => { this.error = code; this.message(code); },
    });
    if (!ok) return false;                       // отказ микрофона — оверлей не взводится
    this.armed = true;
    this.context = context;
    this.log.push({ event: 'overlay-shown', constants: this.c, theta: CR.THETA });
    if (!this.asr.hasInterim()) this.message('degraded');
    this.emitUpdate();
    return true;
  }

  // Пересчёт только при появлении нового слова.
  onInterim(text) {
    const n = R.tokenize(R.normalize(text)).length;
    if (n <= this.tokenCount) return;
    this.tokenCount = n;
    this.text = text;
    this.recompute();
  }

  recompute() {
    const next = buildOverlay(this.text, this.context, this.tasks);
    if (this.frozenStack) next.stack = this.frozenStack;   // порядок заморожен
    this.overlay = next;
    this.recomputes++;
    this.emitUpdate();
  }

  move(dy) {
    this.dy = dy;
    if (dy >= this.c.FREEZE_DY_PX && !this.frozenStack) this.frozenStack = this.overlay.stack;
    this.emitUpdate();
  }

  // Отпускание: сначала финал, потом решение.
  release(dy) {
    this.dy = dy;
    const stopped = this.asr.stop();
    if (stopped && typeof stopped.then === 'function') {
      return stopped.then(() => this.resolveRelease(dy));
    }
    return this.resolveRelease(dy);
  }

  resolveRelease(dy) {
    if (this.error) return { action: 'cancel', why: this.error };

    if (this.finalText !== null && this.finalText !== this.text) {
      this.text = this.finalText;
      this.tokenCount = R.tokenize(R.normalize(this.finalText)).length;
      this.recompute();
    }
    if (!this.asr.hasInterim()) this.overlay = { finger: { kind: 'fallback' }, stack: [] };

    const sel = G.selectAt(dy, this.overlay.stack.length, this.c);
    if (sel.zone === 'cancel') { this.log.push({ event: 'cancelled' }); return { action: 'cancel' }; }

    if (sel.zone === 'candidate') {
      const pick = this.overlay.stack[sel.index];
      return { action: 'command', command: pick.command, label: pick.label };
    }

    const f = this.overlay.finger;
    if (f.kind === 'fallback') return { action: 'fallback', text: this.text };
    if (f.kind === 'blocked') { this.message(f.why); return { action: 'cancel', why: f.why }; }
    return { action: 'command', command: f.command, label: f.label };
  }
}

// Проверка актуальности снимка перед исполнением.
function validate(command, tasks) {
  const ids = [command.actId, command.payload && command.payload.parentId].filter(
    (v) => v != null && v !== 'inbox');
  return ids.every((id) => tasks.some((t) => t.id === id));
}

export { VoiceSession, buildOverlay, validate, MESSAGES };
