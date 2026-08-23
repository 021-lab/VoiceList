// gesture.js — распознавание жестов и геометрия оверлея.
// Реализация INPUT_HANDLING_STANDARD.md §3–§4. Чистые функции без DOM:
// то же самое использует и приём касаний, и автотест.

'use strict';

const C = {
  LONGPRESS_MS: 400,
  ARM_SLOP_PX: 10,
  CONTEXT_DX_PX: -40,
  CONTEXT_RATIO: 2,
  CONTEXT_WINDOW_MS: 250,
  DEADZONE_PX: 16,
  ROW_H_PX: 48,
  CANCEL_ZONE_PX: 56,
  FREEZE_DY_PX: 24,
  MIN_HOLD_MS: 150,
  FINAL_TIMEOUT_MS: 1500,
};

// Выбор строки по смещению. dy вверх положительно.
// n — длина стека кандидатов.
function selectAt(dy, n, c = C) {
  if (dy <= -c.CANCEL_ZONE_PX) return { zone: 'cancel', index: null };
  if (dy < c.DEADZONE_PX) return { zone: 'as-is', index: null };
  if (n === 0) return { zone: 'as-is', index: null };
  const raw = Math.floor((dy - c.DEADZONE_PX) / c.ROW_H_PX);
  return { zone: 'candidate', index: Math.min(raw, n - 1) };
}

// Состояния: 'idle' -> 'pending' -> 'armed' | 'scroll' | 'tap'
class GestureRecognizer {
  constructor(c = C) {
    this.c = c;
    this.reset();
  }
  reset() {
    this.state = 'idle';
    this.mode = null;       // 'global' | 'context'
    this.contextId = null;
    this.t0 = 0; this.x0 = 0; this.y0 = 0;
    this.onElement = null;
    this.dy = 0;
    this.frozen = false;
  }

  down({ t, x, y, elementId = null }) {
    this.reset();
    this.state = 'pending';
    this.t0 = t; this.x0 = x; this.y0 = y;
    this.onElement = elementId;
    return this.state;
  }

  move({ t, x, y }) {
    if (this.state === 'scroll' || this.state === 'idle') return this.state;
    const dx = x - this.x0;
    const dyUp = this.y0 - y;          // вверх положительно

    if (this.state === 'armed') {
      this.dy = dyUp;
      if (dyUp >= this.c.FREEZE_DY_PX) this.frozen = true;
      return this.state;
    }

    // до взвода
    const dt = t - this.t0;
    if (this.onElement && dt <= this.c.CONTEXT_WINDOW_MS &&
        dx <= this.c.CONTEXT_DX_PX && Math.abs(dx) > this.c.CONTEXT_RATIO * Math.abs(dyUp)) {
      this.state = 'armed';
      this.mode = 'context';
      this.contextId = this.onElement;
      this.armedAt = t;
      this.x0 = x; this.y0 = y;        // якорь оверлея — точка взвода
      this.dy = 0;
      return this.state;
    }
    if (Math.abs(dyUp) > this.c.ARM_SLOP_PX && dt < this.c.LONGPRESS_MS) {
      this.state = 'scroll';
      return this.state;
    }
    if (dt >= this.c.LONGPRESS_MS && Math.abs(dyUp) <= this.c.ARM_SLOP_PX) {
      this.state = 'armed';
      this.mode = 'global';
      this.contextId = null;           // удержание не даёт контекста
      this.armedAt = t;
      this.dy = 0;
    }
    return this.state;
  }

  // Тик таймера: взвод по времени без движения пальца.
  tick({ t }) {
    if (this.state !== 'pending') return this.state;
    if (t - this.t0 >= this.c.LONGPRESS_MS) {
      this.state = 'armed';
      this.mode = 'global';
      this.contextId = null;
      this.armedAt = t;
      this.dy = 0;
    }
    return this.state;
  }

  up({ t }) {
    if (this.state === 'armed') {
      const held = t - this.armedAt;
      return { state: 'released', mode: this.mode, contextId: this.contextId,
               dy: this.dy, held, frozen: this.frozen };
    }
    if (this.state === 'pending' && t - this.t0 < this.c.LONGPRESS_MS) {
      this.state = 'tap';
      return { state: 'tap' };
    }
    return { state: this.state };
  }
}

export { C, selectAt, GestureRecognizer };
