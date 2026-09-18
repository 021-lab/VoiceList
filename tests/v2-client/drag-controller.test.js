import { afterEach, describe, expect, it, vi } from 'vitest';
import { DragController, deriveArrangedFromWrappers } from '../../src/v2/client/drag-controller.js';

let drag;
afterEach(() => { drag?.cancel(); vi.useRealTimers(); });
function setup(spec = [['a', 0], ['child', 1], ['b', 0], ['c', 0]]) {
  vi.useFakeTimers();
  const container = document.createElement('div'); document.body.replaceChildren(container);
  for (const [id, level] of spec) {
    const el = document.createElement('div'); el.className = 'list-item-wrapper'; el.dataset.id = id; el.dataset.level = level; el.style.marginLeft = `${level * 24}px`;
    const row = document.createElement('div'); row.className = 'list-item'; row.textContent = id; el.append(row); container.append(el);
    Object.defineProperty(el, 'offsetParent', { get: () => el.hidden || el.style.display === 'none' ? null : container });
    el.getBoundingClientRect = () => {
      const visible = [...container.children].filter(x => !x.hidden && x.style.display !== 'none');
      const y = Number(el.style.transform.match(/translateY\(([-.\d]+)px\)/)?.[1] || 0);
      const top = 100 + visible.indexOf(el) * 50 + y; return { top, bottom: top + 50, height: 50 };
    };
  }
  const onDrop = vi.fn();
  drag = new DragController({ container, header: { getBoundingClientRect: () => ({ bottom: 60 }) }, window, onDrop });
  return { container, onDrop, el: id => container.querySelector(`[data-id="${id}"]`) };
}
describe('original full-card group drag', () => {
  it('derives complete arranged including hidden descendants from DOM levels', () => {
    const { container, el } = setup(); el('child').hidden = true;
    expect(deriveArrangedFromWrappers([...container.children])).toEqual([
      { id: 'a', parentId: null, order: 10 }, { id: 'child', parentId: 'a', order: 10 },
      { id: 'b', parentId: null, order: 20 }, { id: 'c', parentId: null, order: 30 }
    ]);
  });
  it('moves raised card and midpoint DOM order, then restores everything on cancel without command', () => {
    const { container, el, onDrop } = setup();
    const order = [...container.children];
    drag.start(el('b'), { x: 50, y: 185 }, { x: 50, y: 225 });
    expect(el('b').classList.contains('is-dragging')).toBe(true);
    expect(el('b').style.transform).toMatch(/translateY/);
    expect([...container.children]).not.toEqual(order);
    drag.cancel(); expect([...container.children]).toEqual(order);
    expect(el('b').style.transform).toBe(''); expect(el('b').dataset.level).toBe('0'); expect(onDrop).not.toHaveBeenCalled();
  });
  it('keeps hidden descendants in the moved group and submits once after settle', () => {
    const { container, el, onDrop } = setup(); el('child').hidden = true;
    drag.start(el('a'), { x: 50, y: 110 }, { x: 50, y: 125 });
    expect(drag.state.children).toEqual([el('child')]); expect(el('child').style.display).toBe('none');
    drag.update({ x: 50, y: 165 }); drag.finish(); drag.finish();
    expect(onDrop).not.toHaveBeenCalled(); vi.advanceTimersByTime(221);
    expect(onDrop).toHaveBeenCalledOnce();
    const arranged = onDrop.mock.calls[0][1];
    expect(arranged.find(x => x.id === 'child').parentId).toBe('a');
    expect(arranged.findIndex(x => x.id === 'child')).toBe(arranged.findIndex(x => x.id === 'a') + 1);
    expect(arranged).toHaveLength(container.children.length); expect(el('child').hidden).toBe(true);
  });
  it('only a deliberate right shift changes depth and shifts all descendant levels', () => {
    const { el, onDrop } = setup([['before', 0], ['parent', 0], ['child', 1], ['last', 0]]);
    drag.start(el('parent'), { x: 50, y: 171 }, { x: 50, y: 175 });
    drag.update({ x: 75, y: 171 }); expect(el('parent').dataset.level).toBe('0');
    drag.update({ x: 90, y: 171 }); expect(el('parent').dataset.level).toBe('1');
    drag.finish(); vi.advanceTimersByTime(221);
    expect(el('child').dataset.level).toBe('2');
    expect(onDrop.mock.calls[0][1]).toContainEqual({ id: 'parent', parentId: 'before', order: 10 });
    expect(onDrop.mock.calls[0][1]).toContainEqual({ id: 'child', parentId: 'parent', order: 10 });
  });
  it('cancelling during drop animation restores original layout and prevents delayed submission', () => {
    const { el, onDrop } = setup(); drag.start(el('b'), { x: 50, y: 185 }, { x: 50, y: 225 });
    drag.finish(); drag.cancel(); vi.advanceTimersByTime(300);
    expect(onDrop).not.toHaveBeenCalled(); expect(el('b').classList.contains('is-dragging')).toBe(false);
  });
});
