/** DOM-only port of main's full-card/group drag. Never owns a task store. */
export function deriveArrangedFromWrappers(wrappers) {
  const stack = [], orders = new Map();
  return wrappers.map(el => {
    const id = el.dataset.id, level = Number(el.dataset.level || 0);
    while (stack.length > level) stack.pop();
    const parentId = level === 0 ? null : stack[level - 1], order = (orders.get(parentId) || 0) + 10;
    orders.set(parentId, order); stack[level] = id; stack.length = level + 1;
    return { id, parentId, order };
  });
}
export class DragController {
  constructor({ container, header, window: win, onDrop, onRestored }) { Object.assign(this, { container, header, win, onDrop, onRestored }); this.state = null; }
  get active() { return !!this.state; }
  wrappers() { return [...this.container.querySelectorAll('.list-item-wrapper')]; }
  start(wrapper, point, origin = point) {
    this.cancel();
    const all = this.wrappers(), index = all.indexOf(wrapper), originalLevel = Number(wrapper.dataset.level || 0), children = [];
    if (index < 0) return;
    for (let i = index + 1; i < all.length && Number(all[i].dataset.level || 0) > originalLevel; i++) children.push(all[i]);
    this.state = { wrapper, children, originalLevel, pendingLevel: originalLevel, offsetY: 0,
      lastClientY: origin.y, nestBaseX: origin.x, rightShifted: false, originalNodes: [...this.container.childNodes],
      initial: new Map(all.map(el => [el, { level: el.dataset.level, style: el.getAttribute('style'), hidden: el.hidden }])),
      childDisplay: new Map(children.map(el => [el, el.style.display])) };
    wrapper.querySelector('.list-item').style.transition = 'none'; wrapper.classList.add('is-dragging'); wrapper.style.transition = 'none';
    for (const child of children) child.style.display = 'none';
    this.update(point); this.raf = this.win.requestAnimationFrame(() => this.autoScroll());
  }
  transform() {
    const s = this.state; if (!s) return;
    const value = `translateY(${s.offsetY}px)`; s.wrapper.style.transform = value;
    for (const child of s.children) child.style.transform = value;
  }
  moveBefore(reference) {
    const s = this.state, top = s.wrapper.getBoundingClientRect().top;
    this.container.insertBefore(s.wrapper, reference);
    for (const child of s.children) this.container.insertBefore(child, reference);
    // Preserve the raised card's screen position across variable-height row swaps.
    s.offsetY += top - s.wrapper.getBoundingClientRect().top; this.transform();
  }
  clamp() {
    const s = this.state, rect = s.wrapper.getBoundingClientRect(), top = this.header.getBoundingClientRect().bottom - 30;
    if (rect.top < top) s.offsetY += top - rect.top;
    else if (rect.bottom > this.win.innerHeight + 30) s.offsetY -= rect.bottom - this.win.innerHeight - 30;
    this.transform();
  }
  checkSwap() {
    const s = this.state, all = this.wrappers();
    const visible = all.filter(el => el === s.wrapper || (!s.children.includes(el) && !el.hidden && el.style.display !== 'none' && el.offsetParent !== null));
    const index = visible.indexOf(s.wrapper), rect = s.wrapper.getBoundingClientRect(), midpoint = rect.top + rect.height / 2;
    if (index > 0) {
      const previous = visible[index - 1], box = previous.getBoundingClientRect();
      if (midpoint < box.top + box.height / 2) { this.moveBefore(previous); return; }
    }
    if (index < visible.length - 1) {
      const next = visible[index + 1], box = next.getBoundingClientRect();
      if (midpoint > box.top + box.height / 2) this.moveBefore(next.nextElementSibling);
    }
  }
  anchor() {
    const s = this.state, all = this.wrappers(), moving = new Set([s.wrapper, ...s.children]);
    for (let index = all.indexOf(s.wrapper) - 1; index >= 0; index--) {
      const candidate = all[index];
      if (!moving.has(candidate) && !candidate.hidden && candidate.style.display !== 'none' && candidate.offsetParent !== null) return candidate;
    }
    return null;
  }
  updateLevel() {
    const s = this.state, anchor = this.anchor(), anchorLevel = Number(anchor?.dataset.level || 0);
    const level = anchor ? anchorLevel + (s.rightShifted ? 1 : 0) : 0;
    s.pendingLevel = level; s.wrapper.dataset.level = String(level); s.wrapper.style.marginLeft = `${level * 24}px`;
  }
  update(point) {
    const s = this.state; if (!s || s.settling) return;
    s.offsetY += (point.y - s.lastClientY) * 2; s.lastClientY = point.y;
    this.transform(); this.clamp(); this.checkSwap();
    s.rightShifted = point.x - s.nestBaseX > 36;
    this.updateLevel();
  }
  dropPlacement() {
    const s = this.state, moving = new Set([s.wrapper, ...s.children]);
    const remaining = this.wrappers().filter(el => !moving.has(el));
    const anchor = this.anchor();
    if (!anchor) return { level: 0, reference: remaining[0] || null };
    const anchorIndex = remaining.indexOf(anchor), anchorLevel = Number(anchor.dataset.level || 0);
    if (s.rightShifted) return { level: anchorLevel + 1, reference: remaining[anchorIndex + 1] || null };
    let end = anchorIndex;
    while (end + 1 < remaining.length && Number(remaining[end + 1].dataset.level || 0) > anchorLevel) end++;
    return { level: anchorLevel, reference: remaining[end + 1] || null };
  }
  autoScroll() {
    const s = this.state; if (!s || s.settling) return;
    const rect = s.wrapper.getBoundingClientRect(), top = this.header.getBoundingClientRect().bottom;
    const all = this.wrappers(), index = all.indexOf(s.wrapper); let speed = 0;
    if (rect.top < top && index > 0) speed = -Math.min((top - rect.top) / 5 + 2, 14);
    else if (rect.bottom > this.win.innerHeight && index + s.children.length < all.length - 1) speed = Math.min((rect.bottom - this.win.innerHeight) / 5 + 2, 14);
    if (speed) {
      const before = this.win.scrollY; this.win.scrollBy(0, speed); s.offsetY += this.win.scrollY - before;
      this.transform(); this.clamp(); this.checkSwap(); this.updateLevel();
    }
    this.raf = this.win.requestAnimationFrame(() => this.autoScroll());
  }
  stopAnimation() { this.win.cancelAnimationFrame(this.raf); this.raf = null; clearTimeout(this.dropTimer); }
  cancel() {
    this.stopAnimation(); const s = this.state; if (!s) return; this.state = null;
    for (const node of s.originalNodes) this.container.appendChild(node);
    for (const [el, before] of s.initial) {
      el.dataset.level = before.level; el.hidden = before.hidden;
      if (before.style === null) el.removeAttribute('style'); else el.setAttribute('style', before.style);
      el.classList.remove('is-dragging'); el.querySelector('.list-item')?.style.removeProperty('transition');
    }
    this.onRestored?.();
  }
  finish(point) {
    const s = this.state; if (!s || s.settling) return; this.stopAnimation();
    if (point && Number.isFinite(point.x)) s.rightShifted = point.x - s.nestBaseX > 36;
    const placement = this.dropPlacement();
    s.pendingLevel = placement.level;
    s.wrapper.dataset.level = String(placement.level);
    s.wrapper.style.marginLeft = `${placement.level * 24}px`;
    this.moveBefore(placement.reference);
    const delta = s.pendingLevel - s.originalLevel;
    for (const child of s.children) {
      const level = Number(s.initial.get(child).level) + delta;
      child.dataset.level = String(level); child.style.marginLeft = `${level * 24}px`;
      child.style.display = s.childDisplay.get(child); child.style.transform = ''; child.style.transition = 'transform 0.22s cubic-bezier(.4,0,.2,1)';
    }
    s.wrapper.querySelector('.list-item').style.transition = '';
    s.wrapper.style.transition = 'transform 0.22s cubic-bezier(.4,0,.2,1)'; s.wrapper.style.transform = ''; s.settling = true;
    this.dropTimer = setTimeout(() => {
      if (this.state !== s) return;
      s.wrapper.classList.remove('is-dragging'); s.wrapper.style.transition = '';
      for (const child of s.children) child.style.transition = '';
      const arranged = deriveArrangedFromWrappers(this.wrappers()); this.state = null; this.onDrop?.(s.wrapper.dataset.id, arranged);
    }, 220);
  }
}
