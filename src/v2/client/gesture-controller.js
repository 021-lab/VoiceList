/** Local input state machine. It never interprets speech or changes task data. */
export class GestureController {
  constructor({ startVoice, finishVoice, cancelVoice, editVoice, drag, swipe, tap, holdMs = 300 }) {
    Object.assign(this, { startVoice, finishVoice, cancelVoice, editVoice, drag, swipe, tap, holdMs });
    this.state = 'idle';
  }
  begin(point, target) {
    this.cancel();
    this.origin = point;
    this.point = point;
    this.target = target;
    this.state = 'holding';
    this.timer = setTimeout(() => {
      if (this.state !== 'holding') return;
      this.state = 'recording';
      this.startVoice?.(target, point);
    }, this.holdMs);
  }
  move(point) {
    if (this.state === 'idle') return;
    this.point = point;
    const dx = point.x - this.origin.x;
    const dy = point.y - this.origin.y;
    if (this.state === 'holding' && Math.abs(dx) > 20 && Math.abs(dx) > Math.abs(dy)) {
      clearTimeout(this.timer);
      this.state = 'swiping';
    } else if (this.state === 'holding' && Math.abs(dy) > 15) {
      this.cancel(); // ordinary scrolling before the hold threshold
      return;
    }
    if (this.state === 'recording' && dy < -18 && this.target.draggable) {
      this.state = 'dragging';
      this.cancelVoice?.();
    } else if (this.state === 'recording' && dy > 24) {
      this.state = 'editing';
    }
    if (this.state === 'editing' && dy > 120) {
      this.state = 'cancelled';
      this.cancelVoice?.();
    }
    if (this.state === 'dragging') this.drag?.('move', this.target, point, { dx, dy });
    if (this.state === 'swiping') this.swipe?.('move', this.target, point, { dx, dy });
    return this.state;
  }
  async end(point = this.point) {
    clearTimeout(this.timer);
    const state = this.state;
    const target = this.target;
    this.state = 'idle';
    if (state === 'recording') await this.finishVoice?.(target);
    if (state === 'editing') await this.editVoice?.(target);
    if (state === 'holding') this.tap?.(target);
    if (state === 'dragging') this.drag?.('end', target, point, { dx: point.x - this.origin.x, dy: point.y - this.origin.y });
    if (state === 'swiping') this.swipe?.('end', target, point, { dx: point.x - this.origin.x, dy: point.y - this.origin.y });
    this.target = null;
  }
  cancel() {
    clearTimeout(this.timer);
    if (['recording', 'editing'].includes(this.state)) this.cancelVoice?.();
    if (this.state === 'dragging') this.drag?.('cancel', this.target, this.point, {});
    if (this.state === 'swiping') this.swipe?.('cancel', this.target, this.point, {});
    this.state = 'idle';
    this.target = null;
  }
}
