import { BrowserASR } from '../../asr-browser.js';
import { MockASR } from '../../asr-mock.js';
import { deadlineDaysLabel, deadlineFromToday } from '../../task-deadline.js';
import { GestureController } from './gesture-controller.js';
import { DragController } from './drag-controller.js';
import { clientStyles } from './styles.js';

const statuses = ['Open', 'Focus', 'Pause', 'Done', 'Archive', 'Info'];
const colors = { Open: '#007aff', Focus: '#ff9500', Pause: '#5856d6', Done: '#34c759', Archive: '#8e8e93', Info: '#0a84ff' };
const point = (event) => ({ x: event.clientX, y: event.clientY });
const walk = (node, fn) => { if (!node) return; fn(node); for (const child of node.children || []) walk(child, fn); };

/** Browser boundary. Owns view/input state, never a mutable task projection. */
export class Client {
  constructor({ document: dom = globalThis.document, fetch: fetcher = globalThis.fetch?.bind(globalThis), storage = globalThis.sessionStorage, pollMs = 1200, asrFactory } = {}) {
    this.dom = dom;
    this.win = dom.defaultView;
    this.fetch = fetcher;
    this.storage = storage;
    this.pollMs = pollMs;
    this.asrFactory = asrFactory || (() => this.win.__voiceTest ? new MockASR(this.win.__voiceTest) : new BrowserASR());
    this.clientKey = this.readStorage('voicelist.v02.client') || this.win.crypto.randomUUID();
    this.writeStorage('voicelist.v02.client', this.clientKey);
    this.seq = Number(this.readStorage('voicelist.v02.seq') || 0);
    this.viewContext = { view: 'list' };
    this.nodes = new Map();
    this.backStack = [];
    this.pending = new Map();
    try { for (const body of JSON.parse(this.readStorage('voicelist.v02.pending') || '[]')) this.pending.set(body.key.seq, body); } catch { /* ignore corrupt browser cache */ }
    this.collapsed = new Map();
    this.seenActions = new Set();
    this.outcomes = new Map();
    this.requests = new Map();
    this.cursor = 0;
    this.connected = false;
    this.navigationGeneration = 0;
    this.gesture = new GestureController({
      startVoice: (target, p) => this.startVoice(target, p), finishVoice: (target) => this.finishVoice(target),
      cancelVoice: () => this.cancelVoice(), editVoice: (target) => this.finishVoice(target, true),
      tap: (target) => this.tap(target), drag: (...args) => this.drag(...args), swipe: (...args) => this.swipe(...args)
    });
  }
  readStorage(key) { try { return this.storage?.getItem(key); } catch { return null; } }
  writeStorage(key, value) { try { this.storage?.setItem(key, value); } catch { /* unavailable storage is non-fatal */ } }
  $(id) { return this.dom.getElementById(id); }
  node(tag, props = {}, text) {
    const el = this.dom.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === 'className') el.className = value;
      else if (key.startsWith('data-') || key.startsWith('aria-')) el.setAttribute(key, value);
      else el[key] = value;
    }
    if (text !== undefined) el.textContent = String(text);
    return el;
  }
  async request(path, options) {
    const response = await this.fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error?.message || data.error || `Ошибка ${response.status}`);
      error.status = response.status; error.data = data; throw error;
    }
    return data;
  }
  async connect() {
    if (this.connected) return;
    this.connected = true;
    this.mount();
    try { await this.loadDocument(); await this.flush(); this.connection(''); }
    catch (error) { this.connection('Нет соединения. Повторяем подключение…'); this.showToast(error.message); }
    this.schedulePoll();
  }
  disconnect() {
    this.connected = false;
    clearTimeout(this.pollTimer);
    this.gesture.cancel();
    this.dragController?.cancel();
    this.realtime?.stop();
  }
  async resume(cursor = this.cursor) {
    if (this.polling) return;
    this.polling = true;
    try {
      await this.flush();
      const update = await this.request(`/api/v2/updates?${new URLSearchParams({ cursor, clientKey: this.clientKey })}`);
      const changed = update.nextCursor !== undefined && String(update.nextCursor) !== String(this.cursor);
      for (const event of update.events || []) {
        if (event.requestId && ['result', 'outcome', 'action-result'].includes(event.type)) this.outcomes.set(event.requestId, event);
      }
      for (const action of update.actions || []) {
        const id = action.actionId || action.id;
        if (!this.seenActions.has(id)) { this.seenActions.add(id); this.showToast(action.label || action.text || 'Действие выполнено', id); }
      }
      for (const effect of update.uiEffects || []) await this.navigate(effect.view || 'list', effect);
      this.deferredRefresh ||= changed || update.reset || !this.document;
      if (this.deferredRefresh && this.gesture.state === 'idle' && !this.dragController?.active && !this.editorDirty && !this.transcriptEditor) { await this.loadDocument(); this.deferredRefresh = false; }
      this.cursor = update.nextCursor ?? this.cursor;
      this.connection('');
    } catch { this.connection('Нет соединения. Команды будут отправлены при восстановлении связи.'); }
    finally { this.polling = false; }
  }
  schedulePoll() {
    clearTimeout(this.pollTimer);
    if (!this.connected || !this.pollMs) return;
    this.pollTimer = setTimeout(async () => { await this.resume(); this.schedulePoll(); }, this.pollMs);
  }
  async waitForCompletion(requestId, timeoutMs = 30000) {
    const until = Date.now() + timeoutMs;
    while (Date.now() < until) {
      const body = this.requests.get(requestId);
      const receipt = body ? await this.request('/api/v2/input', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) : null;
      if (receipt?.status === 'completed') {
        this.requests.delete(requestId);
        if (receipt.error) throw new Error(receipt.error.message || receipt.error);
        await this.loadDocument();
        return { status: 'applied', newTarget: receipt.target || receipt.actions?.at(-1)?.target };
      }
      await this.resume();
      const result = this.outcomes.get(requestId);
      if (result) {
        if (['failed', 'rejected'].includes(result.status) || result.error) throw new Error(result.error?.message || result.error || 'Не удалось выполнить команду');
        return { status: 'applied', newTarget: result.newTarget || result.target };
      }
      await new Promise((resolve) => setTimeout(resolve, 350));
    }
    throw new Error('Выполнение команды ещё не подтверждено. Проверьте журнал.');
  }
  async loadDocument() {
    const generation = ++this.navigationGeneration;
    const query = new URLSearchParams({ ...this.viewContext, clientKey: this.clientKey });
    const document = await this.request(`/api/v2/document?${query}`);
    if (generation === this.navigationGeneration) this.render(document.document || document);
    return document;
  }
  context(target) {
    return { elementId: target?.id || target?.elementId || 'menu:list', view: this.viewContext.view, revision: this.document?.revision || 0,
      ...(this.viewContext.actionId ? { actionId: this.viewContext.actionId } : {}) };
  }
  async submit(input) {
    if (typeof input.text === 'string' && !input.text.trim()) return null;
    const body = { key: { clientKey: this.clientKey, seq: ++this.seq }, context: input.context || this.context(),
      ...(input.command ? { command: input.command } : { text: input.text.trim() }) };
    this.writeStorage('voicelist.v02.seq', String(this.seq));
    this.pending.set(body.key.seq, body);
    this.savePending();
    return this.deliver(body);
  }
  savePending() { this.writeStorage('voicelist.v02.pending', JSON.stringify([...this.pending.values()])); }
  async deliver(body) {
    try {
      const receipt = await this.request('/api/v2/input', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (receipt.requestId) this.requests.set(receipt.requestId, body);
      this.pending.delete(body.key.seq); this.savePending();
      if (receipt.status === 'rejected' || receipt.error) throw Object.assign(new Error(receipt.error?.message || receipt.error || 'Команда отклонена'), { status: 400 });
      this.editorDirty = false;
      await this.loadDocument();
      return receipt;
    } catch (error) {
      if (error.status >= 400 && error.status < 500) { this.pending.delete(body.key.seq); this.savePending(); }
      this.showToast(error.message || 'Не удалось отправить. Повторим после подключения.');
      throw error;
    }
  }
  async flush() { for (const body of this.pending.values()) await this.deliver(body); }
  handleInput(input) {
    return this.submit(input).catch(() => null);
  }
  async command(command, taskId = null, payload = {}, target = null) {
    try {
      const receipt = await this.submit({ context: this.context(target || (taskId ? { id: `task:${taskId}` } : null)), command: { command, actId: taskId, actType: taskId ? 'task' : 'list', payload } });
      return receipt.status === 'accepted' ? await this.waitForCompletion(receipt.requestId) : receipt;
    } catch (error) { this.showToast(error.message); return null; }
  }
  async navigate(view, params = {}, remember = true) {
    if (remember) this.backStack.push({ context: { ...this.viewContext }, scroll: this.win.scrollY, logScroll: this.$('action-log-panel')?.scrollTop || 0 });
    this.editorDirty = false;
    this.viewContext = { ...params, view };
    delete this.viewContext.clientKey;
    await this.loadDocument();
    if (view === 'log') { const panel = this.$('action-log-panel'); panel.scrollTop = panel.scrollHeight; }
  }
  async close() {
    const back = this.backStack.pop() || { context: { view: 'list' }, scroll: 0 };
    await this.navigate(back.context.view, back.context, false);
    this.win.scrollTo?.(0, back.scroll);
    if (this.$('action-log-panel')) this.$('action-log-panel').scrollTop = back.logScroll || 0;
  }
  connection(text) { if (this.$('v02-connection')) { this.$('v02-connection').textContent = text; this.$('v02-connection').hidden = !text; } }
  mount() {
    if (this.mounted) return;
    this.mounted = true;
    const style = this.node('style', {}, clientStyles); this.dom.head.append(style);
    const root = this.$('app-root');
    this.dragController = new DragController({ container: this.$('list-container'), header: root.querySelector('header'), window: this.win,
      onDrop: (id, arranged) => { this.applyCollapse(); void this.command('reorderItems', id, { arranged }); }, onRestored: () => this.applyCollapse() });
    root.prepend(this.node('div', { id: 'v02-connection', className: 'v02-connection', role: 'status', hidden: true }));
    root.append(this.node('div', { id: 'v02-toast-stack', className: 'v02-toast-stack', 'aria-live': 'polite' }));
    root.append(this.node('div', { id: 'v02-transcript', className: 'v02-transcript', hidden: true, 'aria-live': 'polite' }));
    const search = this.node('form', { className: 'v02-search', id: 'v02-search' });
    const searchInput = this.node('input', { type: 'search', placeholder: 'Поиск задач', 'aria-label': 'Поиск задач' });
    search.append(searchInput, this.node('button', { className: 'v02-primary', type: 'submit' }, 'Найти'));
    search.onsubmit = (event) => { event.preventDefault(); void this.navigate('search', { query: searchInput.value }); };
    root.querySelector('header').after(search);
    const bind = (id, handler) => { if (this.$(id)) this.$(id).onclick = handler; };
    bind('frontier-tab-btn', () => this.navigate(this.viewContext.view === 'frontier' ? 'list' : 'frontier'));
    bind('view-toggle-btn', () => this.navigate(this.viewContext.view === 'log' ? 'list' : 'log'));
    bind('settings-btn', () => this.navigate('settings'));
    bind('settings-close', () => this.close());
    bind('add-btn', () => this.openAdd());
    bind('btn-cancel', () => this.$('modal-overlay').classList.remove('open'));
    bind('btn-confirm', () => this.saveAdd());
    bind('task-page-close', () => this.close());
    bind('task-page-save', () => this.saveTask());
    bind('task-page-add-child', async () => {
      const input = this.$('task-page-child-input'); if (!input.value.trim()) return;
      await this.command('addChild', this.editorNode?.props.taskId, { line1: input.value.trim(), line2: '' }); input.value = '';
    });
    bind('undo-btn', () => this.command('undo'));
    bind('workflowy-import-btn', async () => {
      const status = this.$('workflowy-import-status'); status.textContent = 'Импорт…';
      const result = await this.command('importWorkflowy', null, { url: this.$('workflowy-url-input').value.trim() });
      status.textContent = result ? 'Импорт завершён' : 'Не удалось импортировать';
    });
    for (const id of ['input-line1', 'input-line2']) if (this.$(id)) this.$(id).onkeydown = (event) => { if (event.key === 'Enter') void this.saveAdd(); };
    for (const id of ['task-page-line1', 'task-page-status']) if (this.$(id)) this.$(id).oninput = () => { this.editorDirty = true; };
    if (this.$('task-page-line1')) {
      const details = this.node('textarea', { id: 'task-page-line2', className: 'v02-task-details', placeholder: 'Подробности', 'aria-label': 'Подробности задачи' });
      details.oninput = () => { this.editorDirty = true; };
      this.$('task-page-line1').parentElement.append(details);
    }
    const menus = { 'frontier-tab-btn': 'frontier', 'settings-btn': 'settings', 'view-toggle-btn': 'log', 'add-btn': 'add', 'dialogues-tab-btn': 'dialogues' };
    for (const [id, menu] of Object.entries(menus)) if (this.$(id)) this.$(id).dataset.componentId = `menu:${menu}`;
    this.bindGestures(root);
    this.win.addEventListener('pagehide', () => this.disconnect());
    // A microphone permission prompt blurs the window. Cancelling the whole
    // gesture here would stop ASR immediately, so blur only aborts an active
    // drag. Page hiding still cancels every in-progress input safely.
    this.win.addEventListener('blur', () => {
      if (this.gesture.state === 'dragging') this.gesture.cancel();
      else this.dragController.cancel();
    });
    this.dom.addEventListener('visibilitychange', () => { if (this.dom.visibilityState === 'hidden') { this.gesture.cancel(); this.dragController.cancel(); } });
    this.win.addEventListener('online', () => void this.resume());
  }
  render(document) {
    if (document.schemaVersion !== 1 || !document.root) throw new Error('Неподдерживаемый документ интерфейса');
    if (!this.document) this.cursor = document.cursor ?? this.cursor;
    this.document = document;
    const renderKey = JSON.stringify([document.view, document.root]);
    if (renderKey === this.renderKey) return;
    this.renderKey = renderKey;
    const logPanel = this.$('action-log-panel');
    const keepLogAtEnd = this.$('app-root').dataset.viewMode === 'log' && logPanel.scrollTop + logPanel.clientHeight >= logPanel.scrollHeight - 3;
    const logScroll = logPanel.scrollTop;
    this.nodes.clear(); walk(document.root, (node) => this.nodes.set(node.id, node));
    const view = document.view?.view || document.view || this.viewContext.view;
    this.$('app-root').dataset.viewMode = view;
    this.$('app-root').style.setProperty('--v02-header-height', `${this.dom.querySelector('header').getBoundingClientRect().height}px`);
    const list = this.$('list-container'); list.replaceChildren();
    const log = this.$('action-log-list'); log.replaceChildren();
    this.$('action-log-panel').hidden = view !== 'log';
    this.$('list-container').hidden = !['list', 'frontier', 'search'].includes(view);
    if (!['dialogues'].includes(view)) this.$('app-root').dataset.dialoguesOpen = 'false';
    this.$('v02-search').hidden = !['list', 'frontier', 'search'].includes(view);
    this.$('settings-overlay').classList.toggle('open', view === 'settings');
    this.$('settings-overlay').setAttribute('aria-hidden', String(view !== 'settings'));
    this.$('task-page').classList.toggle('open', view === 'edit');
    this.$('task-page').setAttribute('aria-hidden', String(view !== 'edit'));
    this.$('v02-action-page')?.remove();
    this.$('frontier-tab-btn').classList.toggle('active', view === 'frontier');
    this.$('view-toggle-btn').textContent = view === 'log' ? 'Список' : 'Журнал';
    for (const child of document.root.children || []) this.renderComponent(child, list);
    if (!list.children.length && ['list', 'frontier', 'search'].includes(view)) list.append(this.node('div', { className: 'empty-state' }, view === 'search' ? 'Ничего не найдено.' : 'Пока нет задач. Нажмите +, чтобы добавить.'));
    this.applyCollapse();
    if (view === 'log') logPanel.scrollTop = keepLogAtEnd ? logPanel.scrollHeight : logScroll;
    if (view === 'dialogues') this.realtime?.openDialogues();
  }
  renderComponent(component, parent) {
    const p = component.props || {};
    switch (component.type) {
      case 'application': for (const child of component.children || []) this.renderComponent(child, parent); break;
      case 'task-list':
        if (p.query) parent.append(this.node('div', { className: 'search-summary' }, `Поиск: ${p.query}`));
        if (p.focusHighlights?.length) parent.append(this.node('div', { className: 'frontier-focus-strip' }, `Фокус: ${p.focusHighlights.map(x => x.line1).join(' · ')}`));
        for (const child of component.children || []) this.renderComponent(child, parent); break;
      case 'toolbar': this.$('undo-btn').disabled = p.canUndo === false; if (p.title) this.dom.querySelector('header h1').textContent = p.title; break;
      case 'task': parent.append(this.renderTask(component)); break;
      case 'action-list': for (const child of component.children || []) this.renderComponent(child, this.$('action-log-list')); break;
      case 'action': parent.append(this.renderAction(component)); break;
      case 'action-page': this.renderActionPage(component); break;
      case 'task-editor': this.renderEditor(component); break;
      case 'text': parent.append(this.node('div', { className: p.className === 'frontier-focus-strip' ? p.className : 'search-summary', 'data-component-id': component.id }, p.text || '')); break;
      case 'settings': case 'dialogues': break;
      default: parent.append(this.node('div', { className: 'v02-error' }, `Неизвестный компонент: ${component.type}`));
    }
  }
  renderTask(component) {
    const p = component.props;
    const wrap = this.node('div', { className: 'list-item-wrapper', 'data-id': p.taskId, 'data-act-id': p.taskId, 'data-act-type': 'task', 'data-level': p.level || 0, 'data-component-id': component.id });
    wrap.style.marginLeft = `${(p.level || 0) * 24}px`;
    const row = this.node('div', { className: 'list-item', 'data-act-id': p.taskId, 'data-act-type': 'task', tabIndex: 0 });
    const head = this.node('div', { className: 'item-head' });
    const copy = this.node('div', { className: 'item-copy' });
    const title = this.node('div', { className: 'item-line1' });
    if (p.hasChildren) title.append(this.node('span', { className: 'chevron' }, (this.collapsed.get(p.taskId) ?? p.collapsed) ? '▶' : '▼'));
    title.append(this.dom.createTextNode(p.line1 || ''));
    copy.append(title);
    if (p.line2) copy.append(this.node('div', { className: 'item-line2' }, p.line2));
    if (p.tags?.length) { const tags = this.node('div', { className: 'item-tags' }); for (const tag of p.tags) tags.append(this.node('span', { className: 'item-tag' }, tag)); copy.append(tags); }
    const side = this.node('div', { className: 'item-side' });
    const badge = this.node('span', { className: 'status-badge' }, p.status || 'Open'); badge.style.setProperty('--badge-color', colors[p.status] || colors.Open);
    side.append(badge, this.node('div', { className: 'item-index' + (this.viewContext.view === 'frontier' ? ' item-deadline' : '') }, this.viewContext.view === 'frontier' ? deadlineDaysLabel(p.deadline) : p.index || ''));
    head.append(copy, side); row.append(head); wrap.append(row);
    row.onkeydown = (event) => { if (event.key === 'Enter') void this.navigate('edit', { taskId: p.taskId }); };
    row.ondblclick = () => this.navigate('edit', { taskId: p.taskId });
    return wrap;
  }
  applyCollapse() {
    const hiddenAt = [];
    for (const wrap of this.$('list-container').querySelectorAll('.list-item-wrapper')) {
      const level = Number(wrap.dataset.level);
      while (hiddenAt.length && hiddenAt.at(-1) >= level) hiddenAt.pop();
      wrap.hidden = hiddenAt.length > 0;
      const node = this.nodes.get(wrap.dataset.componentId);
      if (this.collapsed.get(wrap.dataset.id) ?? node?.props.collapsed) hiddenAt.push(level);
    }
  }
  renderAction(component) {
    const p = component.props; const id = p.actionId || component.id.replace(/^action:/, '');
    this.seenActions.add(id);
    const row = this.node('div', { className: 'action-log-row', 'data-log-id': id, 'data-component-id': component.id, tabIndex: 0 });
    row.append(this.node('div', { className: 'action-log-label' }, p.transcript || p.text || 'Реплика пользователя'), this.node('div', { className: 'action-log-status' }, p.status || ''), this.node('div', { className: 'action-log-meta' }, p.createdAt || ''));
    row.onclick = () => this.navigate('action', { actionId: id });
    row.onkeydown = (event) => { if (event.key === 'Enter') row.click(); };
    return row;
  }
  renderActionPage(component) {
    const p = component.props;
    const page = this.node('section', { id: 'v02-action-page', className: 'v02-action-page', 'data-component-id': component.id, 'aria-label': 'Действие' });
    const bar = this.node('div', { className: 'task-page-bar' });
    const close = this.node('button', { id: 'action-close', className: 'task-page-close' }, 'Закрыть'); close.onclick = () => this.close();
    const rollback = this.node('button', { id: 'action-rollback', className: 'task-page-save', disabled: p.canRollback === false }, 'Откатить');
    rollback.onclick = () => this.handleInput({ context: this.context(component), command: { command: 'rollbackAction', actId: p.actionId, actType: 'action', payload: { actionId: p.actionId } } });
    bar.append(close, rollback); page.append(bar);
    const body = this.node('div', { className: 'v02-action-body' });
    body.append(this.node('h2', {}, p.title || p.label || 'Действие'));
    for (const record of p.records || []) {
      const block = this.node('section', { className: 'v02-journal-record', 'data-entry-id': record.id });
      if (record.kind === 'text') {
        block.append(this.node('small', {}, record.corrects ? 'Корректировка' : 'Реплика пользователя'), this.node('div', { className: 'v02-message', 'data-role': 'user' }, record.userText || ''));
        const context = this.node('pre', { className: 'v02-model-context', hidden: true }, JSON.stringify(record.modelContext || {}, null, 2));
        const structured = { ...(record.answer ? { answer: record.answer } : {}), ...(record.commands?.length ? { commands: record.commands } : {}) };
        const answer = this.node('button', { className: 'v02-action-card v02-model-answer', type: 'button', 'aria-expanded': 'false' }, JSON.stringify(structured, null, 2));
        answer.onclick = () => { context.hidden = !context.hidden; answer.setAttribute('aria-expanded', String(!context.hidden)); };
        block.append(context, answer);
      } else {
        block.append(this.node('small', {}, record.corrects ? 'Корректировка интерфейса' : 'Команда интерфейса'), this.node('pre', { className: 'v02-action-card' }, JSON.stringify(record.command || {}, null, 2)));
      }
      body.append(block);
    }
    const form = this.node('form', { className: 'v02-correction-form' });
    const input = this.node('textarea', { id: 'action-correction-input', placeholder: 'Что нужно исправить?', 'aria-label': 'Корректировка действия' });
    input.oninput = () => { this.editorDirty = !!input.value; };
    form.append(input, this.node('button', { type: 'submit' }, 'Отправить'));
    form.onsubmit = async (event) => { event.preventDefault(); if (!input.value.trim()) return; await this.handleInput({ text: input.value, context: this.context(component) }); };
    body.append(form, this.node('small', {}, 'Или удерживайте элемент действия и говорите. Ответ появится текстом.'));
    page.append(body); this.$('app-root').append(page);
  }
  renderEditor(component) {
    this.editorNode = component; const p = component.props;
    this.$('task-page').dataset.componentId = component.id;
    this.$('task-page-line1').value = p.line1 || '';
    this.$('task-page-line2').value = p.line2 || '';
    this.$('task-page-status').value = p.status || 'Open';
    const link = this.$('task-page-parent'); link.hidden = !p.parent; link.textContent = p.parent?.line1 || '';
    link.onclick = (event) => { event.preventDefault(); void this.navigate('edit', { taskId: p.parent.id || p.parent.taskId }); };
    const children = this.$('task-page-subtasks'); children.replaceChildren();
    for (const child of p.subtasks || []) {
      const row = this.node('div', { className: 'task-page-subtask', 'data-id': child.id || child.taskId, 'data-component-id': `task:${child.id || child.taskId}` });
      row.append(this.node('span', {}, child.line1), this.node('small', {}, child.status));
      row.onclick = () => this.navigate('edit', { taskId: child.id || child.taskId }); children.append(row);
    }
  }
  async saveTask() {
    const p = this.editorNode?.props; if (!p) return;
    const line1 = this.$('task-page-line1').value.trim(); if (!line1) return;
    const line2 = this.$('task-page-line2').value;
    const status = this.$('task-page-status').value;
    if (p.mode === 'add' || !p.taskId) { await this.command(p.parentId ? 'addChild' : 'addItem', p.parentId, { line1, line2, status }); return this.close(); }
    // Capture every draft before the first reply renders the page again.
    if (line1 !== p.line1 || line2 !== (p.line2 || '')) await this.command('editItem', p.taskId, { line1, line2 });
    if (status !== p.status) await this.command('setStatus', p.taskId, { status });
    this.showToast('Сохранено');
  }
  openAdd(parentId = null) {
    this.addParentId = parentId;
    this.$('input-line1').value = ''; this.$('input-line2').value = '';
    this.$('modal-title').textContent = parentId ? 'Новая подзадача' : 'Новый элемент';
    this.$('modal-overlay').classList.add('open'); this.$('input-line1').focus();
  }
  async saveAdd() {
    const line1 = this.$('input-line1').value.trim(); if (!line1) return;
    const result = await this.command(this.addParentId ? 'addChild' : 'addItem', this.addParentId, { line1, line2: this.$('input-line2').value.trim() });
    if (result) this.$('modal-overlay').classList.remove('open');
  }
  showToast(message, actionId) {
    const button = this.node('button', { className: 'v02-toast', type: 'button' }, message);
    if (actionId) button.onclick = () => this.navigate('action', { actionId });
    this.$('v02-toast-stack')?.append(button); setTimeout(() => button.remove(), 2000);
  }
  bindGestures(root) {
    // Pointer preventDefault alone cannot stop native touch panning. Once the
    // hold has selected voice/drag, keep the touch stream; pre-hold moves still
    // use native scrolling and cancel the pending hold normally.
    root.addEventListener('touchmove', (event) => {
      if (['recording', 'editing', 'dragging', 'swiping', 'cancelled'].includes(this.gesture.state)) event.preventDefault();
    }, { passive: false });
    root.addEventListener('pointerdown', (event) => {
      // A new deliberate press must not be swallowed by the preceding gesture's click guard.
      this.suppressClickUntil = 0;
      if (event.button !== 0 || event.target.closest('input,textarea,select')) return;
      this.edgeStart = event.clientY <= 35 ? point(event) : null;
      if (this.edgeStart) return;
      const element = event.target.closest('[data-component-id]'); if (!element) return;
      const node = this.nodes.get(element.dataset.componentId);
      const taskId = node?.props.taskId || element.closest('[data-id]')?.dataset.id;
      this.pointerId = event.pointerId;
      this.gesture.begin(point(event), { id: element.dataset.componentId, element, node, taskId, draggable: !!taskId && this.viewContext.view === 'list' });
    });
    this.dom.addEventListener('pointermove', (event) => {
      if (this.edgeStart) return;
      if (this.gesture.state !== 'idle' && event.pointerId === this.pointerId) {
        const state = this.gesture.move(point(event));
        if (['recording', 'editing', 'dragging', 'swiping', 'cancelled'].includes(state)) event.preventDefault();
        if (state === 'editing') this.transcriptHint('Отпустите, чтобы исправить текст. Ещё ниже — отмена', 'editing');
        if (state === 'cancelled') this.transcriptHint('Ввод отменён', 'cancelled');
      }
    }, { passive: false });
    this.dom.addEventListener('pointerup', (event) => {
      if (this.edgeStart) { if (event.clientY - this.edgeStart.y > 50) void this.navigate('log'); this.edgeStart = null; return; }
      if (event.pointerId !== this.pointerId) return;
      const cancelled = this.gesture.state === 'cancelled';
      const suppress = this.gesture.state !== 'holding' && this.gesture.state !== 'idle';
      if (suppress) this.suppressClickUntil = Date.now() + 400;
      void this.gesture.end(point(event));
      if (cancelled) this.$('v02-transcript').hidden = true;
    });
    this.dom.addEventListener('pointercancel', () => { this.edgeStart = null; this.gesture.cancel(); });
    root.addEventListener('click', (event) => { if (Date.now() < (this.suppressClickUntil || 0)) { event.preventDefault(); event.stopImmediatePropagation(); } }, true);
    let logStart;
    const panel = this.$('action-log-panel');
    panel.addEventListener('pointerdown', (event) => { logStart = { y: event.clientY, bottom: panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 3 }; });
    const finishLogSwipe = (y) => { if (this.viewContext.view === 'log' && logStart?.bottom && logStart.y - y > 65) { this.gesture.cancel(); this.suppressClickUntil = Date.now() + 400; void this.close(); } logStart = null; };
    panel.addEventListener('pointerup', (event) => finishLogSwipe(event.clientY));
    // Touch scrolling can cancel the pointer stream; the native touchend still closes at the boundary.
    panel.addEventListener('touchstart', (event) => { logStart = { y: event.touches[0].clientY, bottom: panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 3 }; }, { passive: true });
    panel.addEventListener('touchend', (event) => { if (event.changedTouches[0]) finishLogSwipe(event.changedTouches[0].clientY); }, { passive: true });
  }
  tap(target) {
    if (!target.taskId || target.element.closest('#task-page')) return;
    if (this.viewContext.view === 'frontier' && target.node?.props.parent) {
      const existing = target.element.previousElementSibling;
      if (existing?.classList.contains('frontier-parent-wrapper')) { existing.remove(); target.element.style.marginLeft = '0px'; return; }
      const parent = target.node.props.parent;
      const contextRow = this.node('div', { className: 'list-item-wrapper frontier-parent-wrapper', 'data-id': `parent:${parent.id}` });
      const row = this.node('div', { className: 'list-item frontier-parent-item' }); row.append(this.node('div', { className: 'item-line1' }, parent.line1)); contextRow.append(row);
      contextRow.onclick = () => { if (parent.id !== '__root__') void this.navigate('edit', { taskId: parent.id }); };
      target.element.before(contextRow); target.element.style.marginLeft = '24px'; return;
    }
    if (target.node?.props.hasChildren) {
      this.collapsed.set(target.taskId, !(this.collapsed.get(target.taskId) ?? target.node.props.collapsed));
      this.applyCollapse(); const chevron = target.element.querySelector('.chevron'); if (chevron) chevron.textContent = this.collapsed.get(target.taskId) ? '▶' : '▼';
    } else void this.navigate('edit', { taskId: target.taskId });
  }
  startVoice(target, p) {
    this.realtime?.stop();
    this.voice = { target, text: '', final: '', error: false, asr: this.asrFactory() };
    const voice = this.voice;
    target.element.classList.add('v02-voice-target');
    const overlay = this.$('v02-transcript'); overlay.hidden = false; overlay.dataset.state = 'recording';
    overlay.style.left = `${Math.max(8, Math.min(p.x - 100, this.win.innerWidth - 280))}px`;
    overlay.style.top = `${Math.max(8, p.y - 75)}px`; overlay.textContent = 'Говорите…';
    voice.asr.start({
      onInterim: (text) => { voice.text = text; if (this.voice === voice) overlay.textContent = voice.final ? `${voice.final} ${text}` : text; },
      onFinal: (text) => { voice.final = text; voice.text = text; if (this.voice === voice) overlay.textContent = text; },
      onError: (code) => { voice.error = true; if (this.voice === voice) overlay.textContent = code === 'denied' ? 'Микрофон недоступен' : 'Не удалось распознать речь'; }
    });
    voice.asr.speak?.();
  }
  transcriptHint(text, state) {
    const el = this.$('v02-transcript'); if (!el) return;
    el.hidden = false; el.dataset.state = state;
    el.textContent = state === 'cancelled' ? text : `${this.voice?.text || ''}\n${text}`;
  }
  cancelVoice() {
    const voice = this.voice; this.voice = null;
    if (voice) { voice.target.element.classList.remove('v02-voice-target'); void voice.asr.stop(); }
    if (this.$('v02-transcript')) this.$('v02-transcript').hidden = true;
  }
  async finishVoice(target, edit = false) {
    const voice = this.voice; if (!voice) return;
    // Opening and focusing the editor must stay in the pointer-release call
    // stack. Mobile browsers may refuse to show the keyboard after the first
    // await because the user activation has already expired.
    const initialDraft = (voice.final || voice.text).trim();
    const editor = edit ? this.editTranscript(initialDraft, target) : null;
    await voice.asr.stop();
    const text = (voice.final || voice.text).trim();
    if (this.voice === voice) {
      this.voice = null;
      voice.target.element.classList.remove('v02-voice-target');
      if (this.$('v02-transcript')) this.$('v02-transcript').hidden = true;
    }
    if (edit) {
      // A final ASR result may arrive during stop(). Apply it only while the
      // user has not changed the draft that was shown on release.
      if (text && !voice.error && this.transcriptEditor === editor.overlay && editor.input.value === initialDraft) {
        editor.input.value = text;
        const end = editor.input.value.length;
        editor.input.setSelectionRange(end, end);
      }
      // Editing mode is also the text-input fallback. Keep its focused editor
      // open when recognition yields no text or reports an error.
      if (voice.error) this.showToast('Речь не отправлена: проверьте доступ к микрофону.');
      return;
    }
    if (!text || voice.error) { if (voice.error) this.showToast('Речь не отправлена: проверьте доступ к микрофону.'); return; }
    await this.handleInput({ text, context: this.context(target) });
  }
  editTranscript(text, target) {
    const overlay = this.node('div', { className: 'v02-edit-transcript', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Редактирование распознанного текста' });
    const box = this.node('section'); const input = this.node('textarea', { id: 'transcript-edit-input', value: text });
    const nav = this.node('nav'); const cancel = this.node('button', { className: 'btn btn-cancel' }, 'Отмена'); const send = this.node('button', { className: 'v02-primary' }, 'Отправить');
    const close = () => { overlay.remove(); this.transcriptEditor = null; };
    cancel.onclick = close; send.onclick = async () => { if (!input.value.trim()) return; const value = input.value; close(); await this.handleInput({ text: value, context: this.context(target) }); };
    nav.append(cancel, send); box.append(this.node('h2', {}, 'Исправить текст'), input, nav); overlay.append(box); this.$('app-root').append(overlay); this.transcriptEditor = overlay;
    try { input.focus({ preventScroll: true }); } catch { input.focus(); }
    const end = input.value.length;
    input.setSelectionRange(end, end);
    return { overlay, input, close };
  }
  drag(phase, target, p, delta) {
    const wrap = target.element.closest('.list-item-wrapper'); if (!wrap) return;
    if (phase === 'move') {
      if (!this.dragController.active) this.dragController.start(wrap, p, { x: p.x - delta.dx, y: p.y - delta.dy });
      else this.dragController.update(p);
      return;
    }
    if (phase === 'end') this.dragController.finish(p); else this.dragController.cancel();
  }
  swipe(phase, target, p, delta) {
    if (!target.taskId) return;
    const panel = delta.dx >= 0 ? this.$('drop-zone-panel') : this.$('tag-panel');
    if (phase === 'move') {
      if (this.swipePanel !== panel) {
        this.swipePanel?.classList.remove('visible'); this.swipePanel = panel; panel.replaceChildren();
        const status = target.node?.props.status;
        const toggles = ['Done', status === 'Pause' ? 'Open' : 'Pause', status === 'Focus' ? 'Open' : 'Focus', status === 'Info' ? 'Open' : 'Info', 'Archive'];
        let actions = delta.dx >= 0 ? [
          ...toggles.map((next) => [next, () => this.command('setStatus', target.taskId, { status: next }), colors[next]]),
          ['Edit', () => this.navigate('edit', { taskId: target.taskId }), '#007aff']
        ] : ['Важное', 'Срочно', 'Купить', 'Дом', 'Работа', 'Отложить'].map((tag) => [tag, () => this.command('setTags', target.taskId, { tag }), '#5856d6']);
        if (status === 'Focus' && delta.dx < 0) actions = [[0, 'Сегодня'], [1, 'Завтра'], [3, 'Три дня'], [7, 'Неделя'], [30, 'Месяц']].map(([days, label]) => [label, () => this.command('setDeadline', target.taskId, { deadline: deadlineFromToday(days) }), '#5856d6']);
        const rowHeight = Math.min(72, (this.win.innerHeight - 80) / actions.length);
        for (const [label, run, color] of actions) {
          const item = this.node('div', { className: 'panel-item' }); item.append(this.node('span', { className: 'panel-label' }, label));
          item.style.background = color; item.style.height = `${rowHeight}px`; item.runAction = run; panel.append(item);
        }
        panel.classList.add('visible'); panel.style.top = `${Math.max(72, Math.min(p.y - 100, this.win.innerHeight - actions.length * rowHeight - 8))}px`;
      }
      const item = [...panel.children].find((child) => { const rect = child.getBoundingClientRect(); return p.y >= rect.top && p.y < rect.bottom; });
      for (const child of panel.children) child.classList.toggle('active', child === item);
      if (item) this.swipeSelection = item;
      return;
    }
    if (phase === 'end') this.swipeSelection?.runAction?.();
    this.swipePanel?.classList.remove('visible'); this.swipePanel = null; this.swipeSelection = null;
  }
  /** Read-only bridge for the reused Realtime transport, not a client task store. */
  getTaskState() { return { snapshot: { items: [...this.nodes.values()].filter((node) => node.type === 'task').map((node) => ({ ...node.props, id: node.props.taskId })) } }; }
}
