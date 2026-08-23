// snapshot-adapter.js — приведение снимка хоста к форме, которую ждёт resolver.js
// Реализация PROPOSAL_voice-input.md §3. resolver.js не переписывается под хост.

'use strict';

// Статусы хоста -> статусы резолвера.
// Done — выполнено, Archive — снято с работы: целями команд быть не должны.
// Focus и Pause — рабочие состояния, задача остаётся доступной.
const STATUS_MAP = {
  Open: 'open',
  Focus: 'open',
  Pause: 'open',
  Done: 'closed',
  Archive: 'superseded',
};

function adaptItem(item) {
  return {
    id: item.id,
    title: item.line1 ?? '',
    status: STATUS_MAP[item.status] ?? 'open',
  };
}

function adaptSnapshot(items) {
  return items.map(adaptItem);
}

// Все потомки rootId включительно — для защиты от циклов при setParent.
function subtree(items, rootId) {
  const out = new Set([rootId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const it of items) {
      if (it.parentId && out.has(it.parentId) && !out.has(it.id)) {
        out.add(it.id);
        grew = true;
      }
    }
  }
  return out;
}

// Снимок для выбора нового родителя: без самого элемента и его поддерева.
function adaptForReparent(items, movingId) {
  const banned = subtree(items, movingId);
  return adaptSnapshot(items.filter((it) => !banned.has(it.id)));
}

export { STATUS_MAP, adaptItem, adaptSnapshot, subtree, adaptForReparent };
