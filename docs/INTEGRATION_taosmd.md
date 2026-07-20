# INTEGRATION_taosmd — интеграция VoiceList с бэкендом taosmd

Статус: rev 4. Скоуп v1: storage-модуль (точка входа — [CODER_TASK_storage-module.md](./CODER_TASK_storage-module.md)) + инфраструктура доступа.

## 0. Координаты разработки

| Роль | Репозиторий | Ветка |
|---|---|---|
| Фронт (этот репо) | `021-lab/VoiceList` | `taosmd-backend` |
| Бэкенд taosmd | **`021-lab/TaOS`** — https://github.com/021-lab/TaOS/tree/feat/user-statuses | `feat/user-statuses` |

Канон спецификаций бэкенда — `docs/specs/` в `021-lab/TaOS` (начинать с `00-INDEX-focus-harness.md`): контракт API, жизненный цикл задач, задания бэкенд-кодеру. Здесь копий нет.

**Правила координации:** изменение поведения API сначала попадает амендментом в контракт в TaOS, затем сюда; потребность фронта в новом endpoint или статусе — issue в TaOS, не обход на клиенте. PINNED_COMMIT бэкенда фиксируется в `/deploy/README.md` после реализации статусов focus/pause в TaOS.

## 1. Архитектура доступа (recorder.smileme.ai)

- **Cloudflare Pages** — статика фронта (self-contained `list-manager.html`).
- **Worker-proxy**: route `recorder.smileme.ai/api/*` → strip `/api` → Cloudflare Tunnel → taosmd на Mac Mini (`127.0.0.1:7900`, наружу только через cloudflared). Bearer-токен taosmd — secret Worker'а; в браузер не попадает никогда.
- **Cloudflare Access** на весь хост: allowlist владельца.
- Same-origin по построению: CORS отсутствует; в коде фронта — относительный `/api/*` (конфигурируемый base URL для локальной разработки, без хардкода доменов).

## 2. Модель данных v1

```
Приложение → storage-модуль → localStorage (рабочая реплика)
                     └→ taosmd /tasks (write-through, истина)
                     └→ taosmd /a2a  (журнал действий в канал лога)
```

Пустой localStorage → bootstrap из taosmd. Каждая операция меняет localStorage и сразу синкается в `/tasks`. `list-data.js` удалён: seed-снапшотов больше нет, единственный источник начальных данных — бэкенд.

## 3. Структура репозитория

```
list-manager.html            — собранный self-contained фронт (артефакт сборки)
list-manager.template.html   — шаблон сборки
list-manager.css, src/, scripts/, tests/ — исходники (вкл. storage-модуль), сборка, тесты
list-manager-docs.md         — документация приложения
docs/                        — интеграционные спеки (этот файл + задание storage-модуля)
proxy/                       — (создаётся) Worker: wrangler.toml, src/index.js
deploy/                      — (создаётся) README деплоя: Pages, wrangler, cloudflared, PINNED_COMMIT
```

Удаляется: `list-data.js` и все ссылки на него. Сборка: `npm run prepare-preview`, единственный деплой-артефакт — `list-manager.html`.

## 4. Используемое подмножество API (все пути через /api/*)

- `GET /health` — индикатор соединения
- `GET /tasks?limit=` — bootstrap: список задач
- `GET /tasks/edges` — bootstrap: граф рёбер (fork-endpoint)
- `POST /tasks` — создание
- `POST /tasks/{id}` — мутация (status/title/body/priority)
- `POST /tasks/{id}/edges`, `POST /tasks/{id}/edges/remove` — структура дерева
- `POST /a2a/send`, `GET /a2a/messages` — журнал действий (thread `voicelist-log`)

Worker-proxy держит allowlist ровно этого подмножества; остальное — 404. Не используются в v1: `/tasks/ready`, `/tasks/prime` (агентские), память (`/ingest`, `/search`).

## 5. Обязательные правила интеграции

1. **Провенанс**: все мутации фронта — `created_by=user`, журнальные сообщения — `from=user`. Серверные дефолты не используются.
2. **Парность рёбер**: ребёнок создаётся единым хелпером с двумя рёбрами — `parent` + `blocks` ребёнок→родитель (TASK_LIFECYCLE §1). Расщеплять пару запрещено.
3. **Статусы**: маппинг локальных статусов в enum бэкенда — таблица в CODER_TASK_storage-module.md; `focus`/`pause` — нативные статусы бэкенда (амендмент rev 2 контракта), оба вне ready и оба блокируют родителя. Статусы агентов (`in_progress`, `blocked`) фронт не выставляет и не затирает.
4. **Инвариант границы**: только HTTP-контракт; прямой доступ к файлам/БД taosmd запрещён.
5. **Секреты**: токен только в Worker-secret; в бандле и трафике браузера его нет.

## 6. Задачи

A. **Storage-модуль** — главная задача, полное ТЗ: [CODER_TASK_storage-module.md](./CODER_TASK_storage-module.md).
B. **Worker-proxy** (`/proxy`): маршрутизация `/api/*`, allowlist §4, secret `TAOSMD_TOKEN`, таймауты, честный 502/504 при недоступности tunnel.
C. **Деплой** (`/deploy`): Pages-проект на recorder.smileme.ai, `wrangler deploy` с route, конфиг cloudflared (ingress → 127.0.0.1:7900), порядок включения Cloudflare Access, PINNED_COMMIT бэкенда.

## 7. Приёмка инфраструктуры (помимо приёмки storage-модуля)

1. С телефона вне tailnet: recorder.smileme.ai открывается через Access, дерево грузится из живого taosmd.
2. Токен в браузере отсутствует; Authorization добавляет Worker.
3. Прямой запрос к `/api/*` без Access-сессии отклоняется; пути вне allowlist — 404.
