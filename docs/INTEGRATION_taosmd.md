# INTEGRATION_taosmd — интеграция VoiceList с бэкендом taosmd

Статус: rev 3. Скоуп v1: модуль синхронизации (см. [CODER_TASK_sync-module.md](./CODER_TASK_sync-module.md) — точка входа) + инфраструктура доступа.

## 0. Координаты разработки

| Роль | Репозиторий | Ветка |
|---|---|---|
| Фронт (этот репо) | `021-lab/VoiceList` | `taosmd-backend` |
| Бэкенд taosmd | **`021-lab/TaOS`** — https://github.com/021-lab/TaOS/tree/feat/user-statuses | `feat/user-statuses` |

Канон спецификаций бэкенда — `docs/specs/` в `021-lab/TaOS` (начинать с `00-INDEX-focus-harness.md`): контракт API, жизненный цикл задач, задания бэкенд-кодеру. Здесь копий нет.

**Правила координации:** изменение поведения API сначала попадает амендментом в контракт в TaOS, затем сюда; потребность фронта в новом endpoint — issue в TaOS, не обход на клиенте. Пин бэкенда (PINNED_COMMIT) фиксируется в `/deploy/README.md` после реализации статусов в TaOS.

## 1. Архитектура доступа (recorder.smileme.ai)

- **Cloudflare Pages** — статика фронта (self-contained `list-manager.html`).
- **Worker-proxy**: route `recorder.smileme.ai/api/*` → strip `/api` → Cloudflare Tunnel → taosmd на Mac Mini (`127.0.0.1:7900`, наружу только через cloudflared). Bearer-токен taosmd — secret Worker'а, добавляется на проксировании; в браузер не попадает никогда.
- **Cloudflare Access** на весь хост: allowlist владельца.
- Same-origin по построению: CORS отсутствует; в коде фронта — относительный `/api/*` (конфигурируемый base URL для локальной разработки, без хардкода доменов).

## 2. Структура репозитория (фактическая)

```
list-manager.html            — собранный self-contained фронт (артефакт сборки)
list-manager.template.html   — шаблон сборки
list-manager.css, list-data.js, src/, scripts/, tests/ — исходники, seed, сборка, тесты
list-manager-docs.md         — документация приложения, вкл. статусы и фронтир/резолвер (Θ)
docs/                        — интеграционные спеки (этот файл + задание sync-модуля)
proxy/                       — (создаётся) Worker: wrangler.toml, src/index.js
deploy/                      — (создаётся) README деплоя: Pages, wrangler, cloudflared, PINNED_COMMIT
```

Сборка: `npm run prepare-preview`; единственный деплой-артефакт — `list-manager.html`. Sync-модуль живёт в исходниках и собирается в этот же файл.

## 3. Используемое подмножество API (все пути через /api/*)

v1 (sync-модуль):
- `GET /health` — индикатор соединения
- `POST /a2a/send` — пуш записей журнала (thread `voicelist-log`, from `user`)
- `GET /a2a/messages` — bootstrap-чтение канала (пагинация по `since`)

v2 (зарезервировано, не реализуется сейчас): `GET/POST /tasks*`, `GET /tasks/edges` — двусторонний синк с графом задач и статусами focus/pause бэкенда.

Worker-proxy держит allowlist ровно актуального подмножества (v1: три пути выше; расширяется вместе с v2).

## 4. Обязательные правила интеграции

1. **Модель данных v1**: Приложение → localStorage → sync-модуль → taosmd. localStorage — рабочая реплика; долговременная истина — архив taosmd (каждое A2A-сообщение = событие append-only архива). Синк односторонний + bootstrap на пустом localStorage.
2. **Провенанс**: все отправки фронта — от имени `user` (`from=user`; в v2 для мутаций задач — `created_by=user`). Серверные дефолты не используются.
3. **Статусы**: локальные статусы приложения (`Open/Done/Focus/Archive/Pause`) в v1 не маппятся — едут внутри операций журнала как есть. Бэкенд-enum (с нативными `focus`/`pause`, амендмент rev 2 контракта) задействуется в v2.
4. **Фронтир/резолвер (Θ)**: логика описана в `list-manager-docs.md` и реализована в исходниках `src/`; в v1 не меняется и считается на клиенте.
5. **Секреты**: токен только в Worker-secret; в бандле и network-трафике браузера его нет.

## 5. Задачи

A. **Sync-модуль** — главная задача, полное ТЗ: [CODER_TASK_sync-module.md](./CODER_TASK_sync-module.md).
B. **Worker-proxy** (`/proxy`): маршрутизация `/api/*`, allowlist §3-v1, secret `TAOSMD_TOKEN`, таймауты, честный 502/504 при недоступности tunnel.
C. **Деплой** (`/deploy`): Pages-проект на recorder.smileme.ai, `wrangler deploy` с route, конфиг cloudflared (ingress → 127.0.0.1:7900), порядок включения Cloudflare Access, PINNED_COMMIT бэкенда.

## 6. Приёмка инфраструктуры (помимо приёмки sync-модуля)

1. С телефона вне tailnet: recorder.smileme.ai открывается через Access, приложение работает, синк доходит до живого taosmd.
2. Токен в браузере отсутствует; Authorization добавляет Worker.
3. Прямой запрос к `/api/*` без Access-сессии отклоняется; пути вне allowlist — 404.
