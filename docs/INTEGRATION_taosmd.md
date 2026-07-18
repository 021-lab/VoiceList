# INTEGRATION_taosmd — интеграция VoiceList с бэкендом taosmd

Статус: rev 2. Скоуп v1: отображение графа + фронтир; focus/pause и смена статусов.

## 0. Координаты разработки

| Роль | Репозиторий | Ветка |
|---|---|---|
| Фронт (этот репо) | `021-lab/VoiceList` | `taosmd-backend` |
| Бэкенд taosmd (форк) | **`021-lab/TaOS`** — https://github.com/021-lab/TaOS/tree/feat/user-statuses | `feat/user-statuses` |

Канон спецификаций бэкенда — `TAOSMD_API_CONTRACT.md` и `TASK_LIFECYCLE.md` в `021-lab/TaOS` (docs/), ссылки по PINNED_COMMIT SHA. Этот документ описывает только интеграцию фронта.

**Правила координации:** любое изменение поведения API (новые статусы, поля, endpoints) сначала попадает амендментом в контракт в `021-lab/TaOS`, затем — в этот документ ссылкой на SHA; фронт не подстраивается под незадокументированное поведение. Обратно: потребность фронта в новом endpoint оформляется issue в `021-lab/TaOS`, не форком логики на клиенте. Пин бэкенда, против которого собран фронт, фиксируется в `/deploy/README.md`.

## 1. Архитектура доступа (recorder.smileme.ai)

- **Cloudflare Pages** — статика фронта (pure HTML+JS, как есть).
- **Worker-proxy**: route `recorder.smileme.ai/api/*` → strip `/api` → Cloudflare Tunnel hostname → taosmd на Mac Mini (`127.0.0.1:7900`, наружу только через cloudflared). Bearer-токен taosmd хранится secret'ом Worker'а и добавляется в Authorization на проксировании — в браузер токен не попадает никогда.
- **Cloudflare Access** на весь хост: allowlist владельца (email/passkey).
- Same-origin по построению: CORS в проекте отсутствует.

## 2. Структура репозитория

```
/                — фронт (index.html, js/, css/) — как есть
/docs/           — этот документ
/proxy/          — Worker: wrangler.toml, src/index.js
/deploy/         — README деплоя: Pages, wrangler, cloudflared; PINNED_COMMIT бэкенда
```

Работа в ветке `taosmd-backend`, PR в main после приёмки.

## 3. Используемое подмножество API (все пути через /api/*)

- `GET /health` — индикатор соединения
- `GET /tasks?limit=&status=&project=` — полный список для построения дерева
- `GET /tasks/edges` — граф рёбер (fork-endpoint)
- `GET /tasks/{id}` — задача с embedded edges (fork-endpoint)
- `POST /tasks` — создание `{title, body?, project?, priority?}`
- `POST /tasks/{id}` — мутация `{status?, body?, priority?}`
- `POST /tasks/{id}/edges`, `POST /tasks/{id}/edges/remove`

Не используются в v1: `/tasks/ready`, `/tasks/prime` (агентские), память (`/ingest`, `/search`), A2A. Worker-proxy держит allowlist ровно этого подмножества.

## 4. Обязательные правила интеграции (из TASK_LIFECYCLE)

1. **Провенанс**: каждая мутация фронта — `created_by=user`. Как HTTP-слой принимает `created_by` (параметр тела или заголовок) — берётся из результатов шага 0 контракт-задания в `021-lab/TaOS`; если сервер значение не принимает, его добавляет Worker-proxy. Оставлять серверный дефолт запрещено.
2. **Статусы**: enum `{open, in_progress, blocked, closed, superseded, focus, pause}` (амендмент rev 2, ветка `feat/user-statuses` в `021-lab/TaOS`). `focus`/`pause` ставит только пользователь; фронт не ставит их на задачи в `in_progress`.
3. **Создание ребёнка** — единый хелпер: `POST /tasks` + ребро `parent` + ребро `blocks` ребёнок→родитель. Пара обязательна и неделима (lifecycle §1); никакой код не создаёт ребёнка одним ребром.
4. **Фронтир** вычисляется на клиенте из полного графа (tasks + edges): видимые открытые листья parent-дерева с применением focus/pause-правил существующего Θ-резолвера (focus/pause теперь читаются из поля `status`, не из локального состояния). НЕ через `/tasks/ready` — это агентский срез.
5. **Инвариант stateless-фронта**: слой приложения не хранит данных — никакого localStorage/IndexedDB/кэшей состояния между сессиями; источник истины всегда taosmd, reload = полная загрузка с сервера.

## 5. Задачи

A. **Worker-proxy** (`/proxy`): маршрутизация `/api/*`, allowlist §3, secret `TAOSMD_TOKEN`, таймауты, честный 502/504 при недоступности tunnel.
B. **Адаптер данных фронта**: заменить текущий бэкенд-слой на подмножество §3; статусы focus/pause маппятся в существующий резолвер из `status`.
C. **Хелпер create-child** по правилу §4.3 — единственная точка создания детей.
D. **Деплой** (`/deploy`): Pages-проект на recorder.smileme.ai, `wrangler deploy` Worker'а с route, конфиг cloudflared (ingress → 127.0.0.1:7900), порядок включения Cloudflare Access, зафиксированный PINNED_COMMIT бэкенда из `021-lab/TaOS`.
E. **Smoke-чеклист** по §6.

## 6. Приёмка

1. С телефона вне tailnet: recorder.smileme.ai открывается через Access, дерево грузится из живого taosmd.
2. Создание задачи и ребёнка: пара рёбер видна в `GET /api/tasks/edges`.
3. focus/pause/закрытие меняют статус в taosmd; после reload состояние идентично серверному (клиентского хранения нет).
4. Токен в браузере отсутствует (network-инспекция + грep бандла); Authorization добавляет Worker.
5. Фронтир после focus/pause соответствует правилам Θ-резолвера.
6. В архиве taosmd все мутации фронта видны с `created_by=user`.
7. Прямой запрос к `/api/*` без Access-сессии отклоняется.

## 7. Зависимости от бэкенда (`021-lab/TaOS`)

- Ветка `feat/user-statuses` (статусы focus/pause + `tasks reset`) собрана и задеплоена на Mac Mini за Tunnel.
- Результат шага 0 контракт-задания: механизм передачи `created_by` в HTTP-слое.
- Fork-endpoints `GET /tasks/edges` и `GET /tasks/{id}` доступны в задеплоенной сборке.
