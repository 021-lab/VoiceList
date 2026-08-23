# Backend: единый Durable Object в Cloudflare

Документ описывает backend в ветке fix/issue-3-voice-button. Он является
источником истины для дерева задач и обслуживает обычный интерфейс, а также
создание голосовой Realtime-сессии.

## Архитектура

Все запросы к документу используют один экземпляр ListDocumentDO с постоянным
именем main:

    Browser UI
      -> WebSocket /ws
      -> Cloudflare Worker
      -> LIST_DOCUMENT.getByName(main)
      -> ListDocumentDO
      -> Durable Object Storage

В Durable Object хранится снимок списка, журнал действий и информация,
необходимая для повторной доставки команд. Все подключённые вкладки получают
новый полный снимок после каждого применённого изменения.

Realtime-сессия идёт отдельным путём:

    Browser
      -> POST /api/realtime/session (SDP + дерево задач)
      -> Worker
      -> OpenAI Realtime Calls API
      -> SDP answer
      -> Browser establishes WebRTC connection with OpenAI

Worker не проксирует аудиопоток. Он создаёт сессию от имени приложения, передаёт
скрытый контекст задач и хранит серверные настройки.

## Дерево и WebSocket-команды

Элемент списка содержит стабильный id, parentId, порядок, статус, заголовок,
дополнительный текст, tags и состояние сворачивания. Статусы: Open, Focus,
Pause, Done, Archive и Info.

Браузер открывает WebSocket на /ws. После соединения Worker отправляет текущее
сообщение state. Клиент затем отправляет:

    {
      "type": "command",
      "clientKey": "stable-id-of-browser-tab",
      "seq": 17,
      "input": {
        "actId": "task-id-or-list",
        "actType": "task-or-list",
        "command": "addItem | addChild | editItem | setStatus | setParent | ...",
        "payload": {},
        "source": "ui-or-openai-realtime"
      }
    }

Ответ содержит ack со статусом applied или rejected, после чего всем клиентам
рассылается новый state. Пара clientKey + seq делает повторную отправку
идемпотентной. CloudflareDocumentClient хранит не подтверждённые команды в
localStorage и повторяет их после переподключения.

Команды, созданные голосовым агентом, идут тем же WebSocket-путём с
source: openai-realtime. Это важно: голосовой агент не изменяет снимок в
браузере напрямую и не использует отдельный HTTP write API.

## HTTP-маршруты

| Маршрут | Назначение |
| --- | --- |
| GET / и GET /index.html | HTML-приложение. |
| GET /health | Проверка Worker; возвращает ok. |
| GET /api/tasks/tree.json | Полное вложенное дерево из id, title, status и children. Это read-only API; голосовой UI не загружает через него дерево при старте. |
| GET /api/realtime/key/status | Состояние конфигурации OpenAI-ключа. |
| POST /api/realtime/key | Одноразовое сохранение OpenAI API key по setup-токену. |
| GET /api/realtime/prompt | Текущий системный промпт; если сохранённого нет, возвращает встроенный промпт. |
| POST /api/realtime/prompt | Сохраняет пользовательский системный промпт для следующих голосовых сессий. |
| POST /api/realtime/session | Создаёт SDP-ответ для OpenAI Realtime-сессии. |
| GET /ws с Upgrade: websocket | Канал чтения и изменения документа. |
| POST /reset | Только тестовый сброс; требует TEST_RESET_TOKEN. |

Маршруты ключа, промпта и сессии отвечают с Cache-Control: no-store. Для
создания сессии и настройки ключа Worker проверяет Origin, если он передан.

## Realtime-конфигурация

Браузер отправляет на /api/realtime/session SDP и компактное дерево:
id, title, status, children. Worker передаёт его модели внутри системных
инструкций как скрытый контекст. Модель не должна автоматически озвучивать
дерево.

Целевая конфигурация этой ветки использует gpt-realtime-2.1, аудиовыход marin,
русскую транскрибацию gpt-live-transcribe, near-field noise reduction и
function tools для операций с задачами. Конкретный набор tools определён в
worker/openai-realtime.js.

Сохранённый в настройках пользовательский промпт не заменяет обязательные
инструкции VoiceList: он добавляется перед ними. Поэтому правила о точных id,
неоднозначности и подтверждении результата инструмента сохраняются.

## Ключ и промпт

OPENAI_API_KEY из окружения Worker имеет приоритет. Если он не задан, ключ
можно единожды сохранить в Durable Object через ссылку с одноразовым
setup-токеном. Браузер не сохраняет ключ в localStorage и после успешного
сохранения удаляет setup-токен из URL.

Пользовательский системный промпт хранится в том же Durable Object. Пустое
значение означает встроенный системный промпт. Обе настройки общие для
документа main, а не персональные для отдельного браузера.

## Эксплуатационные ограничения

- Контекст задач создаётся в момент старта голосовой сессии. Во время уже
  открытой сессии он не обновляется автоматически при изменении дерева другой
  вкладкой.
- Дерево в Realtime-контексте ограничено 2 000 задачами.
- Один Durable Object main означает единый общий документ. Разделение задач по
  пользователям или рабочим пространствам в этой ветке не реализовано.
- Диалоги голосового агента не хранятся в Durable Object: они локальны для
  браузера и описаны в docs/Voice-Realtime-Button.md.

## Сборка

Worker отдаёт worker/generated-html.js, а не исходный list-manager.html.
После изменения фронтенда нужно обновлять generated asset штатной командой
build:cloudflare. Конфигурация deploy для issue-ветки задаётся локально и не
должна добавляться в Git.
