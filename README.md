# VoiceList

`VoiceList` — одностраничный HTML-интерфейс для иерархических списков задач с touch-first UX и CI-проверкой готового превью.

## Что внутри

- вложенный список задач с drag, swipe и fullscreen-редактированием;
- статусы `Open`, `Done`, `Focus`, `Archive`, `Pause`;
- отдельные вкладки списка, фронтира и журнала действий;
- единый контракт ввода через интерпретатор команд;
- self-contained `list-manager.html` для деплоя.

## Бэкенд: taosmd (ветка `taosmd-backend`)

Состояние хранится в [taosmd](https://github.com/021-lab/TaOS/tree/feat/user-statuses) (архив-первая память + граф задач + A2A). В браузере работает storage-модуль: localStorage как рабочая реплика, write-through в `/tasks` на каждую операцию, bootstrap из бэкенда на пустом localStorage, журнал действий — в A2A-канал лога. Seed-файла `list-data.js` больше нет.

См. `docs/CODER_TASK_storage-module.md` и `docs/INTEGRATION_taosmd.md` в ветке `taosmd-backend`.

## Локальная разработка

```bash
npm ci
npm run test:unit
npm run prepare-preview
npm run test:e2e
```

Переопределяйте тексты, цвета и интеграции в `src/` и шаблоне `list-manager.template.html`; deployable HTML собирается через `npm run prepare-preview`.

## Preview

Финальный preview для каждой версии строится от точного `commit SHA`, а не от имени ветки.
