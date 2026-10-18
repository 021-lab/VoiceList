When the user says “в журнале баг” or asks to fix a bug in the journal, inspect comments attached to action-log entries, not the action labels, transcripts, or generic "Нераспознано" rows. A comment on an entry is the authoritative bug report. Read its exact text, identify the entry it annotates, and derive expected versus actual behavior from that comment before changing code. Do not infer a bug from an unannotated log row. Then create a GitHub issue for the bug and process it with the `fix github issue VoiceList` skill (`fix-github-issue-voicelist`).

## Формат отчёта

Отвечай по работе только этими четырьмя пунктами, кратко: архитектурные решения с обоснованием;
что протестировано самим (непроверенное называть непроверенным); что готово к тестированию и что
исправлено; ссылка на актуальный адрес для тестирования. Без пересказа процесса и извинений.

## Подпись коммита

Тема коммита — понятное пользователю описание изменения: что стало иначе в приложении, а не что
поменялось в коде. «Новая задача появляется вверху списка», а не «fix order in addItem».

Одна правка — один коммит. Тема становится подписью версии на проде (`npm run rollback:prod -- --list`),
и по ней пользователь выбирает, куда откатиться; коммит из трёх правок под одной подписью откатить
по частям нельзя.
