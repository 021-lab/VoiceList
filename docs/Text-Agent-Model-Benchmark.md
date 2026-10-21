# Модели текстового агента: точность, повторяемость, скорость

Дата: 2026-10-21. Инструкция — `AGENT_RULES` из `worker/v2/model.js` (версия #31), корпус —
`docs/text-agent-corpus.json` (45 фраз). Прогон — весь корпус целиком, затем снова весь
корпус: 3 прогона у всех моделей, у лидеров ещё 5. Запросы те же, что на проде
(`resolveOpenAI`); у других провайдеров меняются только адрес и параметры.

    node scripts/eval-text-agent.mjs --provider <openai|gemini|openrouter|ollama> --model <модель> \
      --params '{"temperature":0,"seed":1}' --runs 3 [--concurrency 1] [--json out.json]

- **Верно по прогонам** — сколько фраз из 45 дали ожидаемые вызовы в каждом прогоне.
- **Прогоны совпали** — у каждой фразы в каждом прогоне те же вызовы, а также наличие или
  отсутствие «Не выполнено» и вопроса; формулировка текста не сравнивается. Если нет —
  перечислены фразы, давшие разный ответ.
- **Медиана / p90, мс** — время ответа от отправки до разбора, из облачного контейнера,
  4 запроса параллельно.
- **Цена** — $ за 1M токенов вход / выход, по OpenRouter. Один запрос — около 3 тыс.
  токенов входа: 1000 команд стоят 0,3 $ на входе при цене входа 0,10 $.

| Модель | Провайдер | Параметры | Прогонов | Верно по прогонам | Прогоны совпали | Медиана, мс | p90, мс | Цена $/1M | Ошибки |
|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | openrouter | temperature=0, seed=1 | 5 | 45 · 45 · 45 · 45 · 45 | да | 1765 | 2558 | 0.10 / 0.50 | — |
| gemini-3.5-flash-lite | gemini | reasoning_effort=low, temperature=0 | 5 | 45 · 44 · 45 · 44 · 44 | нет: mix-1 | 735 | 920 | 0.30 / 2.50 | mix-1 |
| gemini-3.8-flash | gemini | reasoning_effort=none, temperature=0 | 5 | 45 · 44 · 45 · 45 · 45 | нет: add-3 | 1495 | 2205 | 0.75 / 3.75 | add-3 |
| gpt-6-luna | openai | temperature=0, seed=1 | 5 | 43 · 43 · 43 · 44 · 42 | нет: ask-1, rest-1, mix-1 | 1209 | 1771 | 0.10 / 0.50 | ask-1, rest-1, mix-1 |
| gpt-5.6-luna | openai | temperature=0, seed=1 | 5 | 43 · 43 · 43 · 42 · 43 | нет: add-10 | 1649 | 2818 | 0.20 / 1.20 | rest-1, rest-2, add-10 |
| gemini-3.8-flash | gemini | reasoning_effort=none, temperature=0 | 3 | 45 · 45 · 45 | да | 1547 | 2631 | 0.75 / 3.75 | — |
| anthropic/claude-haiku-5.5 | openrouter | temperature=0, seed=1 | 3 | 45 · 45 · 45 | да | 1711 | 2660 | 0.10 / 0.50 | — |
| gemini-3.5-flash-lite | gemini | reasoning_effort=low, temperature=0 | 3 | 44 · 44 · 44 | да | 543 | 643 | 0.30 / 2.50 | mix-1 |
| gpt-6-luna | openai | temperature=0, seed=1 | 3 | 44 · 44 · 44 | да | 890 | 1142 | 0.10 / 0.50 | ask-1 |
| gpt-5.6-luna | openai | temperature=0, seed=1 | 3 | 43 · 43 · 43 | да | 860 | 1137 | 0.20 / 1.20 | rest-1, rest-2 |
| gpt-4.1-mini | openai | seed=1 | 3 | 44 · 43 · 43 | нет: add-3 | 602 | 797 | 0.40 / 1.60 | ask-1, add-3 |
| gemini-3.1-flash-lite | gemini | reasoning_effort=none, temperature=0 | 3 | 44 · 44 · 43 | нет: add-1 | 636 | 776 | 0.25 / 1.50 | add-3, add-1 |
| google/gemma-4-26b-a4b-it | openrouter | temperature=0, seed=1 | 3 | 43 · 42 · 43 | нет: add-10, rest-1 | 703 | 1297 | 0.08 / 0.26 | add-3, ask-1, rest-1 |
| gpt-6-luna | openai | по умолчанию | 3 | 43 · 42 · 44 | нет: ren-7, ask-1, rest-1 | 916 | 1423 | 0.10 / 0.50 | ask-1, rest-1, ren-7 |
| qwen/qwen3-30b-a3b-instruct-2507 | openrouter | temperature=0, seed=1 | 3 | 38 · 38 · 38 | нет: add-4, add-9, add-10, ask-2 | 2098 | 3342 | 0.09 / 0.30 | add-2, add-3, add-10, ren-2, ren-7, ask-1, ask-4, add-4 |
| gpt-5.4-nano | openai | temperature=0, seed=1 | 3 | 38 · 37 · 38 | нет: ren-7, ask-4 | 590 | 806 | 0.20 / 1.25 | add-3, add-10, ren-7, st-8, ask-1, ask-2, rest-1, ask-4 |
| gpt-5.4-mini | openai | temperature=0, seed=1 | 3 | 36 · 37 · 39 | нет: add-2, ren-6, rest-2 | 598 | 793 | 0.75 / 4.50 | add-2, add-3, add-8, ren-4, ren-6, ask-1, rest-1, rest-2, mix-1 |
| mistralai/ministral-8b-2512 | openrouter | temperature=0, seed=1 | 3 | 36 · 35 · 36 | нет: add-10, ren-7, st-3, dl-2, dl-3, mv-3, ask-2 | 677 | 1119 | 0.15 / 0.15 | add-3, add-6, add-9, ren-2, st-8, dl-2, mv-2, ask-4, mix-1, add-10 |
| mistralai/mistral-small-3.2-24b-instruct | openrouter | temperature=0, seed=1 | 3 | 35 · 36 · 36 | нет: ren-7, mv-3 | 1206 | 2069 | 0.09 / 0.25 | add-2, add-3, add-4, add-6, ren-2, ren-7, st-3, dl-2, ask-2, rest-2 |
| openai/gpt-oss-20b | openrouter | reasoning_effort=low, temperature=0, seed=1 | 3 | 35 · 35 · 32 | нет: add-2, add-3, add-5, st-7, mv-2, ask-1 | 6049 | 10280 | 0.02 / 0.09 | add-3, add-5, add-10, ren-7, st-7, st-8, dl-2, rest-1, rest-2, mix-1, mv-2, ask-1, add-2 |
| gpt-4.1-nano | openai | seed=1 | 3 | 27 · 28 · 28 | нет: add-2, add-4, add-8, st-7, st-9, ask-3, mix-1 | 666 | 907 | 0.10 / 0.40 | add-1, add-3, add-4, add-6, add-8, add-10, ren-7, st-8, st-9, dl-2, dl-4, mv-2, mv-3, ask-1, ask-4, rest-1, rest-2, mix-1, st-7, add-2, ask-3 |

## Выводы

- Все прогоны совпали и дали 45/45 только у **claude-haiku-5.5** при temperature 0: 8 полных
  прогонов. Медиана 1,8 с через OpenRouter, цена как у gpt-6-luna.
- Остальные облачные модели совпадали три прогона подряд, но расходились на пятом-восьмом.
  Облачный API не гарантирует повторяемость даже с temperature 0 и seed: сервер собирает
  запросы в пакеты, и от пакета зависит арифметика. «Всегда точно» облако может показать
  только на опыте, но не обещать.
- Гарантирует повторяемость только локальная модель: Ollama, temperature 0, фиксированный
  seed, один запрос за раз, та же машина.
- gpt-6-luna, которая сейчас на проде: с параметрами по умолчанию прогоны расходятся;
  temperature 0 и seed почти выравнивают их, но не до конца. Стабильно ошибается в ask-1
  (две «Клубники» — выбирает одну молча).
- Открытые модели, которые можно запустить в Ollama (qwen3-30b-a3b, gemma-4-26b,
  mistral-small-3.2, ministral-8b, gpt-oss-20b), в облаке дали 32–43 из 45. Локально
  ожидается та же точность при полной повторяемости. Скорость на Mac mini не замерена.

## Локальные модели (Ollama на Mac mini)

Не прогнаны: эта сессия работает в облачном контейнере и до Mac mini не достаёт. Команда
для Mac mini (64 ГБ):

    for m in qwen3:30b-a3b-instruct-2507-q4_K_M gemma4:26b mistral-small3.2:24b gpt-oss:20b; do
      ollama pull $m
      node scripts/eval-text-agent.mjs --provider ollama --model $m --concurrency 1 \
        --params '{"store":null,"reasoning_effort":null,"temperature":0,"seed":1}' --quiet
    done
