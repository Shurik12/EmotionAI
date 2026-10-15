---
name: EmotionAI Run
description: Быстрая пересборка и запуск EmotionAI на localhost через WSL (бэкенд на :80, фронтенд Vite dev на :3000). Используй, когда нужно пересобрать C++/фронтенд или поднять/остановить сервис для проверки в браузере.
---

# EmotionAI Run

Запуск и пересборка проекта EmotionAI в WSL. Всё, что нужно знать, чтобы не
наступить на грабли этой связки.

## Модель окружения

| Что | Где |
|---|---|
| Исходники (правятся тут) | `D:\projects\EmotionAI` == `/mnt/d/projects/EmotionAI` |
| Сборка и запуск (только Linux) | `/home/rinn/projects/EmotionAI` |
| Бинарник | `<repo>/build/emotionai` |
| Скрипт запуска сервера | `/home/rinn/start_emotionai.sh` |
| Логи запуска/сборки | `~/emotionai-run/` (**не** `/tmp` — его сносит при рестарте WSL) |
| Бэкенд | порт **80** (`http://localhost/api/...`) |
| Фронтенд (dev) | порт **3000** (`http://localhost:3000`) |
| Dragonfly (Redis) | docker-контейнер `dragonfly`, `127.0.0.1:6379` |

Windows-копия и WSL-копия — это **разные деревья**. Правки в `D:\...` не видны
сборке, пока не сделаешь `sync`.

## Быстрый путь

```powershell
# из Windows: синк правок + пересборка + подъём
wsl -d Ubuntu -- bash /mnt/d/projects/EmotionAI/.opencode/skills/emotionai-run/scripts/install.sh
wsl -d Ubuntu -- bash ~/emotionai-run/scripts/sync.sh
wsl -d Ubuntu -- bash ~/emotionai-run/scripts/rebuild.sh
wsl -d Ubuntu -- bash ~/emotionai-run/scripts/up.sh
```

Открыть: **http://localhost:3000/** (SPA) — dev-сервер проксирует `/api` на порт 80.
Прод-вариант через сам бэкенд: **http://localhost/**.

Остановить: `wsl -d Ubuntu -- bash ~/emotionai-run/scripts/down.sh`.

## Правила, без которых ломается

1. **Не удаляй `build/.ninja_log`.** Если ninja пишет
   `premature end of file; recovering` — это повреждённый лог, но удаление
   вызывает полную пересборку всех 58 файлов и она часто не доживает. Лучше
   `touch` нужных `src/**` (см. `rebuild.sh`).
2. **Долгие команды — в tmux.** `wsl.exe` может вернуться/оборваться раньше
   времени; вложенный процесс при этом умирает. Скрипты `up.sh` и запуск
   сборки используют detached tmux, поэтому остаются жить.
3. **Логи — в `~/emotionai-run/`, не в `/tmp`.** `/tmp` в WSL не переживает
   остановку VM (а она случается).
4. **WSL иногда падает** (`Wsl/Service/CreateInstance/0x80072746`,
   `E_ABORT`, «connection forcibly closed»). Тогда команды возвращают пустой
   вывод и exit 1. Лечится перезагрузкой Windows или, от админа,
   `wsl --shutdown` + `net stop WSLService` + `net start WSLService`.
   Проверка живости: `wsl -d Ubuntu -- bash -c "echo ready; uptime"`.
5. **`A localhost proxy configuration was detected…`** — это безвредное
   предупреждение wsl.exe, не ошибка.
6. **Один `wsl -d Ubuntu -- ...` = одна команда.** Не цепляй `&&`/`;` через
   `echo`-разделители: у тулов это ломает разбор и права.

## Скрипты

| Скрипт | Что делает |
|---|---|
| `scripts/install.sh` | Копирует скрипты в `~/emotionai-run/scripts`, чинит CRLF, ставит +x |
| `scripts/sync.sh` | `rsync` правок `src/`, `frontend/src/`, `tests/` из Windows-дерева в WSL |
| `scripts/rebuild.sh` | Только бэкенд (`emotionai`), `-j${JOBS:-4}`, лог в `~/emotionai-run/build.log` |
| `scripts/up.sh` | Dragonfly → бэкенд в tmux → Vite dev в tmux → health-check |
| `scripts/prod.sh` | Раздавать собранный SPA с бэкенда (:80): путь `frontend/dist` + `npm run build` + рестарт бэкенда |
| `scripts/down.sh` | Останавливает tmux-сессии бэкенда и фронта |

## Проверка

```bash
curl -s http://localhost/api/health          # {"status": "healthy"}
curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/   # 200
tmux ls                                       # emotionai, emotionai-front
```

## Фича «ревью» (EMO-26)

- `POST /api/review` (multipart: `file`, `self_score` 1..10) → `{task_id}`;
  прогресс: `GET /api/progress/<id>`; готово, когда `complete: true` **и** есть
  объект `review` `{self_score, model_score, delta, verdict, adjusted_score}`.
- Страница: `http://localhost:3000/review`.
- Работает для **видео и изображений**; аудио не поддерживается (нет
  `average_emotions`, аудио-модель может быть не загружена).
- Оценка по VA-головам модели (`valence`/`arousal`, диапазон ~[-1,1]).

## Частые грабли

- **Аудио-модель не загружена** — в логе `Audio model not found`: audio-пути
  (`/upload_burnout`, `/upload_external_influence`) работать не будут.
- **`npm run build`** запускать только из `<repo>/frontend` (там `package.json`),
  иначе `ENOENT: package.json`.
- **Прод-статика `http://localhost/` и ошибка**
  `Failed to load module script: … responded with a MIME type of "text/html"`:
  бэкенд отдаёт `index.html` вместо JS, потому что `paths.frontend` в
  `config.yaml` указывает не на `frontend/dist`. Лечение (одной командой —
  `scripts/prod.sh`):
  1. `sed -i 's|frontend: "./frontend"|frontend: "./frontend/dist"|' <repo>/config.yaml`
  2. `cd <repo>/frontend && npm run build`
  3. перезапустить бэкенд-сессию `emotionai`
  Проверка: `curl -s -o /dev/null -w '%{content_type}' http://localhost/static/main-*.js`
  → должно быть `application/javascript`.
  Dev-сервер на :3000 этой проблемы не имеет — он раздаёт исходники напрямую.
- **`exit 1` у `explorer.exe`** при открытии папки — это успех.
