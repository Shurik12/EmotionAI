# Compass: требования к экспорту и сборке (зафиксированная конфигурация)

Документ нужен тому, кто **переэкспортирует** Compass из генератора (нейросети/конструктора) в
репозиторий. Его задача — чтобы новый экспорт **встал в текущий стек и пайплайн без переделки**
зависимостей, сборки и деплоя.

Источник истины — файлы `compass/frontend/package.json`, `package-lock.json`, `vite.config.ts`,
`tsconfig.json`, `index.html`. При расхождении с этим документом правы они; после осознанного
изменения конфигурации документ обновляется в том же коммите.

## 1. Стек — не менять

| Что | Требование |
|---|---|
| UI | React 18.3.1 + React DOM 18.3.1, клиентский SPA |
| Сборка | Vite 5.4 + `@vitejs/plugin-react`, без SSR и серверных плагинов |
| Язык | TypeScript 5.9, `strict`, компоненты — `.tsx` |
| Рантайм Node.js | 20 (в CI `actions/setup-node@v7`, `node-version: 20`), npm 10 |
| Пакетный менеджер | `npm ci` по закоммиченному `package-lock.json` (`lockfileVersion: 3`) |
| Роутинг | только hash-роуты, реализованы руками в `src/App.tsx` |
| Иконки | `lucide-react` через явную карту `src/components/icons.ts` |
| Стили | обычный CSS в `src/styles.css` |
| Графики | SVG-компонент `src/components/Chart.tsx` |

Чего в проекте быть **не должно**: Next.js (SSR, RSC, `app/`, `pages/api/`), `react-router`/
`react-router-dom`, Tailwind/PostCSS, CSS-in-JS, UI-киты (MUI/shadcn/antd), библиотеки графиков
(recharts/chart.js), state-менеджеры (redux/zustand), i18n-пакеты, серверный код любого вида.

## 2. Зависимости (точные)

```json
{
  "dependencies": {
    "html-react-parser": "6.1.9",
    "lucide-react": "^0.468.0",
    "react": "18.3.1",
    "react-dom": "18.3.1"
  },
  "devDependencies": {
    "@types/node": "^20.19.0",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.7",
    "@vitejs/plugin-react": "^4.3.4",
    "playwright": "^1.64.1",
    "typescript": "^5.9.3",
    "vite": "^5.4.11"
  }
}
```

Правила:

- `react` и `react-dom` — **точные версии** (`18.3.1`, без `^`): от них зависит совместимость с
  `@types/react` 18 и с плагином Vite.
- Новая зависимость допускается только если её нельзя заменить 20 строками своего кода; после
  добавления обновляется `package-lock.json` и этот документ.
- `playwright` — только для `npm run test:browser`, в прод-сборку не попадает, браузеры в CI не
  скачиваются.
- `html-react-parser` нужен движку (`src/prototype/engine.js` рендерит сохранённую разметку
  экранов); не заменять на `dangerouslySetInnerHTML` без причины.

Сейчас в lock-файле 133 пакета. Это бюджет: экспорт, тянущий сотни пакетов, не принимается.

## 3. Структура каталога (обязательная)

```
compass/frontend/
  index.html                    # единственная точка входа
  src/main.tsx                  # createRoot(...).render(<App/>) + import './styles.css'
  src/App.tsx                   # hash-роутинг: /screens/P03 | /capture | редирект
  src/components/               # компоненты UI; icons.ts — карта иконок
  src/prototype/engine.js       # логика 68 экранов (JS), + engine.d.ts
  src/data/screens.json         # реестр экранов (68 записей)
  src/styles.css                # стили
  src/lib/api.ts                # типизированный адаптер к бэкенду
  tools/check-engine.mjs        # npm test
  tools/browser-smoke.cjs       # npm run test:browser
  vite.config.ts
  tsconfig.json
  package.json
  package-lock.json
  .env.example
```

- Файлы не переименовывать и не разносить по другим каталогам: на эти пути завязаны `Makefile`,
  `tools/check-engine.mjs` и документация.
- `src/prototype/engine.js` остаётся **JavaScript** с декларацией `engine.d.ts` (в `tsconfig`
  `allowJs: true`, `checkJs: false`). Переписывание движка на TSX — отдельная работа, не часть
  экспорта.
- У Compass **свои** `node_modules` и `package-lock.json`; он не делит зависимости с лендингом
  (`frontend/`).

## 4. `vite.config.ts` — требования

```ts
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: './',
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  build: {
    outDir: fileURLToPath(new URL('../../frontend/dist/compass', import.meta.url)),
    emptyOutDir: true,
    sourcemap: false,
    minify: 'esbuild',
    target: 'es2020',
    chunkSizeWarningLimit: 512,
  },
  esbuild: { drop: mode === 'production' ? ['console', 'debugger'] : [] },
}));
```

- **`base: './'` обязателен.** Сборка лежит в подкаталоге `/compass/` и грузится с
  `compass/index.html`; относительные пути — единственное, что работает без правки nginx.
- **`outDir` — `../../frontend/dist/compass`**, а не локальный `dist/`. Так существующий деплой
  (`scp frontend/dist/*` на VM) публикует Compass по адресу `https://razuma.tech/compass/`.
- **`sourcemap: false`.** Лендинг собирается со `sourcemap: true`; для Compass это означало бы
  публикацию исходников.
- `minify: 'esbuild'`, `target: 'es2020'`, `chunkSizeWarningLimit: 512` — зафиксированы.
- `drop: ['console', 'debugger']` в production. Никаких `drop_console`-хаков в коде.
- Кроме `@vitejs/plugin-react` плагинов нет. Запрещены `vite-plugin-ssr`, `vite-plugin-pwa`,
  legacy-плагины, `server.proxy` в прод-конфиге.

## 5. `tsconfig.json` — как есть

```json
{"compilerOptions":{"target":"ES2022","lib":["DOM","DOM.Iterable","ES2022"],"allowJs":true,"checkJs":false,"skipLibCheck":true,"strict":true,"noEmit":true,"esModuleInterop":true,"module":"ESNext","moduleResolution":"Bundler","resolveJsonModule":true,"isolatedModules":true,"jsx":"react-jsx","types":["vite/client"],"paths":{"@/*":["./src/*"]}},"include":["src","vite.config.ts"]}
```

Обязательные свойства: `strict`, `jsx: react-jsx`, `moduleResolution: Bundler`, `noEmit`,
`allowJs` + `checkJs: false` (движок на JS), `resolveJsonModule` (`screens.json`), `types:
["vite/client"]`, алиас `@/*`.

Скрипты `package.json`:

```json
{
  "dev": "vite",
  "build": "tsc --noEmit && vite build",
  "preview": "vite preview",
  "typecheck": "tsc --noEmit",
  "test": "node tools/check-engine.mjs",
  "test:browser": "node tools/browser-smoke.cjs"
}
```

`build` всегда сначала типизирует (`tsc --noEmit`), потом собирает. Экспорт, который не проходит
`tsc` без `@ts-ignore`/`any`, не принимается.

## 6. Роутинг — жёсткое ограничение

- Только hash: `/#/screens/P03`, `/#/capture`. Навигация — `location.hash`, `hashchange`.
- **Никаких pushState/History-роутов** вида `/compass/screens/P03`: при `base: './'` ассеты
  резолвятся относительно документа, и такой роут ищет их в `/compass/screens/`.
- Пустой хеш → редирект на `/screens/P03`; неизвестный id → экран «Экран не найден», не пустая
  страница.
- Список экранов — `src/data/screens.json` (68 записей, id вида `P03`, `D01`, `U01`); новые экраны
  добавляются туда, иначе роут отбросит их как неизвестные.

## 7. Окружение и данные

- Только публичные переменные `VITE_*`. Единственная используемая — `VITE_API_BASE_URL`
  (см. `.env.example`); пустая = «сервер API не подключён».
- `.env.example` коммитится, `.env.local` — в `.gitignore`.
- **Секретов и ключей в `VITE_*` быть не может** — они видны посетителю в бандле.
- Обращения к бэкенду — только через `src/lib/api.ts`.
- Приватность: в сеть уходят медиа и тексты задач, кадры камеры обрабатываются по контракту
  Focus-сессий, ничего не пишется на диск. Экспорт не должен добавлять внешнюю аналитику/CDN.

## 8. Что мы уже изменили относительно исходного экспорта генератора

Исходник генератора был **Next.js**-прототипом. Ниже — приведение к текущей конфигурации; при
переэкспорте это должно быть сделано **сразу в генераторе**, а не повторяться руками.

| Было в экспорте | Стало (требование) |
|---|---|
| Next.js (`next` в зависимостях) | Vite 5 + `@vitejs/plugin-react` (Next удалён) |
| `app/`/`pages/` + серверные API-маршруты | плоский `src/` без серверного кода |
| `next/link`, `next/image`, `next/font`, `use client`/`use server` | обычные `<a>`, `<img>`, `<style>` |
| файловый роутинг Next | hash-роутинг в `src/App.tsx` |
| `process.env.*` | `import.meta.env.VITE_*` |
| `base: '/'`, локальный `dist/` | `base: './'`, `outDir: ../../frontend/dist/compass` |
| `sourcemap: true` | `sourcemap: false` |
| `import * as icons from 'lucide-react'` (бандл 1 002 630 B) | явная карта `icons.ts` (бандл ≈342 KiB), `check-engine.mjs` падает на неизвестной иконке |
| TS «как получится» | `strict`, `tsc --noEmit` в `build` |
| `npm install` | `npm ci` по закоммиченному lock-файлу |
| JS-движок без типов | `engine.js` + `engine.d.ts`, `allowJs`/`checkJs: false` |

## 9. Бюджет сборки

- Один JS-чанк. Сейчас: **342 KiB** (349 614 B), gzip ≈ **103 KiB**; CSS 23 KiB (gzip 5 KiB).
- Ориентир: JS не больше **~400 KiB** raw / 120 KiB gzip. `chunkSizeWarningLimit: 512` — это
  предупреждение, а не цель.
- Новый экран не должен тянуть новую библиотеку. Иконки — только через `src/components/icons.ts`
  (сейчас 21 запись; `npm test` проверяет, что движок не просит иконку вне карты).

## 10. Приёмка экспорта (definition of done)

```sh
cd compass/frontend && npm ci && npm run typecheck && npm test && npm run build
# из корня репозитория:
make build_frontend      # лендинг, затем compass; проверяет наличие frontend/dist/compass/index.html
```

Чек-лист:

- [ ] `npm run typecheck` — без ошибок и без новых `@ts-ignore`;
- [ ] `npm test` — `PASS: 68 screens, roles, demo OTP, escaped text, 19 icons mapped` (число —
      сколько иконок реально рендерит движок, сейчас 19 из 21 записи карты);
- [ ] `npm run build` — кладёт `index.html` + `assets/` в `frontend/dist/compass/`;
- [ ] в `frontend/dist/compass/` нет `*.map`;
- [ ] `grep -r "next/" src` — пусто; `grep -r "process.env" src` — пусто;
- [ ] бандл в бюджете (раздел 9);
- [ ] `make build_frontend` проходит (лендинг собирается первым, иначе его `vite build` очистит
      `frontend/dist` и удалит Compass);
- [ ] проверка в браузере: `cd frontend/dist && python3 -m http.server 8080` →
      `http://localhost:8080/compass/#/screens/P03` (по HTTP, не через `file://`: камера и микрофон
      требуют HTTPS или localhost).

## 11. Ловушки экспорта (типовые)

1. `import * as icons from 'lucide-react'` — бандл раздувается до 1 МБ. Только именованные импорты и
   карта.
2. `base: '/'` — приложение «работает» локально и отдаёт 404 на все ассеты на `razuma.tech/compass/`.
3. `sourcemap: true`, скопированный из конфига лендинга, — публикация исходников.
4. `react-router-dom` «для порядка» — лишние 40 KiB и конфликт с hash-роутингом и относительными
   путями.
5. PushState-роуты — ломают загрузку ассетов в подкаталоге (раздел 6).
6. `process.env` вместо `import.meta.env` — в бандле `undefined`, API-адрес теряется.
7. Шрифты/картинки с внешних CDN — деплой должен оставаться самодостаточным статическим набором.
8. Правка `frontend/dist/compass/*` руками — каталог пересоздаётся при каждой сборке.
9. Переименование `engine.js`/`screens.json` — ломает `check-engine.mjs` и роутинг без ошибки
   сборки.
