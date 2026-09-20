# RAZUMA frontend

React 18 + Vite 5 single-page app for the EmotionAI service. The build output is served by the C++
binary itself: any path that is not `/api/...` returns `dist/index.html`, so all client-side routes
(including deep links such as `/detector`) work without server-side configuration.

## Commands

```bash
npm install
npm run dev        # dev server on :3000, /api proxied to API_URL or http://localhost:5000
npm run build      # vite build -> dist/ (also available as `make build_frontend`)
npm run preview    # serve the built dist/ locally
```

`npm test` (jest) and `npm run lint` (eslint) are declared in `package.json` but not configured:
there is no jest config, no eslint config and no test file in the repo — both commands fail.

## Structure

```
src/
├── index.jsx                 entry point
├── components/
│   ├── App.jsx               providers + page switch + header/footer/cookie consent
│   ├── Home.jsx              landing page (all anchored sections)
│   ├── Header.jsx            brand lockup, section nav, language select, contact CTA
│   ├── Footer.jsx            copyright + section links (the element with id="about")
│   ├── Detector.jsx          demo workspace: upload, progress, results, charts
│   ├── Contact.jsx           legal info + contact details
│   ├── Privacy.jsx, Features.jsx
│   └── ...                   burnout / external-influence results, modal, errors
├── context/
│   ├── NavigationContext.jsx currentPage + pushState routing, navigateTo, navigateToSection
│   └── LanguageContext.jsx   RU/EN state
├── hooks/                    useLanguage, useNavigation, useFileUpload, useProgress
├── api/
│   ├── client.js             fetch wrapper, base URL = VITE_API_URL or /api
│   └── endpoints.js          upload / progress endpoint paths
├── styles/
│   ├── global.css            tokens (--layout-padding-x, --color-*, spacing), .app layout
│   └── components/           one stylesheet per area (landing.css = the Home page)
└── utils/translations.js     all RU and EN strings
```

## Navigation

Two helpers from `useNavigation()`:

- `navigateTo(page)` — switch the top-level page: `home`, `features`, `detector`, `privacy`,
  `contact` (anything else renders `home`).
- `navigateToSection(id)` — go to the home page if needed, then smooth-scroll to the element with
  that `id`. Sections must keep `scroll-margin-top: 92px` so the sticky header doesn't cover them.

Header and footer menus (`solutions`, `industries`, `technology`, `cases`, `about`) and the hero
links point at landing-page ids; a renamed id breaks the menu silently (the scroll target no longer
exists). The mobile menu (below 1100px) also closes on every section click.

## Landing page sections

| Anchor | Section | Translation prefix |
|---|---|---|
| `technology` | hero: headline, lead, "Try the demo" + "Discuss a pilot" buttons | `landing.hero*` |
| `solutions` | three analysis cards | `landing.analysis.*` |
| `industries` | four industry cards | `landing.industries.*` |
| `cases` | four benefit cards + decision banner | `landing.benefits.*`, `landing.decision` |
| `demo` | CTA band: RAZUMA / Skolkovo lockup, demo and contact blocks | `landing.cta.*` |
| `contact` | contact column of the CTA band | `landing.cta.contact*` |

Both "Обсудить пилот" buttons and the header CTA call `navigateTo('contact')` — the app never sends
email; the contact page only shows `mailto:` / `tel:` links.

## Conventions

- One named-export component per file (`export const Home = () => ...`); one stylesheet per area in
  `src/styles/components/`, keeping the component's class prefix (`landing-*`, `header-*`, …).
- UI strings go through `t('key')` and live in both the RU and EN halves of
  `src/utils/translations.js`. A missing key renders as the raw key.
- Card lists are translated as arrays of objects and paired with the icon arrays in `Home.jsx`
  (`ANALYSIS_ICONS`, `BENEFIT_ICONS`) by index — reordering cards means reordering icons.
- Images go to `public/static/` and are referenced as `/static/<file>`; photos are `.webp`.
  The hero background is set in `landing.css` via `url('/static/hero.webp')`, not in JSX.
- `react-router-dom` is in `package.json` (and in the `react-vendor` chunk) but is not imported —
  routing is `NavigationContext` + `history.pushState`.

## Build & deploy notes

- `vite.config.js`: output `dist/`, assets under `static/`, sourcemaps on, `console`/`debugger`
  dropped in production, manual chunks for react and axios.
- `dist/` is gitignored and served from the backend's `paths.frontend` — rebuild after every source
  change; the running site does not hot-reload.
- The dev server proxies `/api` to `API_URL` (default `http://localhost:5000`); set `API_URL` to
  your backend origin (the backend default is port 80).
