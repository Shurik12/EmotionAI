# App assets (source of truth)

These are the **only** files to edit when branding changes. `npm run assets`
(`@capacitor/assets`) regenerates every Android mipmap/splash and the iOS
`AppIcon`/`Splash` images from them.

| File | Size | Purpose |
|---|---|---|
| `icon-only.png` | 1024×1024 | iOS app icon (full bleed; iOS applies its own mask) |
| `icon-foreground.png` | 1024×1024 | Android adaptive-icon foreground (transparent, logo inside the safe zone) |
| `icon-background.png` | 1024×1024 | Android adaptive-icon background (solid white) |
| `splash.png` | 2732×2732 | light-mode launch screen |
| `splash-dark.png` | 2732×2732 | dark-mode launch screen (navy `#062E60`) |

All five were rendered from `frontend/public/static/razuma.svg` on a white or
navy background. To refresh them, re-render that SVG at the sizes above and run:

```bash
cd mobile && npm run assets
```

Then `npm run sync` and commit the regenerated native files. Do not hand-edit
files under `android/app/src/main/res/` or `ios/App/App/Assets.xcassets/`.
