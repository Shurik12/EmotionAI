# Razuma mobile apps (Android + iOS)

A [Capacitor](https://capacitorjs.com) shell that packages the existing React SPA as
native apps. There is **no separate UI** here: the app renders the shared product
screens from `../frontend/src` (Detector, Focus, Privacy) inside a native WebView.

```
mobile/                      Capacitor project (this directory)
  capacitor.config.ts        appId, webDir, scheme
  assets/                    icon/splash sources for @capacitor/assets
  android/                   generated Android Studio project
  ios/                       generated Xcode project
  STORE_LISTING.md           store copy + review notes
../frontend/mobile-app/      web entry point (index.html) for the app bundle
../frontend/src/mobile/      mobile shell: app bar + bottom tab bar
../frontend/vite.mobile.config.js
../frontend/dist-mobile/     build output that Capacitor copies into the apps
```

## How it fits together

| Piece | What it does |
|---|---|
| `npm run build:mobile` (in `frontend/`) | Vite build of `src/mobile/` into `dist-mobile/` |
| `npx cap sync` (in `mobile/`) | copies `dist-mobile/` into `android/` and `ios/` |
| `npx cap open android` | opens Android Studio |
| `npx cap open ios` | opens Xcode (macOS only) |
| `npm run assets` | regenerates icons/splash from `assets/` |

From the repo root:

```bash
make build_mobile   # frontend/dist-mobile
make mobile_sync    # build + copy into both native projects
make mobile_android # also build a debug APK (Android SDK + JDK 21)
```

## Prerequisites

| Target | Needs |
|---|---|
| Web bundle only | Node 20+, `npm` |
| Android | Android Studio + **JDK 21** (Gradle 8.11.1 rejects JDK 24+), an emulator or device |
| iOS | macOS + Xcode 15+, CocoaPods (`sudo gem install cocoapods`), an Apple Developer account |

On a machine where `java -version` reports 25 (as on the current dev box), point Gradle
at 21 in the gitignored `mobile/android/local.properties`:

```properties
sdk.dir=/usr/lib/android-sdk
org.gradle.java.home=/usr/lib/jvm/java-21-openjdk-amd64
```

`make mobile_android` uses `JAVA_HOME_ANDROID` (default
`/usr/lib/jvm/java-21-openjdk-amd64`) and builds a debug APK. The full Android SDK
(`platforms;android-35`, `build-tools;35.0.0`) is required; `platform-tools` alone is
not enough.

## First run

```bash
cd frontend && npm install
cd ../mobile && npm install
npx cap sync
npx cap open android      # then Run ▶ in Android Studio
npx cap open ios          # on macOS: set a signing team, then Run ▶
```

The native projects are already committed, so `cap add android|ios` is only needed if
you delete them.

## API endpoint

The app talks to `VITE_API_URL` from `frontend/.env.mobile`, which must be **absolute and
HTTPS** (`https://alpha.razuma.pro/api` by default). A relative `/api` cannot work: inside
the WebView the origin is `https://localhost` (Android) or `capacitor://localhost` (iOS),
not your server. Change that one line to retarget a different deployment.

Android blocks cleartext HTTP, so a local backend needs HTTPS (e.g. a reverse proxy) or a
debug-only `network_security_config.xml`.

Because the WebView origin differs from the API host, every request is cross-origin and
the backend must allow the SPA's analytics headers (`X-Client-Id`, `X-Traffic-Source`,
`X-Referrer`) in `Access-Control-Allow-Headers` — `Server::sendHttpResponse` lists them.

## Security (auth + rate limiting)

The backend has a `security:` config section (`config_template.yaml`):

- **`api_token`** — when non-empty, every `/api/` request must send
  `Authorization: Bearer <token>` (or `X-API-Key`). `/api/health` is exempt.
  Set `VITE_API_TOKEN` in `frontend/.env.mobile` to match. The token is embedded in
  the app bundle, so it is extractable — it is a revocable gate, not per-user auth.
- **Rate limiting** — a per-client token bucket over all requests
  (`rate_limit_per_minute`, `rate_limit_burst`). Rejections are `429` with
  `Retry-After`.
- **`trust_proxy_headers`** — set `true` only behind `config/nginx`, which sets
  `X-Real-IP`; otherwise every visitor shares the proxy's bucket.

## Resumable uploads

Files over 6 MB go through `POST /api/upload/init → chunk* → complete` instead of a
single POST (`frontend/src/api/client.js`). Chunks are retried with exponential
backoff and resume from the server's reported offset, so a dropped cellular
connection no longer loses the whole file. The one-shot endpoints remain for small
files.

## Permissions

Wired up because the Focus camera uses `getUserMedia()`:

- **Android** — `CAMERA`, `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS` in
  `android/app/src/main/AndroidManifest.xml`. Capacitor's `BridgeWebChromeClient`
  requests the runtime grant when the page calls `getUserMedia()`.
- **iOS** — `NSCameraUsageDescription`, `NSMicrophoneUsageDescription`,
  `NSPhotoLibrary*UsageDescription` in `ios/App/App/Info.plist`. `getUserMedia` in
  WKWebView requires **iOS 14.3+**; `allowsInlineMediaPlayback` is already enabled by
  Capacitor, so the preview plays inline.

Verify the camera on a **physical device** — simulator/emulator camera support is
unreliable and says nothing about the store build.

## Branding

Icons and launch screens live in `mobile/assets/` (rendered from
`frontend/public/static/razuma.svg`) and are regenerated with:

```bash
cd mobile && npm run assets && npm run sync
```

See `mobile/assets/README.md`. Do not hand-edit the generated native images.

## Before publishing

See [`STORE_LISTING.md`](STORE_LISTING.md) for store copy, the wellness
positioning, Apple review notes and the Play data-safety answers. Code-side
prerequisites are done:

1. **Authentication + rate limiting** — implemented (`security:` section).
2. **Large uploads** — resumable chunked upload implemented.
3. **JDK 21** — documented and wired into `make mobile_android`.
4. **Icons/splash + in-app wellness disclaimer** — done.

Still required, and store-account specific:

- Confirm the production API host and the public privacy-policy URL.
- Signing: Android keystore, iOS certificates + provisioning profiles
  (`Build → Generate Signed Bundle` for a `.aab`).
- Decide whether the backend runs with `security.api_token` set.

## Not included

- The `compass/frontend` clinical prototype is not part of the app.
- Push notifications, deep links and offline caching are not configured; Capacitor
  plugins (`@capacitor/app`, `@capacitor/push-notifications`, `@capacitor/filesystem`)
  can be added without touching the backend.
