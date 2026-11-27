# Razuma mobile apps (Android + iOS)

A [Capacitor](https://capacitorjs.com) shell that packages the existing React SPA as
native apps. There is **no separate UI** here: the app renders the shared product
screens from `../frontend/src` (Detector, Focus, Privacy) inside a native WebView.

```
mobile/                      Capacitor project (this directory)
  capacitor.config.ts        appId, webDir, scheme
  android/                   generated Android Studio project
  ios/                       generated Xcode project
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

From the repo root:

```bash
make build_mobile   # frontend/dist-mobile
make mobile_sync    # build + copy into both native projects
```

## Prerequisites

| Target | Needs |
|---|---|
| Web bundle only | Node 20+, `npm` |
| Android | Android Studio + **JDK 21** (Gradle 8.11.1 rejects JDK 24+), an emulator or device |
| iOS | macOS + Xcode 15+, CocoaPods (`sudo gem install cocoapods`), an Apple Developer account |

On a machine where `java -version` reports 25 (as on the current dev box), point Gradle at 21:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
# or permanently, in mobile/android/gradle.properties:
# org.gradle.java.home=/usr/lib/jvm/java-21-openjdk-amd64
```

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

## Permissions

Already wired up, because the Focus camera uses `getUserMedia()`:

- **Android** — `CAMERA`, `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS` in
  `android/app/src/main/AndroidManifest.xml`. Capacitor's `BridgeWebChromeClient`
  requests the runtime grant when the page calls `getUserMedia()`.
- **iOS** — `NSCameraUsageDescription`, `NSMicrophoneUsageDescription`,
  `NSPhotoLibrary*UsageDescription` in `ios/App/App/Info.plist`. `getUserMedia` in
  WKWebView requires **iOS 14.3+**; `allowsInlineMediaPlayback` is already enabled by
  Capacitor, so the preview plays inline.

Verify the camera on a **physical device** — simulator/emulator camera support is
unreliable and says nothing about the store build.

## Before publishing — not optional

1. **Authentication.** The backend currently accepts unauthenticated requests and trusts
   any `user_id` in `/api/burnout/baseline`. Shipping a public app without auth exposes
   the GPU and the LLM key to anyone who finds the URL. Add a token/API-key check and
   rate limiting first.
2. **Large uploads.** A single `POST /api/upload` up to 300 MB will fail on mobile data.
   Compress on-device or add resumable/chunked upload.
3. **Store metadata.** Privacy policy URL, data-safety form (Play), camera justification
   and sensitive-data review notes (Apple). Emotion inference is treated as sensitive —
   position it as wellness, never diagnosis.
4. **Signing.** Android: generate a keystore and build a signed `.aab`
   (`Build → Generate Signed Bundle`). iOS: certificates + provisioning profiles via Xcode.

## Not included

- The `compass/frontend` clinical prototype is not part of the app.
- Push notifications, deep links and offline caching are not configured; Capacitor
  plugins (`@capacitor/app`, `@capacitor/push-notifications`, `@capacitor/filesystem`)
  can be added without touching the backend.
