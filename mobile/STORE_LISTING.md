# Store submission pack (Android + iOS)

Copy for the App Store / Google Play listings, plus the review notes that keep
the app in the **wellness** category. Everything here is metadata — no code
changes are needed to use it.

> Confirm the production domain before submitting. The links below assume
> `https://razuma.tech`; the API host in `frontend/.env.mobile` may differ.

## Positioning (read this first)

Razuma analyses facial and voice signals to give the user **self-reflection
insights** and a focus/wellness nudge. It is **not** a medical device, does not
diagnose, treat or predict any condition, and its burnout score is a heuristic,
not a clinical assessment.

Always use: *wellness*, *self-reflection*, *focus*, *productivity*, *insight*.
Never use: *diagnosis*, *detection of depression*, *treatment*, *clinical*,
*medical*, *screening*, *disorder*, *patient*.

This wording is also required inside the app: the Privacy screen shows the
`mobile.wellnessNote` disclaimer (`frontend/src/utils/translations.js`).

## Shared metadata

| Field | Value |
|---|---|
| App name | Razuma |
| Bundle / application ID | `tech.razuma.emotionai` |
| Category | Health & Fitness (primary), Productivity (secondary) |
| Privacy policy URL | `https://razuma.tech/privacy` |
| Support URL | `https://razuma.tech/contact` |
| Marketing URL | `https://razuma.tech` |
| Contact email | support@razuma.tech |
| Age rating | 12+ (emotion/wellness content, no user-generated content) |
| Price | Free |

## App Store (RU)

**Название:** Razuma

**Подзаголовок:** Эмоции, фокус и ресурс — в одном приложении

**Описание:**
Razuma помогает замечать, как задачи и нагрузка влияют на ваше состояние.
Загрузите фото, видео или аудио — приложение покажет эмоциональный профиль,
а анализ речевых признаков подскажет, как меняется ваш ресурс в течение дня.

Razuma Focus помогает работать в своём темпе: камера оценивает динамику
эмоций во время задачи и мягко предлагает паузу, если становится тяжело.
Обработка происходит только после вашего явного согласия, а кадры камеры не
покидают устройство без вашего действия.

• Анализ эмоций по лицу и голосу
• Оценка риска выгорания относительно вашей собственной нормы
• Помощь с концентрацией и перерывами
• Русский и английский языки

Razuma — инструмент для благополучия и продуктивности. Он не является
медицинским изделием и не ставит диагноз.

**Ключевые слова:** эмоции, выгорание, фокус, благополучие, продуктивность,
wellness, саморефлексия

## App Store (EN)

**Name:** Razuma

**Subtitle:** Emotions, focus and energy in one app

**Description:**
Razuma helps you notice how tasks and workload affect how you feel. Upload a
photo, video or audio clip and see your emotional profile; speech-based cues
show how your energy changes through the day.

Razuma Focus helps you work at your own pace: the camera tracks emotional
dynamics while you work on a task and gently suggests a break when things get
heavy. Processing only happens after your explicit consent, and camera frames
stay on your device until you choose to analyse something.

• Emotion insights from face and voice
• Burnout risk relative to your own baseline
• Focus and break support
• Russian and English

Razuma is a wellness and productivity tool. It is not a medical device and does
not diagnose any condition.

**Keywords:** emotions, burnout, focus, wellness, productivity, self-reflection

## Apple review notes (App Review Information)

> Razuma is a wellness and productivity app. It does **not** provide medical
> advice, diagnosis or treatment, and the UI never presents its output as a
> clinical result.
>
> Camera (`NSCameraUsageDescription`): used only by the Focus screen while the
> user has the screen open and has pressed "Enable camera". Frames are sent to
> the user's own Razuma backend for emotion inference (1 frame/s) and are not
> stored by the app. The user can stop the camera at any time.
>
> Photo/video/audio library (`NSPhotoLibraryUsageDescription`): the user picks a
> file to analyse. Uploads go only to the Razuma backend.
>
> The app requires an active network connection; there is no account and no
> third-party advertising SDK.
>
> If a reviewer cannot reach the backend, use the demo build at
> `https://razuma.tech` — the same screens are available in the web app.

If the backend sets `security.api_token`, include a demo token in this field
(marked "demo account") and rotate it after review.

## Google Play — Data safety

| Question | Answer |
|---|---|
| Does the app collect or share user data? | Yes — collect, no sharing with third parties |
| Photos and videos | Collected when the user uploads a file for analysis; optional; not shared |
| Audio files | Same as above |
| Health and fitness (emotion/wellness signals) | Collected as derived analysis results; optional; not shared |
| App activity (in-app actions) | Collected for anonymised product analytics; not shared |
| Device or other IDs | Anonymous random client id (`X-Client-Id`, hashed server-side); not shared |
| Approximate location | **Not** collected |
| Personal info (name, email) | **Not** collected |
| Is data encrypted in transit? | Yes (HTTPS only) |
| Can users request deletion? | Yes — see the privacy policy contact |

Declare the app as **not a medical device**.

## Permissions justification

| Platform | Permission | Justification |
|---|---|---|
| Android | `CAMERA` | Focus screen emotion capture (user-initiated) |
| Android | `RECORD_AUDIO` | Optional audio analysis / native picker |
| Android | `MODIFY_AUDIO_SETTINGS` | Part of the WebView audio-capture path |
| iOS | `NSCameraUsageDescription` | Focus screen emotion capture |
| iOS | `NSMicrophoneUsageDescription` | Optional audio recording |
| iOS | `NSPhotoLibraryUsageDescription` | Choosing an image/video to analyse |
| iOS | `NSPhotoLibraryAddUsageDescription` | Saving a result (optional) |

## Screenshots

Take them on a phone at 1080×1920 (Android) and 1290×2796 (iPhone 15 Pro Max):

1. Detector upload screen with a file selected.
2. Emotion result screen (bars + AI interpretation).
3. Focus session with the camera running.
4. Burnout result.
5. Privacy screen with the wellness disclaimer visible.

Keep the wellness disclaimer or the "not a medical device" line visible in at
least one screenshot.

## Pre-submission checklist

- [x] Shared API token support (`security.api_token`) and per-IP rate limiting.
- [x] Resumable uploads for large files on mobile data.
- [x] JDK 21 documented and wired into `make mobile_android`.
- [x] Real app icon / splash (see `mobile/assets/`).
- [x] In-app wellness disclaimer.
- [ ] Production domain and privacy-policy URL confirmed.
- [ ] Signing keys: Android keystore, iOS certificates + provisioning profiles.
- [ ] Backend `security.api_token` decision (leave empty for an open demo).
