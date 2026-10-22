/**
 * P08 detector session: the analysis lives outside React.
 *
 * Two reasons this is a module store rather than `useState`:
 * 1. The engine re-renders the whole page on any publish and `Compass.tsx`
 *    remounts the keyed subtree, so component state would silently drop a file
 *    or an open report when the demo bar changes the clinical scenario.
 * 2. The upload + polling loop must not be tied to the mounted component: a
 *    remount would otherwise lose the task id and keep polling with stale
 *    closures. Every run carries a token, and reset/navigation invalidates it
 *    and aborts the in-flight request instead of letting it schedule another tick.
 */

import {
  fileKind,
  getProgress,
  humanizeError,
  modeAcceptsKind,
  uploadForAnalysis,
  validateFile,
  type DetectorMode,
  type ProgressPayload,
} from '@/lib/detectorApi';

export interface DetectorSessionState {
  file: File | null;
  previewUrl: string | null;
  mode: DetectorMode;
  busy: boolean;
  percent: number | null;
  error: string | null;
  payload: ProgressPayload | null;
}

const POLL_MS = 900;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

const INITIAL: DetectorSessionState = {
  file: null,
  previewUrl: null,
  mode: 'standard',
  busy: false,
  percent: null,
  error: null,
  payload: null,
};

let state: DetectorSessionState = INITIAL;
const listeners = new Set<() => void>();

/** Number of mounted analyzer instances; a remount within one tick keeps state. */
let attached = 0;
/** Invalidates in-flight work: bumped by every stop. */
let runToken = 0;
let timer: number | null = null;
let inFlight: AbortController | null = null;
let taskId: string | null = null;

const emit = (): void => {
  listeners.forEach((listener) => listener());
};

const set = (patch: Partial<DetectorSessionState>): void => {
  state = { ...state, ...patch };
  emit();
};

const isAbort = (cause: unknown): boolean =>
  typeof cause === 'object' && cause !== null && (cause as { name?: string }).name === 'AbortError';

const messageOf = (cause: unknown, fallback: string): string =>
  cause instanceof Error && cause.message ? cause.message : fallback;

const clampPercent = (value: number): number => Math.min(100, Math.max(0, value));

export const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export const getSnapshot = (): DetectorSessionState => state;

const revokePreview = (): void => {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
};

const stop = (): void => {
  runToken += 1;
  if (timer !== null) {
    window.clearTimeout(timer);
    timer = null;
  }
  inFlight?.abort();
  inFlight = null;
  taskId = null;
};

const schedulePoll = (token: number, startedAt: number): void => {
  timer = window.setTimeout(() => {
    timer = null;
    if (token !== runToken || !taskId) return;

    const controller = new AbortController();
    inFlight = controller;
    void (async () => {
      try {
        const progress = await getProgress(taskId, controller.signal);
        if (token !== runToken) return;

        if (typeof progress.progress === 'number') {
          set({ percent: clampPercent(progress.progress) });
        }
        if (progress.complete) {
          if (progress.error) {
            set({ busy: false, percent: null, error: humanizeError(progress.error) });
            return;
          }
          set({ busy: false, percent: null, payload: progress });
          return;
        }
        if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
          set({
            busy: false,
            percent: null,
            error: 'Анализ не завершился за 5 минут. Попробуйте файл короче.',
          });
          return;
        }
        schedulePoll(token, startedAt);
      } catch (cause) {
        if (token !== runToken || isAbort(cause)) return;
        set({
          busy: false,
          percent: null,
          error: messageOf(cause, 'Не удалось получить статус анализа.'),
        });
      } finally {
        if (inFlight === controller) inFlight = null;
      }
    })();
  }, POLL_MS);
};

/** A new file invalidates the previous run; ignored while one is in flight. */
export function selectFile(next: File | null): void {
  if (state.busy) return;
  stop();
  revokePreview();
  set({ file: null, previewUrl: null, payload: null, percent: null, error: null });
  if (!next) return;

  const problem = validateFile(next);
  if (problem) {
    set({ error: problem });
    return;
  }

  const kind = fileKind(next);
  set({
    file: next,
    previewUrl: (next.type || '').startsWith('image/') ? URL.createObjectURL(next) : null,
    mode: modeAcceptsKind(state.mode, kind) ? state.mode : 'standard',
  });
}

export function selectMode(mode: DetectorMode): void {
  if (state.busy) return;
  set({ mode });
}

export function resetSession(): void {
  stop();
  revokePreview();
  state = { ...INITIAL };
  emit();
}

export async function startAnalysis(): Promise<void> {
  const { file, mode, busy } = state;
  if (!file || busy || !modeAcceptsKind(mode, fileKind(file))) return;

  stop();
  const token = runToken;
  set({ error: null, payload: null, busy: true, percent: 0 });

  const controller = new AbortController();
  inFlight = controller;
  try {
    const { task_id } = await uploadForAnalysis(file, mode, controller.signal);
    if (token !== runToken) return;
    taskId = task_id;
    schedulePoll(token, Date.now());
  } catch (cause) {
    if (token !== runToken || isAbort(cause)) return;
    set({
      busy: false,
      percent: null,
      error: messageOf(cause, 'Не удалось загрузить файл.'),
    });
  } finally {
    if (inFlight === controller) inFlight = null;
  }
}

/**
 * Tracks mounted instances so an engine re-render (unmount + mount in the same
 * tick) keeps the session, while actually leaving the screen drops it.
 */
export function attachSession(): () => void {
  attached += 1;
  return () => {
    attached -= 1;
    if (attached > 0) return;
    window.setTimeout(() => {
      if (attached === 0) resetSession();
    }, 0);
  };
}
