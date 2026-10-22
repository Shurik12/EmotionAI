/**
 * Detector client for the real backend (same-origin `/api` in production).
 *
 * P08 borrows the landing's detector flow: POST the file to the mode endpoint,
 * then poll `/api/progress/:id`. Two things the landing gets wrong and this module
 * does not: it checks the response content type before calling `json()` (a 502
 * from nginx or a 501 from a static server would otherwise surface as
 * "unexpected token <"), and it turns the backend's English failures into text a
 * demo viewer can act on.
 */

export type DetectorMode = 'standard' | 'burnout' | 'external_influence';
export type FileKind = 'image' | 'audio' | 'video';

export const MODES: ReadonlyArray<{ value: DetectorMode; label: string }> = [
  { value: 'standard', label: 'Стандартный анализ' },
  { value: 'burnout', label: 'Анализ состояния человека' },
  { value: 'external_influence', label: 'Анализ внешнего влияния' },
];

export const ACCEPT = 'image/*,video/*,audio/*';
export const MAX_FILE_BYTES = 50 * 1024 * 1024;
export const SUPPORTED_FORMATS =
  'JPG, PNG, MP4, AVI, WEBM, MOV, MP3, WAV, AAC, OGG, FLAC (макс. 50 МБ)';

const ENDPOINTS: Record<DetectorMode, string> = {
  standard: '/upload',
  burnout: '/upload_burnout',
  external_influence: '/upload_external_influence',
};

const IMAGE_EXT = ['png', 'jpg', 'jpeg'];
const AUDIO_EXT = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'opus', 'wma'];
const VIDEO_EXT = ['mp4', 'avi', 'webm', 'mov', 'mkv'];

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim().replace(/\/+$/, '') || '/api';

export interface ProgressPayload {
  complete?: boolean;
  progress?: number;
  message?: string;
  error?: string | null;
  type?: string;
  mode?: string;
  task_id?: string;
  image_url?: string;
  result?: any;
  ai_analysis?: any;
  average_emotions?: Record<string, number | string> | null;
  average_main_emotion?: { label?: string; probability?: number } | null;
  results?: Array<{ image_url?: string; result?: any; timestamp?: number; frame?: number }>;
  frames_processed?: number;
  duration?: number;
  fps?: number;
}

export const fileKind = (file: File | null): FileKind | null => {
  if (!file) return null;
  const type = file.type || '';
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';
  if (type.startsWith('audio/')) return 'audio';
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (IMAGE_EXT.includes(ext)) return 'image';
  if (VIDEO_EXT.includes(ext)) return 'video';
  if (AUDIO_EXT.includes(ext)) return 'audio';
  return null;
};

/**
 * Burnout and external influence analyse the audio track, and the server extracts
 * it from video with FFmpeg — so video is accepted there too. Only images have no
 * sound at all. (The landing disables video outside Standard; the backend does not.)
 */
export const modeAcceptsKind = (mode: DetectorMode, kind: FileKind | null): boolean =>
  kind === null || mode === 'standard' || kind === 'audio' || kind === 'video';

export const modeLabel = (mode: DetectorMode): string =>
  MODES.find((m) => m.value === mode)?.label ?? mode;

export const formatBytes = (bytes: number): string => {
  if (!bytes) return '0 Б';
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
};

export const validateFile = (file: File): string | null => {
  if (!file.size) return 'Файл пустой.';
  if (file.size > MAX_FILE_BYTES) return `Файл больше 50 МБ (${formatBytes(file.size)}).`;
  if (!fileKind(file)) return `Формат не поддерживается. Допустимо: ${SUPPORTED_FORMATS}.`;
  return null;
};

/** Backend and proxy failures are English or HTML; say something actionable. */
export const humanizeError = (raw: string | null | undefined): string => {
  const text = (raw || '').trim();
  if (!text) return 'Не удалось выполнить анализ.';
  if (/no audio stream/i.test(text)) {
    return 'В файле нет звуковой дорожки. Выберите «Стандартный анализ» или файл со звуком.';
  }
  if (/invalid file type/i.test(text)) {
    return `Сервер не поддерживает этот формат. Допустимо: ${SUPPORTED_FORMATS}.`;
  }
  if (/failed to load audio/i.test(text)) return 'Не удалось прочитать звук из файла.';
  if (/file too large|payload too large|413/i.test(text)) return 'Файл больше 50 МБ.';
  return text;
};

const isJson = (response: Response): boolean =>
  (response.headers.get('content-type') || '').includes('json');

const readError = async (response: Response): Promise<string> => {
  if (isJson(response)) {
    const body = await response.json().catch(() => null);
    const message = body?.message || body?.error;
    if (message) return humanizeError(String(message));
  } else {
    // nginx 502, a static server's 501, etc. — never "unexpected token <".
    await response.text().catch(() => '');
  }
  if (response.status === 413) return 'Файл больше 50 МБ.';
  return `Сервис анализа недоступен (HTTP ${response.status}). Попробуйте позже.`;
};

const readJson = async <T,>(response: Response): Promise<T> => {
  if (!isJson(response)) {
    throw new Error(
      `Сервис анализа вернул ответ не в формате данных (HTTP ${response.status}).`,
    );
  }
  return (await response.json()) as T;
};

export async function uploadForAnalysis(
  file: File,
  mode: DetectorMode,
  signal?: AbortSignal,
): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append('file', file);
  form.append('model', 'emotieff');

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${ENDPOINTS[mode]}`, {
      method: 'POST',
      body: form,
      signal,
    });
  } catch (cause) {
    // A cancelled request is not a service failure: the caller dropped the run.
    if (signal?.aborted) throw cause;
    throw new Error('Сервис анализа недоступен: запрос не ушёл. Проверьте соединение.');
  }
  if (!response.ok) throw new Error(await readError(response));

  const body = await readJson<{ task_id?: string }>(response);
  if (!body.task_id) throw new Error('Сервис не вернул идентификатор задачи.');
  return { task_id: body.task_id };
}

export async function getProgress(
  taskId: string,
  signal?: AbortSignal,
): Promise<ProgressPayload> {
  const response = await fetch(`${API_BASE}/progress/${encodeURIComponent(taskId)}`, { signal });
  if (!response.ok) throw new Error(await readError(response));
  return readJson<ProgressPayload>(response);
}
