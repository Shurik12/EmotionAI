import { analyticsHeaders } from '../utils/clientId';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

// Optional shared API token (security.api_token on the server). Embedded in the
// bundle, so treat it as an access-control deterrent / revocable kill switch,
// not as per-user authentication. Left empty for the public demo.
const API_TOKEN = import.meta.env.VITE_API_TOKEN || '';

// Files above this size go through the resumable endpoint (init/chunk/complete)
// instead of a single POST. Mobile networks routinely drop a multi-hundred-MB
// request, and losing the whole upload is much worse than resuming a chunk.
const CHUNKED_UPLOAD_THRESHOLD = 6 * 1024 * 1024;
const MAX_CHUNK_ATTEMPTS = 4;

const authHeaders = () => (API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {});

const baseHeaders = () => ({ ...authHeaders(), ...analyticsHeaders() });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Parse an error response into something the UI can translate. Server errors are
// `{"error": "..."}`; a 413 falls back to the localized size-limit key.
const errorFrom = async (response, fallback = 'errors.uploadFailed') => {
  const text = await response.text().catch(() => '');
  try {
    const parsed = JSON.parse(text);
    if (parsed?.error) return parsed.error;
    if (parsed?.message) return parsed.message;
  } catch {
    // Non-JSON error body (e.g. the plain-text 413 from the body-size guard).
  }
  return response.status === 413 ? 'errors.fileTooLarge' : fallback;
};

export const apiClient = {
  async request(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...baseHeaders(),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || error.message || `API Error: ${response.status}`);
    }

    return response.json();
  },

  // Upload a file, switching to the resumable path for large inputs.
  // `onProgress(percent)` is called with 0–100 while the body is in flight.
  async uploadFile(file, mode = 'standard', { onProgress } = {}) {
    if (file.size > CHUNKED_UPLOAD_THRESHOLD) {
      return this.uploadFileChunked(file, mode, { onProgress });
    }
    return this.uploadFileDirect(file, mode);
  },

  async uploadFileDirect(file, mode = 'standard') {
    const formData = new FormData();
    formData.append('file', file);

    const endpoints = {
      standard: '/upload',
      burnout: '/upload_burnout',
      external_influence: '/upload_external_influence',
    };

    const response = await fetch(`${API_BASE}${endpoints[mode]}`, {
      method: 'POST',
      headers: baseHeaders(),
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await errorFrom(response));
    }

    return response.json();
  },

  // Resumable upload: POST /api/upload/init -> chunk* -> complete.
  // Each chunk is retried with exponential backoff and, if the server already
  // has more bytes than the client thinks (a retry after a lost response), the
  // 409 reply says where to resume.
  async uploadFileChunked(file, mode = 'standard', { onProgress, selfScore } = {}) {
    const initBody = { filename: file.name, size: file.size, mode };
    if (mode === 'review') initBody.self_score = selfScore;

    const initResponse = await fetch(`${API_BASE}/upload/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...baseHeaders() },
      body: JSON.stringify(initBody),
    });
    if (!initResponse.ok) {
      throw new Error(await errorFrom(initResponse));
    }

    const init = await initResponse.json();
    const uploadId = init.upload_id;
    const chunkSize = init.chunk_size > 0 ? init.chunk_size : 4 * 1024 * 1024;

    let offset = 0;
    try {
      while (offset < file.size) {
        const end = Math.min(offset + chunkSize, file.size);
        const blob = file.slice(offset, end);

        for (let attempt = 1; ; attempt += 1) {
          try {
            const response = await fetch(`${API_BASE}/upload/chunk`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/octet-stream',
                'X-Upload-Id': uploadId,
                'X-Chunk-Offset': String(offset),
                ...baseHeaders(),
              },
              body: blob,
            });

            if (response.status === 409) {
              const data = await response.json().catch(() => ({}));
              if (typeof data.received === 'number') {
                offset = data.received;
                break; // re-slice from the new offset
              }
              throw new Error('Upload lost synchronisation');
            }
            if (!response.ok) {
              throw new Error(await errorFrom(response));
            }

            const data = await response.json();
            offset = typeof data.received === 'number' ? data.received : end;
            if (onProgress) onProgress(Math.min(100, Math.round((offset / file.size) * 100)));
            break;
          } catch (error) {
            if (attempt >= MAX_CHUNK_ATTEMPTS) throw error;
            await sleep(500 * 2 ** attempt);
          }
        }
      }

      const completeResponse = await fetch(`${API_BASE}/upload/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...baseHeaders() },
        body: JSON.stringify({ upload_id: uploadId }),
      });
      if (!completeResponse.ok) {
        throw new Error(await errorFrom(completeResponse));
      }
      if (onProgress) onProgress(100);
      return completeResponse.json();
    } catch (error) {
      // Best-effort cleanup; the server also expires stale sessions.
      fetch(`${API_BASE}/upload/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...baseHeaders() },
        body: JSON.stringify({ upload_id: uploadId }),
      }).catch(() => {});
      throw error;
    }
  },

  async uploadReview(file, selfScore, mode = 'standard', { onProgress } = {}) {
    if (file.size > CHUNKED_UPLOAD_THRESHOLD) {
      return this.uploadFileChunked(file, 'review', { onProgress, selfScore });
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('self_score', String(selfScore));
    formData.append('mode', mode);

    const response = await fetch(`${API_BASE}/review`, {
      method: 'POST',
      headers: baseHeaders(),
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await errorFrom(response, 'errors.uploadFailed'));
    }

    return response.json();
  },

  async getProgress(taskId) {
    return this.request(`/progress/${taskId}`);
  },

  // ---- Focus camera session (EMO-17) ----
  // A session aggregates emotional dynamics on the server; frames are streamed
  // into it and the client polls the accumulated state.
  async createFocusSession() {
    return this.request('/focus/session', { method: 'POST', body: '{}' });
  },

  async getFocusSession(sessionId) {
    return this.request(`/focus/session/${sessionId}`);
  },

  async sendFocusFrame(sessionId, blob) {
    const formData = new FormData();
    formData.append('file', blob, 'focus-frame.jpg');

    const response = await fetch(`${API_BASE}/focus/session/${sessionId}/frame`, {
      method: 'POST',
      headers: baseHeaders(),
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || error.message || `Frame upload failed: ${response.status}`);
    }

    return response.json();
  },

  async closeFocusSession(sessionId) {
    return this.request(`/focus/session/${sessionId}/close`, { method: 'POST', body: '{}' });
  },

  // ---- Focus AI task breakdown (EMO-19) ----
  // Only the task text and the block duration are sent to the model; camera
  // frames never leave the Razuma core. Throws on error so the caller can
  // fall back to the local templates.
  async breakdownFocusTask(text, duration = '', lang = 'ru') {
    return this.request('/focus/breakdown', {
      method: 'POST',
      body: JSON.stringify({ text, duration, lang }),
    });
  },

  // ---- Burnout baseline ----
  async getBurnoutBaseline(userId) {
    return this.request(`/burnout/baseline?user_id=${encodeURIComponent(userId)}`);
  },

  async saveBurnoutBaseline(userId, baseline) {
    return this.request('/burnout/baseline', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, baseline }),
    });
  },
};
