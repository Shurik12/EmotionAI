import { analyticsHeaders } from '../utils/clientId';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

// Optional shared API token (security.api_token on the server). Sent as
// X-API-Key, never as Authorization: Authorization carries the per-device
// token so the two layers cannot be confused. Embedded in the bundle, so treat
// it as an access-control deterrent / revocable kill switch, not user auth.
const API_TOKEN = import.meta.env.VITE_API_TOKEN || '';

// Files above this size go through the resumable endpoint (init/chunk/complete)
// instead of a single POST. Mobile networks routinely drop a multi-hundred-MB
// request, and losing the whole upload is much worse than resuming a chunk.
const CHUNKED_UPLOAD_THRESHOLD = 6 * 1024 * 1024;
const MAX_CHUNK_ATTEMPTS = 4;

const DEVICE_TOKEN_KEY = 'razuma_device_token';
const DEVICE_USER_KEY = 'razuma_device_user_id';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// Device identity
//
// The server issues an opaque token once (POST /api/auth/device) and derives
// `user_id` from it. Storing it locally is what lets the device come back to
// its own burnout baseline; the server ignores client-supplied ids, so the
// token is also what prevents one device from touching another's data.
// ---------------------------------------------------------------------------

let identity = null; // { token, userId }
let identityPromise = null;
let identityRetryAt = 0;

const readStoredIdentity = () => {
  try {
    const token = localStorage.getItem(DEVICE_TOKEN_KEY);
    if (!token) return null;
    return { token, userId: localStorage.getItem(DEVICE_USER_KEY) || '' };
  } catch {
    return null;
  }
};

const storeIdentity = (next) => {
  try {
    localStorage.setItem(DEVICE_TOKEN_KEY, next.token);
    if (next.userId) localStorage.setItem(DEVICE_USER_KEY, next.userId);
  } catch {
    // Private mode / storage disabled: identity lives for this tab only.
  }
};

const clearIdentity = () => {
  identity = null;
  try {
    localStorage.removeItem(DEVICE_TOKEN_KEY);
    localStorage.removeItem(DEVICE_USER_KEY);
  } catch {
    // ignore
  }
};

const appKeyHeaders = () => (API_TOKEN ? { 'X-API-Key': API_TOKEN } : {});

const authHeaders = () => (identity ? { Authorization: `Bearer ${identity.token}` } : {});

const requestHeaders = () => ({ ...appKeyHeaders(), ...authHeaders(), ...analyticsHeaders() });

async function ensureIdentity() {
  if (identity) return identity;

  const stored = readStoredIdentity();
  if (stored) {
    identity = stored;
    return identity;
  }

  if (identityPromise) return identityPromise;
  if (Date.now() < identityRetryAt) return null;

  identityPromise = (async () => {
    try {
      const response = await fetch(`${API_BASE}/auth/device`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...appKeyHeaders(), ...analyticsHeaders() },
        body: '{}',
      });

      if (response.status === 503) {
        // Backend has device auth disabled: stop asking for an hour.
        identityRetryAt = Date.now() + 60 * 60 * 1000;
        return null;
      }
      if (!response.ok) {
        throw new Error(`device bootstrap failed: ${response.status}`);
      }

      const data = await response.json();
      if (!data.device_token) throw new Error('device bootstrap returned no token');

      identity = { token: data.device_token, userId: data.user_id || '' };
      storeIdentity(identity);
      identityRetryAt = 0;
      return identity;
    } catch (error) {
      // Fail soft: anonymous endpoints still work; identity-dependent ones
      // return 401 and the UI surfaces that.
      console.warn('Device identity unavailable:', error);
      identityRetryAt = Date.now() + 60 * 1000;
      return null;
    } finally {
      identityPromise = null;
    }
  })();

  return identityPromise;
}

// fetch() with the device token attached. A 401 flagged "invalid_device_token"
// means the stored token was revoked or expired: re-bootstrap once and retry.
async function authFetch(url, init = {}, retried = false) {
  await ensureIdentity();

  const response = await fetch(url, {
    ...init,
    headers: { ...requestHeaders(), ...(init.headers || {}) },
  });

  if (response.status === 401 && !retried) {
    const errorBody = await response.clone().json().catch(() => ({}));
    if (errorBody.error === 'invalid_device_token' || errorBody.error === 'device_token_required') {
      clearIdentity();
      await ensureIdentity();
      return authFetch(url, init, true);
    }
  }

  return response;
}

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
    const response = await authFetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
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

    const response = await authFetch(`${API_BASE}${endpoints[mode]}`, {
      method: 'POST',
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

    const initResponse = await authFetch(`${API_BASE}/upload/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
            const response = await authFetch(`${API_BASE}/upload/chunk`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/octet-stream',
                'X-Upload-Id': uploadId,
                'X-Chunk-Offset': String(offset),
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

      const completeResponse = await authFetch(`${API_BASE}/upload/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upload_id: uploadId }),
      });
      if (!completeResponse.ok) {
        throw new Error(await errorFrom(completeResponse));
      }
      if (onProgress) onProgress(100);
      return completeResponse.json();
    } catch (error) {
      // Best-effort cleanup; the server also expires stale sessions.
      authFetch(`${API_BASE}/upload/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

    const response = await authFetch(`${API_BASE}/review`, {
      method: 'POST',
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

    const response = await authFetch(`${API_BASE}/focus/session/${sessionId}/frame`, {
      method: 'POST',
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
  // Identity comes from the device token; no user_id is sent or trusted.
  async getBurnoutBaseline() {
    return this.request('/burnout/baseline');
  },

  async saveBurnoutBaseline(baseline) {
    return this.request('/burnout/baseline', {
      method: 'POST',
      body: JSON.stringify({ baseline }),
    });
  },

  // ---- Device identity lifecycle ----
  // Revokes the token and deletes the derived baseline. The next request
  // bootstraps a fresh identity.
  async revokeDeviceIdentity() {
    const response = await authFetch(`${API_BASE}/auth/revoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });

    if (!response.ok) {
      throw new Error(await errorFrom(response, 'errors.uploadFailed'));
    }

    const result = await response.json();
    clearIdentity();
    return result;
  },
};
