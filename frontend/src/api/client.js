const API_BASE = import.meta.env.VITE_API_URL || '/api';

export const apiClient = {
  async request(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `API Error: ${response.status}`);
    }

    return response.json();
  },

  async uploadFile(file, mode = 'standard') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', 'emotieff');

    const endpoints = {
      standard: '/upload',
      burnout: '/upload_burnout',
      external_influence: '/upload_external_influence',
    };

    const response = await fetch(`${API_BASE}${endpoints[mode]}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || 'Upload failed');
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
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.message || `Frame upload failed: ${response.status}`);
    }

    return response.json();
  },

  async closeFocusSession(sessionId) {
    return this.request(`/focus/session/${sessionId}/close`, { method: 'POST', body: '{}' });
  },
};