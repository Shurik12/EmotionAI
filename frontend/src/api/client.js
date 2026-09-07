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
      realtime: '/upload_realtime',
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
};