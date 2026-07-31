// Small fetch wrapper + job poller shared by the mode pages.
(function (global) {
  'use strict';

  async function unwrap(response) {
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; } catch (err) { payload = { error: text }; }
    if (!response.ok) {
      throw new Error(payload.error || response.statusText || 'Request failed');
    }
    return payload;
  }

  const api = {
    async get(url) {
      return unwrap(await fetch(url, { headers: { Accept: 'application/json' } }));
    },
    async post(url, body) {
      return unwrap(await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
      }));
    },
    async upload(url, formData) {
      return unwrap(await fetch(url, { method: 'POST', body: formData }));
    },

    // Poll a session job until it leaves the "running" state.
    async waitForJob(sessionId, onProgress, intervalMs) {
      const wait = intervalMs || 600;
      for (;;) {
        const job = await api.get(`/api/sessions/${sessionId}/job`);
        if (onProgress) onProgress(job);
        if (job.status === 'error') throw new Error(job.error || job.message);
        if (job.status !== 'running') return job;
        await new Promise((resolve) => setTimeout(resolve, wait));
      }
    }
  };

  function showError(err) {
    const banner = document.getElementById('error-banner');
    console.error(err);
    if (!banner) { alert(err.message || err); return; }
    banner.textContent = err.message || String(err);
    banner.hidden = false;
    banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    const banner = document.getElementById('error-banner');
    if (banner) banner.hidden = true;
  }

  // Wrap a handler so any rejection lands in the error banner.
  function guard(fn) {
    return function (...args) {
      clearError();
      try {
        const out = fn.apply(this, args);
        if (out && typeof out.catch === 'function') out.catch(showError);
        return out;
      } catch (err) {
        showError(err);
        return undefined;
      }
    };
  }

  global.SamuraiAPI = api;
  global.showError = showError;
  global.clearError = clearError;
  global.guard = guard;
})(window);
