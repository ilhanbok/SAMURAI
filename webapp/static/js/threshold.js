// Mode (ii): threshold lesion finding, then per-candidate SAM2 segmentation.
(function () {
  'use strict';

  const api = window.SamuraiAPI;
  const state = {
    sessionId: null,
    volume: null,
    lesions: [],
    current: null,
    view: null,
    points: [],     // extra user points for the current candidate
    mode: 1,
    logitThreshold: 2.0,
    scatter: null
  };

  const $ = (selector) => document.querySelector(selector);

  function adoptSession(summary) {
    state.sessionId = summary.id;
    state.volume = summary.volume;
    $('#threshold-form').hidden = false;
  }

  $('#upload-form').addEventListener('submit', guard(async (event) => {
    event.preventDefault();
    const data = new FormData();
    data.append('file', event.target.file.files[0]);
    adoptSession(await api.upload('/api/sessions', data));
  }));

  $('#server-form').addEventListener('submit', guard(async (event) => {
    event.preventDefault();
    adoptSession(await api.post('/api/sessions', { path: event.target.path.value.trim() }));
  }));

  $('#threshold-form').mode.addEventListener('change', (event) => {
    const defaults = (window.SAMURAI_MODE_DEFAULTS || {})[event.target.value];
    if (!defaults) return;
    $('#threshold-form').cutoff.value = defaults.cutoff;
    $('#threshold-form').zero_slices_below.value = defaults.zero_slices_below;
  });

  $('#threshold-form').addEventListener('submit', guard(async (event) => {
    event.preventDefault();
    const form = event.target;
    state.logitThreshold = Number(form.logit_threshold.value);
    $('#step-progress').hidden = false;
    $('#progress-bar').style.width = '0%';
    await api.post(`/api/sessions/${state.sessionId}/threshold/run`, {
      mode: form.mode.value,
      cutoff: form.cutoff.value,
      output_csv: form.output_csv.value,
      sigma: form.sigma.value,
      zero_slices_below: form.zero_slices_below.value,
      flip: form.flip.checked,
      skullstrip: form.skullstrip.checked,
      volume_min: form.volume_min.value,
      volume_max: form.volume_max.value
    });
    const job = await api.waitForJob(state.sessionId, onProgress);
    $('#step-progress').hidden = true;
    $('#catalog-meta').textContent =
      `${job.result.num_features} connected components, ${job.result.candidates} `
      + `within the volume gate. Summary CSV: ${job.result.csv}`;
    await loadLesions();
    $('#step-2').hidden = false;
    await refreshFiles();
  }));

  function onProgress(job) {
    $('#progress-bar').style.width = Math.round((job.progress || 0) * 100) + '%';
    $('#progress-message').textContent = job.message || job.status;
  }

  $('#cancel-job').onclick = guard(async () => {
    await api.post(`/api/sessions/${state.sessionId}/job/cancel`);
  });

  // -- candidate list ---------------------------------------------------
  async function loadLesions() {
    const response = await api.get(`/api/sessions/${state.sessionId}/threshold/lesions`);
    state.lesions = response.lesions;
    const body = $('#lesion-rows');
    body.innerHTML = '';
    response.lesions.forEach((lesion) => {
      const row = document.createElement('tr');
      row.className = lesion.segmented ? 'segmented' : '';
      row.innerHTML =
        `<td>${lesion.position + 1}</td><td>${lesion.label}</td>`
        + `<td>${lesion.volume.toFixed(1)}</td>`
        + `<td>${lesion.com.map((c) => c.toFixed(1)).join(', ')}</td>`
        + `<td><button type="button">Inspect</button></td>`;
      row.onclick = guard(() => openLesion(lesion.position, row));
      body.appendChild(row);
    });
    renderResults(response.results);
  }

  async function openLesion(position, row) {
    document.querySelectorAll('#lesion-rows tr').forEach((r) => r.classList.remove('active'));
    if (row) row.classList.add('active');
    const detail = await api.get(
      `/api/sessions/${state.sessionId}/threshold/lesions/${position}`);
    state.current = detail;
    state.points = [];
    $('#lesion-detail').hidden = false;
    $('#lesion-label').textContent = `${detail.label} (${detail.volume.toFixed(1)} mm³)`;
    $('#lesion-frame').textContent = detail.frame;
    $('#lesion-voxels').textContent = detail.voxels_on_frame;

    if (!state.view) {
      state.view = new SliceCanvas($('#lesion-view canvas'));
      state.view.on('click', guard(async (point) => {
        state.points.push({ x: point.x, y: point.y, label: state.mode });
        state.view.setMarkers(state.points);
      }));
    }
    await state.view.setBase(detail.slice_url);
    state.scatter = detail.scatter;
    state.view.setScatter($('#show-scatter').checked ? detail.scatter : null);
    state.view.setMarkers(state.points);
  }

  $('#show-scatter').onchange = (event) => {
    if (state.view) state.view.setScatter(event.target.checked ? state.scatter : null);
  };

  $('#flip-mode').onclick = () => {
    state.mode = 1 - state.mode;
    $('#prompt-mode-label').textContent =
      state.mode === 1 ? 'Selection: Foreground' : 'Selection: Background';
  };

  $('#undo').onclick = () => {
    state.points.pop();
    if (state.view) state.view.setMarkers(state.points);
  };

  $('#clear-points').onclick = () => {
    state.points = [];
    if (state.view) state.view.setMarkers(state.points);
  };

  $('#skip').onclick = guard(async () => {
    const next = state.current ? state.current.position + 1 : 0;
    if (next < state.lesions.length) {
      const rows = document.querySelectorAll('#lesion-rows tr');
      await openLesion(next, rows[next]);
    } else {
      $('#lesion-detail').hidden = true;
    }
  });

  $('#approve').onclick = guard(async () => {
    if (!state.current) return;
    $('#step-progress').hidden = false;
    $('#progress-bar').style.width = '0%';
    await api.post(
      `/api/sessions/${state.sessionId}/threshold/lesions/${state.current.position}/segment`,
      { points: state.points, logit_threshold: state.logitThreshold }
    );
    const job = await api.waitForJob(state.sessionId, onProgress);
    $('#step-progress').hidden = true;
    $('#results-panel').hidden = false;
    await loadLesions();
    await refreshFiles();
    const position = state.current.position + 1;
    if (position < state.lesions.length) {
      const rows = document.querySelectorAll('#lesion-rows tr');
      await openLesion(position, rows[position]);
    }
    if (job.result) {
      $('#catalog-meta').textContent =
        `Saved ${job.result.file} (${job.result.voxels.toLocaleString()} voxels, `
        + `${job.result.prompt_points} prompt points).`;
    }
  });

  function renderResults(results) {
    const list = $('#result-list');
    if (!list) return;
    list.innerHTML = '';
    (results || []).forEach((result) => {
      const item = document.createElement('li');
      item.textContent =
        `label ${result.label}: ${result.file} - ${result.voxels.toLocaleString()} voxels `
        + `(threshold volume ${result.volume.toFixed(1)} mm³)`;
      list.appendChild(item);
    });
    if ((results || []).length) $('#results-panel').hidden = false;
  }

  $('#merge').onclick = guard(async () => {
    const response = await api.post(`/api/sessions/${state.sessionId}/threshold/merge`);
    $('#catalog-meta').textContent =
      `Merged ${response.lesions} lesion(s) into ${response.file} `
      + `(${response.voxels.toLocaleString()} voxels).`;
    await refreshFiles();
  });

  async function refreshFiles() {
    const response = await api.get(`/api/sessions/${state.sessionId}/files`);
    const list = $('#file-list');
    list.innerHTML = '';
    response.files.forEach((file) => {
      const item = document.createElement('li');
      item.innerHTML = `<a href="${file.url}">${file.name}</a> `
        + `<span class="meta">${(file.size / 1024).toFixed(1)} kB</span>`;
      list.appendChild(item);
    });
    $('#results-panel').hidden = response.files.length === 0;
  }
})();
