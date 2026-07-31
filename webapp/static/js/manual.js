// Mode (i): manual point / bounding-box segmentation.
// Drives the same five stages as the desktop wrapper: load, plane+slice,
// points, propagation, review & save.
(function () {
  'use strict';

  const api = window.SamuraiAPI;
  const state = {
    sessionId: null,
    volume: null,
    window: [0, 1],
    slices: [0, 0, 0],
    axis: null,
    mode: 1,          // 1 = foreground, 0 = background
    points: [],
    box: null,
    threshold: window.SAMURAI_DEFAULT_THRESHOLD || 0,
    planeViews: [],
    reviewViews: [],
    promptView: null,
    previewView: null
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  // -- step chrome ------------------------------------------------------
  function setStep(step) {
    [1, 2, 3, 4, 5].forEach((n) => {
      const section = document.getElementById('step-' + n);
      if (section) section.hidden = n !== step;
      const chip = document.querySelector(`#steps li[data-step="${n}"]`);
      if (chip) {
        chip.classList.toggle('current', n === step);
        chip.classList.toggle('done', n < step);
      }
    });
  }

  function sliceUrl(axis, index) {
    const [vmin, vmax] = state.window;
    return `/api/sessions/${state.sessionId}/slice.png?axis=${axis}&index=${index}`
         + `&vmin=${vmin}&vmax=${vmax}`;
  }

  function maskUrl(axis, index) {
    return `/api/sessions/${state.sessionId}/mask.png?axis=${axis}&index=${index}`;
  }

  // -- step 1: load -----------------------------------------------------
  function adoptSession(summary) {
    state.sessionId = summary.id;
    state.volume = summary.volume;
    state.window = summary.window;
    state.slices = summary.slices.slice();
    $('#volume-info').textContent =
      `${summary.volume.name} - display shape ${summary.volume.display_shape.join(' x ')}, `
      + `voxel ${summary.volume.zooms.map((z) => z.toFixed(2)).join(' x ')} mm, `
      + `intensity ${summary.volume.vmin.toFixed(2)} to ${summary.volume.vmax.toFixed(2)}`;
    setupWindowSliders();
    setupPlaneViews();
    setStep(2);
    refreshFiles();
  }

  $('#upload-form').addEventListener('submit', guard(async (event) => {
    event.preventDefault();
    const form = event.target;
    const data = new FormData();
    data.append('file', form.file.files[0]);
    if (form.dwi.checked) data.append('dwi', '1');
    form.querySelector('button').disabled = true;
    try {
      adoptSession(await api.upload('/api/sessions', data));
    } finally {
      form.querySelector('button').disabled = false;
    }
  }));

  $('#server-form').addEventListener('submit', guard(async (event) => {
    event.preventDefault();
    const form = event.target;
    adoptSession(await api.post('/api/sessions', {
      path: form.path.value.trim(),
      dwi: form.dwi.checked ? '1' : ''
    }));
  }));

  $('#browse-btn').addEventListener('click', guard(async () => {
    const box = $('#browser');
    box.hidden = false;
    await renderBrowser(null);
  }));

  async function renderBrowser(dir) {
    const box = $('#browser');
    const listing = await api.get('/api/browse' + (dir ? '?dir=' + encodeURIComponent(dir) : ''));
    box.innerHTML = '';
    const header = document.createElement('div');
    header.className = 'meta';
    header.textContent = listing.dir;
    box.appendChild(header);
    if (listing.parent) {
      const up = document.createElement('div');
      up.className = 'dir';
      up.textContent = '.. (up)';
      up.onclick = guard(() => renderBrowser(listing.parent));
      box.appendChild(up);
    }
    listing.dirs.forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'dir';
      row.textContent = entry.name + '/';
      row.onclick = guard(() => renderBrowser(entry.path));
      box.appendChild(row);
    });
    listing.files.forEach((entry) => {
      const row = document.createElement('div');
      row.textContent = `${entry.name}  (${(entry.size / 1048576).toFixed(1)} MB)`;
      row.onclick = () => { $('#server-form').path.value = entry.path; box.hidden = true; };
      box.appendChild(row);
    });
  }

  // -- step 2: plane and slice -----------------------------------------
  function setupWindowSliders() {
    const [lo, hi] = [state.volume.vmin, state.volume.vmax];
    const min = $('#win-min');
    const max = $('#win-max');
    [min, max].forEach((slider) => { slider.min = lo; slider.max = hi; });
    min.value = state.window[0];
    max.value = state.window[1];
    $('#win-min-out').value = Number(min.value).toFixed(2);
    $('#win-max-out').value = Number(max.value).toFixed(2);

    const onChange = guard(async () => {
      state.window = [Number(min.value), Number(max.value)];
      $('#win-min-out').value = state.window[0].toFixed(2);
      $('#win-max-out').value = state.window[1].toFixed(2);
      await Promise.all([0, 1, 2].map(refreshPlane));
    });
    min.oninput = debounce(onChange, 120);
    max.oninput = debounce(onChange, 120);
  }

  function setupPlaneViews() {
    if (state.planeViews.length) {
      // a second volume in the same tab: rebind limits, keep the listeners
      state.planeViews.forEach((entry) => {
        entry.input.max = state.volume.display_shape[entry.axis] - 1;
        entry.input.value = state.slices[entry.axis];
      });
      [0, 1, 2].forEach(refreshPlane);
      return;
    }
    state.planeViews = $$('#plane-views .viewer').map((figure) => {
      const axis = Number(figure.dataset.axis);
      const view = new SliceCanvas(figure.querySelector('canvas'));
      const input = figure.querySelector('.slice-input');
      input.max = state.volume.display_shape[axis] - 1;
      input.value = state.slices[axis];

      view.on('wheel', guard((step) => {
        setSlice(axis, state.slices[axis] + step);
      }));
      input.addEventListener('change', guard(() => setSlice(axis, Number(input.value))));
      figure.querySelector('.confirm-plane').onclick = guard(() => confirmPlane(axis));
      return { axis, view, figure, input };
    });
    [0, 1, 2].forEach(refreshPlane);
  }

  function setSlice(axis, value) {
    const limit = state.volume.display_shape[axis] - 1;
    state.slices[axis] = Math.max(0, Math.min(limit, Math.round(value)));
    const entry = state.planeViews.find((v) => v.axis === axis);
    if (entry) entry.input.value = state.slices[axis];
    const review = state.reviewViews.find((v) => v.axis === axis);
    if (review) review.input.value = state.slices[axis];
    return Promise.all([refreshPlane(axis), refreshReview(axis)]);
  }

  async function refreshPlane(axis) {
    const entry = state.planeViews.find((v) => v.axis === axis);
    if (!entry) return;
    entry.figure.querySelector('.slice-label').textContent = state.slices[axis];
    await entry.view.setBase(sliceUrl(axis, state.slices[axis]));
  }

  async function confirmPlane(axis) {
    state.axis = axis;
    setStep(4);
    $('#progress-message').textContent = 'Preparing the JPEG stack and SAM2 state...';
    await api.post(`/api/sessions/${state.sessionId}/plane`, {
      axis, slice: state.slices[axis]
    });
    await api.waitForJob(state.sessionId, onProgress);
    state.points = [];
    state.box = null;
    await enterPromptStep();
  }

  function onProgress(job) {
    $('#progress-bar').style.width = Math.round((job.progress || 0) * 100) + '%';
    $('#progress-message').textContent = job.message || job.status;
  }

  // -- step 3: points ---------------------------------------------------
  async function enterPromptStep() {
    setStep(3);
    $('#prompt-plane').textContent = ['Axial', 'Coronal', 'Sagittal'][state.axis];
    $('#prompt-slice').textContent = state.slices[state.axis];

    if (!state.promptView) {
      state.promptView = new SliceCanvas($('#prompt-view canvas'));
      state.promptView.on('click', guard(async (point) => {
        if (state.promptView.boxMode) return;
        state.points.push({ x: point.x, y: point.y, label: state.mode });
        await pushPoints();
      }));
      state.promptView.on('box', guard(async (box) => {
        state.box = box;
        await pushPoints();
      }));
      state.promptView.on('wheel', guard(async (step) => {
        // Points belong to the slice they were clicked on, so moving off it
        // has to start the object again rather than silently re-anchor.
        if (state.points.length || state.box) {
          const ok = window.confirm(
            'Leaving this slice discards the points selected on it. Continue?');
          if (!ok) return;
          state.points = [];
          state.box = null;
          state.promptView.setBox(null);
          await state.previewView.setOverlay(null);
        }
        await setSlice(state.axis, state.slices[state.axis] + step);
        $('#prompt-slice').textContent = state.slices[state.axis];
        await state.promptView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
        await state.previewView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
        state.promptView.setMarkers(state.points);
        renderPointTable();
      }));
      state.previewView = new SliceCanvas($('#preview-view canvas'));
    }
    await state.promptView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
    state.promptView.setMarkers(state.points);
    state.promptView.setBox(state.box);
    await state.previewView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
    renderPointTable();
  }

  async function pushPoints() {
    const response = await api.post(`/api/sessions/${state.sessionId}/points`, {
      points: state.points,
      box: state.box,
      slice: state.slices[state.axis]
    });
    state.points = response.object.points;
    state.promptView.setMarkers(state.points);
    renderPointTable();
    if (response.preview) await refreshPreview(response.object.obj_id);
  }

  async function refreshPreview(objId) {
    await state.previewView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
    await state.previewView.setOverlay(
      `/api/sessions/${state.sessionId}/preview.png?obj=${objId}`, true
    );
  }

  function renderPointTable() {
    const fg = state.points.filter((p) => Number(p.label) === 1).length;
    const body = $('#point-table tbody');
    body.innerHTML =
      `<tr><th>Foreground (fg:1)</th><td>${fg}</td></tr>`
      + `<tr><th>Background (bg:0)</th><td>${state.points.length - fg}</td></tr>`
      + `<tr><th>Bounding box</th><td>${state.box
          ? state.box.map((v) => v.toFixed(1)).join(', ') : 'none'}</td></tr>`;
  }

  $('#flip-mode').onclick = () => {
    state.mode = 1 - state.mode;
    $('#prompt-mode-label').textContent =
      state.mode === 1 ? 'Selection: Foreground' : 'Selection: Background';
  };

  $('#box-mode').onchange = (event) => {
    if (!state.promptView) return;
    state.promptView.boxMode = event.target.checked;
    if (!event.target.checked) { state.box = null; state.promptView.setBox(null); }
  };

  $('#undo').onclick = guard(async () => {
    const response = await api.post(`/api/sessions/${state.sessionId}/points/undo`);
    state.points = (response.object && response.object.points) || [];
    state.promptView.setMarkers(state.points);
    renderPointTable();
    if (response.object) await refreshPreview(response.object.obj_id);
  });

  $('#redo').onclick = guard(async () => {
    const response = await api.post(`/api/sessions/${state.sessionId}/points/redo`);
    state.points = (response.object && response.object.points) || [];
    state.promptView.setMarkers(state.points);
    renderPointTable();
    if (response.object) await refreshPreview(response.object.obj_id);
  });

  $('#clear-points').onclick = guard(async () => {
    state.points = [];
    state.box = null;
    state.promptView.setBox(null);
    await pushPoints();
    await state.previewView.setOverlay(null);
  });

  $('#add-object').onclick = guard(async () => {
    const response = await api.post(`/api/sessions/${state.sessionId}/objects`, {
      slice: state.slices[state.axis]
    });
    state.points = [];
    state.box = null;
    state.promptView.setBox(null);
    state.promptView.setMarkers([]);
    await state.previewView.setOverlay(null);
    renderPointTable();
    $('#prompt-slice').textContent = state.slices[state.axis];
    $('#add-object').textContent = `Add another object (${response.objects} so far)`;
  });

  $('#import-points').addEventListener('change', guard(async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const data = new FormData();
    data.append('file', file);
    const parsed = await api.upload(`/api/sessions/${state.sessionId}/points/import`, data);
    if (parsed.axis !== null && parsed.axis !== undefined && parsed.axis !== state.axis) {
      showError(new Error(
        `Points file was recorded on axis ${parsed.axis}; you are on axis ${state.axis}. `
        + 'Loaded anyway - re-select the plane if that is wrong.'));
    }
    if (parsed.slice !== null && parsed.slice !== undefined) {
      await setSlice(state.axis, parsed.slice);
      $('#prompt-slice').textContent = state.slices[state.axis];
      await state.promptView.setBase(sliceUrl(state.axis, state.slices[state.axis]));
    }
    state.points = parsed.points;
    await pushPoints();
  }));

  $('#back-to-plane').onclick = () => setStep(2);

  // -- step 4/5: propagate, review, save --------------------------------
  $('#propagate').onclick = guard(async () => {
    if (!state.points.length && !state.box) {
      throw new Error('Add at least one foreground point (or a bounding box) first.');
    }
    setStep(4);
    $('#progress-bar').style.width = '0%';
    await api.post(`/api/sessions/${state.sessionId}/propagate`);
    const job = await api.waitForJob(state.sessionId, onProgress);
    await enterReviewStep(job.result || {});
  });

  $('#cancel-job').onclick = guard(async () => {
    await api.post(`/api/sessions/${state.sessionId}/job/cancel`);
  });

  async function enterReviewStep(result) {
    setStep(5);
    const slider = $('#mask-threshold');
    if (typeof result.logit_min === 'number') {
      slider.min = result.logit_min;
      slider.max = result.logit_max;
      slider.step = (result.logit_max - result.logit_min) / 200 || 0.01;
    }
    state.threshold = typeof result.threshold === 'number' ? result.threshold : 0;
    slider.value = state.threshold;
    $('#mask-threshold-out').value = Number(state.threshold).toFixed(3);

    if (!state.reviewViews.length) {
      state.reviewViews = $$('#review-views .viewer').map((figure) => {
        const axis = Number(figure.dataset.axis);
        const view = new SliceCanvas(figure.querySelector('canvas'));
        const input = figure.querySelector('.slice-input');
        input.max = state.volume.display_shape[axis] - 1;
        view.on('wheel', guard((step) => setSlice(axis, state.slices[axis] + step)));
        input.addEventListener('change', guard(() => setSlice(axis, Number(input.value))));
        view.on('click', guard(async (point) => {
          if (!$('#edit-mode').checked) return;
          const response = await api.post(`/api/sessions/${state.sessionId}/mask/toggle`, {
            axis, index: state.slices[axis], x: point.x, y: point.y
          });
          $('#mask-stats').textContent = `${response.edits} manual voxel edit(s)`;
          await Promise.all([0, 1, 2].map(refreshReview));
        }));
        return { axis, view, figure, input };
      });
    }
    state.reviewViews.forEach((entry) => { entry.input.value = state.slices[entry.axis]; });
    await Promise.all([0, 1, 2].map(refreshReview));
    await refreshMaskStats();
  }

  async function refreshReview(axis) {
    const entry = state.reviewViews.find((v) => v.axis === axis);
    if (!entry) return;
    entry.figure.querySelector('.slice-label').textContent = state.slices[axis];
    await entry.view.setBase(sliceUrl(axis, state.slices[axis]));
    await entry.view.setOverlay(maskUrl(axis, state.slices[axis]), true);
  }

  async function refreshMaskStats() {
    const response = await api.post(`/api/sessions/${state.sessionId}/mask/threshold`, {
      threshold: state.threshold
    });
    $('#mask-stats').textContent = `${response.voxels.toLocaleString()} voxels above threshold`;
  }

  $('#mask-threshold').oninput = debounce(guard(async (event) => {
    state.threshold = Number(event.target.value);
    $('#mask-threshold-out').value = state.threshold.toFixed(3);
    await refreshMaskStats();
    await Promise.all([0, 1, 2].map(refreshReview));
  }), 150);

  $('#reset-edits').onclick = guard(async () => {
    await api.post(`/api/sessions/${state.sessionId}/mask/reset-edits`);
    $('#mask-stats').textContent = 'edits cleared';
    await Promise.all([0, 1, 2].map(refreshReview));
  });

  $('#save').onclick = guard(async () => {
    const response = await api.post(`/api/sessions/${state.sessionId}/save`);
    const times = response.times
      .map((t) => `${t.label}: ${t.seconds.toFixed(1)} s`).join(' | ');
    $('#save-result').innerHTML =
      `<p class="meta">Saved ${response.files.length} file(s) at ${response.timestamp}; `
      + `${response.voxels.toLocaleString()} voxels in the mask.<br>${times}</p>`;
    await refreshFiles();
  });

  async function refreshFiles() {
    if (!state.sessionId) return;
    const response = await api.get(`/api/sessions/${state.sessionId}/files`);
    const list = $('#file-list');
    list.innerHTML = '';
    response.files.forEach((file) => {
      const item = document.createElement('li');
      item.innerHTML = `<a href="${file.url}">${file.name}</a> `
        + `<span class="meta">${(file.size / 1024).toFixed(1)} kB</span>`;
      list.appendChild(item);
    });
    $('#files-panel').hidden = response.files.length === 0;
  }

  function debounce(fn, wait) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), wait);
    };
  }
})();
