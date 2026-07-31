// Utilities.py operations as forms.
(function () {
  'use strict';

  const api = window.SamuraiAPI;

  function renderResult(container, endpoint, result) {
    const rows = [];
    if (result.download) {
      rows.push(`<p><a href="${result.download}">Download ${result.output.split('/').pop()}</a></p>`);
    }
    if (typeof result.voxels === 'number') {
      rows.push(`<p class="meta">${result.voxels.toLocaleString()} non-zero voxels.</p>`);
    }
    if (result.inputs) {
      rows.push(`<p class="meta">Merged: ${result.inputs.join(', ')}</p>`);
    }
    if (result.rows) {
      rows.push('<table><thead><tr><th>File</th><th>Volume (mm³)</th></tr></thead><tbody>'
        + result.rows.map((r) => `<tr><td>${r.Filename}</td><td>${r['Volume (mm^3)'].toFixed(2)}</td></tr>`).join('')
        + '</tbody></table>'
        + `<p class="meta">mean ${result.mean.toFixed(2)} mm³, sd ${result.std.toFixed(2)} mm³</p>`);
    }
    if (result.pairs) {
      rows.push('<table><thead><tr><th>A</th><th>B</th><th>Cohen IRV</th><th>Dice</th></tr></thead><tbody>'
        + result.pairs.map((p) =>
          `<tr><td>${p.mask1}</td><td>${p.mask2}</td>`
          + `<td>${p.cohen_irv.toFixed(4)}</td><td>${p.dice.toFixed(4)}</td></tr>`).join('')
        + '</tbody></table>'
        + `<p class="meta">mean ${result.mean.toFixed(4)} (${result.interpretation}), `
        + `sd ${result.std.toFixed(4)}</p>`);
    }
    container.innerHTML = rows.join('') || '<p class="meta">Done.</p>';
  }

  document.querySelectorAll('form.utility').forEach((form) => {
    form.addEventListener('submit', guard(async (event) => {
      event.preventDefault();
      const button = form.querySelector('button[type=submit], button:not([type])');
      const container = form.parentElement.querySelector('.result');
      container.innerHTML = '<p class="meta">Working...</p>';
      if (button) button.disabled = true;
      try {
        const result = await api.upload(form.dataset.endpoint, new FormData(form));
        renderResult(container, form.dataset.endpoint, result);
      } catch (err) {
        container.innerHTML = '';
        throw err;
      } finally {
        if (button) button.disabled = false;
      }
    }));
  });
})();
