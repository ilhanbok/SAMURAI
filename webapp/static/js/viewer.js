// Canvas slice viewer: base slice PNG + optional overlay PNG + prompt markers.
// Replaces the matplotlib figure the desktop scripts open, including its
// scroll-to-change-slice and click-to-add-point behaviour.
(function (global) {
  'use strict';

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not load ' + url));
      img.src = url;
    });
  }

  class SliceCanvas {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.base = null;
      this.overlay = null;
      this.markers = [];
      this.scatter = null;
      this.box = null;
      this.pendingBox = null;
      this.handlers = { click: [], wheel: [], box: [] };
      this._bind();
    }

    _bind() {
      const canvas = this.canvas;

      canvas.addEventListener('wheel', (event) => {
        if (!this.handlers.wheel.length) return;
        event.preventDefault();
        const step = event.deltaY < 0 ? 1 : -1;
        this.handlers.wheel.forEach((cb) => cb(step, event));
      }, { passive: false });

      canvas.addEventListener('mousedown', (event) => {
        if (!this.boxMode) return;
        const start = this.toImage(event);
        this.pendingBox = [start.x, start.y, start.x, start.y];
        this.redraw();
      });

      canvas.addEventListener('mousemove', (event) => {
        if (!this.boxMode || !this.pendingBox) return;
        const point = this.toImage(event);
        this.pendingBox[2] = point.x;
        this.pendingBox[3] = point.y;
        this.redraw();
      });

      canvas.addEventListener('mouseup', (event) => {
        if (this.boxMode && this.pendingBox) {
          const point = this.toImage(event);
          this.pendingBox[2] = point.x;
          this.pendingBox[3] = point.y;
          this.box = this.pendingBox.slice();
          this.pendingBox = null;
          this.redraw();
          this.handlers.box.forEach((cb) => cb(this.box));
          return;
        }
        const point = this.toImage(event);
        this.handlers.click.forEach((cb) => cb(point, event));
      });
    }

    on(event, callback) {
      this.handlers[event].push(callback);
      return this;
    }

    toImage(event) {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
      };
    }

    async setBase(url, bust) {
      this.base = await loadImage(bust ? url + (url.includes('?') ? '&' : '?') + '_=' + Date.now() : url);
      this.canvas.width = this.base.naturalWidth;
      this.canvas.height = this.base.naturalHeight;
      this.redraw();
    }

    async setOverlay(url, bust) {
      if (!url) { this.overlay = null; this.redraw(); return; }
      this.overlay = await loadImage(bust ? url + (url.includes('?') ? '&' : '?') + '_=' + Date.now() : url);
      this.redraw();
    }

    setMarkers(markers) { this.markers = markers || []; this.redraw(); }
    setScatter(points) { this.scatter = points || null; this.redraw(); }
    setBox(box) { this.box = box || null; this.redraw(); }

    redraw() {
      const ctx = this.ctx;
      if (!this.base) return;
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.drawImage(this.base, 0, 0);
      if (this.overlay) {
        ctx.drawImage(this.overlay, 0, 0, this.canvas.width, this.canvas.height);
      }
      if (this.scatter) {
        ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
        for (const [x, y] of this.scatter) ctx.fillRect(x, y, 1, 1);
      }
      const box = this.pendingBox || this.box;
      if (box) {
        ctx.strokeStyle = '#2fbf71';
        ctx.lineWidth = Math.max(1, this.canvas.width / 250);
        ctx.strokeRect(
          Math.min(box[0], box[2]), Math.min(box[1], box[3]),
          Math.abs(box[2] - box[0]), Math.abs(box[3] - box[1])
        );
      }
      const radius = Math.max(2.5, this.canvas.width / 90);
      for (const marker of this.markers) {
        // green = intra-lesional (fg:1), red = extra-lesional (bg:0),
        // matching show_points() in the desktop scripts
        ctx.beginPath();
        ctx.arc(marker.x, marker.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = Number(marker.label) === 1 ? '#38d17a' : '#ff4d4d';
        ctx.fill();
        ctx.lineWidth = Math.max(1, radius / 3);
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
    }
  }

  global.SliceCanvas = SliceCanvas;
})(window);
