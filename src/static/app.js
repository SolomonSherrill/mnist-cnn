const SIZE = 400;
const BRUSH = 18;

const pad = document.getElementById('pad');
const chart = document.getElementById('chart');
const hint = document.getElementById('hint');
const verdict = document.getElementById('verdict');
const verdictConf = document.getElementById('verdict-conf');
const latency = document.getElementById('latency');

/* ---------- canvas ---------- */

const dpr = Math.min(window.devicePixelRatio || 1, 2);
pad.width = SIZE * dpr;
pad.height = SIZE * dpr;

const ctx = pad.getContext('2d', { alpha: false, willReadFrequently: false });
ctx.scale(dpr, dpr);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = BRUSH;
ctx.strokeStyle = '#000';

function wipe() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.restore();
}
wipe();

let drawing = false;
let inked = false;
let last = null;

function pointAt(e) {
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (SIZE / r.width), y: (e.clientY - r.top) * (SIZE / r.height) };
}

pad.addEventListener('pointerdown', (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  last = pointAt(e);
  // A tap with no movement should still leave a mark.
  ctx.beginPath();
  ctx.arc(last.x, last.y, BRUSH / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#000';
  ctx.fill();
  ink();
});

pad.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  for (const ev of events) {
    const p = pointAt(ev);
    ctx.lineTo(p.x, p.y);
    last = p;
  }
  ctx.stroke();
  ink();
});

function release(e) {
  if (!drawing) return;
  drawing = false;
  if (e && e.pointerId !== undefined && pad.hasPointerCapture(e.pointerId)) {
    pad.releasePointerCapture(e.pointerId);
  }
  schedule();
}

pad.addEventListener('pointerup', release);
pad.addEventListener('pointercancel', release);

function ink() {
  if (!inked) {
    inked = true;
    hint.classList.add('hidden');
  }
  schedule();
}

/* ---------- bars ---------- */

function span(cls, text) {
  const el = document.createElement('span');
  el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

const cols = [];
for (let d = 0; d < 10; d++) {
  const col = span('col dim');
  const val = span('val', '0.0%');
  const track = span('track');
  const fill = span('fill');
  track.appendChild(fill);
  col.append(val, track, span('digit', String(d)));
  chart.appendChild(col);
  cols.push({ root: col, val, fill });
}

function render(probs, top) {
  for (let d = 0; d < 10; d++) {
    const p = probs[d];
    const c = cols[d];
    c.fill.style.height = (p * 100).toFixed(2) + '%';
    c.val.textContent = (p * 100).toFixed(1) + '%';
    c.root.classList.toggle('max', d === top);
    c.root.classList.toggle('dim', p < 0.001);
  }
  if (top === null || top === undefined) {
    verdict.textContent = '—';
    verdict.classList.remove('live');
    verdictConf.textContent = 'awaiting input';
  } else {
    verdict.textContent = String(top);
    verdict.classList.add('live');
    verdictConf.textContent = (probs[top] * 100).toFixed(1) + '% confidence';
  }
}

render(new Array(10).fill(0), null);

/* ---------- inference: at most one request in flight ---------- */

let inFlight = false;
let dirty = false;
let frame = 0;

function schedule() {
  dirty = true;
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    pump();
  });
}

function snapshot() {
  return new Promise((resolve) => pad.toBlob(resolve, 'image/png'));
}

async function pump() {
  if (inFlight || !dirty) return;
  inFlight = true;
  dirty = false;
  const t0 = performance.now();
  try {
    const blob = await snapshot();
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'image/png' },
      body: blob,
    });
    const data = await res.json();
    render(data.probs, data.top);
    latency.textContent = Math.round(performance.now() - t0) + ' ms';
    latency.classList.add('on');
  } catch (err) {
    console.error(err);
  } finally {
    inFlight = false;
    // Coalesce: whatever arrived while we were busy gets one fresh pass.
    if (dirty) schedule();
  }
}

/* ---------- clear ---------- */

document.getElementById('clear').addEventListener('click', () => {
  wipe();
  inked = false;
  dirty = false;
  hint.classList.remove('hidden');
  latency.classList.remove('on');
  render(new Array(10).fill(0), null);
});
