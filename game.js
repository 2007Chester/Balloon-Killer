import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("webcam");
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const hintEl = document.getElementById("hint");
const overlay = document.getElementById("overlay");
const overlayStart = document.getElementById("overlay-start");
const readyOverlay = document.getElementById("ready-overlay");
const readyCalibrate = document.getElementById("ready-calibrate");
const calibOverlay = document.getElementById("calib-overlay");
const calibTitle = document.getElementById("calib-title");
const calibText = document.getElementById("calib-text");
const calibCapture = document.getElementById("calib-capture");
const calibProgress = document.getElementById("calib-progress");

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const LM = {
  INDEX_TIP: 8,
};

/** @type {'menu' | 'ready' | 'calib' | 'playing'} */
let gamePhase = "menu";
let handLandmarker = null;
let lastVideoTime = -1;
let score = 0;

let aimRaw = { x: null, y: null, visible: false };
let aim = { x: null, y: null, visible: false };

let shotCooldown = 0;
const SHOT_COOLDOWN_MS = 260;

/** Резкий «выстрел» — палец/прицел движется вверх по экрану (y уменьшается), пикс/мс */
const FLICK_V_UP = -1.05;
let flickLast = /** @type {{ t: number; y: number } | null} */ (null);
let flickPrevVy = 0;
let flickWarmup = 0;

/** Сглаживание прицела (уменьшает дрожание MediaPipe) */
let aimSmoothPlay = /** @type {{ x: number; y: number } | null} */ (null);
let aimSmoothCalib = /** @type {{ x: number; y: number } | null} */ (null);
const AIM_SMOOTH_PLAY_ALPHA = 0.2;
const AIM_SMOOTH_PLAY_MAXSTEP = 44;
const AIM_SMOOTH_CALIB_ALPHA = 0.3;
const AIM_SMOOTH_CALIB_MAXSTEP = 58;

function smoothFollow(prev, tx, ty, alpha, maxStep) {
  if (!prev) return { x: tx, y: ty };
  let cx = tx;
  let cy = ty;
  const dx = tx - prev.x;
  const dy = ty - prev.y;
  const d = Math.hypot(dx, dy);
  if (d > maxStep && d > 0) {
    const k = maxStep / d;
    cx = prev.x + dx * k;
    cy = prev.y + dy * k;
  }
  return {
    x: prev.x + (cx - prev.x) * alpha,
    y: prev.y + (cy - prev.y) * alpha,
  };
}

/** Масштаб и смещение: экран = raw * s + o */
let cal = { sx: 1, sy: 1, ox: 0, oy: 0 };

let calibStep = 0;
const calibSteps = [
  {
    fx: 0.5,
    fy: 0.5,
    title: "Шаг 1 из 3 — центр",
    text: "Наведите кончик указательного пальца на яркий круг в центре экрана. Когда совпадёт — нажмите «Зафиксировать».",
  },
  {
    fx: 0.5,
    fy: 0.82,
    title: "Шаг 2 из 3 — низ",
    text: "То же самое с кругом внизу по центру.",
  },
  {
    fx: 0.88,
    fy: 0.5,
    title: "Шаг 3 из 3 — справа",
    text: "Наведите палец на круг справа по центру и зафиксируйте.",
  },
];
/** @type {{ screen: { x: number; y: number }; raw: { x: number; y: number } }[]} */
let calibSamples = [];
/** @type {{ x: number; y: number }[]} */
let sampleBuffer = [];
const CALIB_BUFFER_MAX = 45;
const CALIB_MIN_FRAMES = 12;

const balloons = [];
let spawnAcc = 0;
const SPAWN_INTERVAL_MS = 900;

const palette = ["#ff4d6d", "#7cf5ff", "#c4ff4d", "#b388ff", "#ffb84d", "#4dff9e"];

const shotFx = [];
const SHOT_FX_MS = 200;

const particles = [];

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", resize);
resize();

function rand(a, b) {
  return a + Math.random() * (b - a);
}

function spawnBalloon() {
  const margin = 48;
  const r = rand(22, 38);
  balloons.push({
    x: rand(margin, window.innerWidth - margin),
    y: window.innerHeight + r + rand(0, 80),
    r,
    vx: rand(-0.35, 0.35),
    vy: rand(-1.8, -1.1),
    wobble: rand(0, Math.PI * 2),
    wobbleSpeed: rand(0.04, 0.09),
    knotDx: rand(-3, 3),
    color: palette[(Math.random() * palette.length) | 0],
  });
}

function dist(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return Math.hypot(dx, dy);
}

function addShotFx(x, y) {
  shotFx.push({ x, y, t: 0 });
}

function updateShotFx(dt) {
  for (let i = shotFx.length - 1; i >= 0; i--) {
    const s = shotFx[i];
    s.t += dt;
    if (s.t >= SHOT_FX_MS) shotFx.splice(i, 1);
  }
}

function drawShotFx() {
  for (const s of shotFx) {
    const k = s.t / SHOT_FX_MS;
    const alpha = (1 - k) * 0.95;
    const r = 8 + k * 120;
    ctx.strokeStyle = `rgba(255, 230, 120, ${alpha})`;
    ctx.lineWidth = 3 * (1 - k * 0.5);
    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = `rgba(124, 245, 255, ${alpha * 0.6})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(s.x, s.y, r * 0.45, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function tryShoot() {
  if (shotCooldown > 0 || aim.x == null || !aim.visible) return;
  shotCooldown = SHOT_COOLDOWN_MS;
  addShotFx(aim.x, aim.y);
  for (let i = balloons.length - 1; i >= 0; i--) {
    const b = balloons[i];
    if (dist(aim.x, aim.y, b.x, b.y) < b.r + 14) {
      balloons.splice(i, 1);
      score += 10;
      scoreEl.textContent = String(score);
      spawnParticles(b.x, b.y, b.color);
      return;
    }
  }
}

function spawnParticles(x, y, color) {
  const n = 12;
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n + rand(-0.2, 0.2);
    const sp = rand(2, 5);
    particles.push({
      x,
      y,
      vx: Math.cos(a) * sp,
      vy: Math.sin(a) * sp,
      life: 1,
      color,
    });
  }
}

function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.12;
    p.life -= dt * 0.002;
    if (p.life <= 0) particles.splice(i, 1);
  }
}

function drawParticles() {
  for (const p of particles) {
    ctx.globalAlpha = Math.max(0, p.life);
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawBalloon(b) {
  const g = ctx.createRadialGradient(b.x - b.r * 0.35, b.y - b.r * 0.35, b.r * 0.1, b.x, b.y, b.r);
  g.addColorStop(0, "#ffffffcc");
  g.addColorStop(0.35, b.color);
  g.addColorStop(1, b.color + "99");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.15)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.beginPath();
  ctx.moveTo(b.x, b.y + b.r);
  ctx.lineTo(b.x + b.knotDx, b.y + b.r + 18);
  ctx.stroke();
}

function drawCrosshairAt(x, y, stroke, fill) {
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  const s = 14;
  ctx.beginPath();
  ctx.moveTo(x - s, y);
  ctx.lineTo(x - 4, y);
  ctx.moveTo(x + 4, y);
  ctx.lineTo(x + s, y);
  ctx.moveTo(x, y - s);
  ctx.lineTo(x, y - 4);
  ctx.moveTo(x, y + 4);
  ctx.lineTo(x, y + s);
  ctx.stroke();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawCrosshair() {
  if (aim.x == null || !aim.visible) return;
  drawCrosshairAt(aim.x, aim.y, "rgba(255,255,255,0.85)", "rgba(124,245,255,0.35)");
}

function drawCalibTarget() {
  const step = calibSteps[calibStep];
  if (!step) return;
  const cx = step.fx * window.innerWidth;
  const cy = step.fy * window.innerHeight;
  const pulse = 0.85 + Math.sin(performance.now() * 0.006) * 0.15;
  const R = 40 * pulse;
  ctx.strokeStyle = "rgba(124, 245, 255, 0.95)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(124, 245, 255, 0.12)";
  ctx.beginPath();
  ctx.arc(cx, cy, R - 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.stroke();
}

function drawBackground() {
  const grd = ctx.createLinearGradient(0, 0, 0, window.innerHeight);
  grd.addColorStop(0, "#121826");
  grd.addColorStop(1, "#0a0d12");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  for (let i = 0; i < 40; i++) {
    const sx = ((i * 997) % window.innerWidth) | 0;
    const sy = ((i * 631 + performance.now() * 0.02) % window.innerHeight) | 0;
    ctx.fillRect(sx, sy, 2, 2);
  }
}

function rawIndexPixels(lm) {
  const ix = lm[LM.INDEX_TIP].x;
  const iy = lm[LM.INDEX_TIP].y;
  // X зеркалим как в селфи; Y инвертируем — у части камер/драйверов картинка «перевёрнута» по вертикали
  return {
    x: (1 - ix) * window.innerWidth,
    y: (1 - iy) * window.innerHeight,
  };
}

function applyCalib(raw) {
  return {
    x: raw.x * cal.sx + cal.ox,
    y: raw.y * cal.sy + cal.oy,
  };
}

function finalizeCalibration() {
  if (calibSamples.length !== 3) return;
  const [p0, p1, p2] = calibSamples;
  const w = window.innerWidth;
  const h = window.innerHeight;

  let sx = (p2.screen.x - p0.screen.x) / (p2.raw.x - p0.raw.x);
  if (!Number.isFinite(sx) || Math.abs(p2.raw.x - p0.raw.x) < w * 0.03) {
    sx = 1;
  }
  const ox = p0.screen.x - p0.raw.x * sx;

  let sy = (p1.screen.y - p0.screen.y) / (p1.raw.y - p0.raw.y);
  if (!Number.isFinite(sy) || Math.abs(p1.raw.y - p0.raw.y) < h * 0.03) {
    sy = 1;
  }
  const oy = p0.screen.y - p0.raw.y * sy;

  cal = { sx, sy, ox, oy };
}

function averageBuffer(buf) {
  if (buf.length === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const p of buf) {
    sx += p.x;
    sy += p.y;
  }
  const n = buf.length;
  return { x: sx / n, y: sy / n };
}

function syncCalibPanel() {
  const step = calibSteps[calibStep];
  if (!step) return;
  calibTitle.textContent = step.title;
  calibText.textContent = step.text;
  calibProgress.textContent = `Кадров в буфере: ${sampleBuffer.length} (нужно ≥${CALIB_MIN_FRAMES})`;
}

function showCalib() {
  calibOverlay.hidden = false;
  calibStep = 0;
  calibSamples = [];
  sampleBuffer = [];
  cal = { sx: 1, sy: 1, ox: 0, oy: 0 };
  aimSmoothCalib = null;
  syncCalibPanel();
}

function hideCalib() {
  calibOverlay.hidden = true;
}

function showReadyScreen() {
  readyOverlay.classList.add("visible");
}

function hideReadyScreen() {
  readyOverlay.classList.remove("visible");
}

async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  const opts = (delegate) => ({
    baseOptions: {
      modelAssetPath: MODEL_URL,
      ...(delegate ? { delegate } : {}),
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
  } catch {
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts());
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function processHands() {
  if (!handLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const result = handLandmarker.detectForVideo(video, performance.now());

  aim.visible = false;
  aimRaw.visible = false;

  if (!result.landmarks || result.landmarks.length === 0) {
    flickLast = null;
    flickPrevVy = 0;
    flickWarmup = 0;
    aimSmoothPlay = null;
    aimSmoothCalib = null;
    return;
  }

  const lm = result.landmarks[0];
  const raw = rawIndexPixels(lm);

  if (gamePhase === "calib") {
    aimSmoothCalib = smoothFollow(aimSmoothCalib, raw.x, raw.y, AIM_SMOOTH_CALIB_ALPHA, AIM_SMOOTH_CALIB_MAXSTEP);
    aimRaw.x = aimSmoothCalib.x;
    aimRaw.y = aimSmoothCalib.y;
    aimRaw.visible = true;
    sampleBuffer.push({ x: aimSmoothCalib.x, y: aimSmoothCalib.y });
    if (sampleBuffer.length > CALIB_BUFFER_MAX) sampleBuffer.shift();
    calibProgress.textContent = `Кадров в буфере: ${sampleBuffer.length} (нужно ≥${CALIB_MIN_FRAMES})`;
    return;
  }

  if (gamePhase === "playing") {
    const c = applyCalib(raw);
    const w = window.innerWidth;
    const h = window.innerHeight;

    aimSmoothPlay = smoothFollow(aimSmoothPlay, c.x, c.y, AIM_SMOOTH_PLAY_ALPHA, AIM_SMOOTH_PLAY_MAXSTEP);
    aim.x = Math.max(0, Math.min(w, aimSmoothPlay.x));
    aim.y = Math.max(0, Math.min(h, aimSmoothPlay.y));
    aim.visible = true;

    const now = performance.now();
    flickWarmup += 1;
    if (flickLast) {
      const dt = now - flickLast.t;
      if (dt >= 6 && dt <= 100) {
        const vy = (c.y - flickLast.y) / dt;
        if (flickWarmup >= 5 && vy < FLICK_V_UP && flickPrevVy > FLICK_V_UP) {
          tryShoot();
        }
        flickPrevVy = vy;
      } else if (dt > 100) {
        flickPrevVy = 0;
      }
    }
    flickLast = { t: now, y: c.y };
  }
}

let lastT = performance.now();
let rafStarted = false;

function frame(now) {
  const dt = now - lastT;
  lastT = now;
  if (shotCooldown > 0) shotCooldown = Math.max(0, shotCooldown - dt);

  processHands();
  updateShotFx(dt);

  if (gamePhase === "playing") {
    spawnAcc += dt;
    while (spawnAcc >= SPAWN_INTERVAL_MS) {
      spawnAcc -= SPAWN_INTERVAL_MS;
      spawnBalloon();
    }
    for (const b of balloons) {
      b.wobble += b.wobbleSpeed;
      b.x += b.vx + Math.sin(b.wobble) * 0.4;
      b.y += b.vy;
    }
    for (let i = balloons.length - 1; i >= 0; i--) {
      if (balloons[i].y < -80) balloons.splice(i, 1);
    }
  }

  updateParticles(dt);

  drawBackground();

  if (gamePhase === "playing") {
    for (const b of balloons) drawBalloon(b);
    drawParticles();
    drawShotFx();
    drawCrosshair();
  } else if (gamePhase === "calib") {
    drawCalibTarget();
    if (aimRaw.visible) {
      drawCrosshairAt(aimRaw.x, aimRaw.y, "rgba(255, 190, 100, 0.9)", "rgba(255, 140, 60, 0.25)");
    }
  } else if (gamePhase === "ready") {
    drawBackground();
  }

  requestAnimationFrame(frame);
}

function startLoop() {
  if (rafStarted) return;
  rafStarted = true;
  lastT = performance.now();
  requestAnimationFrame(frame);
}

overlayStart.addEventListener("click", async () => {
  overlayStart.disabled = true;
  hintEl.textContent = "Загрузка модели…";
  try {
    await initHandLandmarker();
    await startCamera();
    overlay.classList.remove("visible");
    hideCalib();
    gamePhase = "ready";
    showReadyScreen();
    hintEl.textContent = "Нажмите «Калибровать», затем наведите палец на три круга.";
    startLoop();
  } catch (e) {
    console.error(e);
    hintEl.textContent =
      "Не удалось запустить камеру или модель. Проверьте разрешения и HTTPS / localhost.";
    overlayStart.disabled = false;
  }
});

readyCalibrate.addEventListener("click", () => {
  if (gamePhase !== "ready") return;
  hideReadyScreen();
  gamePhase = "calib";
  showCalib();
  hintEl.textContent = "Калибровка: наведите указательный палец на круг и нажмите «Зафиксировать».";
});

calibCapture.addEventListener("click", () => {
  if (gamePhase !== "calib") return;
  if (sampleBuffer.length < CALIB_MIN_FRAMES) {
    calibProgress.textContent = `Мало кадров (${sampleBuffer.length}). Держите руку в кадре и наведите палец на круг.`;
    return;
  }
  const avg = averageBuffer(sampleBuffer);
  if (!avg) return;
  const step = calibSteps[calibStep];
  const screen = { x: step.fx * window.innerWidth, y: step.fy * window.innerHeight };
  calibSamples.push({ screen, raw: { x: avg.x, y: avg.y } });
  sampleBuffer = [];
  calibStep += 1;
  if (calibStep >= calibSteps.length) {
    finalizeCalibration();
    gamePhase = "playing";
    flickLast = null;
    flickPrevVy = 0;
    flickWarmup = 0;
    aimSmoothPlay = null;
    hideCalib();
    hintEl.textContent = "Прицел — указательный. Выстрел — резко поднимите палец вверх. Пробел — выстрел.";
    scoreEl.textContent = String(score);
  } else {
    syncCalibPanel();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && gamePhase === "playing" && aim.visible) {
    e.preventDefault();
    tryShoot();
  }
});
