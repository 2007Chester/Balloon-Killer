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
const calibProgress = document.getElementById("calib-progress");

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const LM = {
  WRIST: 0,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
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

/** Выстрел: опустить большой палец вниз */
const THUMB_DOWN_ON = 0.02;
const THUMB_DOWN_OFF = -0.002;
let thumbDownLatch = false;
let thumbWasDownShot = false;

/** Сглаживание прицела (уменьшает дрожание MediaPipe) */
let aimSmoothPlay = /** @type {{ x: number; y: number } | null} */ (null);
let aimSmoothCalib = /** @type {{ x: number; y: number } | null} */ (null);
/** Медиана по последним кадрам после калибровки — убирает выбросы в стороны */
const AIM_MEDIAN_LEN = 4;
/** @type {{ x: number; y: number }[]} */
let aimMedianBuf = [];

const AIM_PLAY_AX = 0.22;
const AIM_PLAY_AY = 0.24;
const AIM_PLAY_MX = 36;
const AIM_PLAY_MY = 44;
const AIM_CALIB_AX = 0.24;
const AIM_CALIB_AY = 0.28;
const AIM_CALIB_MX = 34;
const AIM_CALIB_MY = 44;

/** Один кадр жеста «большой палец вниз»; обновляет защёлку */
function updateThumbShotGesture(lm) {
  const tip = lm[LM.THUMB_TIP];
  const mcp = lm[LM.THUMB_MCP];
  const wrist = lm[LM.WRIST];
  const yDelta = tip.y - mcp.y;
  const thumbVisible = Math.hypot(tip.x - wrist.x, tip.y - wrist.y) > 0.05;
  if (!thumbVisible) {
    thumbDownLatch = false;
    thumbWasDownShot = false;
    return false;
  }
  if (yDelta > THUMB_DOWN_ON) thumbDownLatch = true;
  else if (yDelta < THUMB_DOWN_OFF) thumbDownLatch = false;
  const isDown = thumbDownLatch;
  const fire = isDown && !thumbWasDownShot;
  thumbWasDownShot = isDown;
  return fire;
}

function smoothFollowAxis(prev, tx, ty, alphaX, alphaY, maxStepX, maxStepY) {
  if (!prev) return { x: tx, y: ty };
  let cx = tx;
  let cy = ty;
  const dx = tx - prev.x;
  const dy = ty - prev.y;
  if (Math.abs(dx) > maxStepX) cx = prev.x + Math.sign(dx) * maxStepX;
  if (Math.abs(dy) > maxStepY) cy = prev.y + Math.sign(dy) * maxStepY;
  return {
    x: prev.x + (cx - prev.x) * alphaX,
    y: prev.y + (cy - prev.y) * alphaY,
  };
}

function median1d(values) {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = (s.length - 1) >> 1;
  return s.length % 2 ? s[m] : (s[m] + s[m + 1]) / 2;
}

function pushAimMedian(p) {
  aimMedianBuf.push({ x: p.x, y: p.y });
  while (aimMedianBuf.length > AIM_MEDIAN_LEN) aimMedianBuf.shift();
}

function medianAimPoint() {
  if (aimMedianBuf.length === 0) return null;
  return {
    x: median1d(aimMedianBuf.map((q) => q.x)),
    y: median1d(aimMedianBuf.map((q) => q.y)),
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
    text: "Наведите оранжевый прицел на голубой круг и выстрелите: опустите большой палец вниз.",
  },
  {
    fx: 0.5,
    fy: 0.82,
    title: "Шаг 2 из 3 — низ",
    text: "То же самое с нижним кругом.",
  },
  {
    fx: 0.88,
    fy: 0.5,
    title: "Шаг 3 из 3 — справа",
    text: "То же самое с правым кругом.",
  },
];
/** Радиус попадания прицела в метку (пиксели), круг на экране ~40px */
const CALIB_HIT_R = 54;
/** @type {{ screen: { x: number; y: number }; raw: { x: number; y: number } }[]} */
let calibSamples = [];

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

/**
 * Точка прицела: смесь TIP + DIP + PIP.
 * X по умолчанию без горизонтального зеркала: x = (1-ix)*ширина.
 * Y по умолчанию без зеркала: y = iy * высота (как в кадре MediaPipe, вниз = вниз).
 * Если горизонталь наоборот — добавьте в URL ?flipX=1 (будет ix*ширина).
 * Если вертикаль наоборот — добавьте в URL ?flipY=1 (будет (1−iy)*высота).
 */
const AIM_FLIP_X = new URLSearchParams(location.search).get("flipX") === "1";
const AIM_FLIP_Y = new URLSearchParams(location.search).get("flipY") === "1";

function rawIndexPixels(lm) {
  const t = lm[LM.INDEX_TIP];
  const d = lm[LM.INDEX_DIP];
  const p = lm[LM.INDEX_PIP];
  const wT = 0.62;
  const wD = 0.24;
  const wP = 0.14;
  const ix = wT * t.x + wD * d.x + wP * p.x;
  const iy = wT * t.y + wD * d.y + wP * p.y;
  const nx = AIM_FLIP_X ? ix : 1 - ix;
  const ny = AIM_FLIP_Y ? 1 - iy : iy;
  return {
    x: nx * window.innerWidth,
    y: ny * window.innerHeight,
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

function syncCalibPanel() {
  const step = calibSteps[calibStep];
  if (!step) return;
  calibTitle.textContent = step.title;
  calibText.textContent = step.text;
  calibProgress.textContent = "Пробел — выстрел с клавиатуры, если прицел на круге.";
}

function calibTargetScreen() {
  const step = calibSteps[calibStep];
  const w = window.innerWidth;
  const h = window.innerHeight;
  return { x: step.fx * w, y: step.fy * h };
}

function finishCalibAndPlay() {
  finalizeCalibration();
  gamePhase = "playing";
  thumbDownLatch = false;
  thumbWasDownShot = false;
  aimSmoothPlay = null;
  aimMedianBuf.length = 0;
  hideCalib();
  hintEl.textContent = "Прицел — указательный. Выстрел — опустите большой палец вниз. Пробел — выстрел.";
  scoreEl.textContent = String(score);
}

function captureCalibSample() {
  if (gamePhase !== "calib" || !aimRaw.visible) return;
  if (shotCooldown > 0) return;
  const { x: tcx, y: tcy } = calibTargetScreen();
  if (dist(aimRaw.x, aimRaw.y, tcx, tcy) > CALIB_HIT_R) return;

  shotCooldown = 320;
  const step = calibSteps[calibStep];
  const w = window.innerWidth;
  const h = window.innerHeight;
  const screen = { x: step.fx * w, y: step.fy * h };
  const raw = { x: aimSmoothCalib.x, y: aimSmoothCalib.y };
  calibSamples.push({ screen, raw });
  addShotFx(aimRaw.x, aimRaw.y);
  spawnParticles(screen.x, screen.y, "#7cf5ff");

  calibStep += 1;
  if (calibStep >= calibSteps.length) {
    finishCalibAndPlay();
  } else {
    syncCalibPanel();
    calibProgress.textContent = "Попадание! Следующая метка — снова наведите и выстрелите.";
  }
}

function tryCalibShootOrMiss() {
  if (gamePhase !== "calib" || !aimRaw.visible) return;
  const { x: tcx, y: tcy } = calibTargetScreen();
  if (dist(aimRaw.x, aimRaw.y, tcx, tcy) <= CALIB_HIT_R) {
    captureCalibSample();
  } else {
    if (shotCooldown <= 0) {
      shotCooldown = 200;
      addShotFx(aimRaw.x, aimRaw.y);
    }
    calibProgress.textContent = "Мимо — наведите прицел на голубой круг и выстрельте ещё раз.";
  }
}

function tryCalibSpaceCapture() {
  if (gamePhase !== "calib" || !aimRaw.visible || shotCooldown > 0) return;
  const { x: tcx, y: tcy } = calibTargetScreen();
  if (dist(aimRaw.x, aimRaw.y, tcx, tcy) <= CALIB_HIT_R) captureCalibSample();
  else {
    shotCooldown = 180;
    addShotFx(aimRaw.x, aimRaw.y);
    calibProgress.textContent = "Сначала наведите прицел на круг.";
  }
}

function showCalib() {
  calibOverlay.hidden = false;
  calibStep = 0;
  calibSamples = [];
  cal = { sx: 1, sy: 1, ox: 0, oy: 0 };
  aimSmoothCalib = null;
  aimMedianBuf.length = 0;
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
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.62,
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
    thumbDownLatch = false;
    thumbWasDownShot = false;
    aimSmoothPlay = null;
    aimSmoothCalib = null;
    aimMedianBuf.length = 0;
    return;
  }

  const lm = result.landmarks[0];
  const raw = rawIndexPixels(lm);

  if (gamePhase === "calib") {
    aimSmoothCalib = smoothFollowAxis(
      aimSmoothCalib,
      raw.x,
      raw.y,
      AIM_CALIB_AX,
      AIM_CALIB_AY,
      AIM_CALIB_MX,
      AIM_CALIB_MY,
    );
    aimRaw.x = aimSmoothCalib.x;
    aimRaw.y = aimSmoothCalib.y;
    aimRaw.visible = true;
    aim.x = aimRaw.x;
    aim.y = aimRaw.y;
    aim.visible = true;

    if (updateThumbShotGesture(lm)) tryCalibShootOrMiss();
    return;
  }

  if (gamePhase === "playing") {
    const c = applyCalib(raw);
    pushAimMedian(c);
    const cStable = medianAimPoint() ?? c;
    const w = window.innerWidth;
    const h = window.innerHeight;

    aimSmoothPlay = smoothFollowAxis(
      aimSmoothPlay,
      cStable.x,
      cStable.y,
      AIM_PLAY_AX,
      AIM_PLAY_AY,
      AIM_PLAY_MX,
      AIM_PLAY_MY,
    );
    aim.x = Math.max(0, Math.min(w, aimSmoothPlay.x));
    aim.y = Math.max(0, Math.min(h, aimSmoothPlay.y));
    aim.visible = true;

    if (updateThumbShotGesture(lm)) tryShoot();
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
    drawShotFx();
    drawParticles();
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
    hintEl.textContent = "Нажмите «Калибровать», затем три раза: прицел на круг — выстрел жестом.";
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
  hintEl.textContent = "Калибровка: наведите прицел на круг и выстрелите большим пальцем (или пробел на круге).";
});

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  if (gamePhase === "playing" && aim.visible) {
    e.preventDefault();
    tryShoot();
  } else if (gamePhase === "calib") {
    e.preventDefault();
    tryCalibSpaceCapture();
  }
});
