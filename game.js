import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("webcam");
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const hintEl = document.getElementById("hint");
const overlay = document.getElementById("overlay");
const overlayStart = document.getElementById("overlay-start");

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const LM = {
  THUMB_TIP: 4,
  INDEX_TIP: 8,
};

let handLandmarker = null;
let lastVideoTime = -1;
let score = 0;
let aim = { x: null, y: null, visible: false };
let pinchPrev = false;
let shotCooldown = 0;
const PINCH_THRESHOLD = 0.055;
const SHOT_COOLDOWN_MS = 220;

const balloons = [];
let spawnAcc = 0;
const SPAWN_INTERVAL_MS = 900;

const palette = ["#ff4d6d", "#7cf5ff", "#c4ff4d", "#b388ff", "#ffb84d", "#4dff9e"];

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

function tryShoot() {
  if (shotCooldown > 0 || aim.x == null || !aim.visible) return;
  shotCooldown = SHOT_COOLDOWN_MS;
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

const particles = [];

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

function drawCrosshair() {
  if (aim.x == null || !aim.visible) return;
  const { x, y } = aim;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
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
  ctx.fillStyle = "rgba(124,245,255,0.35)";
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fill();
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

function mapHandToScreen(lm) {
  const ix = lm[LM.INDEX_TIP].x;
  const iy = lm[LM.INDEX_TIP].y;
  const tx = lm[LM.THUMB_TIP].x;
  const ty = lm[LM.THUMB_TIP].y;
  const mirroredX = 1 - ix;
  const mirroredTx = 1 - tx;
  return {
    ax: mirroredX * window.innerWidth,
    ay: iy * window.innerHeight,
    pinchDist: Math.hypot(mirroredX - mirroredTx, iy - ty),
  };
}

function processHands() {
  if (!handLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const result = handLandmarker.detectForVideo(video, performance.now());
  aim.visible = false;
  if (result.landmarks && result.landmarks.length > 0) {
    const { ax, ay, pinchDist } = mapHandToScreen(result.landmarks[0]);
    aim.x = ax;
    aim.y = ay;
    aim.visible = true;
    const pinched = pinchDist < PINCH_THRESHOLD;
    if (pinched && !pinchPrev) tryShoot();
    pinchPrev = pinched;
  } else {
    pinchPrev = false;
  }
}

let lastT = performance.now();

function frame(now) {
  const dt = now - lastT;
  lastT = now;
  if (shotCooldown > 0) shotCooldown = Math.max(0, shotCooldown - dt);

  processHands();

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

  updateParticles(dt);

  drawBackground();
  for (const b of balloons) drawBalloon(b);
  drawParticles();
  drawCrosshair();

  requestAnimationFrame(frame);
}

overlayStart.addEventListener("click", async () => {
  overlayStart.disabled = true;
  hintEl.textContent = "Загрузка модели распознавания руки…";
  try {
    await initHandLandmarker();
    await startCamera();
    overlay.classList.remove("visible");
    hintEl.textContent = "Указательный палец — прицел. Щипок — выстрел.";
    requestAnimationFrame(frame);
  } catch (e) {
    console.error(e);
    hintEl.textContent =
      "Не удалось запустить камеру или модель. Проверьте разрешения и попробуйте Chrome по HTTPS или localhost.";
    overlayStart.disabled = false;
  }
});

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && aim.visible) {
    e.preventDefault();
    tryShoot();
  }
});
