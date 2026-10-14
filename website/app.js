/* redact-tool landing — animations & simulated demos */
(function () {
  "use strict";
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  /* ---------- cursor glow ---------- */
  const glow = $("#cursorGlow");
  let gx = innerWidth / 2, gy = 200, tx = gx, ty = gy;
  addEventListener("pointermove", (e) => { tx = e.clientX; ty = e.clientY; });
  (function glowLoop() {
    gx += (tx - gx) * 0.08; gy += (ty - gy) * 0.08;
    if (glow) glow.style.transform = `translate(${gx - 260}px,${gy - 260}px)`;
    requestAnimationFrame(glowLoop);
  })();

  /* ---------- mobile nav ---------- */
  const burger = $("#burger"), navMobile = $("#navMobile");
  if (burger) burger.addEventListener("click", () => navMobile.classList.toggle("open"));
  $$("#navMobile a").forEach((a) => a.addEventListener("click", () => navMobile.classList.remove("open")));

  /* ---------- scroll reveal ---------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.12 });
  $$(".reveal").forEach((el, i) => { el.style.transitionDelay = (i % 4) * 60 + "ms"; io.observe(el); });

  /* ---------- animated counters ---------- */
  const cio = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const el = e.target, target = +el.dataset.count;
      const t0 = performance.now(), dur = 1400;
      (function tick(t) {
        const p = Math.min(1, (t - t0) / dur), ease = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * ease) + (target === 100 ? "" : "");
        if (p < 1) requestAnimationFrame(tick); else el.textContent = target + (target === 100 ? "" : "");
      })(t0);
      cio.unobserve(el);
    });
  }, { threshold: 0.5 });
  $$("[data-count]").forEach((el) => cio.observe(el));

  /* ---------- hero typed command ---------- */
  const cmds = [
    'python redact-tool.py inspect --video leak.mp4 --out inv.json  # eyes for a blind model',
    'python redact-tool.py render --video in.mp4 --out safe.mp4 --spec pii.json',
    'python redact-tool.py verify --video safe.mp4 --spec pii.json --source in.mp4  # exit 0 ✓',
  ];
  const typedEl = $("#typedCmd");
  let ci = 0, ch = 0, del = false;
  if (typedEl) (function type() {
    const full = cmds[ci];
    typedEl.textContent = full.slice(0, ch);
    if (!del) { ch++; if (ch > full.length + 14) del = true; }
    else { ch -= 2; if (ch <= 0) { ch = 0; del = false; ci = (ci + 1) % cmds.length; } }
    setTimeout(type, del ? 22 : 42);
  })();

  /* ---------- marquee seamless ---------- */
  const mt = $("#marqueeTrack");
  if (mt) mt.innerHTML += mt.innerHTML;

  /* ---------- HERO PLAYER simulation ---------- */
  const stage = $("#playerStage");
  const person = $("#scenePerson"), tbFace = $("#tbFace");
  const wc1 = $("#wc1"), wc2 = $("#wc2"), tbWord = $("#tbWord"), tbEmail = $("#tbEmail");
  const hudF = $("#hudFrame"), hudT = $("#hudTime");
  const tlp = $("#tlProgress"), tlpH = $("#tlPlayhead"), tlCur = $("#tlCur");
  const heroBtn = $("#heroPlayBtn");
  let heroPlaying = true, heroT = 0;
  const FPS = 30, DUR = 32;
  if (heroBtn) heroBtn.addEventListener("click", () => {
    heroPlaying = !heroPlaying;
    heroBtn.textContent = heroPlaying ? "❚❚ pause" : "▶ play";
  });
  function heroFrame(dt) {
    if (heroPlaying && stage) {
      heroT = (heroT + dt) % DUR;
      const t = heroT, W = stage.clientWidth, H = stage.clientHeight;
      // slow natural drift: gentle walk across frame, subtle bob, faint zoom
      const px = W * (0.08 + 0.30 * (t / DUR)) + Math.sin(t * 0.45) * 6;
      const py = 34 + Math.abs(Math.sin(t * 0.55)) * -5;
      const sc = 1 + 0.12 * (t / DUR); // gentle 1.0 → 1.12 drift-zoom
      person.style.transform = `translate(${px - 60}px,${-py * 0.15}px) scale(${sc})`;
      person.style.left = "0";
      // face box rides the head
      const fx = px - 46 * sc + 8, fy = H * 0.52 - 66 * sc + Math.sin(t * 0.45) * 3;
      tbFace.style.left = fx + "px"; tbFace.style.top = fy + "px";
      tbFace.style.width = 92 * sc + "px"; tbFace.style.height = 102 * sc + "px";
      // word chips drift slowly (scrolling doc); masks shadow each PII chip
      const w2x = W * 0.60 + Math.sin(t * 0.35) * 12 - (t / DUR) * 30;
      const w2y = H * 0.62 + Math.cos(t * 0.45) * 6;
      wc2.style.left = w2x + "px"; wc2.style.top = w2y + "px";
      const w1x = W * 0.52 + Math.cos(t * 0.3) * 10;
      wc1.style.left = w1x + "px";
      tbWord.style.left = (w2x - 10) + "px"; tbWord.style.top = (w2y - 6) + "px";
      tbWord.style.width = (wc2.offsetWidth + 20) + "px";
      tbWord.style.height = (wc2.offsetHeight + 12) + "px";
      if (tbEmail) {
        tbEmail.style.left = (w1x - 10) + "px"; tbEmail.style.top = (wc1.offsetTop - 6) + "px";
        tbEmail.style.width = (wc1.offsetWidth + 20) + "px";
        tbEmail.style.height = (wc1.offsetHeight + 12) + "px";
      }
      // HUD + timeline
      const fr = Math.floor(t * FPS);
      hudF.textContent = String(fr).padStart(4, "0");
      hudT.textContent = t.toFixed(2) + "s";
      const pct = (t / DUR) * 100;
      tlp.style.width = pct + "%"; tlpH.style.left = pct + "%";
      tlCur.textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    }
  }
  // single rAF driver — one tick per frame (never schedule inside heroFrame)
  let lastH = performance.now();
  (function loopH(now) {
    const dt = Math.min(0.1, (now - lastH) / 1000);
    lastH = now;
    heroFrame(dt);
    requestAnimationFrame(loopH);
  })(lastH);

  /* ---------- DEMO stage ---------- */
  const dActor = $("#demoActor"), dFace = $("#demoFaceBox"), dDoc = $("#demoDoc"), dText = $("#demoTextBox");
  const dBy = document.querySelector(".demo-bystander");
  const dPlay = $("#demoPlay"), dLose = $("#demoLose"), dLog = $("#demoLog"), dToast = $("#demoToast");
  const repF = $("#repFace"), repT = $("#repText");
  let dPlaying = true, dT = 0, dLost = false, fCount = 0, wCount = 0;
  const modes = { face: "blur", text: "black" };
  function log(msg, cls) {
    if (!dLog) return;
    const d = document.createElement("div");
    if (cls) d.className = cls;
    d.textContent = msg;
    dLog.appendChild(d); dLog.scrollTop = dLog.scrollHeight;
    while (dLog.children.length > 40) dLog.removeChild(dLog.firstChild);
  }
  function fillFor(mode) {
    return mode === "black" ? "mode-black-fill" : mode === "blur" ? "mode-blur-fill" : "mode-pixelate-fill";
  }
  function paintDemoBoxes() {
    [["face", dFace], ["text", dText]].forEach(([k, box]) => {
      if (!box) return;
      box.className = "demo-box " + (k === "face" ? "demo-face" : "demo-text") + " mode-" + modes[k];
      let f = box.querySelector(".fill");
      if (!f) { f = document.createElement("div"); f.className = "fill"; box.appendChild(f); }
      f.className = "fill " + fillFor(modes[k]);
      const lab = box.querySelector("label span");
      if (lab) lab.textContent = modes[k];
    });
    const fm = $("#demoFaceMode"), tm = $("#demoTextMode");
    if (fm) fm.textContent = modes.face;
    if (tm) tm.textContent = modes.text;
  }
  $$(".seg").forEach((seg) => {
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", seg).forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      modes[seg.dataset.target] = b.dataset.mode;
      paintDemoBoxes();
      log(`mode → ${seg.dataset.target} = ${b.dataset.mode}`, "ok");
    });
  });
  if (dPlay) dPlay.addEventListener("click", () => {
    dPlaying = !dPlaying; dPlay.textContent = dPlaying ? "❚❚ pause" : "▶ play";
  });
  if (dLose) dLose.addEventListener("click", () => {
    dLost = true;
    log("⚠ subject left frame — appearance check failed → STOP_TRACKING (no drift)", "warn");
    if (dToast) dToast.textContent = "track lost → stopped · bystander untouched ✓";
    setTimeout(() => { dLost = false; log("↩ new anchor needed for re-entry (lost stays lost)", "warn"); }, 3500);
  });
  paintDemoBoxes();
  log("inspect → 2 faces, 14 PII words (emails, plates) as JSON — model sees text only", "ok");
  log('plan → subject face @ frame 0 + regex PII `\\S+@\\S+`, plates', "");
  let lastD = performance.now(), logTick = 0;
  (function loopD(now) {
    const dt = Math.min(0.05, (now - lastD) / 1000); lastD = now;
    if (dPlaying && dActor) {
      dT += dt;
      const st = $("#demoStage"), W = st.clientWidth;
      const sr = st.getBoundingClientRect();
      const span = Math.max(0, Math.min(W * 0.34,
        (dBy ? dBy.getBoundingClientRect().left - sr.left : W) - dActor.offsetWidth - 20));
      const ax = 8 + span * (0.5 + 0.5 * Math.sin(dT * 0.22));
      dActor.style.left = ax + "px";
      dActor.style.transform = `rotate(${Math.sin(dT * 0.45) * 1.5}deg)`;
      dDoc.style.transform = `translateX(calc(-50% + ${Math.sin(dT * 0.35) * 24}px))`;
      // masks measured from live layout so they sit exactly on target at any size
      const ar = dActor.getBoundingClientRect();
      const FW = 112, FH = 126;
      dFace.style.width = FW + "px"; dFace.style.height = FH + "px";
      dFace.style.left = (ar.left - sr.left + ar.width / 2 - FW / 2) + "px";
      dFace.style.top = (ar.top - sr.top + 14) + "px";
      dFace.style.display = dLost ? "none" : "block";
      const dr = dDoc.getBoundingClientRect();
      dText.style.left = (dr.left - sr.left - 8) + "px";
      dText.style.top = (dr.top - sr.top - 6) + "px";
      dText.style.width = (dr.width + 16) + "px";
      if (!dLost) {
        fCount += dt * 30; wCount += dt * 60; // 30fps video clock, not rAF speed
        if (repF) repF.textContent = Math.floor(fCount) + " frames";
        if (repT) repT.textContent = Math.floor(wCount) + " words";
        if (dToast) dToast.textContent = `tracker locked · frame ${Math.floor(fCount)} · ${modes.face}/${modes.text}`;
      }
      logTick += dt;
      if (logTick > 2.4) {
        logTick = 0;
        const msgs = [
          ["CSRT ✓ + face overlap ✓ — bystander ignored", "ok"],
          ["teleport candidate rejected (>60px/frame)", "warn"],
          ["word boxes clipped vs track rect — no bleed into face", ""],
          ["anchor still recognises target — model may keep learning", "ok"],
        ];
        const m = msgs[Math.floor(Math.random() * msgs.length)];
        log(m[0], m[1]);
      }
    }
    requestAnimationFrame(loopD);
  })(lastD);

  /* ---------- compare slider ---------- */
  const cr = $("#compareRange"), ca = $("#compareAfter"), chh = $("#compareHandle");
  function setCompare(v) {
    // wipe reveal: after-layer stays full-size, clip-path does the cutting
    ca.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    chh.style.left = v + "%";
  }
  if (cr) {
    setCompare(+cr.value);
    cr.addEventListener("input", () => setCompare(+cr.value));
  }

  /* ---------- spec tabs ---------- */
  const specs = {
    full: {
      note: "blind spec: face track + PII regex mask",
      raw: `{\n  "version": 1,\n  "ocr": { "engine": "paddle", "device": "gpu:0" },\n  "defaults": { "mode": "blur", "blur_strength": 0.9 },\n  "actions": [\n    { "type": "track_box", "id": "subject", "target": "face",\n      "anchor_frame": 0,\n      "anchor_box_px": { "x": 100, "y": 200, "w": 80, "h": 120 },\n      "track_lost_behavior": "stop_tracking" },\n    { "type": "text_mask", "id": "pii",\n      "match": { "regex": ["\\\\S+@\\\\S+", "[A-Z]{2}[^.]{0,3}\\\\d{2}[^.]{0,3}\\\\d{4}"],\n        "contains": ["Main St"] }, "mode": "black" }\n  ]\n}`,
    },
    track: {
      note: "follows turning / zooming faces — others untouched",
      raw: `{\n  "type": "track_box",\n  "id": "subject",\n  "when": { "from": 0.0, "to": 180.0 },\n  "anchor_frame": 0,\n  "anchor_box_px": { "x": 100, "y": 200, "w": 80, "h": 120 },\n  "target": "face",\n  "mode": "blur",\n  "blur_strength": 0.9,\n  "track_lost_behavior": "stop_tracking"\n}`,
    },
    text: {
      note: "word-level PII match — neighbours stay sharp",
      raw: `{\n  "type": "text_mask",\n  "id": "pii",\n  "when": "always",\n  "match": {\n    "contains": ["Main St"],\n    "regex": ["\\\\S+@\\\\S+"],\n    "terms": [],\n    "exclude_text_contains": []\n  },\n  "mode": "black"\n}`,
    },
    fixed: {
      note: "static PII overlay for a time window",
      raw: `{\n  "type": "fixed_box",\n  "id": "watermark",\n  "when": { "from": 9.0, "to": 15.5 },\n  "box_norm": { "x": 0.321, "y": 0.543, "w": 0.0245, "h": 0.0507 },\n  "mode": "blur",\n  "blur_strength": 0.95\n}`,
    },
  };
  function highlight(raw) {
    return raw
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/("(?:\\.|[^"\\])*")(\s*:)?/g, (m, s, c) =>
        c ? `<span class="tok-k">${s}</span><span class="tok-p">:</span>` : `<span class="tok-s">${s}</span>`)
      .replace(/\b(\d+\.?\d*)\b/g, '<span class="tok-n">$1</span>');
  }
  const specBody = $("#specBody"), specNote = $("#specNote");
  let curSpec = "full";
  function showSpec(k) {
    curSpec = k;
    if (specBody) specBody.innerHTML = highlight(specs[k].raw);
    if (specNote) specNote.textContent = specs[k].note;
    $$(".spec-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.spec === k));
  }
  $$(".spec-tabs button").forEach((b) => b.addEventListener("click", () => showSpec(b.dataset.spec)));
  showSpec("full");

  /* ---------- terminal auto-log ---------- */
  const termLines = [
    ['$ python redact-tool.py render --video in.mp4 --out safe.mp4 --spec pii.json', "c-lime"],
    ["validating spec v1 … 2 actions OK (subject, pii)", "c-dim"],
    ["tracking subject … CSRT + YuNet assist · 1.5× zoom ✓ · 30° rot ✓", ""],
    ["OCR PP-OCRv6 … 214 words · 14 PII hits (emails, plates)", "c-cyan"],
    ["masking … subject <span class='c-ok'>1820 frames</span> · pii <span class='c-ok'>14 words</span> · bystander <span class='c-ok'>0 touched</span>", ""],
    ["verify … leftovers: none · retrace match ✓ → <span class='c-ok'>exit 0 — ship it</span>", "c-ok"],
  ];
  const termBody = $("#termBody");
  let ti = 0;
  function typeTerm() {
    if (!termBody) return;
    termBody.innerHTML = "";
    ti = 0;
    (function next() {
      if (ti >= termLines.length) { setTimeout(typeTerm, 9000); return; }
      const [txt, cls] = termLines[ti++];
      const div = document.createElement("div");
      if (cls) div.className = cls;
      div.innerHTML = txt;
      termBody.appendChild(div);
      setTimeout(next, 750);
    })();
  }
  typeTerm();

  /* ---------- blind agent log loop ---------- */
  const blindScript = [
    ["user", "👤 USER", "“Remove her face plus any emails, phones and plates. Shareable cut, keep the story.”"],
    ["model", "🤖 BLIND MODEL ⊘", "No video opened. Running: <code>inspect --every 15</code>…"],
    ["tool", "◧ INSPECT → JSON", "<code>faces: 2 tracklets</code> (f0–118, f22–340) · <code>texts: anna@mail.com, MH·12·4821…</code>"],
    ["model", "🤖 BLIND MODEL", "Two faces — I must not guess. Ask: “top-left (0–4s) or right column?”"],
    ["user", "👤 USER", "“Top-left woman, 0–4s. And everything that looks like PII.”"],
    ["model", "🤖 BLIND MODEL", "Anchor = tracklet start_box @ first_frame. Emitting <code>pii.json</code>…"],
    ["tool", "◧ RENDER", "<code>subject 1820 frames · pii 14 words · bystander 0 touched</code>"],
    ["ok", "✓ VERIFY → EXIT 0", "No readable leftovers. Retrace matches. Safe to share."],
  ];
  const blindBox = $("#blindLines");
  function playBlind() {
    if (!blindBox) return;
    blindBox.innerHTML = "";
    let i = 0;
    (function next() {
      if (i >= blindScript.length) { setTimeout(playBlind, 8000); return; }
      const [cls, who, html] = blindScript[i++];
      const d = document.createElement("div");
      d.className = "bl-line " + cls;
      d.innerHTML = `<span class="who">${who}</span>${html}`;
      blindBox.appendChild(d);
      setTimeout(next, 1100);
    })();
  }
  const bio = new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting) { playBlind(); bio.disconnect(); } });
  }, { threshold: 0.3 });
  if (blindBox) bio.observe(blindBox);

  /* ---------- copy buttons ---------- */
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".copy"); if (!b) return;
    const txt = b.dataset.copy || (b.hasAttribute("data-copy-spec") ? specs[curSpec].raw : "");
    if (navigator.clipboard) navigator.clipboard.writeText(txt).catch(() => {});
    const old = b.textContent; b.textContent = "copied ✓";
    setTimeout(() => (b.textContent = old), 1400);
  });

  /* ---------- tilt ---------- */
  $$(".tilt").forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
      card.style.transform = `perspective(800px) rotateX(${-y * 7}deg) rotateY(${x * 7}deg) translateY(-6px)`;
    });
    card.addEventListener("pointerleave", () => (card.style.transform = ""));
  });
})();
