const state = {
  view: "recipes",
  recipes: [],
  robots: [],
  selected: null,
  starter: null,
  expanded: null,
  runs: [],
  run: null,
  poll: null,
  stream: null,
};

const main = document.getElementById("main");

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function pill(recipe) {
  if (recipe.runnable) return `<span class="pill live">runnable</span>`;
  return `<span class="pill blocked">compile / later phase</span>`;
}

function renderRecipes() {
  const cards = state.recipes
    .map(
      (r) => `
      <article class="card">
        ${pill(r)}
        <h2>${r.title}</h2>
        <p>${r.summary}</p>
        <div class="actions">
          <button class="primary" data-open="${r.id}">Open</button>
        </div>
      </article>`
    )
    .join("");
  main.innerHTML = `
    <h1>Recipes</h1>
    <p class="lede">
      Pick a known-good task. Drag objects when a scene exists. Train writes
      a job spec and an eval video — MuJoCo and Isaac Lab stay engines.
    </p>
    <div class="grid">${cards}</div>
  `;
  main.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => openRecipe(btn.dataset.open));
  });
}

async function openRecipe(id) {
  const detail = await api(`/api/recipes/${id}`);
  state.selected = detail.recipe;
  state.starter = detail.starter_spec;
  await refreshExpanded();
  state.view = "recipe";
  setActive("recipes");
  renderRecipe();
}

async function refreshExpanded() {
  state.expanded = await api("/api/specs/expand", {
    method: "POST",
    body: JSON.stringify({ spec: state.starter }),
  });
}

function sceneHTML(spec) {
  const objects = spec?.scene?.objects;
  if (!objects || !objects.length) return "";
  const tokens = objects
    .map((obj) => {
      const left = ((Number(obj.x) || 0) + 0.6) / 1.2 * 100;
      const top = ((Number(obj.y) || 0) + 0.3) / 0.6 * 100;
      return `<button type="button" class="token" data-id="${escapeHtml(obj.id)}" style="left:${left}%;top:${top}%">${escapeHtml(obj.id)}</button>`;
    })
    .join("");
  return `
    <h2>Scene</h2>
    <p class="lede">Drag objects on the counter. That writes <code>scene.objects</code> in the spec.</p>
    <div class="canvas-wrap" id="scene-canvas">
      <span class="canvas-label">${escapeHtml(spec.scene.template || "scene")}</span>
      ${tokens}
    </div>
  `;
}

function renderRecipe() {
  const r = state.selected;
  const spec = state.expanded?.spec || state.starter;
  main.innerHTML = `
    <h1>${r.title}</h1>
    <p class="lede">${r.summary}</p>
    <div class="detail">
      <section>
        <p>${pill(r)} &nbsp; robot <code>${r.robot}</code></p>
        <p class="lede">${r.language || ""}</p>
        ${sceneHTML(spec)}
        <div class="actions" style="margin-top:16px">
          <button class="primary" id="train">Train this recipe</button>
          <button class="ghost" data-view="recipes" id="back">Back to catalog</button>
        </div>
        <p class="status" id="train-status"></p>
        <p class="error" id="train-error"></p>
      </section>
      <section>
        <h2>Job spec</h2>
        <p class="lede">Edit freely. Apply re-expands recipe defaults under your overlay.</p>
        <textarea class="spec" id="spec-json">${escapeHtml(JSON.stringify(state.starter, null, 2))}</textarea>
        <div class="actions" style="margin-top:8px">
          <button class="ghost" id="apply-spec">Apply spec</button>
        </div>
        <h2>Expanded</h2>
        <pre>${escapeHtml(JSON.stringify(spec, null, 2))}</pre>
      </section>
    </div>
  `;
  document.getElementById("train").addEventListener("click", trainCurrent);
  document.getElementById("back").addEventListener("click", () => switchView("recipes"));
  document.getElementById("apply-spec").addEventListener("click", applySpecEditor);
  bindSceneDrag();
}

async function applySpecEditor() {
  const err = document.getElementById("train-error");
  try {
    state.starter = JSON.parse(document.getElementById("spec-json").value);
    await refreshExpanded();
    renderRecipe();
  } catch (error) {
    err.textContent = error.message;
  }
}

function bindSceneDrag() {
  const canvas = document.getElementById("scene-canvas");
  if (!canvas || !state.starter?.scene?.objects) return;
  canvas.querySelectorAll(".token").forEach((token) => {
    token.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const id = token.dataset.id;
      const move = (ev) => {
        const box = canvas.getBoundingClientRect();
        const nx = ((ev.clientX - box.left) / box.width) * 1.2 - 0.6;
        const ny = ((ev.clientY - box.top) / box.height) * 0.6 - 0.3;
        const obj = state.starter.scene.objects.find((item) => item.id === id);
        if (!obj) return;
        obj.x = Math.max(-0.55, Math.min(0.55, nx));
        obj.y = Math.max(-0.28, Math.min(0.28, ny));
        token.style.left = `${((obj.x + 0.6) / 1.2) * 100}%`;
        token.style.top = `${((obj.y + 0.3) / 0.6) * 100}%`;
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        document.getElementById("spec-json").value = JSON.stringify(state.starter, null, 2);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  });
}

async function trainCurrent() {
  const btn = document.getElementById("train");
  const status = document.getElementById("train-status");
  const err = document.getElementById("train-error");
  btn.disabled = true;
  status.textContent = "queued…";
  err.textContent = "";
  try {
    const editor = document.getElementById("spec-json");
    if (editor) {
      state.starter = JSON.parse(editor.value);
    }
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec: state.starter }),
    });
    state.view = "run";
    await showRun(run.run_id);
  } catch (error) {
    err.textContent = error.message;
    btn.disabled = false;
    status.textContent = "";
  }
}

function renderRuns() {
  const rows = state.runs
    .map(
      (run) => `
      <li>
        <div class="run-row" data-run="${run.run_id}">
          <div>
            <strong>${run.recipe || run.run_id}</strong>
            <div class="meta">${run.run_id}</div>
          </div>
          <div class="status ${run.status || ""}">${run.status}</div>
        </div>
      </li>`
    )
    .join("");
  main.innerHTML = `
    <h1>Runs</h1>
    <p class="lede">Every run writes a manifest, engine payload, and — when the adapter can — an eval video.</p>
    <ul class="runs">${rows || "<li class='lede'>No runs yet.</li>"}</ul>
  `;
  main.querySelectorAll("[data-run]").forEach((el) => {
    el.addEventListener("click", () => showRun(el.dataset.run));
  });
}

function paintRun(run, logText) {
  const video = run.artifacts && run.artifacts["eval.mp4"]
    ? `<video controls autoplay muted src="/api/runs/${run.run_id}/artifacts/eval.mp4?t=${Date.now()}"></video>`
    : `<p class="lede">No eval video yet.</p>`;
  const hasMetrics = run.metrics && run.metrics.eval_episodes != null;
  const metrics = hasMetrics
    ? `<p class="status ${run.status}">success_rate=${fmt(run.metrics.success_rate)} mean_return=${fmt(run.metrics.mean_return)} passed=${run.metrics.passed ?? "—"}</p>`
    : "";
  main.innerHTML = `
    <h1>Run</h1>
    <p class="lede">${run.run_id}</p>
    <p class="status ${run.status}">${run.status}</p>
    ${metrics}
    ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
    <div class="detail">
      <section>
        <h2>Eval</h2>
        ${video}
      </section>
      <section>
        <h2>Log</h2>
        <pre class="log" id="run-log">${escapeHtml(logText || run.log || "")}</pre>
      </section>
    </div>
  `;
}

async function showRun(runId) {
  stopPoll();
  state.view = "run";
  setActive("runs");
  const run = await api(`/api/runs/${runId}`);
  state.run = run;
  let logText = run.log || "";
  paintRun(run, logText);
  if (!["queued", "running"].includes(run.status)) return;

  if (window.EventSource) {
    state.stream = new EventSource(`/api/runs/${runId}/events`);
    state.stream.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.chunk) logText += msg.chunk;
      run.status = msg.status || run.status;
      run.metrics = msg.metrics || run.metrics;
      run.error = msg.error;
      run.artifacts = msg.artifacts || run.artifacts;
      const logEl = document.getElementById("run-log");
      if (logEl) {
        logEl.textContent = logText;
        logEl.scrollTop = logEl.scrollHeight;
      } else {
        paintRun(run, logText);
      }
      const statusEl = document.querySelector("main .status");
      if (statusEl) statusEl.textContent = run.status;
      if (!["queued", "running"].includes(run.status)) {
        stopPoll();
        paintRun(run, logText);
      }
    };
  } else {
    const paint = async () => {
      const latest = await api(`/api/runs/${runId}`);
      paintRun(latest, latest.log || "");
      if (["queued", "running"].includes(latest.status)) {
        state.poll = setTimeout(paint, 400);
      }
    };
    state.poll = setTimeout(paint, 400);
  }
}

function renderSpecHelp() {
  main.innerHTML = `
    <h1>Job spec</h1>
    <p class="lede">
      The UI is a projection of a versioned JSON document. Adapters compile
      it into Gymnasium, MuJoCo, Playground, mjlab, or Isaac Lab.
    </p>
    <pre>${escapeHtml(`{
  "spec_version": "0.1.0",
  "name": "g1-stand",
  "robot": { "id": "unitree-g1-29dof", "source": "catalog" },
  "task": { "recipe": "g1-stand" },
  "train": { "method": "hold" },
  "backend": { "prefer": ["mujoco"], "compute": "local" }
}`)}</pre>
  `;
}

function fmt(value) {
  if (value === undefined || value === null) return "—";
  if (typeof value === "number") return value.toFixed(2);
  return String(value);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function stopPoll() {
  if (state.poll) {
    clearTimeout(state.poll);
    state.poll = null;
  }
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

function setActive(view) {
  document.querySelectorAll(".rail-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
}

async function switchView(view) {
  stopPoll();
  state.view = view;
  setActive(view);
  if (view === "recipes") {
    renderRecipes();
  } else if (view === "runs") {
    state.runs = (await api("/api/runs")).runs;
    renderRuns();
  } else {
    renderSpecHelp();
  }
}

document.querySelectorAll(".rail-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

async function boot() {
  try {
    const health = await api("/api/health");
    document.getElementById("health").textContent = health.ok
      ? "local · studio online"
      : "offline";
    const [recipes, robots] = await Promise.all([
      api("/api/recipes"),
      api("/api/robots"),
    ]);
    state.recipes = recipes.recipes;
    state.robots = robots.robots;
    renderRecipes();
  } catch (error) {
    document.getElementById("health").textContent = "offline";
    main.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

boot();
