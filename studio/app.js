const state = {
  view: "tasks",
  recipes: [],
  robots: [],
  robot: null,
  selected: null,
  starter: null,
  expanded: null,
  runs: [],
  run: null,
  poll: null,
  stream: null,
  dataset: null,
  datasetUri: "",
  keepEpisodes: [],
  recording: false,
  pendingTrajectories: [],
  lastDemoMessage: "",
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
  if (recipe.imitate) return `<span class="pill live">imitation</span>`;
  if (recipe.scene_preview) return `<span class="pill live">scene preview</span>`;
  if (recipe.runnable) return `<span class="pill live">runnable</span>`;
  if (recipe.id === "g1-walk") {
    return `<span class="pill blocked">blocked · GPU walk</span>`;
  }
  return `<span class="pill blocked">compile only · later phase</span>`;
}

function stepsHTML(active) {
  const items = [
    ["robots", "Robot"],
    ["task", "Task"],
    ["scene", "Scene"],
    ["train", "Train"],
  ];
  return `<ol class="steps">${items
    .map(
      ([id, label]) =>
        `<li class="${id === active ? "active" : ""}">${escapeHtml(label)}</li>`
    )
    .join("")}</ol>`;
}

function recipesForRobot() {
  if (!state.robot) return state.recipes;
  return state.recipes.filter((r) => r.robot === state.robot.id);
}

const START_HERE = [
  ["cartpole-balance", "1. Cartpole", "~30s CPU eval video"],
  ["g1-stand", "2. G1 stand", "arms raise + wave (~2s)"],
  ["pick-and-place", "3. Pick and place", "BC steers mustard; arm follows"],
];

function renderRobots() {
  const cards = state.robots
    .map((robot) => {
      const selected = state.robot && state.robot.id === robot.id;
      return `
      <article class="card robot-card ${selected ? "selected" : ""}">
        <h2>${escapeHtml(robot.name)}</h2>
        <p class="meta">${escapeHtml(robot.id)} · ${escapeHtml(robot.kind)} · ${robot.dofs} DoF</p>
        <p>${escapeHtml(robot.summary || "")}</p>
        <div class="actions">
          <button class="primary" data-robot="${escapeHtml(robot.id)}">Use this robot</button>
        </div>
      </article>`;
    })
    .join("");
  main.innerHTML = `
    ${stepsHTML("robots")}
    <h1>Robots</h1>
    <p class="lede">
      Pick a body from the catalog. The studio compiles a job spec for that
      robot — it does not import a new physics engine.
    </p>
    <div class="grid">${cards}</div>
  `;
  main.querySelectorAll("[data-robot]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.robot = state.robots.find((r) => r.id === btn.dataset.robot) || null;
      switchView("tasks");
    });
  });
}

function renderRecipes() {
  const list = recipesForRobot();
  const filter = state.robot
    ? `Showing tasks for <code>${escapeHtml(state.robot.id)}</code>.`
    : "Pick a known-good task. Drag objects when a scene exists.";
  const start = state.robot
    ? ""
    : `<div class="start-here" id="start-here">
        <p>Start here — train writes an eval video. Re-train G1 recipes if an old run looks frozen.</p>
        <div class="actions">
          ${START_HERE.map(
            ([id, label, hint]) =>
              `<button class="primary" data-open="${id}">${label}</button><span class="meta">${hint}</span>`
          ).join("")}
        </div>
      </div>`;
  const cards = list
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
    ${stepsHTML("task")}
    <h1>Tasks</h1>
    <p class="lede">${filter} Train writes a job spec and an eval video.</p>
    ${start}
    <div class="grid">${cards || `<p class="lede">No recipes for this robot yet.</p>`}</div>
  `;
  main.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => openRecipe(btn.dataset.open));
  });
}

async function openRecipe(id) {
  const detail = await api(`/api/recipes/${id}`);
  state.selected = detail.recipe;
  state.starter = detail.starter_spec;
  if (state.robot) {
    state.starter.robot = { id: state.robot.id, source: "catalog" };
  }
  await refreshExpanded();
  state.view = "recipe";
  setActive("tasks");
  renderRecipe();
}

async function refreshExpanded() {
  state.expanded = await api("/api/specs/expand", {
    method: "POST",
    body: JSON.stringify({ spec: state.starter }),
  });
}

function tokenStyle(obj) {
  const y = Number(obj.y) || 0;
  const x = Number(obj.x) || 0;
  const left = ((y + 0.32) / 0.64) * 100;
  const top = ((0.42 - x) / 0.84) * 100;
  return `left:${left}%;top:${top}%`;
}

function tableToPct(pt) {
  const y = Number(pt.y) || 0;
  const x = Number(pt.x) || 0;
  return [((y + 0.32) / 0.64) * 100, ((0.42 - x) / 0.84) * 100];
}

function trailPolylines(extra) {
  const paths = [...state.pendingTrajectories];
  if (extra && extra.length) paths.push(extra);
  return paths
    .map((path) => {
      const pts = path.map((p) => tableToPct(p).join(",")).join(" ");
      return `<polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.2" />`;
    })
    .join("");
}

function sceneHTML(spec) {
  const objects = spec?.scene?.objects;
  if (!objects || !objects.length) {
    return `
      <h2>Scene</h2>
      <p class="lede">This recipe has no movable objects. Train still writes eval video.</p>
    `;
  }
  const tokens = objects
    .map((obj) => {
      const target = recordTargetId();
      const rec = state.recording && obj.id === target ? " recording-target" : "";
      return `<button type="button" class="token${rec}" data-id="${escapeHtml(obj.id)}" style="${tokenStyle(obj)}">${escapeHtml(obj.id)}</button>`;
    })
    .join("");
  return `
    <h2>Scene</h2>
    <p class="lede">Top-down counter. G1 stands at the near edge. Drag objects — that writes <code>scene.objects</code>.</p>
    <div class="canvas-wrap${state.recording ? " recording" : ""}" id="scene-canvas">
      <svg class="trail" id="scene-trail" viewBox="0 0 100 100" preserveAspectRatio="none">${trailPolylines()}</svg>
      <span class="canvas-label">${escapeHtml(spec.scene.template || "scene")}</span>
      <span class="robot-mark">G1</span>
      ${tokens}
    </div>
  `;
}

function renderRecipe() {
  const r = state.selected;
  const spec = state.expanded?.spec || state.starter;
  const hasScene = Boolean(spec?.scene?.objects?.length);
  const trainLabel = r.imitate ? "Train pick eval" : r.scene_preview ? "Preview scene" : "Train this recipe";
  const trainHint = r.imitate
    ? `<p class="lede">Fits linear BC on demos (canvas or scripted). Eval: G1 reaches, then BC steers the mustard into the bowl while the arm follows. Not finger grasping.</p>`
    : r.id === "g1-walk"
      ? `<p class="lede">Blocked on this CPU. Train still compiles Playground / mjlab / Isaac Lab payloads, then stops with a next-step sentence — install Playground on a GPU box, or use <code>g1-stand</code> for a CPU hold preview. This is not a frozen walk clip.</p>`
      : r.runnable
        ? ""
        : `<p class="lede">Compile only on this CPU. Train still runs and will block with a next-step sentence (usually Playground / Isaac Lab + GPU).</p>`;
  const nTakes = state.pendingTrajectories.length;
  const saveBtn = nTakes
    ? `<button class="primary" id="save-demos">Save ${nTakes} demo${nTakes === 1 ? "" : "s"}</button>
       <button class="ghost" id="undo-demo">Undo last</button>`
    : "";
  const demoControls = r.imitate && hasScene
    ? `<div class="actions recipe-bar">
          <button class="ghost" id="toggle-record">${state.recording ? "Stop recording" : "Record a demo"}</button>
          ${saveBtn}
        </div>
        <p class="lede">${state.recording ? "Drag mustard into the bowl. Each pointer-up is one take. The yellow trail is what will be saved." : "Record a take on the canvas, then Save. Scripted demos still live in Data."}</p>
        <p class="status" id="demo-status">${escapeHtml(state.lastDemoMessage || (nTakes ? `${nTakes} take(s) in memory` : ""))}</p>
        <p class="error" id="demo-error"></p>`
    : "";
  main.innerHTML = `
    ${stepsHTML(hasScene ? "scene" : "train")}
    <h1>${r.title}</h1>
    <p class="lede">${r.summary}</p>
    <div class="detail">
      <section>
        <p>${pill(r)} &nbsp; robot <code>${escapeHtml(r.robot)}</code></p>
        <p class="lede">${r.language || ""}</p>
        <div class="actions recipe-bar">
          <button class="primary" id="train">${trainLabel}</button>
          <button class="ghost" id="back">Back to tasks</button>
        </div>
        <p class="status" id="train-status"></p>
        <p class="error" id="train-error"></p>
        ${trainHint}
        ${demoControls}
        ${sceneHTML(spec)}
      </section>
      <section>
        <details class="advanced" id="advanced">
          <summary>Advanced · job spec</summary>
          <p class="lede">The UI is a projection of this document. Edit only if you need to.</p>
          <textarea class="spec" id="spec-json">${escapeHtml(JSON.stringify(state.starter, null, 2))}</textarea>
          <div class="actions" style="margin-top:8px">
            <button class="ghost" id="apply-spec">Apply spec</button>
          </div>
          <h2>Expanded</h2>
          <pre>${escapeHtml(JSON.stringify(spec, null, 2))}</pre>
        </details>
      </section>
    </div>
  `;
  document.getElementById("train").addEventListener("click", trainCurrent);
  document.getElementById("back").addEventListener("click", () => switchView("tasks"));
  document.getElementById("apply-spec").addEventListener("click", applySpecEditor);
  document.getElementById("toggle-record")?.addEventListener("click", () => {
    state.recording = !state.recording;
    state.lastDemoMessage = state.recording
      ? "recording — drag mustard into the bowl"
      : state.pendingTrajectories.length
        ? `${state.pendingTrajectories.length} take(s) in memory`
        : "";
    renderRecipe();
  });
  document.getElementById("save-demos")?.addEventListener("click", saveCanvasDemos);
  document.getElementById("undo-demo")?.addEventListener("click", () => {
    state.pendingTrajectories.pop();
    state.lastDemoMessage = state.pendingTrajectories.length
      ? `${state.pendingTrajectories.length} take(s) in memory`
      : "";
    renderRecipe();
  });
  bindSceneDrag();
}

async function applySpecEditor() {
  const err = document.getElementById("train-error");
  try {
    state.starter = JSON.parse(document.getElementById("spec-json").value);
    await refreshExpanded();
    renderRecipe();
    document.getElementById("advanced")?.setAttribute("open", "");
  } catch (error) {
    err.textContent = error.message;
  }
}

function recordTargetId() {
  return state.expanded?.spec?.task?.success?.object || "mustard";
}

function canvasPoint(canvas, event) {
  const box = canvas.getBoundingClientRect();
  const nx = (event.clientX - box.left) / box.width;
  const ny = (event.clientY - box.top) / box.height;
  return {
    y: Math.max(-0.28, Math.min(0.28, nx * 0.64 - 0.32)),
    x: Math.max(-0.36, Math.min(0.36, 0.42 - ny * 0.84)),
  };
}

function applyTokenStyle(token, obj) {
  token.style.left = `${((obj.y + 0.32) / 0.64) * 100}%`;
  token.style.top = `${((0.42 - obj.x) / 0.84) * 100}%`;
}

function bindSceneDrag() {
  const canvas = document.getElementById("scene-canvas");
  if (!canvas || !state.starter?.scene?.objects) return;
  canvas.querySelectorAll(".token").forEach((token) => {
    token.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const id = token.dataset.id;
      const recordingMustard = state.recording && id === recordTargetId();
      const path = [];
      if (recordingMustard) path.push(canvasPoint(canvas, event));
      const move = (ev) => {
        const pt = canvasPoint(canvas, ev);
        const obj = state.starter.scene.objects.find((item) => item.id === id);
        if (!obj) return;
        if (recordingMustard) {
          path.push(pt);
          applyTokenStyle(token, pt);
          const svg = document.getElementById("scene-trail");
          if (svg) svg.innerHTML = trailPolylines(path);
          return;
        }
        obj.y = pt.y;
        obj.x = pt.x;
        applyTokenStyle(token, obj);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        if (recordingMustard) {
          if (path.length >= 2) {
            state.pendingTrajectories.push(path);
            state.lastDemoMessage = `${state.pendingTrajectories.length} take(s) in memory`;
          }
          const start = state.starter.scene.objects.find((item) => item.id === id);
          if (start) applyTokenStyle(token, start);
          renderRecipe();
          return;
        }
        const editor = document.getElementById("spec-json");
        if (editor) editor.value = JSON.stringify(state.starter, null, 2);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  });
}

async function saveCanvasDemos() {
  const err = document.getElementById("demo-error");
  const status = document.getElementById("demo-status");
  if (!state.pendingTrajectories.length) return;
  if (status) status.textContent = "saving…";
  if (err) err.textContent = "";
  try {
    const result = await api("/api/datasets/record", {
      method: "POST",
      body: JSON.stringify({
        spec: state.starter,
        trajectories: state.pendingTrajectories,
      }),
    });
    if (!result.ok) throw new Error(result.error || "record failed");
    state.starter.data = state.starter.data || {};
    state.starter.data.datasets = [result.path || result.dest];
    state.datasetUri = result.path || result.dest;
    state.dataset = result;
    state.keepEpisodes = (result.episodes || [])
      .filter((ep) => ep.success !== false)
      .map((ep) => ep.episode_index);
    if (state.dataset && state.dataset.ok) {
      state.starter.data.keep_episodes = state.keepEpisodes;
    }
    state.pendingTrajectories = [];
    state.recording = false;
    state.lastDemoMessage = `wrote ${result.total_episodes} canvas demos → ${result.path || result.dest}`;
    renderRecipe();
  } catch (error) {
    if (err) err.textContent = error.message;
    if (status) status.textContent = "";
  }
}

async function trainCurrent() {
  const btn =
    document.getElementById("train") ||
    document.getElementById("train-from-data") ||
    document.getElementById("train-again");
  const status = document.getElementById("train-status") || document.getElementById("record-status");
  const err = document.getElementById("train-error") || document.getElementById("record-error");
  if (btn) btn.disabled = true;
  if (status) status.textContent = "queued…";
  if (err) err.textContent = "";
  try {
    const editor = document.getElementById("spec-json");
    if (editor) {
      state.starter = JSON.parse(editor.value);
    } else if (state.run?.run_id) {
      const specRes = await fetch(`/api/runs/${state.run.run_id}/artifacts/spec.json`);
      if (specRes.ok) state.starter = await specRes.json();
    }
    if (!state.starter) {
      const recipeId = state.run?.recipe || "cartpole-balance";
      const detail = await api(`/api/recipes/${recipeId}`);
      state.starter = detail.starter_spec;
      state.selected = detail.recipe;
    }
    if (state.dataset && state.dataset.ok) {
      state.starter.data = state.starter.data || {};
      state.starter.data.keep_episodes = state.keepEpisodes;
    }
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec: state.starter }),
    });
    state.view = "run";
    await showRun(run.run_id);
  } catch (error) {
    if (err) err.textContent = error.message;
    if (btn) btn.disabled = false;
    if (status) status.textContent = "";
  }
}

function renderData() {
  const datasets = state.starter?.data?.datasets || state.expanded?.spec?.data?.datasets || [];
  const datasetList = datasets.length
    ? `<ul class="runs">${datasets
        .map((d) => `<li class="run-row"><code>${escapeHtml(d)}</code></li>`)
        .join("")}</ul>`
    : `<p class="lede">No datasets on the current spec. Pick-and-place imitation needs a LeRobot folder with <code>meta/info.json</code>.</p>`;
  const inspected = state.dataset;
  let body = "";
  if (inspected && inspected.ok) {
    const episodes = inspected.episodes || [];
    const total = episodes.length;
    const keep = state.keepEpisodes.length
      ? state.keepEpisodes
      : episodes.map((ep) => ep.episode_index);
    const keptEps = episodes.filter((ep) => keep.includes(ep.episode_index));
    const keptOk = keptEps.filter((ep) => ep.success !== false).length;
    const keptMiss = keptEps.filter((ep) => ep.success === false).length;
    const trainSummary =
      keptMiss && !keptOk
        ? `<p class="error">Keeping only misses (${keptMiss}) — train pick eval should fail mustard-in-bowl. Prefer success takes unless you are proving keep/drop.</p>`
        : keptMiss
          ? `<p class="lede">Train will fit BC on ${keep.length} of ${total} episodes (${keptOk} ok · ${keptMiss} miss). Drop misses unless you are proving keep/drop.</p>`
          : `<p class="lede">Train will fit BC on ${keep.length} of ${total} episodes. That filter writes <code>data.keep_episodes</code>.</p>`;
    const rows = episodes
      .map((ep) => {
        const idx = ep.episode_index;
        const checked = keep.includes(idx);
        const ok = ep.success === false ? "miss" : ep.success ? "ok" : "—";
        return `<tr>
          <td><input type="checkbox" data-ep="${idx}" ${checked ? "checked" : ""} /></td>
          <td>${idx}</td>
          <td>${ep.length ?? "—"}</td>
          <td>${escapeHtml((ep.tasks || []).join(", ") || "—")}</td>
          <td>${ok}</td>
        </tr>`;
      })
      .join("");
    body = `
      <p class="status completed">${escapeHtml(inspected.format)} · ${inspected.total_episodes} episodes · fps=${inspected.fps ?? "—"} · robot=${escapeHtml(inspected.robot_type || "—")}</p>
      <p class="meta">path <code>${escapeHtml(state.datasetUri || inspected.path || "")}</code></p>
      <table class="data-table">
        <thead><tr><th>keep</th><th>#</th><th>length</th><th>tasks</th><th>success</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${trainSummary}
      <div class="actions">
        <button class="primary" id="train-from-data">Train pick eval (${keep.length} kept)</button>
        <button class="ghost" id="keep-successes">Keep successes only</button>
      </div>
    `;
  } else if (inspected && inspected.error) {
    body = `<p class="error">${escapeHtml(inspected.error)}</p>`;
  }
  main.innerHTML = `
    <h1>Data</h1>
    <p class="lede">
      Demonstrations live in the LeRobot dataset format — we do not invent one.
      Record scripted mustard→bowl demos here, drop bad takes, then train.
      Or record by dragging mustard on the task scene. Gamepad teleop is still later;
      these demos fit linear BC that steers the mustard; the G1 arm follows with poses, not finger grasping.
    </p>
    <section>
      <h2>On this job</h2>
      ${datasetList}
      <div class="actions" style="margin-top:12px">
        <button class="primary" id="record-ds">Record scripted demos</button>
      </div>
      <p class="status" id="record-status"></p>
      <p class="error" id="record-error"></p>
    </section>
    <section style="margin-top:24px">
      <h2>Inspect local dataset</h2>
      <p class="lede">Path to a directory that contains <code>meta/info.json</code>.</p>
      <div class="actions">
        <input class="path-input" id="dataset-uri" placeholder="file:/path/to/lerobot_dataset" value="${escapeHtml(state.datasetUri)}" />
        <button class="primary" id="inspect-ds">Inspect</button>
      </div>
      <div id="dataset-body">${body}</div>
    </section>
  `;
  document.getElementById("inspect-ds").addEventListener("click", inspectDataset);
  document.getElementById("record-ds")?.addEventListener("click", recordDemos);
  document.getElementById("train-from-data")?.addEventListener("click", trainCurrent);
  document.getElementById("keep-successes")?.addEventListener("click", () => {
    if (!state.dataset?.episodes) return;
    state.keepEpisodes = state.dataset.episodes
      .filter((ep) => ep.success !== false)
      .map((ep) => ep.episode_index);
    if (state.starter) {
      state.starter.data = state.starter.data || {};
      state.starter.data.keep_episodes = state.keepEpisodes;
    }
    renderData();
  });
  main.querySelectorAll("[data-ep]").forEach((box) => {
    box.addEventListener("change", () => {
      syncKeepEpisodes();
      renderData();
    });
  });
}

async function recordDemos() {
  const status = document.getElementById("record-status");
  const err = document.getElementById("record-error");
  status.textContent = "recording…";
  err.textContent = "";
  try {
    if (!state.starter) {
      const detail = await api("/api/recipes/pick-and-place");
      state.starter = detail.starter_spec;
      state.selected = detail.recipe;
    }
    const result = await api("/api/datasets/record", {
      method: "POST",
      body: JSON.stringify({
        spec: state.starter,
        episodes: 4,
        include_failure: true,
      }),
    });
    if (!result.ok) throw new Error(result.error || "record failed");
    state.starter.data = state.starter.data || {};
    state.starter.data.datasets = [result.path || result.dest];
    state.datasetUri = result.path || result.dest;
    state.dataset = result;
    state.keepEpisodes = (result.episodes || [])
      .filter((ep) => ep.success !== false)
      .map((ep) => ep.episode_index);
    if (state.dataset && state.dataset.ok) {
      state.starter.data.keep_episodes = state.keepEpisodes;
    }
    status.textContent = `wrote ${result.total_episodes} episodes`;
    renderData();
  } catch (error) {
    err.textContent = error.message;
    status.textContent = "";
  }
}

async function inspectDataset() {
  const uri = document.getElementById("dataset-uri").value.trim();
  state.datasetUri = uri;
  try {
    state.dataset = await api("/api/datasets/inspect", {
      method: "POST",
      body: JSON.stringify({ uri }),
    });
    if (state.dataset.ok) {
      const eps = state.dataset.episodes || [];
      const hasSuccessMeta = eps.some((ep) => ep.success === true || ep.success === false);
      state.keepEpisodes = hasSuccessMeta
        ? eps.filter((ep) => ep.success !== false).map((ep) => ep.episode_index)
        : eps.map((ep) => ep.episode_index);
      if (state.starter) {
        state.starter.data = state.starter.data || {};
        state.starter.data.keep_episodes = state.keepEpisodes;
        if (uri) state.starter.data.datasets = [uri.replace(/^file:/, "")];
      }
    }
  } catch (error) {
    state.dataset = { ok: false, error: error.message };
  }
  renderData();
}

function syncKeepEpisodes() {
  const keep = [];
  main.querySelectorAll("[data-ep]").forEach((box) => {
    if (box.checked) keep.push(Number(box.dataset.ep));
  });
  state.keepEpisodes = keep;
  if (state.starter) {
    state.starter.data = state.starter.data || {};
    state.starter.data.keep_episodes = keep;
  }
}

function runDemoHint(run) {
  const notes = (run.notes || []).join(" ");
  const keep = (notes.match(/keep_episodes=(\[[^\]]*\]|all)/) || [])[1];
  const frames = (notes.match(/frames=(\d+)/) || [])[1];
  const bc = (notes.match(/bc_steps=(\d+)/) || [])[1];
  const mode =
    (notes.match(/arm_mode=[\w+]+/) || [])[0] ||
    (notes.includes("playback") ? "arm_mode=playback" : "") ||
    (notes.includes("linear BC") ? "object-BC" : "");
  const bits = [];
  if (mode) bits.push(mode);
  if (keep) bits.push(`keep=${keep}`);
  if (frames) bits.push(`${frames} frames`);
  if (bc) bits.push(`bc_steps=${bc}`);
  if (
    run.recipe === "pick-and-place" &&
    run.status !== "queued" &&
    run.status !== "running" &&
    !notes.includes("playback+BC") &&
    !notes.includes("IK+BC") &&
    !notes.includes("mocap-BC")
  ) {
    bits.push("stale? re-train for BC");
  }
  if (run.recipe === "g1-walk" && run.status === "blocked") {
    bits.push("needs Playground + GPU");
  }
  return bits.length ? ` · ${bits.join(" · ")}` : "";
}

function renderRuns() {
  const rows = state.runs
    .map((run) => {
      const passed = run.metrics && run.metrics.passed === true;
      const statusClass =
        run.status === "passed" || passed
          ? "passed"
          : run.status === "completed"
            ? "completed"
            : run.status || "";
      const hint = runDemoHint(run);
      const failedHint =
        !hint &&
        run.status === "completed" &&
        run.metrics &&
        run.metrics.passed === false
          ? " · failed metrics"
          : "";
      return `
      <li>
        <div class="run-row" data-run="${run.run_id}">
          <div>
            <strong>${run.recipe || run.run_id}</strong>
            <div class="meta">${run.run_id}${run.artifacts && run.artifacts["eval.mp4"] ? " · eval.mp4" : ""}${hint}${failedHint}</div>
          </div>
          <div class="status ${statusClass}">${run.status}${passed ? "" : run.status === "completed" ? " (not passed)" : ""}${run.status === "blocked" ? " (compile only)" : ""}</div>
        </div>
      </li>`;
    })
    .join("");
  main.innerHTML = `
    <h1>Runs</h1>
    <p class="lede">Every run writes a manifest, engine payload, and — when the adapter can — an eval video. Click a row to play it. Prefer pick-and-place notes with <code>arm_mode=playback+BC</code> and a <code>keep=</code> list; older frozen clips lack those. Blocked <code>g1-walk</code> means GPU walk is not available here.</p>
    <ul class="runs">${rows || "<li class='lede'>No runs yet. Train Cartpole from Tasks.</li>"}</ul>
  `;
  main.querySelectorAll("[data-run]").forEach((el) => {
    el.addEventListener("click", () => showRun(el.dataset.run));
  });
}

function paintRun(run, logText) {
  const video = run.artifacts && run.artifacts["eval.mp4"]
    ? `<video controls autoplay muted src="/api/runs/${run.run_id}/artifacts/eval.mp4?t=${Date.now()}"></video>`
    : `<p class="lede">No eval video yet.</p>`;
  const scene = run.artifacts && run.artifacts["composed_scene.xml"]
    ? `<p class="lede"><a href="/api/runs/${run.run_id}/artifacts/composed_scene.xml">composed_scene.xml</a> — open in native MuJoCo.</p>`
    : "";
  const notes = (run.notes || []).map((n) => escapeHtml(n)).join(" · ");
  const hasMetrics = run.metrics && run.metrics.eval_episodes != null;
  const metrics = hasMetrics
    ? `<p class="status ${run.status}">success_rate=${fmt(run.metrics.success_rate)} mean_return=${fmt(run.metrics.mean_return)} passed=${run.metrics.passed ?? "—"}</p>`
    : "";
  const statusClass =
    run.status === "passed" || (run.metrics && run.metrics.passed === true)
      ? "passed"
      : run.status === "completed"
        ? "completed"
        : run.status || "";
  const demoLine = runDemoHint(run)
    ? `<p class="lede">Demos / BC: ${escapeHtml(runDemoHint(run).replace(/^ · /, ""))}</p>`
    : "";
  const blockedHelp =
    run.status === "blocked" && (run.recipe === "g1-walk" || (run.error || "").includes("Playground"))
      ? `<p class="lede">Next step: on a machine with an NVIDIA GPU, <code>pip install playground</code> then run the generated <code>train.sh</code>, or open <code>g1-stand</code> for a CPU hold preview. Blocked is expected here — not a silent failure.</p>`
      : "";
  main.innerHTML = `
    ${stepsHTML("train")}
    <h1>Run</h1>
    <p class="lede">${run.run_id}</p>
    <p class="status ${statusClass}">${run.status}${
      run.status === "completed" && run.metrics && run.metrics.passed === false
        ? " (metrics not passed)"
        : run.status === "blocked"
          ? " (compile only — not trained)"
          : ""
    }</p>
    ${metrics}
    ${demoLine}
    ${notes ? `<p class="lede">${notes}</p>` : ""}
    ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
    ${blockedHelp}
    <div class="actions recipe-bar">
      <button class="primary" id="train-again">Train again</button>
      <button class="ghost" id="back-tasks">Back to tasks</button>
    </div>
    <div class="detail">
      <section>
        <h2>Eval</h2>
        ${video}
        ${scene}
      </section>
      <section>
        <h2>Log</h2>
        <pre class="log" id="run-log">${escapeHtml(logText || run.log || "")}</pre>
      </section>
    </div>
  `;
  document.getElementById("train-again")?.addEventListener("click", trainCurrent);
  document.getElementById("back-tasks")?.addEventListener("click", () => switchView("tasks"));
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
      if (msg.notes) run.notes = msg.notes;
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
  setActive(view === "recipe" ? "tasks" : view);
  if (view === "robots") {
    renderRobots();
  } else if (view === "tasks" || view === "recipes") {
    renderRecipes();
  } else if (view === "data") {
    renderData();
  } else if (view === "runs") {
    state.runs = (await api("/api/runs")).runs;
    renderRuns();
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
