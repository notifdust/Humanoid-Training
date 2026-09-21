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
  keepEpisodes: null, // null = unset (treat as all); [] = keep none
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

function worksHere(recipe) {
  return recipe && recipe.availability === "cpu";
}

function recipeById(id) {
  return (state.recipes || []).find((r) => r.id === id) || null;
}

function readyRecipes(list) {
  return (list || recipesForRobot()).filter((r) => worksHere(r));
}

function laterRecipes(list) {
  return (list || recipesForRobot()).filter((r) => !worksHere(r));
}

function firstReadyRecipe() {
  const all = state.recipes || [];
  return all.find((r) => r.start_here) || all.find((r) => worksHere(r)) || all[0] || null;
}

function firstImitateRecipe() {
  const all = state.recipes || [];
  return all.find((r) => r.imitate && worksHere(r)) || all.find((r) => r.imitate) || null;
}

function englishList(items) {
  const names = (items || []).filter(Boolean);
  if (!names.length) return "";
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
}

function pill(recipe) {
  if (worksHere(recipe)) return `<span class="pill live">works on this computer</span>`;
  if (recipe.availability === "gpu") return `<span class="pill blocked">needs a GPU — skip for now</span>`;
  return `<span class="pill blocked">later</span>`;
}

function stepsHTML(active) {
  const items = [
    ["task", "1. Task"],
    ["train", "2. Train"],
    ["video", "3. Video"],
  ];
  const map = { robots: "task", task: "task", scene: "task", train: "train", video: "video" };
  const now = map[active] || active;
  return `<ol class="steps">${items
    .map(
      ([id, label]) =>
        `<li class="${id === now ? "active" : ""}">${escapeHtml(label)}</li>`
    )
    .join("")}</ol>`;
}

function recipesForRobot() {
  if (!state.robot) return state.recipes;
  return state.recipes.filter((r) => r.robot === state.robot.id);
}

function renderRobots() {
  const cards = state.robots
    .map((robot) => {
      const selected = state.robot && state.robot.id === robot.id;
      return `
      <article class="card robot-card ${selected ? "selected" : ""}">
        <h2>${escapeHtml(robot.name)}</h2>
        <p class="meta">${escapeHtml(robot.kind)} · ${robot.dofs} joints</p>
        <p>${escapeHtml(robot.summary || "")}</p>
        <div class="actions">
          <button class="primary" data-robot="${escapeHtml(robot.id)}">Use this robot</button>
        </div>
      </article>`;
    })
    .join("");
  main.innerHTML = `
    ${stepsHTML("task")}
    <h1>Robots</h1>
    <p class="lede">Start with the Unitree G1. A non-humanoid smoke test is included so Train → video is real before humanoid engines.</p>
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
  const ready = readyRecipes(list);
  const later = laterRecipes(list);
  const readyCards = ready
    .map(
      (r) => `
      <article class="card hero-card">
        ${pill(r)}
        <h2>${escapeHtml(r.title)}</h2>
        <p>${escapeHtml(r.promise || r.summary)}</p>
        <div class="actions">
          <button class="primary" data-open="${escapeHtml(r.id)}">Open and train</button>
        </div>
      </article>`
    )
    .join("");
  const laterCards = later
    .map(
      (r) => `
      <article class="card">
        ${pill(r)}
        <h2>${escapeHtml(r.title)}</h2>
        <p>${escapeHtml(r.blocked_hint || r.promise || r.summary)}</p>
        <div class="actions">
          <button class="ghost" data-open="${escapeHtml(r.id)}">See why it is blocked</button>
        </div>
      </article>`
    )
    .join("");
  const filter = state.robot
    ? `Tasks for ${escapeHtml(state.robot.name || state.robot.id)}.`
    : "";
  const skipLine = later.length
    ? ` ${englishList(later.map((r) => r.title))} ${
        later.length === 1 ? "is" : "are"
      } listed below so ${later.length === 1 ? "it does" : "they do"} not look like silent failures — skip ${
        later.length === 1 ? "it" : "them"
      } on this computer.`
    : "";
  main.innerHTML = `
    ${stepsHTML("task")}
    <h1>What should the robot do?</h1>
    <p class="lede" id="start-here">
      Start here: click a task, then Train. You should get a video. That is the whole product today.
      ${filter}${skipLine}
    </p>
    <div class="grid">${readyCards || `<p class="lede">No recipes for this robot yet.</p>`}</div>
    ${
      laterCards
        ? `<h2 class="later-head">Needs a GPU (skip)</h2>
           <p class="lede">These compile a job for another machine. Train will stop with a next step, not a fake success clip.</p>
           <div class="grid">${laterCards}</div>`
        : ""
    }
  `;
  main.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => openRecipe(btn.dataset.open));
  });
}

async function openRecipe(id) {
  const prevId = state.selected?.id;
  if (prevId && prevId !== id) {
    clearDemoSession();
  }
  const detail = await api(`/api/recipes/${id}`);
  state.selected = detail.recipe;
  state.starter = detail.starter_spec;
  if (state.robot) {
    state.starter.robot = { id: state.robot.id, source: "catalog" };
  }
  // Re-open of the same imitation recipe: put inspected/saved demos back on the starter
  // so Train does not silently fall back to auto-scripted takes.
  if (detail.recipe.imitate) {
    rebindDatasetOntoStarter();
  }
  await refreshExpanded();
  state.view = "recipe";
  setActive("tasks");
  renderRecipe();
}

function clearDemoSession() {
  state.dataset = null;
  state.datasetUri = "";
  state.keepEpisodes = null;
  state.pendingTrajectories = [];
  state.recording = false;
  state.lastDemoMessage = "";
}

function rebindDatasetOntoStarter() {
  if (!state.starter) return;
  const path =
    state.datasetUri ||
    state.dataset?.path ||
    ((state.starter.data || {}).datasets || [])[0];
  if (!path) return;
  state.starter.data = state.starter.data || {};
  state.starter.data.datasets = [String(path).replace(/^file:/, "")];
  if (Array.isArray(state.keepEpisodes) && state.keepEpisodes.length > 0) {
    state.starter.data.keep_episodes = state.keepEpisodes;
  }
}

function isImitationJob(spec, selected) {
  if (selected && selected.imitate) return true;
  return String((spec?.train || {}).method || "") === "imitation";
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
  const recipe = state.selected;
  const obj = objectLabel();
  const dest = recordContainerId();
  const sceneLede =
    (recipe && recipe.scene_hint) ||
    (dest
      ? `Top-down ${spec.scene.template || "scene"}. Drag ${obj} into the ${dest} if you want. Train still works if you do not.`
      : `Top-down ${spec.scene.template || "scene"}. Drag ${obj} if you want. Train still works if you do not.`);
  return `
    <h2>Scene</h2>
    <p class="lede">${escapeHtml(sceneLede)}</p>
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
  const cpu = worksHere(r);
  const trainLabel = cpu ? "Train" : "Compile (will stop — needs GPU)";
  const hintText = r.train_hint || r.promise || r.summary || "";
  const trainHint = hintText ? `<p class="lede">${escapeHtml(hintText)}</p>` : "";
  const boundDs = (state.starter?.data?.datasets || [])[0];
  const boundKeep = state.starter?.data?.keep_episodes;
  const boundHint =
    r.imitate && boundDs
      ? `<p class="meta">Using your recorded demos${
          Array.isArray(boundKeep) ? ` (${boundKeep.length} kept)` : ""
        }.</p>`
      : r.imitate
        ? `<p class="meta">No demos saved yet — Train will use built-in scripted takes. Or record by dragging ${escapeHtml(objectLabel())} below.</p>`
        : "";
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
        <p class="lede">${state.recording ? `Drag ${escapeHtml(objectLabel())}${recordContainerId() ? ` into the ${escapeHtml(recordContainerId())}` : ""}. Each release is one take.` : "Optional: record a take on the canvas, then Save, then Train."}</p>
        <p class="status" id="demo-status">${escapeHtml(state.lastDemoMessage || (nTakes ? `${nTakes} take(s) in memory` : ""))}</p>
        <p class="error" id="demo-error"></p>`
    : "";
  main.innerHTML = `
    ${stepsHTML("train")}
    <h1>${escapeHtml(r.title)}</h1>
    <p class="lede">${escapeHtml(r.language || r.summary || "")}</p>
    <div class="detail">
      <section>
        <p>${pill(r)}</p>
        <div class="actions recipe-bar">
          <button class="primary" id="train">${trainLabel}</button>
          <button class="ghost" id="back">Back</button>
        </div>
        <p class="status" id="train-status"></p>
        <p class="error" id="train-error"></p>
        ${trainHint}
        ${boundHint}
        ${demoControls}
        ${sceneHTML(spec)}
      </section>
      <section>
        <details class="advanced" id="advanced">
          <summary>Advanced · job spec</summary>
          <p class="lede">Researchers: this JSON is what Train sends to the engine. Beginners can ignore it.</p>
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
      ? `recording — drag ${objectLabel()}${recordContainerId() ? ` into the ${recordContainerId()}` : ""}`
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
  const fromSuccess = state.expanded?.spec?.task?.success?.object;
  if (fromSuccess) return String(fromSuccess);
  const objects = state.starter?.scene?.objects || state.expanded?.spec?.scene?.objects || [];
  return objects[0]?.id ? String(objects[0].id) : "";
}

function recordContainerId() {
  return String(state.expanded?.spec?.task?.success?.container || "");
}

function objectLabel() {
  return recordTargetId() || "the object";
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
  if (btn) {
    btn.disabled = true;
    btn.classList.add("busy");
  }
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
      const recipeId = state.run?.recipe || firstReadyRecipe()?.id;
      if (!recipeId) throw new Error("Pick a task first.");
      const detail = await api(`/api/recipes/${recipeId}`);
      state.starter = detail.starter_spec;
      state.selected = detail.recipe;
    }
    if (isImitationJob(state.starter, state.selected)) {
      rebindDatasetOntoStarter();
      state.starter.data = state.starter.data || {};
      // Only stamp keep from the Data-room session when a dataset is inspected.
      // "Train again" from a past run should use that run's own keep_episodes.
      if (state.dataset && state.dataset.ok && Array.isArray(state.keepEpisodes)) {
        if (state.keepEpisodes.length === 0) {
          throw new Error(
            "Keep at least one episode — empty keep_episodes fits no BC frames."
          );
        }
        state.starter.data.keep_episodes = state.keepEpisodes;
      }
      const specKeep = state.starter.data.keep_episodes;
      if (Array.isArray(specKeep) && specKeep.length === 0) {
        throw new Error(
          "Keep at least one episode — empty keep_episodes fits no BC frames."
        );
      }
    } else if (state.starter.data) {
      // Do not leak Data-room keep filters onto non-imitation recipes.
      delete state.starter.data.keep_episodes;
    }
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec: state.starter }),
    });
    state.view = "run";
    await showRun(run.run_id);
  } catch (error) {
    if (err) err.textContent = error.message;
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("busy");
    }
    if (status) status.textContent = "";
  }
}

function keepTrainSummary(emptyKeep, keptMiss, keptOk, keep, total) {
  const check = successCheckPhrase();
  if (emptyKeep) {
    return `<p class="error">No episodes kept — train would fit BC on 0 frames. Check at least one take.</p>`;
  }
  if (keptMiss && !keptOk) {
    return `<p class="error">Keeping only misses (${keptMiss}) — train eval should fail ${escapeHtml(check)}. Prefer success takes unless you are proving keep/drop.</p>`;
  }
  if (keptMiss) {
    return `<p class="lede">Train will fit BC on ${keep.length} of ${total} episodes (${keptOk} ok · ${keptMiss} miss). Drop misses unless you are proving keep/drop.</p>`;
  }
  return `<p class="lede">Train will fit BC on ${keep.length} of ${total} episodes. That filter writes <code>data.keep_episodes</code>.</p>`;
}

function successCheckPhrase() {
  const obj = recordTargetId();
  const box = recordContainerId();
  if (obj && box) return `${obj}-in-${box}`;
  return "the success check";
}

function renderData() {
  const imitateOpen = Boolean(state.selected?.imitate);
  const datasets = state.starter?.data?.datasets || state.expanded?.spec?.data?.datasets || [];
  const datasetList = datasets.length
    ? `<ul class="runs">${datasets
        .map((d) => `<li class="run-row"><code>${escapeHtml(d)}</code></li>`)
        .join("")}</ul>`
    : imitateOpen
      ? `<p class="lede">No demos on this job yet. Train still works — it writes built-in takes. Record here when you want to keep or drop episodes.</p>`
      : `<p class="lede">No demos on this job yet.</p>`;
  const inspected = state.dataset;
  let body = "";
  if (inspected && inspected.ok) {
    const episodes = inspected.episodes || [];
    const total = episodes.length;
    const keepExplicit = Array.isArray(state.keepEpisodes);
    const keep = keepExplicit
      ? state.keepEpisodes
      : episodes.map((ep) => ep.episode_index);
    const emptyKeep = keepExplicit && keep.length === 0;
    const keptEps = episodes.filter((ep) => keep.includes(ep.episode_index));
    const keptOk = keptEps.filter((ep) => ep.success !== false).length;
    const keptMiss = keptEps.filter((ep) => ep.success === false).length;
    const trainSummary = keepTrainSummary(emptyKeep, keptMiss, keptOk, keep, total);
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
      <p class="meta">${escapeHtml(inspected.format)} · ${inspected.total_episodes} episodes · fps=${inspected.fps ?? "—"} · robot=${escapeHtml(inspected.robot_type || "—")}</p>
      <p class="meta">path <code>${escapeHtml(state.datasetUri || inspected.path || "")}</code></p>
      <table class="data-table">
        <thead><tr><th>keep</th><th>#</th><th>length</th><th>tasks</th><th>success</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div id="keep-summary">${trainSummary}</div>
      <div class="actions">
        <button class="primary" id="train-from-data" ${emptyKeep ? "disabled" : ""}>Train (${keep.length} kept)</button>
        <button class="ghost" id="keep-successes">Keep successes only</button>
      </div>
    `;
  } else if (inspected && inspected.error) {
    body = `<p class="error">${escapeHtml(inspected.error)}</p>`;
  }
  const otherReady = readyRecipes(state.recipes)
    .filter((r) => !r.imitate)
    .map((r) => r.title);
  const recordActions = imitateOpen
    ? `<div class="actions" style="margin-top:12px">
        <button class="primary" id="record-ds">Record scripted demos</button>
      </div>`
    : `<div class="actions" style="margin-top:12px">
        <button class="primary" id="open-imitate">Open an imitation task</button>
      </div>
      <p class="lede">Recording belongs on a demonstration recipe.${
        otherReady.length ? ` Train on ${escapeHtml(englishList(otherReady))} from Tasks.` : ""
      }</p>`;
  const dataLede =
    state.selected?.imitate && state.selected.record_hint
      ? state.selected.record_hint
      : "This room is for demonstration data. Open an imitation task, record takes, uncheck the bad ones, then Train.";
  main.innerHTML = `
    <h1>Data</h1>
    <p class="lede">
      ${escapeHtml(dataLede)}
    </p>
    <section>
      <h2>On this job</h2>
      ${datasetList}
      ${recordActions}
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
  document.getElementById("open-imitate")?.addEventListener("click", () => {
    const imitate = firstImitateRecipe();
    if (imitate) openRecipe(imitate.id);
    else switchView("tasks");
  });
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
      refreshKeepSummary();
    });
  });
}

function refreshKeepSummary() {
  const inspected = state.dataset;
  if (!inspected || !inspected.ok) return;
  const episodes = inspected.episodes || [];
  const total = episodes.length;
  const keepExplicit = Array.isArray(state.keepEpisodes);
  const keep = keepExplicit
    ? state.keepEpisodes
    : episodes.map((ep) => ep.episode_index);
  const emptyKeep = keepExplicit && keep.length === 0;
  const keptEps = episodes.filter((ep) => keep.includes(ep.episode_index));
  const keptOk = keptEps.filter((ep) => ep.success !== false).length;
  const keptMiss = keptEps.filter((ep) => ep.success === false).length;
  const summary = keepTrainSummary(emptyKeep, keptMiss, keptOk, keep, total);
  const host = document.getElementById("keep-summary");
  if (host) host.innerHTML = summary;
  const btn = document.getElementById("train-from-data");
  if (btn) {
    btn.disabled = emptyKeep;
    btn.textContent = `Train (${keep.length} kept)`;
  }
}

async function recordDemos() {
  const status = document.getElementById("record-status");
  const err = document.getElementById("record-error");
  status.textContent = "recording…";
  err.textContent = "";
  try {
    if (!state.starter) {
      const imitate = firstImitateRecipe();
      if (!imitate) throw new Error("No imitation recipe in the catalog.");
      const detail = await api(`/api/recipes/${imitate.id}`);
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

function hasBcEvidence(run) {
  const facts = runFacts(run);
  if (facts.kind === "imitation") return true;
  if (facts.arm_mode) return true;
  if (typeof facts.bc_steps === "number") return true;
  // Older manifests only had English notes — do not stale-label those.
  const notes = (run.notes || []).join(" ");
  if (/arm_mode=(playback|IK|mocap)[+_]BC/.test(notes)) return true;
  if (/bc_steps=\d+/.test(notes)) return true;
  if (notes.includes("linear BC")) return true;
  if (/frames=\d+/.test(notes) && /keep_episodes=/.test(notes)) return true;
  if (notes.includes("arm_ik=on")) return true;
  return false;
}

function runFacts(run) {
  return (run && run.facts) || {};
}

function formatKeep(value) {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}

function runDemoHint(run) {
  const facts = runFacts(run);
  const notes = (run.notes || []).join(" ");
  let mode = facts.arm_mode || "";
  if (mode) mode = `arm_mode=${mode}`;
  if (!mode) {
    const modeMatch = notes.match(/arm_mode=(playback\+BC|IK\+BC|mocap-BC|[-\w+]+)/);
    mode = modeMatch ? `arm_mode=${modeMatch[1]}` : "";
    if (!mode && notes.includes("linear BC")) mode = "linear-BC";
    if (!mode && notes.includes("arm_ik=on")) mode = "legacy-BC";
  }
  const keep =
    formatKeep(facts.keep_episodes) ||
    (notes.match(/keep_episodes=(\[[^\]]*\]|all)/) || [])[1] ||
    "";
  const frames =
    facts.frames != null
      ? String(facts.frames)
      : (notes.match(/frames=(\d+)/) || [])[1] || "";
  const bc =
    facts.bc_steps != null
      ? String(facts.bc_steps)
      : (notes.match(/bc_steps=(\d+)/) || [])[1] || "";
  const bits = [];
  if (mode) bits.push(mode);
  if (keep) bits.push(`keep=${keep}`);
  if (frames) bits.push(`${frames} frames`);
  if (bc) bits.push(`bc_steps=${bc}`);
  const rec = recipeById(run.recipe);
  if (
    rec &&
    rec.imitate &&
    run.status !== "queued" &&
    run.status !== "running" &&
    !hasBcEvidence(run)
  ) {
    bits.push("stale? re-train for BC");
  }
  if (facts.runner === "docker") bits.push("Docker");
  if (run.status === "blocked") {
    if (rec && rec.availability === "gpu") {
      bits.push(rec.blocked_hint || "needs a GPU");
    }
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
            <strong>${escapeHtml(prettyRecipe(run.recipe))}</strong>
            <div class="meta">${run.run_id}${run.artifacts && run.artifacts["eval.mp4"] ? " · eval.mp4" : ""}${hint}${failedHint}</div>
          </div>
          <div class="status ${statusClass}">${englishRunStatus(run)}</div>
        </div>
      </li>`;
    })
    .join("");
  main.innerHTML = `
    <h1>Runs</h1>
    <p class="lede">Click a run to watch the video. Green means the task succeeded. Orange means it finished but failed, or it cannot train on this computer.</p>
    <ul class="runs">${rows || `<li class='lede'>No runs yet. Open ${escapeHtml(firstReadyRecipe()?.title || "a task")} from Tasks and click Train.</li>`}</ul>
  `;
  main.querySelectorAll("[data-run]").forEach((el) => {
    el.addEventListener("click", () => showRun(el.dataset.run));
  });
}

function englishRunStatus(run) {
  const passed = run.metrics && run.metrics.passed === true;
  if (run.status === "queued" || run.status === "running") return "training…";
  if (run.status === "passed" || passed) return "worked";
  if (run.status === "blocked") return "can't train here";
  if (run.status === "failed") return "broke";
  if (run.status === "completed") return "finished — did not pass";
  return run.status || "";
}

function paintRun(run, logText) {
  const video = run.artifacts && run.artifacts["eval.mp4"]
    ? `<video controls autoplay muted src="/api/runs/${run.run_id}/artifacts/eval.mp4?t=${Date.now()}"></video>`
    : `<p class="lede">${
        run.status === "blocked"
          ? "No video — this task cannot train on this computer."
          : ["queued", "running"].includes(run.status)
            ? "Video appears when training finishes."
            : "No eval video. On a machine without a display, Train still scores success but skips the clip."
      }</p>`;
  const scene = run.artifacts && run.artifacts["composed_scene.xml"]
    ? `<p class="lede"><a href="/api/runs/${run.run_id}/artifacts/composed_scene.xml">Scene file</a> — open in MuJoCo if you want.</p>`
    : "";
  const notes = (run.notes || []).map((n) => escapeHtml(n)).join(" · ");
  const hasMetrics = run.metrics && run.metrics.eval_episodes != null;
  const metrics = hasMetrics
    ? `<p class="meta">score ${fmt(run.metrics.success_rate)} · passed=${run.metrics.passed ?? "—"}</p>`
    : "";
  const statusClass =
    run.status === "passed" || (run.metrics && run.metrics.passed === true)
      ? "passed"
      : run.status === "completed"
        ? "completed"
        : run.status || "";
  const headline = englishRunStatus(run);
  const readyTitles = readyRecipes(state.recipes).map((r) => r.title);
  const laterTitles = laterRecipes(state.recipes).map((r) => r.title);
  const blockedHelp =
    run.status === "blocked"
      ? `<p class="lede">This is expected.${
          readyTitles.length ? ` Use ${escapeHtml(englishList(readyTitles))} on this computer.` : ""
        }${laterTitles.length ? ` ${escapeHtml(englishList(laterTitles))} need a GPU box.` : ""}</p>`
      : "";
  main.innerHTML = `
    ${stepsHTML("video")}
    <h1>${escapeHtml(prettyRecipe(run.recipe))}</h1>
    <p class="status ${statusClass}">${escapeHtml(headline)}</p>
    ${metrics}
    ${run.error ? `<p class="error">${escapeHtml(plainError(run.error))}</p>` : ""}
    ${blockedHelp}
    <div class="actions recipe-bar">
      <button class="primary" id="train-again">Train again</button>
      <button class="ghost" id="back-tasks">Back to tasks</button>
    </div>
    <div class="detail">
      <section>
        <h2>Did it work?</h2>
        ${video}
        ${scene}
      </section>
      <section>
        <details class="advanced" ${["queued", "running"].includes(run.status) ? "open" : ""}>
          <summary>Log</summary>
          <pre class="log" id="run-log">${escapeHtml(logText || run.log || "")}</pre>
          ${notes ? `<p class="meta">${notes}</p>` : ""}
        </details>
      </section>
    </div>
  `;
  document.getElementById("train-again")?.addEventListener("click", trainCurrent);
  document.getElementById("back-tasks")?.addEventListener("click", () => switchView("tasks"));
}

function prettyRecipe(id) {
  const hit = (state.recipes || []).find((r) => r.id === id);
  return (hit && hit.title) || id || "Run";
}

function plainError(err) {
  const text = String(err);
  if (text.includes("keep_episodes is empty")) {
    return "Keep at least one demo, then Train.";
  }
  const rec = recipeById(state.run?.recipe) || state.selected;
  if (rec && rec.availability === "gpu" && rec.blocked_hint && runIsEngineBlock(text)) {
    return rec.blocked_hint;
  }
  return text;
}

function runIsEngineBlock(text) {
  return /Playground|mjlab|Isaac|GPU/i.test(text);
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
      if (msg.facts) run.facts = msg.facts;
      const logEl = document.getElementById("run-log");
      if (logEl) {
        logEl.textContent = logText;
        logEl.scrollTop = logEl.scrollHeight;
      } else {
        paintRun(run, logText);
      }
      const statusEl = document.querySelector("main .status");
      if (statusEl) statusEl.textContent = englishRunStatus(run);
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
