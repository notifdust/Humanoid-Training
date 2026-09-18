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
      Pick a known-good task. Beginners train a template. The job spec is
      the source of truth — Isaac Lab and MuJoCo are engines, not the product.
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
  state.expanded = await api("/api/specs/expand", {
    method: "POST",
    body: JSON.stringify({ spec: detail.starter_spec }),
  });
  state.view = "recipe";
  setActive("recipes");
  renderRecipe();
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
        <div class="actions">
          <button class="primary" id="train">Train this recipe</button>
          <button class="ghost" data-view="recipes">Back to catalog</button>
        </div>
        <p class="status" id="train-status"></p>
        <p class="error" id="train-error"></p>
      </section>
      <section>
        <h2>Expanded spec</h2>
        <pre>${escapeHtml(JSON.stringify(spec, null, 2))}</pre>
      </section>
    </div>
  `;
  document.getElementById("train").addEventListener("click", trainCurrent);
}

async function trainCurrent() {
  const btn = document.getElementById("train");
  const status = document.getElementById("train-status");
  const err = document.getElementById("train-error");
  btn.disabled = true;
  status.textContent = "queued…";
  err.textContent = "";
  try {
    const spec = state.starter;
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ spec }),
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

async function showRun(runId) {
  stopPoll();
  state.view = "run";
  setActive("runs");
  const paint = async () => {
    const run = await api(`/api/runs/${runId}`);
    state.run = run;
    const video = run.artifacts && run.artifacts["eval.mp4"]
      ? `<video controls src="/api/runs/${run.run_id}/artifacts/eval.mp4"></video>`
      : `<p class="lede">No eval video yet.</p>`;
    const metrics = run.metrics
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
          <pre class="log">${escapeHtml(run.log || "")}</pre>
        </section>
      </div>
    `;
    const live = ["queued", "running"].includes(run.status);
    if (live) {
      state.poll = setTimeout(paint, 800);
    }
  };
  await paint();
}

function renderSpecHelp() {
  main.innerHTML = `
    <h1>Job spec</h1>
    <p class="lede">
      The UI is a projection of a versioned JSON document. Adapters compile
      that document into Gymnasium, MuJoCo Playground, or Isaac Lab payloads.
      Unknown fields are preserved.
    </p>
    <pre>${escapeHtml(`{
  "spec_version": "0.1.0",
  "name": "cartpole-balance",
  "robot": { "id": "cartpole", "source": "catalog" },
  "task": { "recipe": "cartpole-balance" },
  "train": { "method": "rl" },
  "backend": { "prefer": ["gymnasium"], "compute": "local" }
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
