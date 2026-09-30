/* Betterment gates: pure Train / compare / deploy / catalog decisions.
 * Loaded before app.js; tested via Node (tests/test_studio_gates.py). */
(function (root) {
  const HTGates = root.HTGates || {};

  /** Local laptop/GPU launch — not OSMO harvest. */
  HTGates.isLocalLaunch = function isLocalLaunch(recipe) {
    return Boolean(
      recipe && recipe.launch_here && recipe.launch_path !== "osmo"
    );
  };

  /** OSMO harvest can submit from this machine. */
  HTGates.isHarvestLaunch = function isHarvestLaunch(recipe) {
    return Boolean(
      recipe && recipe.launch_here && recipe.launch_path === "osmo"
    );
  };

  /**
   * Split catalog cards for Tasks: local ready / remote harvest / blocked.
   * @param {object[]} recipes
   */
  HTGates.groupCatalogRecipes = function groupCatalogRecipes(recipes) {
    const list = Array.isArray(recipes) ? recipes : [];
    return {
      ready: list.filter((r) => HTGates.isLocalLaunch(r)),
      harvest: list.filter((r) => HTGates.isHarvestLaunch(r)),
      later: list.filter((r) => !r || !r.launch_here),
    };
  };

  /**
   * Decide what Train should do when canvas demos may be unsaved.
   * Prefer block + Save over OK=discard confirm (B0 footgun).
   * @param {number} pendingCount
   */
  HTGates.decideUnsavedDemoTrain = function decideUnsavedDemoTrain(pendingCount) {
    const n = Number(pendingCount) || 0;
    if (n <= 0) {
      return { action: "proceed" };
    }
    const takes = n === 1 ? "1 unsaved demo take" : `${n} unsaved demo takes`;
    return {
      action: "block_save_first",
      message:
        `You have ${takes}. Click Save so Train uses your demos — ` +
        "Train will not silently fall back to built-in scripted demos.",
      scriptedLabel: "Discard unsaved and train on built-in scripted demos",
    };
  };

  HTGates.unsavedTakesHint = function unsavedTakesHint(pendingCount) {
    const n = Number(pendingCount) || 0;
    if (n <= 0) return "";
    const label = n === 1 ? "1 unsaved take" : `${n} unsaved takes`;
    return (
      `You have ${label}. Click Save before Train — or Train will stop and ask you to Save.`
    );
  };

  /**
   * Same-recipe compare gate for Runs.
   * @param {string[]} ids
   * @param {{run_id: string, recipe?: string}[]} runs
   */
  HTGates.decideCompareSelection = function decideCompareSelection(ids, runs) {
    const list = Array.isArray(ids) ? ids : [];
    const byId = Object.fromEntries((runs || []).map((r) => [r.run_id, r]));
    const picked = list.map((id) => byId[id]).filter(Boolean);
    if (list.length !== 2 || picked.length !== 2) {
      return { ok: false, reason: "Pick two runs to compare.", runs: [] };
    }
    if (picked[0].recipe !== picked[1].recipe) {
      return {
        ok: false,
        reason: "Pick two runs of the same task.",
        runs: picked,
      };
    }
    return { ok: true, reason: "", runs: picked, recipe: picked[0].recipe };
  };

  /**
   * Deploy button enablement follows preflight — never always-clickable.
   * @param {{status?: string}} run
   * @param {{ok?: boolean, reasons?: string[], error?: string}|null} preflight
   */
  HTGates.decideDeployButton = function decideDeployButton(run, preflight) {
    const status = run && run.status;
    if (status === "queued" || status === "running") {
      return { disabled: true, title: "Wait until training finishes." };
    }
    if (status === "blocked") {
      return { disabled: true, title: "This run never trained — nothing to deploy." };
    }
    if (!preflight) {
      return { disabled: true, title: "Checking hardware gate…" };
    }
    if (preflight.ok === true) {
      return {
        disabled: false,
        title:
          "Checklist passed, but the Unitree driver is still unwired — click to confirm fail-closed.",
      };
    }
    const reason =
      (Array.isArray(preflight.reasons) && preflight.reasons[0]) ||
      preflight.error ||
      "Stays sim-only — see why below.";
    return { disabled: true, title: String(reason) };
  };

  /**
   * Next unused start_here recipe that launches locally (Cartpole → stand → mustard).
   * @param {object[]} recipes
   * @param {{recipe?: string, status?: string}[]} runs
   * @param {string|null} currentRecipeId
   */
  HTGates.nextLadderRecipe = function nextLadderRecipe(
    recipes,
    runs,
    currentRecipeId
  ) {
    const done = new Set();
    (runs || []).forEach((r) => {
      if (!r || !r.recipe) return;
      const st = String(r.status || "").toLowerCase();
      if (st === "passed" || st === "completed") done.add(r.recipe);
    });
    if (currentRecipeId) done.add(currentRecipeId);
    const ladder = (recipes || []).filter(
      (r) => r && r.start_here && HTGates.isLocalLaunch(r)
    );
    return ladder.find((r) => !done.has(r.id)) || null;
  };

  /**
   * Primary CTA on a recipe Train page.
   * @param {object} recipe
   * @param {object|null} firstLocalReady
   */
  HTGates.decideRecipePrimary = function decideRecipePrimary(
    recipe,
    firstLocalReady
  ) {
    if (!recipe) {
      return { action: "back_tasks", label: "Back to tasks" };
    }
    if (HTGates.isLocalLaunch(recipe)) {
      return { action: "train", label: "Train" };
    }
    if (HTGates.isHarvestLaunch(recipe)) {
      return { action: "train", label: "Submit for OSMO harvest" };
    }
    if (firstLocalReady && firstLocalReady.id) {
      return {
        action: "open_ready",
        label: `Try ${firstLocalReady.title || "a ready task"}`,
        recipeId: firstLocalReady.id,
      };
    }
    return { action: "back_tasks", label: "Back to ready tasks" };
  };

  /**
   * Primary CTA on a finished/blocked run detail.
   * @param {{status?: string, recipe?: string}} run
   * @param {object|null} firstLocalReady
   * @param {object|null} nextLadder
   */
  HTGates.decideRunPrimary = function decideRunPrimary(
    run,
    firstLocalReady,
    nextLadder
  ) {
    const status = String((run && run.status) || "").toLowerCase();
    if (status === "blocked") {
      if (firstLocalReady && firstLocalReady.id) {
        return {
          action: "open_ready",
          label: `Try ${firstLocalReady.title || "a ready task"}`,
          recipeId: firstLocalReady.id,
        };
      }
      return { action: "back_tasks", label: "Back to tasks" };
    }
    if (status === "passed" || status === "completed") {
      if (nextLadder && nextLadder.id) {
        return {
          action: "open_ready",
          label: `Next: ${nextLadder.title || "next task"}`,
          recipeId: nextLadder.id,
        };
      }
      return { action: "train_again", label: "Train again" };
    }
    if (status === "queued" || status === "running") {
      return { action: "wait", label: "Training…", disabled: true };
    }
    return { action: "train_again", label: "Train again" };
  };

  /**
   * Whether the Data step should look "done" in the progress strip.
   * @param {object|null} recipe
   * @param {string[]} datasets
   * @param {boolean} visitedData
   */
  HTGates.dataStepDone = function dataStepDone(recipe, datasets, visitedData) {
    if (!recipe || !recipe.imitate) return true;
    if (Array.isArray(datasets) && datasets.length > 0) return true;
    return Boolean(visitedData);
  };

  root.HTGates = HTGates;
})(typeof window !== "undefined" ? window : globalThis);
