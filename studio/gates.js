/* Betterment gates: pure Train / compare / deploy decisions.
 * Loaded before app.js; tested via Node (tests/test_studio_gates.py). */
(function (root) {
  const HTGates = root.HTGates || {};

  /**
   * Decide what Train should do when canvas demos may be unsaved.
   * @param {number} pendingCount
   * @returns {{action: string, message?: string, confirmLabel?: string, cancelLabel?: string}}
   */
  HTGates.decideUnsavedDemoTrain = function decideUnsavedDemoTrain(pendingCount) {
    const n = Number(pendingCount) || 0;
    if (n <= 0) {
      return { action: "proceed" };
    }
    const takes = n === 1 ? "1 unsaved demo take" : `${n} unsaved demo takes`;
    return {
      action: "confirm_discard_or_cancel",
      message:
        `You have ${takes}. Save them first so Train uses your demos — ` +
        "otherwise Train would silently use built-in scripted demos.",
      confirmLabel: "Discard unsaved and train on built-in scripted demos",
      cancelLabel: "Cancel — go Save first",
    };
  };

  HTGates.unsavedTakesHint = function unsavedTakesHint(pendingCount) {
    const n = Number(pendingCount) || 0;
    if (n <= 0) return "";
    const label = n === 1 ? "1 unsaved take" : `${n} unsaved takes`;
    return (
      `You have ${label}. Click Save before Train — or Train will ask whether to discard them.`
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

  root.HTGates = HTGates;
})(typeof window !== "undefined" ? window : globalThis);
