/* Betterment B0: pure Train gates. Loaded before app.js; tested via Node. */
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

  root.HTGates = HTGates;
})(typeof window !== "undefined" ? window : globalThis);
