// lesson.js - Lesson Planner form and lesson results page.
"use strict";

// ---------- Planner: grade selection + client-side validation ----------
(function () {
  const form = document.getElementById("lesson-form");
  if (!form) return;
  const boxes = [...form.querySelectorAll('input[name="grades"]')];
  const selectAll = form.querySelector("[data-select-all-grades]");

  function updateSelectAllLabel() {
    selectAll.textContent = boxes.every((b) => b.checked) ? "Clear all" : "Select all";
  }
  selectAll.addEventListener("click", () => {
    const check = !boxes.every((b) => b.checked);
    boxes.forEach((b) => (b.checked = check));
    updateSelectAllLabel();
    showError("grades", "");
  });
  boxes.forEach((b) => b.addEventListener("change", () => { updateSelectAllLabel(); showError("grades", ""); }));
  updateSelectAllLabel();

  function showError(name, message) {
    const slot = form.querySelector(`[data-error-for="${name}"]`);
    if (slot) slot.textContent = message;
    const input = form.elements[name];
    if (input && input.classList) input.classList.toggle("is-invalid", !!message);
  }

  // The server validates again; this only gives faster, field-level feedback.
  function validate() {
    const problems = {};
    const topic = form.elements.topic.value.trim();
    if (!topic) problems.topic = "Please enter a topic (for example: Fractions).";
    else if (topic.length < 2) problems.topic = "Topic is too short.";
    else if (!/\p{L}/u.test(topic)) problems.topic = "Topic must contain letters.";

    const duration = form.elements.duration;
    const d = Number(duration.value);
    if (!Number.isInteger(d) || d < Number(duration.min) || d > Number(duration.max)) {
      problems.duration = `Class duration must be between ${duration.min} and ${duration.max} minutes.`;
    }
    if (!boxes.some((b) => b.checked)) problems.grades = "Select at least one grade.";

    ["topic", "duration", "grades"].forEach((name) => showError(name, problems[name] || ""));
    return problems;
  }

  form.addEventListener("submit", (e) => {
    const problems = validate();
    const first = Object.keys(problems)[0];
    if (!first) return;
    e.preventDefault();
    (first === "grades" ? boxes[0] : form.elements[first]).focus();
  });
  ["topic", "duration"].forEach((name) =>
    form.elements[name].addEventListener("input", () => showError(name, "")));
})();

// ---------- Accessibility settings: keep preset radio and checkboxes in sync ----------
document.querySelectorAll("[data-acc-root]").forEach((root) => {
  const radios = root.querySelectorAll("[data-acc-preset]");
  const boxes = [...root.querySelectorAll("[data-acc-option]")];
  const preset = (root.querySelector("[data-acc-options]").dataset.preset || "").split(",");
  const badge = root.closest("details").querySelector("[data-acc-badge]");
  const setMode = (value) => radios.forEach((r) => (r.checked = r.value === value));

  function updateBadge() {
    const count = boxes.filter((b) => b.checked).length;
    const mode = [...radios].find((r) => r.checked)?.value;
    badge.textContent = count === 0 ? "Off" : mode === "dyslexia" ? "Dyslexia-Friendly Mode"
      : `${count} adaptation${count === 1 ? "" : "s"}`;
    badge.classList.toggle("badge-primary", count > 0);
  }

  radios.forEach((r) => r.addEventListener("change", () => {
    if (r.value === "dyslexia") boxes.forEach((b) => (b.checked = preset.includes(b.dataset.accOption)));
    if (r.value === "off") boxes.forEach((b) => (b.checked = false));
    updateBadge();
  }));
  boxes.forEach((b) => b.addEventListener("change", () => {
    const on = boxes.filter((x) => x.checked).map((x) => x.dataset.accOption);
    const isPreset = preset.every((k) => on.includes(k)) && on.every((k) => preset.includes(k) || k === "reduce_questions");
    setMode(on.length === 0 ? "off" : isPreset ? "dyslexia" : "custom");
    updateBadge();
  }));
});

// ---------- Results: answers are hidden until the teacher asks (they always print) ----------
document.querySelectorAll("[data-answers-scope]").forEach((panel) => {
  panel.classList.add("answers-hidden");
  const btn = panel.querySelector("[data-toggle-answers]");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const show = panel.classList.toggle("answers-hidden") === false;
    btn.setAttribute("aria-pressed", String(show));
    btn.textContent = show ? "Hide answers" : "Show answers";
  });
});

// ---------- Results: inline editing of generated content ----------
(function () {
  const toggle = document.getElementById("edit-toggle");
  if (!toggle) return;
  const bar = document.getElementById("edit-bar");
  const status = document.getElementById("edit-status");
  const nodes = () => document.querySelectorAll("[data-path]");
  const original = new Map();

  function start() {
    document.body.classList.add("editing");
    bar.hidden = false;
    // Everything must be visible while editing: all grades, answers, collapsed sections.
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = false));
    document.querySelectorAll("[data-answers-scope]").forEach((p) => p.classList.remove("answers-hidden"));
    document.querySelectorAll("details").forEach((d) => (d.open = true));
    nodes().forEach((n) => {
      original.set(n, n.innerText.trim());
      n.contentEditable = "true";
      n.addEventListener("input", () => n.classList.add("edited"));
    });
    toggle.disabled = true;
    bar.scrollIntoView({ block: "nearest" });
  }

  toggle.addEventListener("click", start);
  document.getElementById("edit-cancel").addEventListener("click", () => {
    if (document.querySelector("[data-path].edited") && !confirm("Discard your unsaved changes?")) return;
    window.location.reload();
  });
  document.getElementById("edit-save").addEventListener("click", async () => {
    const edits = {};
    nodes().forEach((n) => {
      const text = n.innerText.trim();
      if (text !== original.get(n)) edits[n.dataset.path] = text;
    });
    if (!Object.keys(edits).length) { status.textContent = "No changes to save."; return; }
    status.textContent = "Saving…";
    try {
      const res = await fetch(toggle.dataset.saveUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edits }),
      });
      const data = await res.json();
      if (data.errors && data.errors.length) {
        status.textContent = "Some changes were not saved: " + data.errors.join(" ");
        if (data.saved) setTimeout(() => { document.body.classList.remove("editing"); window.location.reload(); }, 2500);
        return;
      }
      document.body.classList.remove("editing");
      window.location.reload();
    } catch (err) {
      status.textContent = "Could not save (network error). Your edits are still on the page.";
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (document.body.classList.contains("editing") && document.querySelector("[data-path].edited")) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
})();
