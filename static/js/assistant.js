// assistant.js - Adaptive Classroom Assistant:
// section tabs, editable setup rows, break schedule with live end time,
// weekly-demand counter and the timetable cell editor (edit / swap / lock).
// Backend validation stays authoritative; this only gives immediate feedback.
"use strict";

// ---------- Section tabs: keep ?tab= in the URL so reloads return to the same tab ----------
document.addEventListener("tabchange", (e) => {
  const key = e.target.dataset.atab;
  if (!key) return;
  const url = new URL(window.location);
  url.searchParams.set("tab", key);
  history.replaceState(null, "", url);
});
document.querySelectorAll("[data-goto-tab]").forEach((link) =>
  link.addEventListener("click", (e) => {
    const tab = document.querySelector(`[data-atab="${link.dataset.gotoTab}"]`);
    if (!tab) return;
    e.preventDefault();
    tab.click();
    tab.focus();
  }));

// ---------- Setup form: add/remove rows, load colours, weekly demand, unsaved guard ----------
(function () {
  const form = document.getElementById("setup-form");
  if (!form) return;

  form.addEventListener("click", (e) => {
    const add = e.target.closest("[data-add-row]");
    if (add) {
      const name = add.dataset.addRow;
      const tpl = form.querySelector(`template[data-row-template="${name}"]`);
      const list = form.querySelector(`[data-row-list="${name}"]`);
      list.append(tpl.content.cloneNode(true));
      list.lastElementChild.querySelector("input, select").focus();
      updateDemand();
      return;
    }
    const remove = e.target.closest("[data-remove-row]");
    if (remove) {
      remove.closest(".row-item").remove();
      updateDemand();
    }
  });

  form.addEventListener("change", (e) => {
    const sel = e.target.closest("[data-load-select]");
    if (sel) {
      sel.classList.remove("load-high", "load-medium", "load-low");
      sel.classList.add("load-" + sel.value);
    }
  });

  // "x of y weekly periods used"
  const demand = form.querySelector("[data-demand]");
  function updateDemand() {
    if (!demand) return;
    const periods = parseInt(form.elements.num_periods.value, 10) || 0;
    const days = form.querySelectorAll('input[name="working_days"]:checked').length;
    const fixed = form.querySelectorAll('[data-row-list="fixed"] .row-item').length;
    const slots = periods * days - fixed;
    const used = [...form.querySelectorAll("[data-ppw]")].reduce((sum, i) => sum + (parseInt(i.value, 10) || 0), 0);
    demand.textContent = `${used} of ${slots} weekly periods used${used > slots ? " - too many" : ""}`;
    demand.classList.toggle("over", used > slots);
  }
  form.addEventListener("input", (e) => {
    if (e.target.matches("[data-ppw], [data-periods], input[name='working_days']")) updateDemand();
  });
  form.addEventListener("change", (e) => {
    if (e.target.matches("[data-periods], input[name='working_days']")) updateDemand();
  });

  let dirty = false;
  form.addEventListener("input", () => (dirty = true));
  form.addEventListener("submit", (e) => { if (!e.defaultPrevented) dirty = false; });
  window.addEventListener("beforeunload", (e) => {
    if (dirty) { e.preventDefault(); e.returnValue = ""; }
  });
})();

// ---------- Break schedule: positions depend on the period count; live end time ----------
(function () {
  const root = document.querySelector("[data-break-schedule]");
  if (!root) return;
  const form = root.closest("form");
  const periodsSelect = form.querySelector("[data-periods]");
  const startInput = form.querySelector("[data-start-time]");
  const lengthInput = form.querySelector("[data-period-length]");
  const rowsBox = root.querySelector("[data-break-rows]");
  const template = root.querySelector("template[data-break-template]");
  const addButton = root.querySelector("[data-break-add]");
  const hint = root.querySelector("[data-break-hint]");
  const counter = root.querySelector("[data-break-count]");
  const empty = root.querySelector("[data-break-empty]");
  const endText = root.querySelector("[data-day-end]");
  const detail = root.querySelector("[data-day-detail]");
  const blocked = form.querySelector("[data-setup-blocked]");
  const MIN = Number(root.dataset.minBreak);
  const MAX = Number(root.dataset.maxBreak);

  const rows = () => [...rowsBox.querySelectorAll("[data-break-row]")];
  const positionOf = (row) => parseInt(row.querySelector("[data-break-after]").value, 10);
  const periodCount = () => parseInt(periodsSelect.value, 10) || 1;
  const validPositions = () => Array.from({ length: Math.max(0, periodCount() - 1) }, (_, i) => i + 1);
  const toMinutes = (v) => { const m = /^(\d{1,2}):(\d{2})/.exec(v || ""); return m ? +m[1] * 60 + +m[2] : null; };
  const format12 = (t) => {
    const h = Math.floor(t / 60) % 24, m = t % 60;
    return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
  };
  const option = (value, label) => Object.assign(document.createElement("option"), { value, textContent: label });

  // Rebuild every "After period" list so it shows only valid positions not used by another break.
  function refresh() {
    const n = periodCount();
    const allowed = validPositions();
    const claimed = new Set();
    let invalidCount = 0;

    rows().forEach((row) => {
      const select = row.querySelector("[data-break-after]");
      const minutes = row.querySelector("[data-break-minutes]");
      const message = row.querySelector("[data-break-msg]");
      const current = parseInt(select.value, 10);
      const usedByOthers = new Set(rows().filter((r) => r !== row).map(positionOf));

      select.replaceChildren();
      allowed.forEach((p) => { if (!usedByOthers.has(p) || p === current) select.append(option(p, `After period ${p}`)); });
      let problem = "";
      if (!allowed.includes(current)) {
        select.append(option(current, `Period ${current} (invalid)`));
        problem = allowed.length
          ? `The day has ${n} periods, so a break cannot follow period ${current}. Choose another position or remove this break.`
          : "A one-period day has no gap for a break. Remove this break or add periods.";
      } else if (claimed.has(current)) {
        problem = `Another break already follows period ${current}. Choose a different position.`;
      }
      select.value = String(current);
      if (allowed.includes(current)) claimed.add(current);

      const m = parseInt(minutes.value, 10);
      const badMinutes = !(m >= MIN && m <= MAX);
      minutes.classList.toggle("is-invalid", badMinutes);
      if (!problem && badMinutes) problem = `Duration must be between ${MIN} and ${MAX} minutes.`;

      row.classList.toggle("is-invalid", !!problem);
      message.textContent = problem;
      if (problem) invalidCount += 1;
    });

    const free = allowed.filter((p) => !claimed.has(p));
    counter.textContent = `${claimed.size} of ${allowed.length} positions used`;
    empty.hidden = rows().length > 0;
    addButton.disabled = free.length === 0;
    hint.textContent = !allowed.length
      ? "A one-period day has no gaps between periods, so no break can be added."
      : free.length === 0 ? `Every gap between periods already has a break (${allowed.length} for ${n} periods).` : "";
    updateEndTime(allowed);
    return invalidCount;
  }

  function updateEndTime(allowed) {
    const start = toMinutes(startInput.value);
    const length = parseInt(lengthInput.value, 10) || 0;
    const n = periodCount();
    const counted = new Set();
    let breakMinutes = 0;
    rows().forEach((row) => {
      const p = positionOf(row);
      if (allowed.includes(p) && !counted.has(p)) {
        counted.add(p);
        breakMinutes += parseInt(row.querySelector("[data-break-minutes]").value, 10) || 0;
      }
    });
    if (start === null) {
      endText.textContent = "Set a start time to see when the school day ends.";
      return;
    }
    endText.innerHTML = `School day ends at <strong>${format12(start + n * length + breakMinutes)}</strong>.`;
    detail.textContent = `${n} × ${length} min of teaching + ${breakMinutes} min of breaks`;
  }

  function addBreak() {
    const allowed = validPositions();
    const used = new Set(rows().map(positionOf));
    const free = allowed.filter((p) => !used.has(p));
    if (!free.length) return;
    const last = Math.max(0, ...[...used].filter((p) => allowed.includes(p)));
    const pick = free.find((p) => p > last) ?? free[0]; // next free position after the last break
    rowsBox.append(template.content.cloneNode(true));
    const row = rows().at(-1);
    const select = row.querySelector("[data-break-after]");
    select.append(option(pick, `After period ${pick}`));
    select.value = String(pick);
    refresh();
    const name = row.querySelector('[name="br_name"]');
    name.focus();
    name.select();
  }

  addButton.addEventListener("click", addBreak);
  rowsBox.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-break-remove]");
    if (!btn) return;
    btn.closest("[data-break-row]").remove();
    refresh();
    addButton.focus();
  });
  rowsBox.addEventListener("change", refresh);
  rowsBox.addEventListener("input", (e) => { if (e.target.matches("[data-break-minutes]")) refresh(); });
  [periodsSelect, startInput, lengthInput].forEach((el) => {
    el.addEventListener("change", refresh);
    el.addEventListener("input", refresh);
  });

  // Block saving while a break or the period count is invalid (the server checks again).
  form.addEventListener("submit", (e) => {
    const invalid = refresh();
    const overLimit = /over limit/.test(periodsSelect.selectedOptions[0]?.textContent || "");
    if (!invalid && !overLimit) { blocked.hidden = true; return; }
    e.preventDefault();
    blocked.textContent = overLimit
      ? "Choose a number of periods within the limit before saving."
      : `Fix ${invalid} break${invalid > 1 ? "s" : ""} in the break schedule before saving: remove it or choose another position.`;
    blocked.hidden = false;
    (rowsBox.querySelector(".is-invalid") || periodsSelect).scrollIntoView({ block: "center", behavior: "smooth" });
  });

  refresh();
})();

// ---------- Timetable cell editor: set subject, swap two periods, lock ----------
(function () {
  const editor = document.getElementById("cell-editor");
  if (!editor) return;
  const form = editor.querySelector("form");
  const field = (name) => form.elements[name];
  const label = editor.querySelector("[data-ed-label]");
  const hint = editor.querySelector("[data-ed-hint]");
  const lockButton = editor.querySelector("[data-ed-lock]");
  const setButton = editor.querySelector('[data-ed-action="set"]');
  const swapButton = editor.querySelector("[data-ed-swap]");
  const lessonLink = editor.querySelector("[data-ed-lesson]");
  const defaultHint = hint.textContent;
  let selected = null;
  let swapping = false;

  function clearSelection() {
    document.querySelectorAll(".timetable td.is-selected, .timetable td.is-swap-target")
      .forEach((c) => c.classList.remove("is-selected", "is-swap-target"));
  }

  function select(cell) {
    if (swapping && selected && cell !== selected) {
      if (cell.dataset.target !== selected.dataset.target) {
        hint.textContent = "Pick a period in the same timetable to swap with.";
        return;
      }
      cell.classList.add("is-swap-target");
      field("action").value = "swap";
      field("cell_b").value = cell.dataset.key;
      hint.textContent = `Swapping ${selected.dataset.label} with ${cell.dataset.label}…`;
      form.requestSubmit ? form.requestSubmit() : form.submit();
      return;
    }
    clearSelection();
    swapping = false;
    selected = cell;
    cell.classList.add("is-selected");
    editor.hidden = false;
    field("target").value = cell.dataset.target;
    field("cell_a").value = cell.dataset.key;
    field("cell_b").value = "";

    const isFixed = cell.dataset.kind === "fixed";
    const locked = cell.dataset.locked === "1";
    label.textContent = `${cell.dataset.label} · ${cell.dataset.subject} (${cell.dataset.target === "proposal" ? "proposal" : "current"})`;
    lockButton.lastChild.textContent = locked ? "Unlock" : "Lock";
    lockButton.disabled = isFixed;
    setButton.disabled = isFixed || locked;
    swapButton.disabled = isFixed || locked;
    const subject = field("subject");
    if ([...subject.options].some((o) => o.value === cell.dataset.subject)) subject.value = cell.dataset.subject;
    lessonLink.hidden = !cell.dataset.lesson;
    if (cell.dataset.lesson) lessonLink.href = cell.dataset.lesson;
    hint.textContent = isFixed ? "Fixed activities are changed in Setup."
      : locked ? "Locked periods are kept when you regenerate. Unlock to change this period."
      : defaultHint;
  }

  document.querySelectorAll(".timetable td.cell").forEach((cell) => {
    cell.addEventListener("click", () => select(cell));
    cell.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(cell); }
    });
  });
  editor.querySelectorAll("[data-ed-action]").forEach((btn) =>
    btn.addEventListener("click", () => (field("action").value = btn.dataset.edAction)));
  swapButton.addEventListener("click", () => {
    if (!selected) return;
    swapping = true;
    hint.textContent = `Now click the period to swap with ${selected.dataset.label}.`;
  });
  editor.querySelector("[data-ed-close]").addEventListener("click", () => {
    editor.hidden = true;
    clearSelection();
    if (selected) selected.focus();
    selected = null;
    swapping = false;
  });
})();
