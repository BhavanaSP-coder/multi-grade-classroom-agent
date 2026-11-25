// app.js - shared interactions used on every page:
// mobile navigation, dismissible notifications, confirm dialogs, loading state for
// long-running forms (and double-submit protection), accessible tabs, printing.
"use strict";

// ---------- Mobile navigation ----------
(function () {
  const toggle = document.querySelector("[data-nav-toggle]");
  const backdrop = document.querySelector("[data-nav-close]");
  if (!toggle) return;

  function setOpen(open) {
    document.body.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    backdrop.hidden = !open;
    if (open) document.querySelector("#sidebar a")?.focus();
  }
  toggle.addEventListener("click", () => setOpen(!document.body.classList.contains("nav-open")));
  backdrop.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.body.classList.contains("nav-open")) {
      setOpen(false);
      toggle.focus();
    }
  });
})();

// ---------- Notifications ----------
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-dismiss]");
  if (btn) btn.closest("[data-dismissible]")?.remove();
});

// ---------- Confirm dialog for destructive forms: <form data-confirm="Question?"> ----------
(function () {
  const dialog = document.querySelector("[data-confirm-dialog]");

  document.addEventListener("submit", (e) => {
    const form = e.target;
    if (!form.dataset.confirm || form.dataset.confirmed === "1") return;
    e.preventDefault();
    e.stopImmediatePropagation();

    const proceed = () => {
      form.dataset.confirmed = "1";
      form.requestSubmit ? form.requestSubmit() : form.submit();
    };
    if (!dialog || typeof dialog.showModal !== "function") {
      if (window.confirm(form.dataset.confirm)) proceed();
      return;
    }
    dialog.querySelector("[data-confirm-title]").textContent = form.dataset.confirm;
    dialog.querySelector("[data-confirm-text]").textContent = form.dataset.confirmText || "";
    dialog.querySelector("[data-confirm-ok]").textContent = form.dataset.confirmOk || "Confirm";
    dialog.returnValue = "";
    dialog.showModal();
    dialog.addEventListener("close", () => { if (dialog.returnValue === "ok") proceed(); }, { once: true });
  }, true); // capture: runs before page-specific handlers
})();

// ---------- Loading overlay + double-submit protection: <form data-loading="Title"> ----------
const overlay = document.querySelector("[data-loading-overlay]");

function showLoading(title, text) {
  if (!overlay) return;
  overlay.querySelector("[data-loading-title]").textContent = title || "Working…";
  overlay.querySelector("[data-loading-text]").textContent = text || "This usually takes a few seconds.";
  overlay.hidden = false;
}

document.addEventListener("submit", (e) => {
  const form = e.target;
  if (e.defaultPrevented) return; // page validation stopped it
  if (form.dataset.submitting === "1") { e.preventDefault(); return; }
  form.dataset.submitting = "1";
  form.querySelectorAll('button[type="submit"], button:not([type])').forEach((b) => {
    if (b === e.submitter || !e.submitter) b.classList.add("is-busy");
    b.disabled = true;
  });
  if (form.dataset.loading !== undefined) showLoading(form.dataset.loading, form.dataset.loadingText);
});

// Restore state when the page comes back from the browser cache (Back button).
window.addEventListener("pageshow", () => {
  if (overlay) overlay.hidden = true;
  document.querySelectorAll("form[data-submitting]").forEach((f) => {
    delete f.dataset.submitting;
    delete f.dataset.confirmed;
    f.querySelectorAll("button").forEach((b) => { b.disabled = false; b.classList.remove("is-busy"); });
  });
});

// ---------- Tabs: role="tablist" with buttons role="tab" aria-controls="panel-id" ----------
function selectTab(tab, focus) {
  const list = tab.closest('[role="tablist"]');
  list.querySelectorAll('[role="tab"]').forEach((t) => {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    const panel = document.getElementById(t.getAttribute("aria-controls"));
    if (panel) panel.hidden = !on;
  });
  if (focus) tab.focus();
  tab.dispatchEvent(new CustomEvent("tabchange", { bubbles: true }));
}

document.querySelectorAll('[role="tablist"]').forEach((list) => {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (step) selectTab(tabs[(i + step + tabs.length) % tabs.length], true);
      if (e.key === "Home") selectTab(tabs[0], true);
      if (e.key === "End") selectTab(tabs[tabs.length - 1], true);
    });
  });
});

// ---------- Printing ----------
document.querySelectorAll("[data-print]").forEach((b) => b.addEventListener("click", () => window.print()));
let closedDetails = [];
window.addEventListener("beforeprint", () => {
  closedDetails = [...document.querySelectorAll("details:not([open])")];
  closedDetails.forEach((d) => (d.open = true));
});
window.addEventListener("afterprint", () => {
  closedDetails.forEach((d) => (d.open = false));
  closedDetails = [];
});
