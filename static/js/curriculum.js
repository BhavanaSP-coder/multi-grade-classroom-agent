// curriculum.js - Curriculum Library upload: drag and drop, chosen-file display and
// client-side checks (type and size). The server validates the content again.
"use strict";

(function () {
  const form = document.getElementById("upload-form");
  if (!form) return;
  const input = form.querySelector("[data-file-input]");
  const zone = form.querySelector("[data-dropzone]");
  const chosen = form.querySelector("[data-chosen-file]");
  const error = form.querySelector("[data-upload-error]");
  const maxBytes = Number(form.dataset.maxMb) * 1024 * 1024;

  function check() {
    const file = input.files && input.files[0];
    let message = "";
    if (!file) message = "Choose a .pdf or .txt file first.";
    else if (!/\.(pdf|txt)$/i.test(file.name)) message = `"${file.name}" is not supported. Upload a .pdf or .txt file.`;
    else if (file.size === 0) message = `"${file.name}" is empty.`;
    else if (file.size > maxBytes) message = `"${file.name}" is larger than ${form.dataset.maxMb} MB.`;
    error.textContent = message;
    zone.classList.toggle("is-invalid", !!message && !!file);
    return !message;
  }

  function showChosen() {
    const file = input.files && input.files[0];
    chosen.textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "";
    if (file) check();
  }

  input.addEventListener("change", showChosen);

  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (e) => {
    e.preventDefault();
    zone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("is-dragover")));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!e.dataTransfer.files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(e.dataTransfer.files[0]); // one file at a time
    input.files = transfer.files;
    showChosen();
  });

  form.addEventListener("submit", (e) => {
    if (check()) return;
    e.preventDefault();
    input.focus();
  });
})();
