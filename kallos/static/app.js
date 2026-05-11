// kallos frontend.
// One file, organized in sections. No framework, no build step.

(() => {
  "use strict";

  // --- state ---

  const state = {
    loaded: false,
    settings: defaultSettings(),
    compareMode: "edited",  // edited | sxs | slider
    pendingRender: 0,        // monotonic id; latest one wins
    objectUrl: null,         // for cleanup on next render
  };

  function defaultSettings() {
    return {
      brightness: 0, contrast: 0, warmth: 0, vibrance: 0, clarity: 0,
      sharpen: 0, denoise: 0, ai_deblur: false, wb_preset: "as_shot",
    };
  }

  // --- dom shortcuts ---

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const dropZone   = $("#drop-zone");
  const fileInput  = $("#file-input");
  const viewer     = $("#viewer");
  const imgEdited  = $("#img-edited");
  const imgSxsO    = $("#img-sxs-original");
  const imgSxsE    = $("#img-sxs-edited");
  const imgSlO     = $("#img-slider-original");
  const imgSlE     = $("#img-slider-edited");
  const sliderClip = $("#slider-clip");
  const sliderHandle = $("#slider-handle");
  const btnAuto    = $("#btn-auto");
  const btnReset   = $("#btn-reset");
  const btnCompare = $("#btn-compare");
  const btnSave    = $("#btn-save");
  const status     = $("#status");
  const filenameEl = $("#filename");
  const formatSelect = $("#save-format");
  const sliderInputs = $$("input[type='range'][data-key]");
  const selectInputs = $$("select[data-key]");
  const checkInputs  = $$("input[type='checkbox'][data-key]");

  // --- upload ---

  function setupDropZone() {
    ["dragenter", "dragover"].forEach((ev) =>
      dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragging");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      dropZone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragging");
      })
    );
    dropZone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) uploadFile(file);
    });
    dropZone.addEventListener("click", (e) => {
      if (e.target.tagName !== "LABEL") fileInput.click();
    });
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) uploadFile(file);
    });
  }

  async function uploadFile(file) {
    setStatus("Loading…");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      onUploaded(data);
    } catch (err) {
      setStatus(`Could not load that file: ${err.message}`, true);
    }
  }

  function onUploaded(data) {
    state.loaded = true;
    state.settings = data.settings || defaultSettings();
    syncControlsFromState();
    dropZone.hidden = true;
    viewer.hidden = false;
    filenameEl.textContent = data.source_name;
    [btnAuto, btnReset, btnCompare, btnSave].forEach((b) => (b.disabled = false));

    imgSxsO.src = "/original.jpg";
    imgSlO.src  = "/original.jpg";
    setStatus("");
    renderPreview();
  }

  // --- render ---

  async function renderPreview() {
    if (!state.loaded) return;
    const id = ++state.pendingRender;
    try {
      const res = await fetch("/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.settings),
      });
      if (id !== state.pendingRender) return;  // a newer render is in flight
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = URL.createObjectURL(blob);
      imgEdited.src = state.objectUrl;
      imgSxsE.src   = state.objectUrl;
      imgSlE.src    = state.objectUrl;
    } catch (err) {
      setStatus(`Render failed: ${err.message}`, true);
    }
  }

  const debouncedRender = debounce(renderPreview, 80);

  // --- controls ---

  function syncControlsFromState() {
    for (const inp of sliderInputs) {
      const key = inp.dataset.key;
      inp.value = String(state.settings[key] ?? 0);
      inp.nextElementSibling.value = inp.value;  // <output>
    }
    for (const sel of selectInputs) {
      sel.value = state.settings[sel.dataset.key] ?? "as_shot";
    }
    for (const cb of checkInputs) {
      cb.checked = !!state.settings[cb.dataset.key];
    }
  }

  function wireControls() {
    for (const inp of sliderInputs) {
      inp.addEventListener("input", () => {
        const key = inp.dataset.key;
        const v = parseFloat(inp.value);
        state.settings[key] = v;
        inp.nextElementSibling.value = inp.value;
        debouncedRender();
      });
    }
    for (const sel of selectInputs) {
      sel.addEventListener("change", () => {
        state.settings[sel.dataset.key] = sel.value;
        renderPreview();
      });
    }
    for (const cb of checkInputs) {
      cb.addEventListener("change", () => {
        state.settings[cb.dataset.key] = cb.checked;
        if (cb.dataset.key === "ai_deblur") {
          setStatus(cb.checked ? "AI Deblur will apply on Save." : "");
        }
      });
    }
  }

  // --- buttons ---

  btnAuto.addEventListener("click", async () => {
    btnAuto.disabled = true;
    try {
      const res = await fetch("/auto", { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      state.settings = { ...state.settings, ...(await res.json()) };
      syncControlsFromState();
      await renderPreview();
      setStatus("Auto Enhance applied.");
    } catch (err) {
      setStatus(`Auto failed: ${err.message}`, true);
    } finally {
      btnAuto.disabled = false;
    }
  });

  btnReset.addEventListener("click", async () => {
    const res = await fetch("/reset", { method: "POST" });
    state.settings = (await res.json()).settings;
    syncControlsFromState();
    renderPreview();
    setStatus("Reset to original.");
  });

  // --- compare modes ---

  const COMPARE_MODES = ["edited", "sxs", "slider"];
  const COMPARE_LABELS = {
    edited: "Compare: Edited",
    sxs:    "Compare: Side-by-side",
    slider: "Compare: Slider",
  };

  btnCompare.addEventListener("click", () => {
    const i = COMPARE_MODES.indexOf(state.compareMode);
    state.compareMode = COMPARE_MODES[(i + 1) % COMPARE_MODES.length];
    applyCompareMode();
  });

  function applyCompareMode() {
    btnCompare.textContent = COMPARE_LABELS[state.compareMode];
    $("#view-edited").hidden = state.compareMode !== "edited";
    $("#view-sxs").hidden    = state.compareMode !== "sxs";
    $("#view-slider").hidden = state.compareMode !== "slider";
  }

  // slider-divider drag
  let dragging = false;
  const sliderStack = sliderHandle.parentElement;
  sliderHandle.addEventListener("pointerdown", (e) => {
    dragging = true;
    sliderHandle.setPointerCapture(e.pointerId);
  });
  sliderHandle.addEventListener("pointerup", (e) => {
    dragging = false;
    sliderHandle.releasePointerCapture(e.pointerId);
  });
  sliderHandle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = sliderStack.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    sliderClip.style.width = `${pct * 100}%`;
    sliderHandle.style.left = `${pct * 100}%`;
  });

  // --- save ---

  btnSave.addEventListener("click", async () => {
    btnSave.disabled = true;
    setStatus(state.settings.ai_deblur ? "Saving (AI deblur, this may take a moment)…" : "Saving…");
    try {
      const res = await fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings: state.settings,
          format: formatSelect.value,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStatus(`Saved to ${data.path}`);
    } catch (err) {
      setStatus(`Save failed: ${err.message}`, true);
    } finally {
      btnSave.disabled = false;
    }
  });

  // --- helpers ---

  function debounce(fn, ms) {
    let t = null;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  function setStatus(text, isError = false) {
    status.textContent = text;
    status.style.color = isError ? "var(--danger)" : "var(--muted)";
  }

  // --- boot ---

  setupDropZone();
  wireControls();
  applyCompareMode();
})();
