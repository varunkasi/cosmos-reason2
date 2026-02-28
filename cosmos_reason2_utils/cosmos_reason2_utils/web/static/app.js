// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// ── State ───────────────────────────────────────────────────────────────────

const state = {
  currentPath: "/",
  /** @type {Set<string>} checked items in the file browser */
  browserChecked: new Set(),
  /** @type {Array<{path: string, type: string}>} selected files for inference */
  selectedFiles: [],
  /** @type {{image_tokens: number, video_tokens: number, text_tokens: number}} */
  tokenEstimate: { image_tokens: 0, video_tokens: 0, text_tokens: 0 },
  running: false,
  /** @type {{path: string, type: string} | null} file currently shown in preview */
  previewFile: null,
};

// ── DOM refs ────────────────────────────────────────────────────────────────

const $browsePath = document.getElementById("browse-path");
const $browseUp = document.getElementById("browse-up");
const $browseGo = document.getElementById("browse-go");
const $fileList = document.getElementById("file-list");
const $addSelectedBtn = document.getElementById("add-selected-btn");
const $promptInput = document.getElementById("prompt-input");
const $selectedFiles = document.getElementById("selected-files");
const $fileCount = document.getElementById("file-count");
const $budgetBar = document.getElementById("budget-bar");
const $budgetText = document.getElementById("budget-text");
const $reasoning = document.getElementById("param-reasoning");
const $temperature = document.getElementById("param-temperature");
const $topP = document.getElementById("param-top-p");
const $topK = document.getElementById("param-top-k");
const $maxTokens = document.getElementById("param-max-tokens");
const $fps = document.getElementById("param-fps");
const $maxModelLen = document.getElementById("param-max-model-len");
const $systemPrompt = document.getElementById("param-system-prompt");
const $runBtn = document.getElementById("run-btn");
const $responsePanel = document.getElementById("response-panel");
const $reasoningBlock = document.getElementById("reasoning-block");
const $reasoningContent = document.getElementById("reasoning-content");
const $responseContent = document.getElementById("response-content");
const $responseMeta = document.getElementById("response-meta");
const $modelLabel = document.getElementById("model-label");
const $healthDot = document.getElementById("health-dot");

// Preview panel
const $mediaPreview = document.getElementById("media-preview");
const $previewClose = document.getElementById("preview-close");
const $previewMediaList = document.getElementById("preview-media-list");
const $previewFilename = document.getElementById("preview-filename");
const $previewContent = document.getElementById("preview-content");

// ── Helpers ─────────────────────────────────────────────────────────────────

function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / (1024 * 1024)).toFixed(1) + " MB";
}

function fileIcon(type) {
  if (type === "dir") return "📁";
  if (type === "image") return "🖼";
  if (type === "video") return "🎬";
  return "📄";
}

function isMedia(type) {
  return type === "image" || type === "video";
}

// ── Health check ────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.vllm === "ok") {
      $healthDot.className = "health-dot ok";
      $modelLabel.textContent = data.model;
    } else {
      $healthDot.className = "health-dot err";
      $modelLabel.textContent = "vLLM unavailable";
    }
  } catch {
    $healthDot.className = "health-dot err";
    $modelLabel.textContent = "Disconnected";
  }
}

// ── File browser ────────────────────────────────────────────────────────────

async function browseTo(path) {
  state.currentPath = path;
  state.browserChecked.clear();
  $browsePath.value = path;
  $addSelectedBtn.disabled = true;

  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (data.error) {
      $fileList.innerHTML = `<div style="padding:10px;color:var(--red)">${data.error}</div>`;
      return;
    }
    renderFileList(data.entries, data.path);
  } catch (err) {
    $fileList.innerHTML = `<div style="padding:10px;color:var(--red)">Failed to browse: ${err.message}</div>`;
  }
}

function renderFileList(entries, displayPath) {
  $fileList.innerHTML = "";
  for (const entry of entries) {
    const el = document.createElement("div");
    el.className = "file-entry" + (entry.type === "dir" ? " dir" : "");

    const fullPath = displayPath.replace(/\/$/, "") + "/" + entry.name;

    if (entry.type === "dir") {
      el.innerHTML = `
        <span class="icon">${fileIcon(entry.type)}</span>
        <span class="name" title="${entry.name}">${entry.name}/</span>`;
      el.addEventListener("click", () => browseTo(fullPath));
    } else {
      const selectable = isMedia(entry.type);
      const checked = state.browserChecked.has(fullPath);
      el.innerHTML = `
        ${selectable ? `<input type="checkbox" ${checked ? "checked" : ""}>` : '<span style="width:15px"></span>'}
        <span class="icon">${fileIcon(entry.type)}</span>
        <span class="name" title="${entry.name}">${entry.name}</span>
        <span class="size">${formatBytes(entry.size || 0)}</span>`;
      if (selectable) {
        const cb = el.querySelector("input");
        cb.addEventListener("change", () => {
          if (cb.checked) state.browserChecked.add(fullPath);
          else state.browserChecked.delete(fullPath);
          $addSelectedBtn.disabled = state.browserChecked.size === 0;
        });
        // clicking anywhere on the row toggles the checkbox
        el.addEventListener("click", (e) => {
          if (e.target === cb) return;
          cb.checked = !cb.checked;
          cb.dispatchEvent(new Event("change"));
        });
      }
    }
    $fileList.appendChild(el);
  }
}

$browseUp.addEventListener("click", () => {
  const parts = state.currentPath.replace(/\/$/, "").split("/");
  parts.pop();
  browseTo(parts.join("/") || "/");
});

$browseGo.addEventListener("click", () => browseTo($browsePath.value.trim() || "/"));
$browsePath.addEventListener("keydown", (e) => {
  if (e.key === "Enter") browseTo($browsePath.value.trim() || "/");
});

// ── Selected files management ───────────────────────────────────────────────

$addSelectedBtn.addEventListener("click", () => {
  for (const path of state.browserChecked) {
    if (!state.selectedFiles.some((f) => f.path === path)) {
      // Determine type from extension
      const ext = path.split(".").pop().toLowerCase();
      const imgExts = ["jpg", "jpeg", "png", "gif", "bmp", "webp"];
      const type = imgExts.includes(ext) ? "image" : "video";
      state.selectedFiles.push({ path, type });
    }
  }
  state.browserChecked.clear();
  $addSelectedBtn.disabled = true;
  // Uncheck all checkboxes in file list
  $fileList.querySelectorAll("input[type=checkbox]").forEach((cb) => (cb.checked = false));
  renderSelectedFiles();
  requestTokenEstimate();
});

function renderSelectedFiles() {
  $fileCount.textContent = `(${state.selectedFiles.length})`;
  $selectedFiles.innerHTML = "";
  for (let i = 0; i < state.selectedFiles.length; i++) {
    const f = state.selectedFiles[i];
    const basename = f.path.split("/").pop();
    const el = document.createElement("div");
    const isPreviewing = state.previewFile && state.previewFile.path === f.path;
    el.className = "selected-file"
      + (isMedia(f.type) ? " previewable" : "")
      + (isPreviewing ? " previewing" : "");
    el.innerHTML = `
      <span class="icon">${fileIcon(f.type)}</span>
      <span class="path" title="${f.path}">${basename}</span>
      <span class="tokens" data-idx="${i}">…</span>
      <button class="remove" title="Remove">&times;</button>`;
    el.querySelector(".remove").addEventListener("click", (e) => {
      e.stopPropagation();
      const wasPreviewTarget = state.previewFile && state.previewFile.path === f.path;
      state.selectedFiles.splice(i, 1);
      if (wasPreviewTarget) {
        const nextMedia = state.selectedFiles.find((sf) => isMedia(sf.type));
        if (nextMedia) openPreview(nextMedia);
        else closePreview();
      }
      renderSelectedFiles();
      requestTokenEstimate();
    });
    if (isMedia(f.type)) {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".remove")) return;
        openPreview(f);
      });
    }
    $selectedFiles.appendChild(el);
  }
}

// ── Token estimation ────────────────────────────────────────────────────────

let _estimateTimer = null;

function requestTokenEstimate() {
  clearTimeout(_estimateTimer);
  _estimateTimer = setTimeout(doTokenEstimate, 300);
}

async function doTokenEstimate() {
  const images = state.selectedFiles.filter((f) => f.type === "image").map((f) => f.path);
  const videos = state.selectedFiles.filter((f) => f.type === "video").map((f) => f.path);
  const prompt = $promptInput.value;
  const fps = parseFloat($fps.value) || 2.0;
  const maxModelLen = parseInt($maxModelLen.value) || 16384;
  const maxTokens = parseInt($maxTokens.value) || 4096;

  try {
    const res = await fetch("/api/estimate-tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images, videos, prompt, fps, max_model_len: maxModelLen, max_tokens: maxTokens }),
    });
    const data = await res.json();
    state.tokenEstimate = data;
    updateBudgetDisplay(data);
  } catch {
    // silently fail — estimation is best-effort
  }
}

function updateBudgetDisplay(est) {
  const totalInput = est.image_tokens + est.video_tokens + est.text_tokens;
  const totalUsed = totalInput + (parseInt($maxTokens.value) || 4096);
  const maxLen = est.max_model_len || parseInt($maxModelLen.value) || 16384;
  const pct = Math.min(100, (totalUsed / maxLen) * 100);

  $budgetBar.style.width = pct + "%";
  $budgetBar.className = "budget-bar" + (pct > 90 ? " danger" : pct > 70 ? " warn" : "");

  $budgetText.textContent =
    `Images: ~${est.image_tokens}  Video: ~${est.video_tokens}  Text: ~${est.text_tokens}` +
    `  |  ${totalUsed.toLocaleString()} / ${maxLen.toLocaleString()}`;
}

// Debounced estimation on parameter/prompt changes
$promptInput.addEventListener("input", requestTokenEstimate);
$fps.addEventListener("change", requestTokenEstimate);
$maxModelLen.addEventListener("change", requestTokenEstimate);
$maxTokens.addEventListener("change", requestTokenEstimate);

// ── Reasoning toggle updates defaults ───────────────────────────────────────

$reasoning.addEventListener("change", async () => {
  try {
    const res = await fetch(`/api/defaults?reasoning=${$reasoning.checked}`);
    const d = await res.json();
    $temperature.value = d.temperature;
    $topP.value = d.top_p;
    $topK.value = d.top_k;
    $maxTokens.value = d.max_tokens;
    $fps.value = d.fps;
    requestTokenEstimate();
  } catch {
    // keep current values
  }
});

// ── Run inference ───────────────────────────────────────────────────────────

$runBtn.addEventListener("click", runInference);

async function runInference() {
  if (state.running) return;
  const prompt = $promptInput.value.trim();
  if (!prompt) {
    alert("Please enter a prompt.");
    return;
  }

  state.running = true;
  $runBtn.disabled = true;
  $runBtn.textContent = "Running…";
  $runBtn.classList.add("loading");
  $responsePanel.hidden = true;

  const images = state.selectedFiles.filter((f) => f.type === "image").map((f) => f.path);
  const videos = state.selectedFiles.filter((f) => f.type === "video").map((f) => f.path);

  const body = {
    prompt,
    images,
    videos,
    reasoning: $reasoning.checked,
    system_prompt: $systemPrompt.value,
    temperature: parseFloat($temperature.value),
    max_tokens: parseInt($maxTokens.value),
    top_p: parseFloat($topP.value),
    top_k: parseInt($topK.value),
    fps: parseFloat($fps.value),
    max_model_len: parseInt($maxModelLen.value),
  };

  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) {
      showError(data.error);
      return;
    }
    showResponse(data);
  } catch (err) {
    showError(err.message);
  } finally {
    state.running = false;
    $runBtn.disabled = false;
    $runBtn.textContent = "Run Inference";
    $runBtn.classList.remove("loading");
  }
}

function showResponse(data) {
  $responsePanel.hidden = false;
  $responseContent.textContent = data.content;

  if (data.reasoning_content) {
    $reasoningBlock.hidden = false;
    $reasoningContent.textContent = data.reasoning_content;
  } else {
    $reasoningBlock.hidden = true;
  }

  const u = data.usage || {};
  $responseMeta.textContent =
    `Tokens: ${u.prompt_tokens || 0} prompt + ${u.completion_tokens || 0} completion = ${u.total_tokens || 0} total` +
    `  |  Time: ${data.duration_s || 0}s`;
}

function showError(msg) {
  $responsePanel.hidden = false;
  $reasoningBlock.hidden = true;
  $responseContent.textContent = "Error: " + msg;
  $responseContent.style.color = "var(--red)";
  $responseMeta.textContent = "";
  // Reset color after next successful response
  setTimeout(() => ($responseContent.style.color = ""), 0);
}

// ── Media preview ───────────────────────────────────────────────────────────

function openPreview(file) {
  state.previewFile = file;
  $mediaPreview.classList.add("visible");
  renderPreviewMediaList();
  renderPreviewContent();
  renderSelectedFiles(); // refresh highlight
}

function closePreview() {
  state.previewFile = null;
  $mediaPreview.classList.remove("visible");
  renderSelectedFiles(); // clear highlight
}

function renderPreviewMediaList() {
  $previewMediaList.innerHTML = "";
  const mediaFiles = state.selectedFiles.filter((f) => isMedia(f.type));
  for (const f of mediaFiles) {
    const chip = document.createElement("button");
    const basename = f.path.split("/").pop();
    const isActive = state.previewFile && state.previewFile.path === f.path;
    chip.className = "preview-media-chip" + (isActive ? " active" : "");
    chip.title = f.path;
    chip.textContent = `${fileIcon(f.type)} ${basename}`;
    chip.addEventListener("click", () => openPreview(f));
    $previewMediaList.appendChild(chip);
  }
}

function renderPreviewContent() {
  if (!state.previewFile) {
    $previewContent.innerHTML = '<div class="preview-empty">Select a media file to preview</div>';
    $previewFilename.textContent = "";
    return;
  }

  const mediaUrl = `/api/media?path=${encodeURIComponent(state.previewFile.path)}`;
  $previewFilename.textContent = state.previewFile.path;
  $previewFilename.title = state.previewFile.path;

  if (state.previewFile.type === "image") {
    $previewContent.innerHTML = `<img src="${mediaUrl}" alt="Preview" draggable="false">`;
  } else if (state.previewFile.type === "video") {
    $previewContent.innerHTML = `<video src="${mediaUrl}" controls preload="metadata"></video>`;
  }
}

$previewClose.addEventListener("click", closePreview);

// ── Init ────────────────────────────────────────────────────────────────────

checkHealth();
setInterval(checkHealth, 30000);
browseTo("/");
requestTokenEstimate();
