// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// ── State ───────────────────────────────────────────────────────────────────

const state = {
  /** @type {"landing"|"workspace-list"|"workspace"|"picker"} */
  view: "landing",
  /** @type {object|null} active workspace {id, name, folders, created_at, updated_at} */
  activeWorkspace: null,

  /** @type {string|null} folder currently being browsed inside the workspace */
  activeFolderRoot: null,
  /** Current browse path within the active folder */
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

  /** Folder picker state */
  pickerPath: "/",
};

// ── DOM refs ────────────────────────────────────────────────────────────────

// Landing view
const $landingView = document.getElementById("landing-view");
const $landingButtons = document.getElementById("landing-buttons");
const $btnLoadWs = document.getElementById("btn-load-ws");
const $btnNewWs = document.getElementById("btn-new-ws");
const $wsListContainer = document.getElementById("ws-list-container");
const $wsList = document.getElementById("ws-list");
const $btnWsBack = document.getElementById("btn-ws-back");

// Workspace view
const $workspaceView = document.getElementById("workspace-view");
const $wsName = document.getElementById("ws-name");
const $btnSaveWs = document.getElementById("btn-save-ws");
const $btnExitWs = document.getElementById("btn-exit-ws");
const $wsFolders = document.getElementById("ws-folders");
const $btnAddFolder = document.getElementById("btn-add-folder");
const $wsBrowserSection = document.getElementById("ws-browser-section");
const $wsBrowseUp = document.getElementById("ws-browse-up");
const $wsBrowsePath = document.getElementById("ws-browse-path");
const $wsFileList = document.getElementById("ws-file-list");
const $addSelectedBtn = document.getElementById("add-selected-btn");

// Picker view
const $pickerView = document.getElementById("picker-view");
const $pickerUp = document.getElementById("picker-up");
const $pickerPath = document.getElementById("picker-path");
const $pickerGo = document.getElementById("picker-go");
const $pickerList = document.getElementById("picker-list");
const $pickerAdd = document.getElementById("picker-add");
const $pickerCancel = document.getElementById("picker-cancel");

// Workspace (center) panel
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

function basename(path) {
  return path.replace(/\/$/, "").split("/").pop() || path;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ── View switching ──────────────────────────────────────────────────────────

function showView(view) {
  state.view = view;
  $landingView.style.display = (view === "landing" || view === "workspace-list") ? "" : "none";
  $workspaceView.style.display = view === "workspace" ? "flex" : "none";
  $pickerView.style.display = view === "picker" ? "flex" : "none";

  // Within landing, toggle between buttons and list
  if (view === "landing") {
    $landingButtons.style.display = "";
    $wsListContainer.style.display = "none";
  } else if (view === "workspace-list") {
    $landingButtons.style.display = "none";
    $wsListContainer.style.display = "";
  }
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

// ── Workspace CRUD ──────────────────────────────────────────────────────────

async function loadWorkspaceList() {
  try {
    const res = await fetch("/api/workspaces");
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) throw new Error("Expected JSON response");
    const data = await res.json();
    renderWorkspaceList(data.workspaces || []);
    showView("workspace-list");
  } catch (err) {
    $wsList.innerHTML = `<div class="ws-error">Failed to load workspaces: ${esc(err.message)}</div>`;
    showView("workspace-list");
  }
}

function renderWorkspaceList(workspaces) {
  $wsList.innerHTML = "";
  if (workspaces.length === 0) {
    $wsList.innerHTML = '<div class="ws-empty">No saved workspaces yet.</div>';
    return;
  }
  for (const ws of workspaces) {
    const el = document.createElement("div");
    el.className = "ws-list-item";
    el.innerHTML = `
      <div class="ws-list-info">
        <span class="ws-list-name">${esc(ws.name)}</span>
        <span class="ws-list-detail">${ws.folders.length} folder${ws.folders.length !== 1 ? "s" : ""}</span>
      </div>
      <button class="ws-list-delete" title="Delete workspace">&times;</button>`;
    el.querySelector(".ws-list-info").addEventListener("click", () => openWorkspace(ws));
    el.querySelector(".ws-list-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete workspace "${ws.name}"?`)) return;
      const delRes = await fetch(`/api/workspaces/${ws.id}`, { method: "DELETE" });
      if (!delRes.ok) console.warn("Delete workspace returned", delRes.status);
      loadWorkspaceList();
    });
    $wsList.appendChild(el);
  }
}

async function createNewWorkspace() {
  try {
    const res = await fetch("/api/workspaces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const ws = await res.json();
    openWorkspace(ws);
  } catch (err) {
    alert("Failed to create workspace: " + err.message);
  }
}

function openWorkspace(ws) {
  state.activeWorkspace = ws;
  state.activeFolderRoot = null;
  state.browserChecked.clear();
  state.selectedFiles = [];
  $wsName.value = ws.name;
  $wsBrowserSection.style.display = "none";
  renderSelectedFiles();
  renderWorkspaceFolders();
  requestTokenEstimate();
  showView("workspace");
}

async function saveWorkspace() {
  if (!state.activeWorkspace) return;
  const ws = state.activeWorkspace;
  ws.name = $wsName.value.trim() || ws.name;
  try {
    const res = await fetch(`/api/workspaces/${ws.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: ws.name, folders: ws.folders }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const updated = await res.json();
    state.activeWorkspace = updated;
    $wsName.value = updated.name;
  } catch (err) {
    alert("Failed to save workspace: " + err.message);
  }
}

async function saveAndExitWorkspace() {
  await saveWorkspace();
  state.activeWorkspace = null;
  state.activeFolderRoot = null;
  state.selectedFiles = [];
  closePreview();
  renderSelectedFiles();
  requestTokenEstimate();
  showView("landing");
}

// ── Workspace folders ───────────────────────────────────────────────────────

function renderWorkspaceFolders() {
  const ws = state.activeWorkspace;
  if (!ws) return;

  $wsFolders.innerHTML = "";
  for (const folder of ws.folders) {
    const el = document.createElement("div");
    const isActive = state.activeFolderRoot === folder;
    el.className = "ws-folder-item" + (isActive ? " active" : "");
    el.innerHTML = `
      <span class="ws-folder-icon">📁</span>
      <span class="ws-folder-path" title="${esc(folder)}">${esc(basename(folder))}</span>
      <button class="ws-folder-remove" title="Remove folder">&times;</button>`;
    el.querySelector(".ws-folder-path").addEventListener("click", () => {
      selectFolder(folder);
    });
    el.querySelector(".ws-folder-remove").addEventListener("click", (e) => {
      e.stopPropagation();
      removeFolder(folder);
    });
    $wsFolders.appendChild(el);
  }
}

function selectFolder(folder) {
  state.activeFolderRoot = folder;
  state.browserChecked.clear();
  $addSelectedBtn.disabled = true;
  $wsBrowserSection.style.display = "";
  renderWorkspaceFolders();
  browseFolderTo(folder);
}

function removeFolder(folder) {
  const ws = state.activeWorkspace;
  if (!ws) return;
  ws.folders = ws.folders.filter((f) => f !== folder);
  if (state.activeFolderRoot === folder) {
    state.activeFolderRoot = null;
    $wsBrowserSection.style.display = "none";
  }
  renderWorkspaceFolders();
}

// ── File browsing (within workspace folder) ─────────────────────────────────

async function browseFolderTo(path) {
  state.currentPath = path;
  state.browserChecked.clear();
  $addSelectedBtn.disabled = true;
  $wsBrowsePath.value = path;

  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    if (data.error) {
      $wsFileList.innerHTML = `<div style="padding:10px;color:var(--red)">${esc(data.error)}</div>`;
      return;
    }
    renderFolderFileList(data.entries, data.path);
  } catch (err) {
    $wsFileList.innerHTML = `<div style="padding:10px;color:var(--red)">Failed to browse: ${esc(err.message)}</div>`;
  }
}

function renderFolderFileList(entries, displayPath) {
  $wsFileList.innerHTML = "";
  for (const entry of entries) {
    const el = document.createElement("div");
    el.className = "file-entry" + (entry.type === "dir" ? " dir" : "");

    const fullPath = displayPath.replace(/\/$/, "") + "/" + entry.name;

    if (entry.type === "dir") {
      el.innerHTML = `
        <span class="icon">${fileIcon(entry.type)}</span>
        <span class="name" title="${esc(entry.name)}">${esc(entry.name)}/</span>`;
      el.addEventListener("click", () => browseFolderTo(fullPath));
    } else {
      const selectable = isMedia(entry.type);
      const checked = state.browserChecked.has(fullPath);
      el.innerHTML = `
        ${selectable ? `<input type="checkbox" ${checked ? "checked" : ""}>` : '<span style="width:15px"></span>'}
        <span class="icon">${fileIcon(entry.type)}</span>
        <span class="name" title="${esc(entry.name)}">${esc(entry.name)}</span>
        <span class="size">${formatBytes(entry.size || 0)}</span>`;
      if (selectable) {
        const cb = el.querySelector("input");
        cb.addEventListener("change", () => {
          if (cb.checked) state.browserChecked.add(fullPath);
          else state.browserChecked.delete(fullPath);
          $addSelectedBtn.disabled = state.browserChecked.size === 0;
        });
        el.addEventListener("click", (e) => {
          if (e.target === cb) return;
          cb.checked = !cb.checked;
          cb.dispatchEvent(new Event("change"));
        });
      }
    }
    $wsFileList.appendChild(el);
  }
}

$wsBrowseUp.addEventListener("click", () => {
  if (!state.activeFolderRoot) return;
  // Don't navigate above the workspace folder root
  const root = state.activeFolderRoot;
  const current = state.currentPath;
  if (current === root || current === root + "/") return;
  const parts = current.replace(/\/$/, "").split("/");
  parts.pop();
  const parent = parts.join("/") || "/";
  // Clamp to folder root
  if (!parent.startsWith(root)) {
    browseFolderTo(root);
  } else {
    browseFolderTo(parent);
  }
});

// ── Folder picker (unrestricted browsing to add folders) ────────────────────

function openFolderPicker() {
  state.pickerPath = "/";
  $pickerPath.value = "/";
  showView("picker");
  pickerBrowseTo("/");
}

async function pickerBrowseTo(path) {
  state.pickerPath = path;
  $pickerPath.value = path;

  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    if (data.error) {
      $pickerList.innerHTML = `<div style="padding:10px;color:var(--red)">${esc(data.error)}</div>`;
      return;
    }
    renderPickerList(data.entries, data.path);
  } catch (err) {
    $pickerList.innerHTML = `<div style="padding:10px;color:var(--red)">Failed to browse: ${esc(err.message)}</div>`;
  }
}

function renderPickerList(entries, displayPath) {
  $pickerList.innerHTML = "";
  // Only show directories in the picker
  const dirs = entries.filter((e) => e.type === "dir");
  if (dirs.length === 0) {
    $pickerList.innerHTML = '<div style="padding:10px;color:var(--text-dim)">No subdirectories</div>';
    return;
  }
  for (const entry of dirs) {
    const el = document.createElement("div");
    el.className = "file-entry dir";
    const fullPath = displayPath.replace(/\/$/, "") + "/" + entry.name;
    el.innerHTML = `
      <span class="icon">📁</span>
      <span class="name" title="${esc(entry.name)}">${esc(entry.name)}/</span>`;
    el.addEventListener("click", () => pickerBrowseTo(fullPath));
    $pickerList.appendChild(el);
  }
}

function pickerAddFolder() {
  const folder = state.pickerPath;
  if (!folder || folder === "/") {
    alert("Please navigate to a specific folder to add.");
    return;
  }
  const ws = state.activeWorkspace;
  if (!ws) return;
  if (ws.folders.includes(folder)) {
    alert("This folder is already in the workspace.");
    showView("workspace");
    return;
  }
  ws.folders.push(folder);
  renderWorkspaceFolders();
  showView("workspace");
}

// Picker event listeners
$pickerUp.addEventListener("click", () => {
  const parts = state.pickerPath.replace(/\/$/, "").split("/");
  parts.pop();
  pickerBrowseTo(parts.join("/") || "/");
});
$pickerGo.addEventListener("click", () => pickerBrowseTo($pickerPath.value.trim() || "/"));
$pickerPath.addEventListener("keydown", (e) => {
  if (e.key === "Enter") pickerBrowseTo($pickerPath.value.trim() || "/");
});
$pickerAdd.addEventListener("click", pickerAddFolder);
$pickerCancel.addEventListener("click", () => showView("workspace"));

// ── Selected files management ───────────────────────────────────────────────

$addSelectedBtn.addEventListener("click", () => {
  for (const path of state.browserChecked) {
    if (!state.selectedFiles.some((f) => f.path === path)) {
      const ext = path.split(".").pop().toLowerCase();
      const imgExts = ["jpg", "jpeg", "png", "gif", "bmp", "webp"];
      const type = imgExts.includes(ext) ? "image" : "video";
      state.selectedFiles.push({ path, type });
    }
  }
  state.browserChecked.clear();
  $addSelectedBtn.disabled = true;
  $wsFileList.querySelectorAll("input[type=checkbox]").forEach((cb) => (cb.checked = false));
  renderSelectedFiles();
  requestTokenEstimate();
});

function renderSelectedFiles() {
  $fileCount.textContent = `(${state.selectedFiles.length})`;
  $selectedFiles.innerHTML = "";
  for (let i = 0; i < state.selectedFiles.length; i++) {
    const f = state.selectedFiles[i];
    const name = basename(f.path);
    const el = document.createElement("div");
    const isPreviewing = state.previewFile && state.previewFile.path === f.path;
    el.className = "selected-file"
      + (isMedia(f.type) ? " previewable" : "")
      + (isPreviewing ? " previewing" : "");
    el.innerHTML = `
      <span class="icon">${fileIcon(f.type)}</span>
      <span class="path" title="${esc(f.path)}">${esc(name)}</span>
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
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Server error ${res.status}`);
    }
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
  $responseContent.style.color = "";
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
}

// ── Media preview ───────────────────────────────────────────────────────────

function openPreview(file) {
  state.previewFile = file;
  $mediaPreview.classList.add("visible");
  renderPreviewMediaList();
  renderPreviewContent();
  renderSelectedFiles();
}

function closePreview() {
  state.previewFile = null;
  $mediaPreview.classList.remove("visible");
  renderSelectedFiles();
}

function renderPreviewMediaList() {
  $previewMediaList.innerHTML = "";
  const mediaFiles = state.selectedFiles.filter((f) => isMedia(f.type));
  for (const f of mediaFiles) {
    const chip = document.createElement("button");
    const name = basename(f.path);
    const isActive = state.previewFile && state.previewFile.path === f.path;
    chip.className = "preview-media-chip" + (isActive ? " active" : "");
    chip.title = f.path;
    chip.textContent = `${fileIcon(f.type)} ${name}`;
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

// ── Landing event listeners ─────────────────────────────────────────────────

$btnLoadWs.addEventListener("click", loadWorkspaceList);
$btnNewWs.addEventListener("click", createNewWorkspace);
$btnWsBack.addEventListener("click", () => showView("landing"));

// ── Workspace event listeners ───────────────────────────────────────────────

$btnSaveWs.addEventListener("click", saveWorkspace);
$btnExitWs.addEventListener("click", saveAndExitWorkspace);
$btnAddFolder.addEventListener("click", openFolderPicker);

// ── Init ────────────────────────────────────────────────────────────────────

checkHealth();
setInterval(checkHealth, 30000);
showView("landing");
requestTokenEstimate();
