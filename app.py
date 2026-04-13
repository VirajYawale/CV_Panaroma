"""
PanoramaWeave — Flask Backend
ORB + FLANN-LSH + RANSAC + Laplacian Blending + OpenCV Stitcher
"""

import gc
import base64
import logging
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Embedded frontend (no static/ folder needed) ──────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PanoramaWeave</title>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --cream: #faf7f2; --warm-white: #fffdf9; --sand: #f0ebe0;
    --taupe: #d9d0c0; --muted-brown: #a89880; --warm-text: #3d2f20;
    --soft-text: #7a6555; --accent: #c07840; --accent-light: #f5e8d8;
    --accent-hover: #a86430; --success: #6a9a6a; --success-light: #e8f2e8;
    --border: #e0d8cc; --shadow: rgba(100,70,40,0.08);
  }
  body { font-family: 'DM Sans', sans-serif; background: var(--cream); color: var(--warm-text); min-height: 100vh; }
  .header { background: var(--warm-white); border-bottom: 1px solid var(--border); padding: 1.25rem 2rem; display: flex; align-items: center; gap: 1rem; }
  .header-icon { width: 40px; height: 40px; background: var(--accent-light); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
  .header h1 { font-family: 'Lora', serif; font-size: 1.25rem; font-weight: 500; }
  .header p { font-size: 0.75rem; color: var(--soft-text); margin-top: 1px; }
  .badge { margin-left: auto; background: var(--accent-light); color: var(--accent); font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 20px; border: 1px solid #e8c89a; }
  .layout { display: grid; grid-template-columns: 260px 1fr; min-height: calc(100vh - 69px); }
  .sidebar { background: var(--warm-white); border-right: 1px solid var(--border); padding: 1.5rem; display: flex; flex-direction: column; gap: 1.5rem; }
  .sidebar-section-title { font-size: 10px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; color: var(--muted-brown); margin-bottom: 0.75rem; }
  .pipeline-step { display: flex; align-items: flex-start; gap: 10px; padding: 6px 6px; cursor: pointer; border-radius: 8px; transition: background 0.15s; }
  .pipeline-step:hover { background: var(--accent-light); }
  .step-dot { width: 22px; height: 22px; border-radius: 50%; background: var(--sand); border: 1.5px solid var(--taupe); display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 500; color: var(--soft-text); flex-shrink: 0; margin-top: 2px; }
  .pipeline-step.active .step-dot { background: var(--accent); border-color: var(--accent); color: white; }
  .pipeline-step.done .step-dot { background: var(--success); border-color: var(--success); color: white; }
  .step-name { font-size: 13px; color: var(--warm-text); }
  .step-desc { font-size: 11px; color: var(--soft-text); margin-top: 1px; }
  .pipeline-step.active .step-name { font-weight: 500; color: var(--accent); }
  .params-card { background: var(--sand); border-radius: 12px; padding: 1rem; border: 1px solid var(--border); }
  .param-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--taupe); }
  .param-row:last-child { border-bottom: none; }
  .param-label { font-size: 12px; color: var(--soft-text); }
  .param-select, .param-input { font-family: 'DM Sans', sans-serif; font-size: 12px; background: var(--warm-white); border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; color: var(--warm-text); }
  .param-input { width: 64px; text-align: right; }
  .param-input:focus, .param-select:focus { outline: 1.5px solid var(--accent); }
  .main { padding: 2rem; display: flex; flex-direction: column; gap: 2rem; }
  .upload-zone { background: var(--warm-white); border: 2px dashed var(--taupe); border-radius: 16px; padding: 2.5rem 2rem; text-align: center; cursor: pointer; transition: all 0.2s; position: relative; }
  .upload-zone:hover, .upload-zone.drag-over { border-color: var(--accent); background: var(--accent-light); }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .upload-icon { width: 54px; height: 54px; background: var(--accent-light); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 24px; }
  .upload-zone h2 { font-family: 'Lora', serif; font-size: 1.1rem; font-weight: 400; margin-bottom: 0.4rem; }
  .upload-zone p { font-size: 13px; color: var(--soft-text); }
  .upload-hint { font-size: 11px; color: var(--muted-brown); margin-top: 0.75rem; }
  .section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem; }
  .section-title { font-family: 'Lora', serif; font-size: 1rem; font-weight: 500; }
  .section-sub { font-size: 12px; color: var(--soft-text); margin-top: 2px; }
  .photo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; }
  .photo-card { background: var(--warm-white); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; position: relative; cursor: grab; transition: all 0.2s; box-shadow: 0 2px 8px var(--shadow); }
  .photo-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px var(--shadow); }
  .photo-card.dragging { opacity: 0.4; }
  .photo-card.drag-target { border: 2px solid var(--accent); }
  .photo-thumb { width: 100%; aspect-ratio: 4/3; overflow: hidden; background: var(--sand); }
  .photo-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .photo-footer { padding: 6px 8px; display: flex; justify-content: space-between; align-items: center; }
  .photo-name { font-size: 10px; color: var(--soft-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 80px; }
  .photo-order { width: 18px; height: 18px; background: var(--accent); color: white; border-radius: 50%; font-size: 10px; font-weight: 500; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .photo-remove { position: absolute; top: 5px; right: 5px; width: 20px; height: 20px; background: rgba(255,255,255,0.9); border-radius: 50%; display: none; align-items: center; justify-content: center; font-size: 11px; cursor: pointer; color: #a05030; border: 1px solid var(--border); }
  .photo-card:hover .photo-remove { display: flex; }
  .pipeline-status { background: var(--warm-white); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.25rem; }
  .stage-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--sand); }
  .stage-row:last-child { border-bottom: none; }
  .stage-icon { font-size: 15px; width: 22px; text-align: center; }
  .stage-name { font-size: 13px; flex: 1; }
  .stage-badge { font-size: 11px; padding: 2px 9px; border-radius: 12px; }
  .stage-badge.pending { background: var(--sand); color: var(--muted-brown); }
  .stage-badge.running { background: #fef3e2; color: #c07040; border: 1px solid #f5d8a0; }
  .stage-badge.done { background: var(--success-light); color: var(--success); }
  .stage-badge.error { background: #fce8e8; color: #c04040; }
  .progress-wrap { background: var(--sand); border-radius: 4px; height: 5px; margin-top: 0.75rem; overflow: hidden; }
  .progress-bar { height: 100%; background: var(--accent); border-radius: 4px; width: 0%; transition: width 0.3s ease; }
  .result-card { background: var(--warm-white); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
  .result-header { padding: 1rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  .result-title { font-family: 'Lora', serif; font-size: 0.95rem; font-weight: 500; }
  .result-tabs { display: flex; gap: 4px; }
  .result-tab { font-family: 'DM Sans', sans-serif; font-size: 12px; padding: 5px 12px; border-radius: 8px; border: 1px solid transparent; cursor: pointer; color: var(--soft-text); background: transparent; transition: all 0.15s; }
  .result-tab.active { background: var(--accent-light); color: var(--accent); border-color: #e8c89a; }
  .result-tab:hover:not(.active) { background: var(--sand); }
  .result-canvas { background: var(--sand); min-height: 180px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  .result-canvas img { max-width: 100%; display: block; }
  .result-placeholder { text-align: center; color: var(--muted-brown); padding: 3rem; font-family: 'Lora', serif; font-style: italic; font-size: 13px; }
  .result-footer { padding: 0.75rem 1.5rem; border-top: 1px solid var(--border); display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .stat-pill { font-size: 11px; background: var(--sand); border: 1px solid var(--border); border-radius: 20px; padding: 3px 10px; color: var(--soft-text); }
  .stat-pill strong { color: var(--warm-text); }
  .error-banner { background: #fce8e8; border: 1px solid #f0b8b8; border-radius: 10px; padding: 0.75rem 1rem; font-size: 13px; color: #a03030; }
  .btn-primary { font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 500; background: var(--accent); color: white; border: none; border-radius: 10px; padding: 9px 20px; cursor: pointer; transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px; }
  .btn-primary:hover { background: var(--accent-hover); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .btn-secondary { font-family: 'DM Sans', sans-serif; font-size: 13px; background: var(--warm-white); color: var(--warm-text); border: 1px solid var(--border); border-radius: 10px; padding: 9px 16px; cursor: pointer; transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px; }
  .btn-secondary:hover { background: var(--sand); }
  .action-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .info-chip { font-size: 11px; color: var(--muted-brown); background: var(--sand); border-radius: 6px; padding: 3px 8px; border: 1px solid var(--border); }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.4); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header class="header">
  <div class="header-icon">🌅</div>
  <div><h1>PanoramaWeave</h1><p>ORB · FLANN-LSH · RANSAC · Laplacian blending</p></div>
  <div class="badge">Flask backend</div>
</header>
<div class="layout">
  <aside class="sidebar">
    <div>
      <div class="sidebar-section-title">Pipeline</div>
      <div id="pipeline-nav"></div>
    </div>
    <div>
      <div class="sidebar-section-title">Parameters</div>
      <div class="params-card">
        <div class="param-row"><span class="param-label">ORB features</span><input class="param-input" type="number" value="3000" min="500" max="8000" step="500" id="p-orb"></div>
        <div class="param-row"><span class="param-label">RANSAC thresh</span><input class="param-input" type="number" value="5.0" min="1" max="20" step="0.5" id="p-ransac"></div>
        <div class="param-row"><span class="param-label">Match ratio</span><input class="param-input" type="number" value="0.75" min="0.5" max="0.95" step="0.05" id="p-ratio"></div>
        <div class="param-row"><span class="param-label">Blending</span>
          <select class="param-select" id="p-blend">
            <option value="laplacian">Laplacian</option>
            <option value="distance">Distance-weighted</option>
            <option value="alpha">Alpha</option>
          </select>
        </div>
        <div class="param-row"><span class="param-label">Crop</span>
          <select class="param-select" id="p-crop">
            <option value="auto">Auto</option>
            <option value="tight">Tight</option>
            <option value="none">None</option>
          </select>
        </div>
        <div class="param-row"><span class="param-label">OpenCV compare</span>
          <select class="param-select" id="p-opencv">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>
      </div>
    </div>
    <div style="margin-top:auto;padding-top:1rem;border-top:1px solid var(--border);">
      <div class="sidebar-section-title">About</div>
      <p style="font-size:11px;color:var(--soft-text);line-height:1.6;">Upload overlapping photos left → right, reorder by drag, then stitch. Powered by ORB keypoints, FLANN-LSH matching, RANSAC homographies, and Laplacian pyramid blending.</p>
    </div>
  </aside>
  <main class="main">
    <!-- Upload -->
    <div>
      <div class="section-header">
        <div><div class="section-title">Upload Photos</div><div class="section-sub">Select all panorama images · JPG / PNG</div></div>
      </div>
      <div class="upload-zone" id="upload-zone">
        <input type="file" id="file-input" accept="image/*" multiple onchange="handleFiles(this.files)">
        <div class="upload-icon">📷</div>
        <h2>Drop your photos here</h2>
        <p>or click to browse</p>
        <p class="upload-hint">Select all overlapping images at once — reorder after upload</p>
      </div>
    </div>
    <!-- Photo tray -->
    <div id="photo-section" style="display:none;">
      <div class="section-header">
        <div><div class="section-title">Arrange Photos</div><div class="section-sub">Drag to reorder left → right</div></div>
        <div class="action-row">
          <span class="info-chip" id="photo-count">0 photos</span>
          <button class="btn-secondary" onclick="clearPhotos()">✕ Clear</button>
          <button class="btn-primary" id="stitch-btn" onclick="runStitch()" disabled>✦ Stitch panorama</button>
        </div>
      </div>
      <div class="photo-grid" id="photo-grid"></div>
    </div>
    <!-- Error -->
    <div id="error-section" style="display:none;">
      <div class="error-banner" id="error-msg"></div>
    </div>
    <!-- Processing -->
    <div id="processing-section" style="display:none;">
      <div class="section-title" style="margin-bottom:0.75rem;">Processing</div>
      <div class="pipeline-status" id="stage-list"></div>
      <div class="progress-wrap"><div class="progress-bar" id="progress-bar"></div></div>
    </div>
    <!-- Result -->
    <div id="result-section" style="display:none;">
      <div class="section-header">
        <div class="section-title">Result</div>
        <div class="action-row">
          <button class="btn-secondary" onclick="downloadResult()">↓ Download</button>
          <button class="btn-primary" onclick="resetAll()">↺ New panorama</button>
        </div>
      </div>
      <div class="result-card">
        <div class="result-header">
          <span class="result-title">Stitched panorama</span>
          <div class="result-tabs">
            <button class="result-tab active" id="tab-custom" onclick="switchTab('custom')">Custom ORB</button>
            <button class="result-tab" id="tab-opencv" onclick="switchTab('opencv')">OpenCV</button>
          </div>
        </div>
        <div class="result-canvas" id="result-canvas">
          <img id="result-img" src="" alt="panorama result" style="display:none;">
          <div class="result-placeholder" id="result-placeholder">No result yet</div>
        </div>
        <div class="result-footer" id="result-stats"></div>
      </div>
    </div>
  </main>
</div>
<script>
const PIPELINE_STEPS = [
  {name:"Upload photos",   desc:"JPG / PNG selection",    icon:"📁"},
  {name:"Preview & sort",  desc:"Left → right order",     icon:"🗂"},
  {name:"ORB detection",   desc:"Keypoint extraction",    icon:"🔍"},
  {name:"FLANN matching",  desc:"Feature correspondence", icon:"🔗"},
  {name:"RANSAC homography",desc:"Geometric alignment",   icon:"📐"},
  {name:"Canvas warp",     desc:"Global projection",      icon:"🗺"},
  {name:"Laplacian blend", desc:"Pyramid blending",       icon:"✨"},
  {name:"Export",          desc:"Download result",        icon:"💾"},
];
const STAGES = [
  {key:"orb",    label:"ORB keypoint detection",     icon:"🔍"},
  {key:"flann",  label:"FLANN-LSH matching",          icon:"🔗"},
  {key:"ransac", label:"RANSAC homography",           icon:"📐"},
  {key:"warp",   label:"Global canvas warp",          icon:"🗺"},
  {key:"blend",  label:"Laplacian pyramid blending",  icon:"✨"},
  {key:"crop",   label:"Auto crop & save",            icon:"✂"},
];

let photos = [];         // {name, previewSrc, file}
let currentStep = 0;
let apiResult = null;
let activeTab = "custom";
let dragSrc = null;

function renderPipeline() {
  document.getElementById("pipeline-nav").innerHTML = PIPELINE_STEPS.map((s, i) => {
    const done   = i < currentStep;
    const active = i === currentStep;
    return `<div class="pipeline-step ${done?'done':active?'active':''}" onclick="setStep(${i})">
      <div class="step-dot">${done?'✓':i+1}</div>
      <div><div class="step-name">${s.icon} ${s.name}</div><div class="step-desc">${s.desc}</div></div>
    </div>`;
  }).join('');
}
function setStep(i) { currentStep = i; renderPipeline(); }

function handleFiles(files) {
  Array.from(files).forEach(f => {
    if (!f.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => {
      photos.push({name: f.name, previewSrc: e.target.result, file: f});
      renderPhotoGrid();
      document.getElementById("photo-section").style.display = "block";
    };
    reader.readAsDataURL(f);
  });
  setStep(1);
}

function renderPhotoGrid() {
  document.getElementById("photo-grid").innerHTML = photos.map((p, i) => `
    <div class="photo-card" draggable="true" data-idx="${i}"
      ondragstart="dragStart(event,${i})" ondragover="dragOver(event)"
      ondrop="dropPhoto(event,${i})" ondragend="dragEnd()">
      <span class="photo-remove" onclick="removePhoto(${i})">✕</span>
      <div class="photo-thumb"><img src="${p.previewSrc}" alt="${p.name}"></div>
      <div class="photo-footer">
        <span class="photo-name">${p.name.replace(/\.[^.]+$/,'')}</span>
        <span class="photo-order">${i+1}</span>
      </div>
    </div>`).join('');
  document.getElementById("photo-count").textContent = `${photos.length} photo${photos.length!==1?'s':''}`;
  document.getElementById("stitch-btn").disabled = photos.length < 2;
}

function dragStart(e, i) { dragSrc = i; e.currentTarget.classList.add("dragging"); }
function dragOver(e) { e.preventDefault(); }
function dropPhoto(e, i) {
  e.preventDefault();
  if (dragSrc === null || dragSrc === i) return;
  const [moved] = photos.splice(dragSrc, 1);
  photos.splice(i, 0, moved);
  renderPhotoGrid();
}
function dragEnd() {
  dragSrc = null;
  document.querySelectorAll(".photo-card").forEach(c => c.classList.remove("dragging"));
}
function removePhoto(i) {
  photos.splice(i, 1); renderPhotoGrid();
  if (!photos.length) { document.getElementById("photo-section").style.display = "none"; setStep(0); }
}
function clearPhotos() {
  photos = [];
  ["photo-section","result-section","processing-section","error-section"].forEach(id =>
    document.getElementById(id).style.display = "none");
  document.getElementById("file-input").value = "";
  setStep(0);
}

async function runStitch() {
  if (photos.length < 2) return;

  // hide previous results/errors, show processing
  document.getElementById("result-section").style.display = "none";
  document.getElementById("error-section").style.display  = "none";
  document.getElementById("processing-section").style.display = "block";
  document.getElementById("stitch-btn").disabled = true;
  document.getElementById("stitch-btn").innerHTML = '<span class="spinner"></span> Stitching…';

  // Render stage list
  document.getElementById("stage-list").innerHTML = STAGES.map(s => `
    <div class="stage-row" id="stage-${s.key}">
      <span class="stage-icon">${s.icon}</span>
      <span class="stage-name">${s.label}</span>
      <span class="stage-badge pending" id="badge-${s.key}">pending</span>
    </div>`).join('');

  const bar = document.getElementById("progress-bar");
  bar.style.width = "5%";

  // Animate stages while waiting for the real API response
  let stageIdx = 0;
  const stageTimer = setInterval(() => {
    if (stageIdx > 0) {
      const prev = STAGES[stageIdx - 1];
      document.getElementById("badge-" + prev.key).className = "stage-badge done";
      document.getElementById("badge-" + prev.key).textContent = "done";
    }
    if (stageIdx < STAGES.length) {
      const cur = STAGES[stageIdx];
      document.getElementById("badge-" + cur.key).className = "stage-badge running";
      document.getElementById("badge-" + cur.key).textContent = "running…";
      setStep(2 + stageIdx);
      bar.style.width = Math.round(((stageIdx + 1) / STAGES.length) * 85) + "%";
      stageIdx++;
    } else {
      clearInterval(stageTimer);
    }
  }, 900);

  // Build FormData
  const form = new FormData();
  photos.forEach(p => form.append("images[]", p.file));
  form.append("nfeatures", document.getElementById("p-orb").value);
  form.append("ransac",    document.getElementById("p-ransac").value);
  form.append("ratio",     document.getElementById("p-ratio").value);
  form.append("blend",     document.getElementById("p-blend").value);
  form.append("crop",      document.getElementById("p-crop").value);
  form.append("opencv",    document.getElementById("p-opencv").value);

  try {
    const res  = await fetch("/stitch", {method: "POST", body: form});
    const data = await res.json();

    clearInterval(stageTimer);

    if (!res.ok) {
      showError(data.error || `Server error ${res.status}`);
      return;
    }

    // Mark all stages done
    STAGES.forEach(s => {
      document.getElementById("badge-" + s.key).className = "stage-badge done";
      document.getElementById("badge-" + s.key).textContent = "done";
    });
    bar.style.width = "100%";

    apiResult = data;
    setTimeout(() => showResult(data), 400);

  } catch (err) {
    clearInterval(stageTimer);
    showError("Network error: " + err.message);
  }
}

function showError(msg) {
  document.getElementById("processing-section").style.display = "none";
  document.getElementById("error-section").style.display = "block";
  document.getElementById("error-msg").textContent = "⚠ " + msg;
  document.getElementById("stitch-btn").disabled = false;
  document.getElementById("stitch-btn").innerHTML = "✦ Stitch panorama";
}

function showResult(data) {
  document.getElementById("processing-section").style.display = "none";
  document.getElementById("result-section").style.display = "block";
  document.getElementById("stitch-btn").disabled = false;
  document.getElementById("stitch-btn").innerHTML = "✦ Stitch panorama";
  setStep(7);
  switchTab("custom");
}

function switchTab(tab) {
  activeTab = tab;
  document.getElementById("tab-custom").classList.toggle("active", tab === "custom");
  document.getElementById("tab-opencv").classList.toggle("active", tab === "opencv");

  if (!apiResult) return;
  const src = tab === "custom"
    ? apiResult.custom?.image
    : apiResult.opencv?.image;

  const img = document.getElementById("result-img");
  const ph  = document.getElementById("result-placeholder");

  if (src) {
    img.src = src;
    img.style.display = "block";
    ph.style.display  = "none";
  } else {
    img.style.display = "none";
    ph.style.display  = "block";
    const status = apiResult.opencv?.status || "unavailable";
    ph.textContent = tab === "opencv" ? `OpenCV stitcher: ${status}` : "No result";
  }

  // Stats
  const d = tab === "custom" ? apiResult.custom : apiResult.opencv;
  if (d && d.width) {
    const inliers = tab === "custom" && apiResult.custom.inlier_counts
      ? `<span class="stat-pill">Avg inliers: <strong>${Math.round(apiResult.custom.inlier_counts.reduce((a,b)=>a+b,0) / (apiResult.custom.inlier_counts.length||1))}</strong></span>`
      : "";
    document.getElementById("result-stats").innerHTML = `
      <span class="stat-pill">Width: <strong>${d.width}px</strong></span>
      <span class="stat-pill">Height: <strong>${d.height}px</strong></span>
      <span class="stat-pill">Images: <strong>${apiResult.custom.n_frames}</strong></span>
      ${inliers}
      <span class="stat-pill">Blend: <strong>${apiResult.custom.blend_mode}</strong></span>`;
  } else {
    document.getElementById("result-stats").innerHTML = "";
  }
}

function downloadResult() {
  if (!apiResult) return;
  const src = activeTab === "custom" ? apiResult.custom?.image : apiResult.opencv?.image;
  if (!src) return;
  const a = document.createElement("a");
  a.href = src;
  a.download = `panorama_${activeTab}.jpg`;
  a.click();
}

function resetAll() { clearPhotos(); apiResult = null; }

// Drag-and-drop on upload zone
const zone = document.getElementById("upload-zone");
zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
zone.addEventListener("drop", e => {
  e.preventDefault(); zone.classList.remove("drag-over");
  handleFiles(e.dataTransfer.files);
});

renderPipeline();
</script>
</body>
</html>
"""

RESIZE_WIDTH   = 640    # resize each input photo to this width (set 0 to skip)
MAX_CANVAS_GB  = 2.5    # refuse stitching if canvas RAM would exceed this


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(file_storage) -> np.ndarray:
    """Read a werkzeug FileStorage object into a BGR numpy array."""
    data = file_storage.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {file_storage.filename}")
    return img


def resize_if_needed(img: np.ndarray, width: int) -> np.ndarray:
    if width and img.shape[1] > width:
        scale = width / img.shape[1]
        img   = cv2.resize(img, (width, int(img.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    return img


def encode_image_b64(img: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    """Encode a BGR numpy array to a base64 data-URI string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


# ── Core pipeline ─────────────────────────────────────────────────────────────

def detect_and_match_orb(img1: np.ndarray, img2: np.ndarray,
                          nfeatures: int = 3000,
                          ratio: float = 0.75,
                          top_k: int = 150):
    """
    ORB keypoint detection + FLANN-LSH matching with Lowe's ratio test.
    Returns (kp1, kp2, good_matches).
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return kp1, kp2, []

    # FLANN-LSH — memory-efficient for binary descriptors
    index_params  = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw   = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda x: x.distance)[:top_k]
    return kp1, kp2, good


def compute_homography_ransac(kp1, kp2, matches,
                              thresh: float = 5.0,
                              min_match: int = 10):
    """RANSAC homography. Returns (H, mask) or (None, None)."""
    if len(matches) < min_match:
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh, maxIters=10000)
    return H, mask


def laplacian_blend(img1: np.ndarray, img2: np.ndarray,
                    mask: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    Laplacian pyramid blending of img1 and img2 using a float mask (0-1).
    mask must be 2-D (H, W); images must be (H, W, 3).
    levels is auto-clamped so no pyramid level shrinks below 2 px.
    """
    assert img1.shape == img2.shape, "Images must match for Laplacian blending"
    h, w = img1.shape[:2]

    # Cap levels to avoid degenerate tiny pyramid levels on large canvases
    max_levels = max(1, int(np.floor(np.log2(min(h, w)))) - 1)
    levels = max(1, min(levels, max_levels))

    # Keep mask as (H, W, 1) float32 so it broadcasts against (H, W, 3) at every level
    mask3 = mask[:, :, np.newaxis].astype(np.float32)
    i1 = img1.astype(np.float32)
    i2 = img2.astype(np.float32)

    def build_gauss(img, n):
        pyr = [img]
        for _ in range(n - 1):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def build_laplacian(gauss):
        lap = []
        for k in range(len(gauss) - 1):
            up = cv2.pyrUp(gauss[k + 1],
                           dstsize=(gauss[k].shape[1], gauss[k].shape[0]))
            lap.append(gauss[k] - up)
        lap.append(gauss[-1])
        return lap

    g1  = build_gauss(i1,    levels)
    g2  = build_gauss(i2,    levels)
    gm  = build_gauss(mask3, levels)  # each level stays (H', W', 1)
    lp1 = build_laplacian(g1)
    lp2 = build_laplacian(g2)

    # gm[k] is (H', W', 1) — broadcasts cleanly against (H', W', 3)
    blended_pyr = [lp1[k] * gm[k] + lp2[k] * (1.0 - gm[k]) for k in range(levels)]

    # Reconstruct
    result = blended_pyr[-1]
    for k in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result,
                           dstsize=(blended_pyr[k].shape[1], blended_pyr[k].shape[0]))
        result = result + blended_pyr[k]  # use + not += to avoid in-place dtype issues

    return np.clip(result, 0, 255).astype(np.uint8)


def stitch_custom(frames: list[np.ndarray],
                  nfeatures: int = 3000,
                  ransac_thresh: float = 5.0,
                  ratio: float = 0.75,
                  blend_mode: str = "laplacian",
                  crop_mode: str = "auto") -> tuple[np.ndarray, dict]:
    """
    Full custom ORB + RANSAC panorama pipeline.

    Parameters
    ----------
    frames       : ordered list of BGR images (left → right)
    nfeatures    : ORB feature count per image
    ransac_thresh: RANSAC reprojection threshold in pixels
    ratio        : Lowe's ratio test threshold
    blend_mode   : 'laplacian' | 'distance' | 'alpha'
    crop_mode    : 'auto' | 'tight' | 'none'

    Returns
    -------
    (panorama_bgr, info_dict)
    """
    log.info("Starting custom stitch: %d frames, features=%d, ransac=%.1f",
             len(frames), nfeatures, ransac_thresh)

    # ── Step A: Chain relative homographies ───────────────────────────────────
    H_chain = [np.eye(3, dtype=np.float64)]
    skipped = 0
    inlier_counts = []

    for i in range(1, len(frames)):
        kp1, kp2, matches = detect_and_match_orb(frames[i - 1], frames[i],
                                                  nfeatures=nfeatures,
                                                  ratio=ratio)
        H_rel, mask = compute_homography_ransac(kp1, kp2, matches,
                                                thresh=ransac_thresh)
        del kp1, kp2, matches
        gc.collect()

        if H_rel is None:
            log.warning("  Pair %d→%d: SKIP (not enough matches)", i - 1, i)
            H_chain.append(H_chain[-1].copy())
            inlier_counts.append(0)
            skipped += 1
            continue

        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_counts.append(n_inliers)
        H_chain.append(H_chain[-1] @ np.linalg.inv(H_rel))
        log.info("  Pair %d→%d: OK  inliers=%d", i - 1, i, n_inliers)

    # ── Step B: Global canvas size ─────────────────────────────────────────────
    all_corners = []
    for i, frm in enumerate(frames):
        h, w = frm.shape[:2]
        c = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(c, H_chain[i]))

    all_corners = np.concatenate(all_corners, axis=0)
    xmin, ymin  = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    xmax, ymax  = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Clamp: no axis may exceed 4x the largest input dimension (prevents drift)
    max_input_dim = max(int(frm.shape[0]) for frm in frames) * 4
    xmin = max(int(xmin), -max_input_dim); ymin = max(int(ymin), -max_input_dim)
    xmax = min(int(xmax),  max_input_dim); ymax = min(int(ymax),  max_input_dim)

    canvas_w, canvas_h = xmax - xmin, ymax - ymin

    # If canvas is portrait (height >> width) the chain drifted vertically.
    # Swap axes so the panorama is always landscape-oriented.
    if canvas_h > canvas_w * 1.5:
        log.warning("Canvas portrait (%dx%d) -> transposing.", canvas_w, canvas_h)
        canvas_w, canvas_h = canvas_h, canvas_w
        xmin, ymin = ymin, xmin

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    ram_gb = int(canvas_w) * int(canvas_h) * 16 / 1e9
    log.info("Canvas: %dx%d  RAM=%.2f GB", canvas_w, canvas_h, ram_gb)
    if ram_gb > MAX_CANVAS_GB:
        raise MemoryError(
            f"Canvas too large ({ram_gb:.1f} GB > {MAX_CANVAS_GB} GB). "
            "Reduce image size or use fewer photos."
        )

    # For very large canvases, Laplacian blending is too memory-intensive.
    LAPLACIAN_MAX_PX = 6_000_000
    if blend_mode == "laplacian" and int(canvas_w) * int(canvas_h) > LAPLACIAN_MAX_PX:
        log.warning("Canvas >6 MP -- falling back to distance blend.")
        blend_mode = "distance"

    # ── Step C: Warp + blend ───────────────────────────────────────────────────
    if blend_mode == "laplacian":
        # Accumulate with distance weights, then apply Laplacian between pairs
        acc_color  = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        acc_weight = np.zeros((canvas_h, canvas_w),    dtype=np.float32)

        warped_list = []
        dist_list   = []

        for i, frm in enumerate(frames):
            h, w   = frm.shape[:2]
            H_abs  = T @ H_chain[i]
            warped = cv2.warpPerspective(frm, H_abs, (canvas_w, canvas_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            fmask  = np.ones((h, w), dtype=np.uint8) * 255
            wmask  = cv2.warpPerspective(fmask, H_abs, (canvas_w, canvas_h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT)
            dist   = cv2.distanceTransform(wmask, cv2.DIST_L2, 5).astype(np.float32)
            warped_list.append(warped)
            dist_list.append(dist)
            acc_color  += warped.astype(np.float32) * dist[:, :, np.newaxis]
            acc_weight += dist
            del fmask, wmask
            gc.collect()
            log.info("  Warped %d/%d", i + 1, len(frames))

        acc_weight = np.maximum(acc_weight, 1e-6)
        pano_uint8 = np.clip(acc_color / acc_weight[:, :, np.newaxis], 0, 255).astype(np.uint8)
        del acc_color, acc_weight
        gc.collect()

        # Refine seams between consecutive image pairs with Laplacian blending
        result = warped_list[0].copy()
        for i in range(1, len(warped_list)):
            d1 = dist_list[i - 1]
            d2 = dist_list[i]
            total = np.maximum(d1 + d2, 1e-6)
            mask  = d1 / total                          # 1 where left image dominates
            result = laplacian_blend(result, warped_list[i], mask)
            log.info("  Laplacian blend %d/%d", i, len(warped_list) - 1)

        panorama = result
        del warped_list, dist_list
        gc.collect()

    elif blend_mode == "distance":
        acc_color  = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        acc_weight = np.zeros((canvas_h, canvas_w),    dtype=np.float32)

        for i, frm in enumerate(frames):
            h, w   = frm.shape[:2]
            H_abs  = T @ H_chain[i]
            warped = cv2.warpPerspective(frm, H_abs, (canvas_w, canvas_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            fmask  = np.ones((h, w), dtype=np.uint8) * 255
            wmask  = cv2.warpPerspective(fmask, H_abs, (canvas_w, canvas_h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT)
            dist   = cv2.distanceTransform(wmask, cv2.DIST_L2, 5).astype(np.float32)
            acc_color  += warped.astype(np.float32) * dist[:, :, np.newaxis]
            acc_weight += dist
            del warped, wmask, dist, fmask
            gc.collect()
            log.info("  Distance-blend %d/%d", i + 1, len(frames))

        acc_weight = np.maximum(acc_weight, 1e-6)
        panorama   = np.clip(acc_color / acc_weight[:, :, np.newaxis], 0, 255).astype(np.uint8)
        del acc_color, acc_weight
        gc.collect()

    else:  # alpha (simple overwrite, left-to-right)
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i, frm in enumerate(frames):
            H_abs  = T @ H_chain[i]
            warped = cv2.warpPerspective(frm, H_abs, (canvas_w, canvas_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            mask   = warped.any(axis=2)
            panorama[mask] = warped[mask]
            del warped
            gc.collect()
            log.info("  Alpha-blend %d/%d", i + 1, len(frames))

    # ── Step D: Crop ───────────────────────────────────────────────────────────
    if crop_mode in ("auto", "tight"):
        gray_p = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thr  = cv2.threshold(gray_p, 1, 255, cv2.THRESH_BINARY)
        coords  = cv2.findNonZero(thr)
        if coords is not None:
            xr, yr, wr, hr = cv2.boundingRect(coords)
            if crop_mode == "tight":
                # shrink bounding box inward by 1% to remove edge artefacts
                pad = max(1, int(min(wr, hr) * 0.01))
                xr  = min(xr + pad, xr + wr - 1)
                yr  = min(yr + pad, yr + hr - 1)
                wr  = max(wr - 2 * pad, 1)
                hr  = max(hr - 2 * pad, 1)
            panorama = panorama[yr:yr + hr, xr:xr + wr]

    info = {
        "width":        panorama.shape[1],
        "height":       panorama.shape[0],
        "n_frames":     len(frames),
        "skipped":      skipped,
        "inlier_counts": inlier_counts,
        "blend_mode":   blend_mode,
        "crop_mode":    crop_mode,
    }
    log.info("Custom stitch done: %dx%d", panorama.shape[1], panorama.shape[0])
    return panorama, info


def stitch_opencv(frames: list[np.ndarray]) -> tuple[np.ndarray | None, str]:
    """Run OpenCV's built-in Stitcher. Returns (image, status_str)."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(frames)

    error_map = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS:           "Need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:      "Homography estimation failed",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:"Camera parameter adjustment failed",
    }

    if status == cv2.Stitcher_OK:
        log.info("OpenCV stitcher OK: %dx%d", pano.shape[1], pano.shape[0])
        return pano, "ok"
    else:
        msg = error_map.get(status, f"Unknown error (code {status})")
        log.warning("OpenCV stitcher failed: %s", msg)
        return None, msg


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the embedded frontend — no static/ folder needed."""
    return Response(FRONTEND_HTML, mimetype="text/html")


@app.route("/stitch", methods=["GET"])
def stitch_get():
    return jsonify(error="Use POST /stitch with multipart/form-data and images[]."), 405


@app.route("/stitch", methods=["POST"])
def stitch():
    """
    POST /stitch
    Form-data:
        images[]   : one or more image files (ordered left → right)
        nfeatures  : int   (default 3000)
        ransac     : float (default 5.0)
        ratio      : float (default 0.75)
        blend      : str   ('laplacian' | 'distance' | 'alpha')
        crop       : str   ('auto' | 'tight' | 'none')
        opencv     : '1' to also run OpenCV stitcher, '0' to skip

    Returns JSON:
        {
          "custom": {
            "image": "data:image/jpeg;base64,...",
            "width": ..., "height": ..., "inlier_counts": [...], ...
          },
          "opencv": {
            "image": "data:image/jpeg;base64,...",  # or null
            "status": "ok" | "<error message>"
          }
        }
    """
    files = request.files.getlist("images[]")
    if not files or len(files) < 2:
        return jsonify(error="Please upload at least 2 images."), 400

    # Parse parameters
    nfeatures    = int(request.form.get("nfeatures",   3000))
    ransac_thresh= float(request.form.get("ransac",    5.0))
    ratio        = float(request.form.get("ratio",     0.75))
    blend_mode   = request.form.get("blend",           "laplacian")
    crop_mode    = request.form.get("crop",            "auto")
    run_opencv   = request.form.get("opencv", "1") == "1"

    if blend_mode not in ("laplacian", "distance", "alpha"):
        blend_mode = "laplacian"
    if crop_mode not in ("auto", "tight", "none"):
        crop_mode = "auto"

    # Decode + resize images
    frames = []
    for f in files:
        try:
            img = decode_image(f)
            img = resize_if_needed(img, RESIZE_WIDTH)
            frames.append(img)
        except ValueError as e:
            return jsonify(error=str(e)), 400

    log.info("Received %d frames. Running pipeline...", len(frames))

    # Custom pipeline
    try:
        pano_custom, info = stitch_custom(
            frames,
            nfeatures=nfeatures,
            ransac_thresh=ransac_thresh,
            ratio=ratio,
            blend_mode=blend_mode,
            crop_mode=crop_mode,
        )
        custom_b64 = encode_image_b64(pano_custom)
        custom_result = {
            "image":         custom_b64,
            "width":         info["width"],
            "height":        info["height"],
            "n_frames":      info["n_frames"],
            "skipped":       info["skipped"],
            "inlier_counts": info["inlier_counts"],
            "blend_mode":    info["blend_mode"],
            "crop_mode":     info["crop_mode"],
        }
    except MemoryError as e:
        return jsonify(error=str(e)), 413
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify(error=f"Custom stitching failed: {str(e)}"), 500

    # OpenCV stitcher (optional)
    opencv_result = {"image": None, "status": "skipped"}
    if run_opencv:
        try:
            pano_cv, status_str = stitch_opencv(frames)
            if pano_cv is not None:
                opencv_result = {
                    "image":  encode_image_b64(pano_cv),
                    "width":  pano_cv.shape[1],
                    "height": pano_cv.shape[0],
                    "status": status_str,
                }
            else:
                opencv_result = {"image": None, "status": status_str}
        except Exception as e:
            log.error(traceback.format_exc())
            opencv_result = {"image": None, "status": f"Exception: {str(e)}"}

    del frames
    gc.collect()

    return jsonify(custom=custom_result, opencv=opencv_result)


@app.route("/health")
def health():
    return jsonify(status="ok", opencv_version=cv2.__version__)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)