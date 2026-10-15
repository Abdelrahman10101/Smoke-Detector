// Smoking Detection Service — Dashboard Controller
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const selectVideo = document.getElementById("select-video");
  const selectMode = document.getElementById("select-mode");
  const btnStart = document.getElementById("btn-start");
  const btnPause = document.getElementById("btn-pause");
  const btnStop = document.getElementById("btn-stop");
  
  const streamImg = document.getElementById("stream-img");
  const cropsImg = document.getElementById("crops-img");
  const cropsEmpty = document.getElementById("crops-empty");
  const playerEmpty = document.getElementById("player-empty-state");
  const liveAlertBanner = document.getElementById("live-alert-banner");

  const kpiFps = document.getElementById("kpi-fps");
  const kpiDetLatency = document.getElementById("kpi-det-latency");
  const kpiClsLatency = document.getElementById("kpi-cls-latency");
  const kpiPersons = document.getElementById("kpi-persons");
  const kpiAlerts = document.getElementById("kpi-alerts");
  
  const sliderThreshold = document.getElementById("slider-threshold");
  const valThreshold = document.getElementById("val-threshold");
  const sliderVoteK = document.getElementById("slider-vote-k");
  const valVoting = document.getElementById("val-voting");
  const selectCropMode = document.getElementById("select-crop-mode");
  const toggleCctvSim = document.getElementById("toggle-cctv-sim");
  
  const alertFeed = document.getElementById("alert-feed");
  const alertsEmpty = document.getElementById("alerts-empty");
  const btnExportAlerts = document.getElementById("btn-export-alerts");

  // Modals
  const uploadModal = document.getElementById("upload-modal");
  const btnOpenUpload = document.getElementById("btn-open-upload");
  const btnCloseUpload = document.getElementById("btn-close-upload");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadProgress = document.getElementById("upload-progress");

  const snapshotModal = document.getElementById("snapshot-modal");
  const btnCloseSnapshot = document.getElementById("btn-close-snapshot");
  const modalSnapshotImg = document.getElementById("modal-snapshot-img");
  const modalSnapshotTitle = document.getElementById("modal-snapshot-title");
  const modalSnapshotInfo = document.getElementById("modal-snapshot-info");

  // App State
  let currentSessionId = null;
  let statusPollInterval = null;
  let sseEventSource = null;
  let loadedAlerts = [];
  let alertCount = 0;

  // 1. Initial Data Fetch
  async function init() {
    await checkReadiness();
    await loadVideos();
    await loadInitialAlerts();
    await loadConfig();
    setupSSE();
  }

  async function checkReadiness() {
    try {
      const res = await fetch("/readyz");
      if (res.ok) {
        const data = await res.json();
        document.getElementById("status-text").textContent = `Engine Ready (${data.providers[0] || 'CPU'})`;
      }
    } catch (e) {
      document.getElementById("status-text").textContent = "Engine Offline";
      document.getElementById("status-dot").style.background = "var(--color-danger)";
    }
  }

  async function loadVideos() {
    try {
      const res = await fetch("/api/videos");
      const data = await res.json();
      selectVideo.innerHTML = '<option value="" disabled selected>Select a demo video...</option>';
      
      if (data.videos && data.videos.length > 0) {
        data.videos.forEach(v => {
          const opt = document.createElement("option");
          opt.value = v.id;
          opt.textContent = `${v.filename} (${v.size_mb} MB, ${v.duration_s}s)`;
          selectVideo.appendChild(opt);
        });
      } else {
        const opt = document.createElement("option");
        opt.value = "";
        opt.disabled = true;
        opt.textContent = "No videos found — please upload one";
        selectVideo.appendChild(opt);
      }
    } catch (e) {
      console.error("Failed to load videos:", e);
    }
  }

  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      const cfg = await res.json();
      sliderThreshold.value = cfg.threshold;
      valThreshold.textContent = Number(cfg.threshold).toFixed(2);
      
      sliderVoteK.value = cfg.vote_k;
      valVoting.textContent = `${cfg.vote_k} of ${cfg.vote_n}`;

      selectCropMode.value = cfg.crop_mode;
      toggleCctvSim.checked = cfg.cctv_sim;
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  }

  async function loadInitialAlerts() {
    try {
      const res = await fetch("/api/events?limit=50");
      const data = await res.json();
      if (data.alerts && data.alerts.length > 0) {
        loadedAlerts = data.alerts;
        alertCount = loadedAlerts.length;
        kpiAlerts.textContent = alertCount;
        alertsEmpty.style.display = "none";
        alertFeed.innerHTML = "";
        loadedAlerts.forEach(addAlertCard);
      }
    } catch (e) {
      console.error("Failed to load alerts:", e);
    }
  }

  // 2. Session Management
  selectVideo.addEventListener("change", () => {
    btnStart.disabled = !selectVideo.value;
  });

  btnStart.addEventListener("click", async () => {
    const videoId = selectVideo.value;
    const mode = selectMode.value;
    if (!videoId) return;

    btnStart.disabled = true;
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, mode: mode, loop: true })
      });
      if (!res.ok) throw new Error("Failed to start session");
      const data = await res.json();
      currentSessionId = data.session_id;

      // Update UI for active session
      playerEmpty.style.display = "none";
      streamImg.style.display = "block";
      streamImg.src = `/api/sessions/${currentSessionId}/stream?t=${Date.now()}`;
      
      cropsEmpty.style.display = "none";
      cropsImg.style.display = "block";
      cropsImg.src = `/api/sessions/${currentSessionId}/crops?t=${Date.now()}`;

      btnPause.disabled = false;
      btnStop.disabled = false;
      btnPause.textContent = "Pause";

      startStatusPolling();
    } catch (e) {
      alert("Error starting session: " + e.message);
      btnStart.disabled = false;
    }
  });

  btnPause.addEventListener("click", async () => {
    if (!currentSessionId) return;
    const isPaused = btnPause.textContent === "Resume";
    const endpoint = isPaused ? "resume" : "pause";
    
    await fetch(`/api/sessions/${currentSessionId}/${endpoint}`, { method: "POST" });
    btnPause.textContent = isPaused ? "Pause" : "Resume";
  });

  btnStop.addEventListener("click", async () => {
    if (!currentSessionId) return;
    await fetch(`/api/sessions/${currentSessionId}/stop`, { method: "POST" });
    stopSessionUI();
  });

  function stopSessionUI() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    currentSessionId = null;
    streamImg.src = "";
    streamImg.style.display = "none";
    cropsImg.src = "";
    cropsImg.style.display = "none";
    cropsEmpty.style.display = "block";
    playerEmpty.style.display = "flex";
    liveAlertBanner.style.display = "none";

    btnStart.disabled = !selectVideo.value;
    btnPause.disabled = true;
    btnStop.disabled = true;

    kpiFps.textContent = "0.0";
    kpiDetLatency.textContent = "0";
    kpiClsLatency.textContent = "0";
    kpiPersons.textContent = "0";
  }

  function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    statusPollInterval = setInterval(async () => {
      if (!currentSessionId) return;
      try {
        const res = await fetch(`/api/sessions/${currentSessionId}`);
        if (!res.ok) {
          stopSessionUI();
          return;
        }
        const data = await res.json();
        const meta = data.meta || {};

        kpiFps.textContent = data.processed_fps.toFixed(1);
        if (meta.latencies) {
          kpiDetLatency.textContent = Math.round(meta.latencies.det_ms);
          kpiClsLatency.textContent = Math.round(meta.latencies.cls_ms);
        }
        if (meta.persons) {
          kpiPersons.textContent = meta.persons.length;
          // Check if any person is in active alert status
          const activeAlert = meta.persons.some(p => p.is_alert);
          liveAlertBanner.style.display = activeAlert ? "inline-flex" : "none";
        }
      } catch (e) {
        console.error("Status poll failed:", e);
      }
    }, 500);
  }

  // 3. SSE Alerts Listener
  function setupSSE() {
    if (sseEventSource) sseEventSource.close();
    sseEventSource = new EventSource("/api/events/stream");

    sseEventSource.onmessage = (e) => {
      try {
        const alertData = JSON.parse(e.data);
        alertCount++;
        kpiAlerts.textContent = alertCount;
        loadedAlerts.unshift(alertData);
        alertsEmpty.style.display = "none";
        addAlertCard(alertData, true);

        // Flash banner
        liveAlertBanner.style.display = "inline-flex";
        setTimeout(() => {
          liveAlertBanner.style.display = "none";
        }, 4000);
      } catch (err) {
        // ping or parsing error
      }
    };
  }

  function addAlertCard(alert, prepend = false) {
    const card = document.createElement("div");
    card.className = "alert-item";
    card.innerHTML = `
      <img src="/api/snapshots/${alert.snapshot_file}" class="alert-thumb" alt="Snapshot" />
      <div class="alert-details">
        <div class="alert-header-row">
          <span class="alert-tag">Track #${alert.track_id} Alert</span>
          <span class="alert-time">${alert.timestamp_s}s</span>
        </div>
        <div class="alert-prob">Smoking: ${(alert.p_smoking * 100).toFixed(1)}%</div>
      </div>
    `;

    // Click to open snapshot modal
    card.querySelector(".alert-thumb").addEventListener("click", () => {
      modalSnapshotImg.src = `/api/snapshots/${alert.snapshot_file}`;
      modalSnapshotTitle.textContent = `Incident at ${alert.timestamp_s}s · Track #${alert.track_id}`;
      modalSnapshotInfo.textContent = `Confidence: ${(alert.p_smoking * 100).toFixed(1)}% | Video: ${alert.video_name} | Logged: ${alert.created_at}`;
      snapshotModal.classList.add("active");
    });

    if (prepend && alertFeed.firstChild) {
      alertFeed.insertBefore(card, alertFeed.firstChild);
    } else {
      alertFeed.appendChild(card);
    }
  }

  // 4. Live Tuning Controls
  let patchDebounce = null;
  function patchConfig(data) {
    clearTimeout(patchDebounce);
    patchDebounce = setTimeout(async () => {
      await fetch("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
    }, 200);
  }

  sliderThreshold.addEventListener("input", () => {
    valThreshold.textContent = Number(sliderThreshold.value).toFixed(2);
    patchConfig({ threshold: parseFloat(sliderThreshold.value) });
  });

  sliderVoteK.addEventListener("input", () => {
    valVoting.textContent = `${sliderVoteK.value} of 5`;
    patchConfig({ vote_k: parseInt(sliderVoteK.value, 10) });
  });

  selectCropMode.addEventListener("change", () => {
    patchConfig({ crop_mode: selectCropMode.value });
  });

  toggleCctvSim.addEventListener("change", () => {
    patchConfig({ cctv_sim: toggleCctvSim.checked });
  });

  // 5. Upload Handling
  btnOpenUpload.addEventListener("click", () => {
    uploadModal.classList.add("active");
  });
  btnCloseUpload.addEventListener("click", () => {
    uploadModal.classList.remove("active");
  });
  uploadDropzone.addEventListener("click", () => {
    fileInput.click();
  });

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });
  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });
  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleUpload(fileInput.files[0]);
    }
  });

  async function handleUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    uploadProgress.style.display = "block";
    uploadProgress.textContent = `Uploading ${file.name}...`;

    try {
      const res = await fetch("/api/videos", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      uploadProgress.textContent = "Upload successful!";
      setTimeout(() => {
        uploadModal.classList.remove("active");
        uploadProgress.style.display = "none";
      }, 800);
      
      await loadVideos();
      selectVideo.value = data.video_id;
      btnStart.disabled = false;
    } catch (e) {
      uploadProgress.textContent = "Error: " + e.message;
    }
  }

  // Snapshot modal close
  btnCloseSnapshot.addEventListener("click", () => {
    snapshotModal.classList.remove("active");
  });

  // Export Alerts
  btnExportAlerts.addEventListener("click", () => {
    if (loadedAlerts.length === 0) {
      alert("No alerts to export.");
      return;
    }
    const jsonStr = JSON.stringify(loadedAlerts, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `smoking_alerts_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  init();
});
