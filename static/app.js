const taskList = document.getElementById("task-list");
const generalVoiceBtn = document.getElementById("general-voice");
const voiceStatus = document.getElementById("voice-status");
const transcriptNode = document.getElementById("transcript");
const promptInput = document.getElementById("prompt-input");
const promptStatus = document.getElementById("prompt-status");
const savePromptBtn = document.getElementById("save-prompt");
const resetPromptBtn = document.getElementById("reset-prompt");

let tasks = [];
let mediaRecorder = null;
let audioChunks = [];
let activeTaskId = null;
let activeMimeType = "";
const API_BASE = window.location.pathname.startsWith("/voicelist/") ? "/voicelist" : "";

function pickRecordingMimeType() {
  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return "";
  }
  for (const mime of preferred) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return "";
}

function extensionForMime(mimeType) {
  if (!mimeType) return "webm";
  if (mimeType.includes("mp4")) return "mp4";
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("wav")) return "wav";
  return "webm";
}

function setStatus(node, message, isError = false) {
  node.textContent = message;
  node.classList.toggle("error", isError);
}

async function loadTasks() {
  const response = await fetch(`${API_BASE}/api/tasks`);
  tasks = await response.json();
  renderTasks();
}

function renderTasks() {
  taskList.innerHTML = "";
  if (!tasks.length) {
    const li = document.createElement("li");
    li.textContent = "No tasks yet. Use voice: add ...";
    taskList.appendChild(li);
    return;
  }

  for (const task of tasks) {
    const item = document.createElement("li");
    item.className = "task-item";
    item.dataset.taskId = task.id;
    item.textContent = task.name;
    item.addEventListener("pointerdown", () => beginRecording(task.id, item));
    item.addEventListener("pointerup", () => stopRecording(item));
    item.addEventListener("pointercancel", () => stopRecording(item));
    taskList.appendChild(item);
  }
}

async function beginRecording(taskId, element) {
  try {
    activeTaskId = taskId;
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    activeMimeType = pickRecordingMimeType();
    mediaRecorder = activeMimeType
      ? new MediaRecorder(stream, { mimeType: activeMimeType })
      : new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      audioChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", async () => {
      const mimeType = mediaRecorder.mimeType || activeMimeType || "audio/webm";
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      await sendVoiceCommand(audioBlob, activeTaskId, mimeType);
      stream.getTracks().forEach((track) => track.stop());
    });
    mediaRecorder.start();
    if (element) {
      element.classList.add("recording");
    }
    setStatus(voiceStatus, "Recording...");
  } catch (error) {
    setStatus(voiceStatus, `Mic error: ${error}`, true);
  }
}

async function stopRecording(element) {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }
  if (element) {
    element.classList.remove("recording");
  }
  setStatus(voiceStatus, "Processing command...");
  mediaRecorder.stop();
}

async function sendVoiceCommand(audioBlob, taskId, mimeType = "audio/webm") {
  const formData = new FormData();
  const ext = extensionForMime(mimeType);
  formData.append("audio", audioBlob, `voice.${ext}`);
  if (taskId) {
    formData.append("selected_item_id", taskId);
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/api/voice-command`, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    setStatus(voiceStatus, `Network error: ${error}`, true);
    return;
  }

  let result;
  try {
    result = await response.json();
  } catch (error) {
    const text = await response.text();
    setStatus(voiceStatus, `Server error: ${text || error}`, true);
    return;
  }
  transcriptNode.textContent = `Transcript: ${result.transcript ?? ""}`;

  if (!response.ok || result.error) {
    setStatus(voiceStatus, result.error || "Command failed", true);
    return;
  }

  tasks = result.tasks;
  renderTasks();
  setStatus(voiceStatus, "Command applied");
}

async function loadPrompt() {
  const response = await fetch(`${API_BASE}/api/prompt`);
  const payload = await response.json();
  promptInput.value = payload.user_prompt;
}

async function savePrompt() {
  const response = await fetch(`${API_BASE}/api/prompt`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_prompt: promptInput.value }),
  });

  const payload = await response.json();
  if (!response.ok) {
    setStatus(promptStatus, payload.detail || "Failed to save prompt", true);
    return;
  }
  setStatus(promptStatus, "Prompt saved");
}

async function resetPrompt() {
  const response = await fetch(`${API_BASE}/api/prompt/reset`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(promptStatus, "Failed to reset prompt", true);
    return;
  }
  promptInput.value = payload.user_prompt;
  setStatus(promptStatus, "Prompt reset to default");
}

savePromptBtn.addEventListener("click", savePrompt);
resetPromptBtn.addEventListener("click", resetPrompt);
generalVoiceBtn.addEventListener("pointerdown", () => beginRecording(null, generalVoiceBtn));
generalVoiceBtn.addEventListener("pointerup", () => stopRecording(generalVoiceBtn));
generalVoiceBtn.addEventListener("pointercancel", () => stopRecording(generalVoiceBtn));

(async () => {
  await Promise.all([loadTasks(), loadPrompt()]);
})();
