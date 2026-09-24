/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vendored from google-gemini/live-api-web-console (src/lib/audio-recorder.ts), types
// removed and the eventemitter3 base class replaced by the onData / onVolume callbacks.

import { audioContext } from './audio-context.js';
import { AudioRecordingWorklet, VolMeterWorket } from './worklets.js';
import { createWorketFromSrc } from './audioworklet-registry.js';

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export class AudioRecorder {
  constructor(sampleRate = 16000) {
    this.sampleRate = sampleRate;
    this.stream = undefined;
    this.audioContext = undefined;
    this.source = undefined;
    this.recording = false;
    this.recordingWorklet = undefined;
    this.vuWorklet = undefined;
    this.starting = null;
    /** Receives each chunk of microphone audio as base64 PCM16. */
    this.onData = () => {};
    /** Receives the input level, for a meter. */
    this.onVolume = () => {};
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Could not request user media');
    }

    this.starting = new Promise(async (resolve) => {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = await audioContext({ sampleRate: this.sampleRate });
      this.source = this.audioContext.createMediaStreamSource(this.stream);

      const workletName = 'audio-recorder-worklet';
      const src = createWorketFromSrc(workletName, AudioRecordingWorklet);

      await this.audioContext.audioWorklet.addModule(src);
      this.recordingWorklet = new AudioWorkletNode(this.audioContext, workletName);

      this.recordingWorklet.port.onmessage = async (ev) => {
        // worklet processes recording floats and messages converted buffer
        const arrayBuffer = ev.data.data.int16arrayBuffer;

        if (arrayBuffer) {
          this.onData(arrayBufferToBase64(arrayBuffer));
        }
      };
      this.source.connect(this.recordingWorklet);

      // vu meter worklet
      const vuWorkletName = 'vu-meter';
      await this.audioContext.audioWorklet.addModule(
        createWorketFromSrc(vuWorkletName, VolMeterWorket)
      );
      this.vuWorklet = new AudioWorkletNode(this.audioContext, vuWorkletName);
      this.vuWorklet.port.onmessage = (ev) => {
        this.onVolume(ev.data.volume);
      };

      this.source.connect(this.vuWorklet);
      this.recording = true;
      resolve();
      this.starting = null;
    });
    return this.starting;
  }

  stop() {
    // its plausible that stop would be called before start completes
    // such as if the websocket immediately hangs up
    const handleStop = () => {
      this.source?.disconnect();
      this.stream?.getTracks().forEach((track) => track.stop());
      this.stream = undefined;
      this.recordingWorklet = undefined;
      this.vuWorklet = undefined;
      this.recording = false;
    };
    if (this.starting) {
      this.starting.then(handleStop);
      return;
    }
    handleStop();
  }
}
