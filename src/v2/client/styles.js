export const clientStyles = `
[hidden]{display:none!important}
#app-root[data-view-mode=log] #action-log-panel{position:fixed;top:var(--v02-header-height,65px);right:0;bottom:0;left:0;overflow-y:auto;overscroll-behavior:contain;z-index:80;background:#f0f2f5;-webkit-overflow-scrolling:touch}
.v02-toast-stack{position:fixed;top:70px;left:12px;right:12px;z-index:2200;display:flex;flex-direction:column;gap:8px;pointer-events:none;align-items:center}
.v02-toast{pointer-events:auto;border:0;border-radius:12px;padding:13px 18px;max-width:540px;background:#1a1a2e;color:white;box-shadow:0 5px 20px #0002;font:inherit;text-align:left;cursor:pointer}
.v02-voice-target{outline:2px solid #007aff;outline-offset:2px}
.v02-transcript{position:fixed;z-index:2300;max-width:min(420px,90vw);padding:10px 14px;background:#fff;border:1px solid #007aff;border-radius:12px;box-shadow:0 5px 20px #0002;color:#1a1a2e;pointer-events:none;font-size:15px}
.v02-transcript[data-state=editing]{border:2px solid #007aff;box-shadow:0 0 0 4px #007aff20,0 5px 20px #0002;background:#f5faff;white-space:pre-line}
.v02-transcript[data-state=cancelled]{border-color:transparent;box-shadow:none;background:#f0f2f5;color:#555}
.v02-action-page{position:fixed;inset:0;background:#f0f2f5;z-index:1400;overflow:auto;padding-bottom:24px}
.v02-action-body{max-width:760px;margin:auto;padding:18px;display:grid;gap:14px}
.v02-action-card,.v02-message{background:#fff;padding:14px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.v02-journal-record{display:grid;gap:8px}.v02-model-answer{width:100%;border:0;font:inherit;color:#1a1a2e;text-align:left;cursor:pointer}.v02-model-context{max-height:260px;overflow:auto;background:#e9edf3;padding:12px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#303744}
.v02-message[data-role=user]{background:#e7f1ff;margin-left:24px}.v02-message[data-role=assistant]{margin-right:24px}
.v02-correction-form{display:flex;gap:8px;align-items:flex-end}.v02-correction-form textarea{flex:1;min-height:78px;border:1px solid #ccd0d5;border-radius:10px;padding:12px;font:inherit}
.v02-correction-form button,.v02-primary{border:0;border-radius:10px;background:#007aff;color:white;padding:12px 16px;font:inherit;cursor:pointer}
.v02-error{background:#fff1f0;color:#9b2020;padding:12px;border-radius:10px;margin:12px}
.v02-edit-transcript{--v02-editor-top:max(12px,env(safe-area-inset-top));position:fixed;inset:0;z-index:2400;background:#0006;display:grid;place-items:start center;padding:var(--v02-editor-top) 20px 0}
.v02-edit-transcript section{width:min(500px,100%);height:min(310px,calc(50vh - var(--v02-editor-top)));max-height:calc(50vh - var(--v02-editor-top));height:min(310px,calc(50dvh - var(--v02-editor-top)));max-height:calc(50dvh - var(--v02-editor-top));background:white;padding:clamp(10px,2.6dvh,22px);border-radius:18px;display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:clamp(6px,1.65dvh,14px);overflow-y:auto}.v02-edit-transcript textarea{width:100%;height:auto;min-height:0;max-height:none;margin-top:1em;padding:12px;font:inherit;overflow-y:auto}.v02-edit-transcript nav{display:flex;justify-content:flex-end;gap:10px}
.v02-task-details{width:100%;min-height:90px;border:1px solid #ddd;padding:12px;border-radius:10px;font:inherit}.v02-search{padding:10px 12px;display:flex;gap:8px}.v02-search input{flex:1;min-width:0;border:1px solid #ddd;border-radius:10px;padding:9px;font:inherit}
.v02-connection{font-size:12px;padding:6px 12px;background:#fff4d9;color:#684200}.v02-status-select{border:0;background:transparent;max-width:90px;font:inherit;color:inherit}
.task-page-subtask{cursor:pointer}.v02-action-label{font-weight:600}.action-log-row{cursor:pointer}.list-item-wrapper{touch-action:pan-y}
button:focus-visible,a:focus-visible{outline:3px solid #007aff;outline-offset:3px}.v02-action-page h2{font-size:20px}.v02-action-page small{color:#555}
`;
