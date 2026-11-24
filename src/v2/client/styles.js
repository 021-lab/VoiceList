export const clientStyles = `
[hidden]{display:none!important}
#app-root[data-view-mode=log] #action-log-panel{position:fixed;top:var(--v02-header-height,65px);right:0;bottom:0;left:0;overflow-y:auto;overscroll-behavior:contain;z-index:80;background:#f0f2f5;-webkit-overflow-scrolling:touch}
.v02-toast-stack{position:fixed;top:70px;left:12px;right:12px;z-index:2200;display:flex;flex-direction:column;gap:8px;pointer-events:none;align-items:center}
.v02-heard{background:#2d6cdf;font-style:normal}
.v02-toast{pointer-events:auto;border:0;border-radius:12px;padding:13px 18px;max-width:540px;background:#1a1a2e;color:white;box-shadow:0 5px 20px #0002;font:inherit;text-align:left;cursor:pointer}
.v02-voice-target{outline:2px solid #007aff;outline-offset:2px}
.v02-transcript{position:fixed;z-index:2300;max-width:min(420px,90vw);padding:10px 14px;background:#fff;border:1px solid #007aff;border-radius:12px;box-shadow:0 5px 20px #0002;color:#1a1a2e;pointer-events:none;font-size:15px}
.v02-transcript[data-state=editing]{border:2px solid #007aff;box-shadow:0 0 0 4px #007aff20,0 5px 20px #0002;background:#f5faff;white-space:pre-line}
.v02-transcript[data-state=cancelled]{border-color:transparent;box-shadow:none;background:#f0f2f5;color:#555}
.v02-action-page{position:fixed;inset:0;background:#f0f2f5;z-index:1400;overflow:auto;padding-bottom:calc(24px + env(safe-area-inset-bottom))}
.v02-action-page .task-page-bar{position:sticky;top:0;z-index:2}.v02-action-page .task-page-save:disabled{background:#d8dde5;color:#667080;cursor:default}
.v02-action-body{max-width:760px;margin:auto;padding:20px 18px 32px;display:grid;gap:18px}.v02-action-body>h2{margin:0;color:#1a1a2e;font-size:24px;letter-spacing:-.02em}
.v02-journal-record{position:relative;display:grid;gap:9px;padding:18px 16px 18px 22px;background:#fff;border-radius:16px;box-shadow:0 2px 12px #1a1a2e0a}.v02-journal-record::before{content:"";position:absolute;left:10px;top:24px;bottom:18px;width:2px;border-radius:2px;background:#dbe8f8}.v02-journal-record::after{content:"";position:absolute;left:7px;top:22px;width:8px;height:8px;border-radius:50%;background:#007aff;box-shadow:0 0 0 3px #e7f1ff}
.v02-record-marker{color:#52606f!important;font-size:12px;font-weight:800;letter-spacing:.025em;text-transform:uppercase}.v02-section-label{margin-top:3px;color:#747d89;font-size:12px;font-weight:700}
.v02-action-card,.v02-message{background:#f5f7fa;padding:13px 14px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}.v02-message[data-role=user]{background:#e7f1ff;color:#15233a;font-size:16px;line-height:1.4}
.v02-context-disclosure{width:100%;min-height:42px;display:flex;align-items:center;justify-content:space-between;gap:12px;border:0;border-top:1px solid #edf0f4;border-bottom:1px solid #edf0f4;background:transparent;padding:9px 2px;color:#485568;font:inherit;font-size:13px;font-weight:700;text-align:left;cursor:pointer}.v02-context-disclosure:disabled{color:#9299a3;cursor:default}.v02-context-state{color:#007aff;font-size:12px;font-weight:700}
.v02-model-context{max-height:300px;overflow:auto;background:#20252e;padding:13px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.45;color:#e8edf4;user-select:text;-webkit-user-select:text}
.v02-model-output{display:grid;gap:9px}
.v02-model-answer{width:100%;border:1px solid #e5e8ed;background:#fff;font:inherit;line-height:1.45;color:#1a1a2e;text-align:left;cursor:pointer;box-shadow:0 1px 5px #1a1a2e0a}.v02-model-answer:active,.v02-command-card:active{background:#f0f6ff}
.v02-command-list{display:grid;gap:8px}.v02-command-card{width:100%;display:grid;gap:6px;border:1px solid #dfe4eb;border-left:4px solid #007aff;border-radius:11px;background:#fff;padding:12px 13px;color:#1a1a2e;font:inherit;text-align:left}.v02-command-card.v02-context-trigger{cursor:pointer}.v02-command-heading{font-size:15px;font-weight:800;line-height:1.3}.v02-command-target{color:#4d5968;font-size:13px;font-weight:600}.v02-command-field{display:grid;grid-template-columns:minmax(90px,.42fr) minmax(0,1fr);gap:10px;padding-top:6px;border-top:1px solid #eef0f3;font-size:13px;line-height:1.35}.v02-command-field-label{color:#737c88}.v02-command-field-value{color:#202734;font-weight:650;overflow-wrap:anywhere}
.v02-correction-form{display:flex;gap:8px;align-items:flex-end}.v02-correction-form textarea{flex:1;min-height:78px;border:1px solid #ccd0d5;border-radius:10px;padding:12px;font:inherit}
.v02-correction-form button,.v02-primary{border:0;border-radius:10px;background:#007aff;color:white;padding:12px 16px;font:inherit;cursor:pointer}
.v02-error{background:#fff1f0;color:#9b2020;padding:12px;border-radius:10px;margin:12px}
.v02-edit-transcript{--v02-editor-top:max(12px,env(safe-area-inset-top));position:fixed;inset:0;z-index:2400;background:#0006;display:grid;place-items:start center;padding:var(--v02-editor-top) 20px 0}
.v02-edit-transcript section{width:min(500px,100%);height:min(310px,calc(50vh - var(--v02-editor-top)));max-height:calc(50vh - var(--v02-editor-top));height:min(310px,calc(50dvh - var(--v02-editor-top)));max-height:calc(50dvh - var(--v02-editor-top));background:white;padding:clamp(10px,2.6dvh,22px);border-radius:18px;display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:clamp(6px,1.65dvh,14px);overflow-y:auto}.v02-edit-transcript textarea{width:100%;height:auto;min-height:0;max-height:none;margin-top:1em;padding:12px;font:inherit;overflow-y:auto}.v02-edit-transcript nav{display:flex;justify-content:flex-end;gap:10px}
.v02-task-details{width:100%;min-height:90px;border:1px solid #ddd;padding:12px;border-radius:10px;font:inherit}.v02-search{padding:10px 12px;display:flex;gap:8px}.v02-search input{flex:1;min-width:0;border:1px solid #ddd;border-radius:10px;padding:9px;font:inherit}
.v02-connection{font-size:12px;padding:6px 12px;background:#fff4d9;color:#684200}.v02-status-select{border:0;background:transparent;max-width:90px;font:inherit;color:inherit}
.task-page-subtask{cursor:pointer}.v02-action-label{font-weight:600}.action-log-row{cursor:pointer}.list-item-wrapper{touch-action:pan-y}
button:focus-visible,a:focus-visible{outline:3px solid #007aff;outline-offset:3px}.v02-action-page small{color:#555}
@media(max-width:520px){.v02-action-body{padding:16px 12px 28px}.v02-journal-record{border-radius:14px;padding-right:13px}.v02-command-field{grid-template-columns:1fr;gap:3px}.v02-correction-form{align-items:stretch;flex-direction:column}.v02-correction-form button{align-self:flex-end}}
.v02-grok{margin-top:20px;background:#fff;border-radius:14px;padding:14px}.v02-grok h3{margin:0 0 10px;font-size:17px}
.v02-grok-thread{display:flex;flex-direction:column;gap:10px}.v02-grok-message{padding:9px 12px;border-radius:12px;background:#f0f2f5}
.v02-grok-message[data-role=grok]{background:#eef5ff}.v02-grok-message[data-role=system]{background:none;padding:0 4px;color:#8e8e93}
.v02-grok-meta{font-size:12px;color:#8e8e93;margin-bottom:3px}.v02-grok-text{white-space:pre-wrap;overflow-wrap:anywhere}
.v02-grok-working,.v02-grok-status{font-size:13px;color:#8e8e93}.v02-grok-retry{margin-top:6px;border:1px solid #007aff;background:none;color:#007aff;border-radius:8px;padding:4px 10px;font:inherit}
.v02-grok-form{display:flex;flex-direction:column;gap:8px;margin-top:12px}.v02-grok-form textarea{min-height:64px;font:inherit;padding:10px;border-radius:10px;border:1px solid #d0d4da}
.v02-grok-form button{align-self:flex-end}.v02-grok-form button:disabled{opacity:.5}
.v02-grok-summary{margin:-4px 0 6px 12px;padding:6px 10px;border-left:3px solid #8e8e93;font-size:13px;color:#3a3a3c;cursor:pointer}
.v02-grok-summary-text{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;white-space:pre-wrap}
.v02-grok-time{font-size:11px;color:#8e8e93}.v02-grok-working.v02-grok-summary,.v02-grok-summary.v02-grok-working{color:#8e8e93}.v02-grok-error{border-left-color:#ff3b30;color:#ff3b30}
.v02-grok-actions{display:flex;justify-content:space-between;align-items:center;gap:8px}.v02-grok-open{color:#007aff;text-decoration:none;font-size:15px}.v02-grok-open[aria-disabled=true]{opacity:.4;pointer-events:none}
.v02-grok-pages{margin-top:10px}.v02-grok-page{padding:4px 0}.v02-grok-page a{color:#007aff}.v02-grok-page small{color:#8e8e93}
`;
