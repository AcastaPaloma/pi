import { applyCopy, expandAll, focusPassage, previewPassage, refreshReadingMeasurements } from './app.js?v=20261017-r1';
import { serializeBody, validateCopy } from './rich-text.mjs';
import { steps, details } from './content.js?v=20261017-r1';

const initialPassage = location.hash.slice(1).split('/').pop();
const ids = new Set([...steps, ...details].map(item => item.id));
const draftKey = 'pi-text-editor-draft-v1';
let saved, doc, sha, token, dirty = false, busy = false, activeBody, selection;
let draftTimer, measureFrame;
const notes = document.querySelector('.notes');
document.body.classList.add('editing');
// Every passage is editable, so every component note stays open in the editor.
expandAll();
const toolbar = document.createElement('div'); toolbar.className = 'editor-toolbar';
toolbar.innerHTML = '<div class="editor-actions"><span class="editor-label">Text editor</span><a href="https://acastapaloma.github.io/pi/" target="_blank" rel="noopener">View live ↗</a><button id="save-live" disabled>Save to live</button></div><div class="format-tools" role="toolbar" aria-label="Text formatting"><button data-command="bold" title="Bold (⌘/Ctrl B)" aria-label="Bold"><strong>B</strong></button><button data-command="italic" title="Italic (⌘/Ctrl I)" aria-label="Italic"><em>I</em></button><button data-command="underline" title="Underline (⌘/Ctrl U)" aria-label="Underline"><u>U</u></button><select id="editor-font" aria-label="Font"><option value="Georgia">Serif</option><option value="Arial">Sans serif</option><option value="Courier New">Monospace</option></select><button data-command="indent" title="Indent paragraph" aria-label="Indent">Indent</button><button data-command="outdent" title="Outdent paragraph" aria-label="Outdent">Outdent</button><button data-command="insertUnorderedList" title="Bulleted list">List</button><button data-command="undo" title="Undo">Undo</button><button data-command="redo" title="Redo">Redo</button></div><div class="editor-feedback"><span id="save-status" role="status">Connecting to GitHub…</span><button id="download-draft" hidden>Download draft</button></div>';
notes.insertBefore(toolbar, document.querySelector('#reader'));
const saveButton = document.querySelector('#save-live'), status = document.querySelector('#save-status');
function message(text, error = false) { status.textContent = text; status.classList.toggle('is-error', error); }
function updateButton() { saveButton.disabled = !dirty || busy; }
async function request(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The request failed. Try again.');
  return result;
}
function storeDraft() {
  if (!doc) return;
  try {
    if (dirty) localStorage.setItem(draftKey, JSON.stringify({ sha, doc }));
    else localStorage.removeItem(draftKey);
  } catch { message('Draft storage is unavailable. Keep this tab open until you save.', true); }
}
function changed() {
  if (!activeBody || !doc) return;
  doc.sections[activeBody.dataset.section] = serializeBody(activeBody);
  dirty = JSON.stringify(doc) !== saved;
  clearTimeout(draftTimer); draftTimer = setTimeout(storeDraft, 200);
  message(dirty ? 'Unsaved changes · draft kept on this computer' : 'No unsaved changes');
  updateButton();
  if (!measureFrame) measureFrame = requestAnimationFrame(() => { measureFrame = 0; refreshReadingMeasurements(); });
}
function retainSelection() {
  const current = window.getSelection();
  if (!current.rangeCount) return;
  const node = current.anchorNode?.nodeType === 1 ? current.anchorNode : current.anchorNode?.parentElement;
  const body = node?.closest('.passage-body');
  if (body) { activeBody = body; selection = current.getRangeAt(0).cloneRange(); }
}
document.addEventListener('selectionchange', () => {
  retainSelection();
  for (const button of toolbar.querySelectorAll('[data-command]')) {
    if (['bold', 'italic', 'underline'].includes(button.dataset.command))
      button.setAttribute('aria-pressed', String(document.queryCommandState(button.dataset.command)));
  }
});
function command(name, value) {
  if (!activeBody) { message('Click a passage to start editing.'); return; }
  activeBody.focus({ preventScroll: true });
  if (selection && activeBody.contains(selection.commonAncestorContainer)) {
    const current = window.getSelection(); current.removeAllRanges(); current.addRange(selection);
  }
  document.execCommand(name, false, value);
  retainSelection(); changed();
}
toolbar.addEventListener('pointerdown', event => {
  if (event.target.closest('[data-command]')) event.preventDefault();
});
toolbar.addEventListener('click', event => {
  const button = event.target.closest('[data-command]');
  if (button) command(button.dataset.command);
});
document.querySelector('#editor-font').addEventListener('change', event => command('fontName', event.target.value));
document.querySelector('#article').addEventListener('focusin', event => {
  const body = event.target.closest('.passage-body');
  if (body) { activeBody = body; previewPassage(body.dataset.section); }
});
document.querySelector('#article').addEventListener('input', event => {
  activeBody = event.target.closest('.passage-body'); changed();
});
document.querySelector('#article').addEventListener('paste', event => {
  if (!event.target.closest('.passage-body')) return;
  event.preventDefault();
  // Paste plain text; imported document styles and embedded media stay out.
  document.execCommand('insertText', false, event.clipboardData.getData('text/plain'));
});
document.querySelector('#article').addEventListener('keydown', event => {
  if (event.key === 'Tab' && event.target.closest('.passage-body')) {
    // Keep normal Tab navigation; indentation has explicit toolbar controls.
    retainSelection();
  }
});
document.querySelector('#download-draft').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = 'pi-text-draft.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function waitForPublication(commit) {
  for (let attempt = 0; attempt < 120; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 5000));
    const deployment = await request('/api/deployment/' + commit);
    if (deployment.status === 'completed') {
      if (deployment.conclusion !== 'success') throw new Error('Text saved on GitHub, but deployment did not finish successfully. Check GitHub Actions before retrying.');
      return;
    }
  }
  throw new Error('Text saved on GitHub. Publishing is taking longer than expected; check GitHub Actions.');
}
saveButton.addEventListener('click', async () => {
  if (!dirty || busy) return;
  clearTimeout(draftTimer); storeDraft();
  const snapshot = structuredClone(doc), snapshotJSON = JSON.stringify(snapshot);
  busy = true; updateButton(); message('Saving to GitHub…');
  try {
    validateCopy(snapshot, ids);
    const result = await request('/api/publish', {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Editor-Token': token },
      body: JSON.stringify({ sha, doc: snapshot })
    });
    sha = result.sha; saved = snapshotJSON; dirty = JSON.stringify(doc) !== saved; storeDraft();
    if (result.commit) {
      message('Saved · publishing to the live site…');
      await waitForPublication(result.commit);
    }
    message(dirty ? 'Published · you have newer unsaved changes' : 'Published · your changes are live');
  } catch (error) {
    message(error.message, true); document.querySelector('#download-draft').hidden = false;
  } finally { busy = false; updateButton(); }
});
window.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault(); if (!saveButton.disabled) saveButton.click();
  }
});
window.addEventListener('beforeunload', event => {
  if (dirty) { storeDraft(); event.preventDefault(); event.returnValue = ''; }
});
try {
  const state = await request('/api/state');
  token = state.token; sha = state.sha; doc = validateCopy(state.doc, ids); saved = JSON.stringify(doc);
  let draft;
  try { draft = JSON.parse(localStorage.getItem(draftKey)); } catch {}
  if (draft && draft.sha === sha) {
    doc = validateCopy(draft.doc, ids); dirty = JSON.stringify(doc) !== saved;
    message(dirty ? 'Restored your unsaved draft' : 'Select any passage to edit');
  } else message('Select any passage to edit · Save to live publishes your text');
  applyCopy(doc);
  for (const body of document.querySelectorAll('.passage-body')) {
    body.contentEditable = 'true'; body.spellcheck = true;
    body.setAttribute('role', 'textbox'); body.setAttribute('aria-multiline', 'true');
    body.setAttribute('aria-label', 'Edit ' + body.previousElementSibling.textContent);
  }
  if (draft && draft.sha !== sha) {
    // Keep stale drafts recoverable instead of silently publishing over new text.
    const recovery = document.createElement('button'); recovery.textContent = 'Download older draft';
    recovery.addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(draft.doc, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = 'pi-older-draft.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    toolbar.querySelector('.editor-feedback').append(recovery);
    message('Loaded latest text. An older local draft is available to download.');
  }
  focusPassage(initialPassage);
  updateButton();
} catch (error) { message(error.message + ' Reload to reconnect.', true); }
