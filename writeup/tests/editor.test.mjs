import test from 'node:test';
import http from 'node:http';
import assert from 'node:assert/strict';
import { createEditorServer } from '../editor-server.mjs';
import { validateCopy, emptyCopy } from '../rich-text.mjs';
import { steps, details } from '../content.js';
import { readFile } from 'node:fs/promises';
const ids = new Set([...steps, ...details].map(x => x.id));
const example = { version: 1, sections: { vlm: [
  { tag: 'p', children: [{ tag: 'strong', children: [{ text: 'Bold' }] }, { tag: 'u', font: 'sans', children: [{ text: ' Underlined' }] }] },
  { tag: 'blockquote', children: [{ tag: 'p', children: [{ text: 'Indented' }] }] },
  { tag: 'ul', children: [{ tag: 'li', children: [{ text: 'Item' }] }] },
  { equation: 'flow-loss' }
] } };
test('text schema accepts supported formatting and rejects executable or arbitrary content', () => {
  assert.equal(validateCopy(example, ids), example);
  for (const bad of [
    { version: 1, sections: { unknown: [] } },
    { version: 1, sections: { vlm: [{ tag: 'script', children: [] }] } },
    { version: 1, sections: { vlm: [{ tag: 'span', font: 'url(evil)', children: [] }] } },
    { version: 1, sections: { vlm: [{ tag: 'p', onclick: 'alert(1)', children: [] }] } },
    { version: 1, sections: { vlm: [{ text: 2 }] } },
    { version: 1, sections: {}, geometry: [] }
  ]) assert.throws(() => validateCopy(bad, ids));
});
test('published text document validates against actual section IDs', async () => {
  validateCopy(JSON.parse(await readFile(new URL('../text.json', import.meta.url))), ids);
});
test('publisher limits writes to text, detects conflicts, and blocks foreign requests', async t => {
  let remote = { sha: 'old-blob', doc: emptyCopy() }, writes = [];
  const api = async (path, method, body) => {
    if (method === 'PUT') {
      writes.push({ path, body });
      remote = { sha: 'new-blob', doc: JSON.parse(Buffer.from(body.content, 'base64').toString()) };
      return { content: { sha: remote.sha }, commit: { sha: 'a'.repeat(40) } };
    }
    if (path.includes('/actions/')) return { workflow_runs: [{ status: 'completed', conclusion: 'success' }] };
    return { sha: remote.sha, content: Buffer.from(JSON.stringify(remote.doc)).toString('base64') };
  };
  const server = createEditorServer({ api, port: 0 });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const origin = 'http://127.0.0.1:' + server.address().port;
  const state = await (await fetch(origin + '/api/state')).json();
  const headers = { Origin: origin, 'Content-Type': 'application/json', 'X-Editor-Token': state.token };
  const send = (data, extra = {}) => fetch(origin + '/api/publish', { method: 'POST', headers: { ...headers, ...extra }, body: JSON.stringify(data) });
  assert.equal((await send({ sha: state.sha, doc: example }, { Origin: 'https://evil.test' })).status, 403);
  assert.equal((await send({ sha: state.sha, doc: example }, { 'X-Editor-Token': 'wrong' })).status, 403);
  const foreignHostStatus = await new Promise(resolve => {
    http.get(origin + '/api/state', { headers: { Host: 'evil.test' } }, res => { res.resume(); resolve(res.statusCode); });
  });
  assert.equal(foreignHostStatus, 403);
  assert.equal((await send({ sha: 'stale', doc: example })).status, 409);
  assert.equal((await send({ sha: state.sha, doc: { version: 1, sections: { unknown: [] } } })).status, 400);
  assert.equal(writes.length, 0);
  const result = await send({ sha: state.sha, doc: example });
  assert.equal(result.status, 200);
  assert.equal((await result.json()).commit, 'a'.repeat(40));
  assert.equal(writes.length, 1);
  assert.equal(writes[0].path, 'repos/AcastaPaloma/pi/contents/writeup/text.json');
  assert.equal(writes[0].body.sha, 'old-blob');
  assert.equal(writes[0].body.branch, 'main');
  assert.deepEqual(remote.doc, example);
  assert.equal((await send({ sha: state.sha, doc: example })).status, 409);
  assert.equal((await send({ sha: 'new-blob', doc: example })).status, 200);
  assert.equal(writes.length, 1, 'unchanged saves do not create commits');
  assert.equal((await fetch(origin + '/api/deployment/not-a-sha')).status, 400);
  const deployment = await (await fetch(origin + '/api/deployment/' + 'a'.repeat(40))).json();
  assert.equal(deployment.conclusion, 'success');
  assert.equal((await fetch(origin + '/editor-server.mjs')).status, 404);
  assert.equal((await fetch(origin + '/author-copy.json')).status, 404);
});
