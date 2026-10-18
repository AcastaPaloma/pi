import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { steps, details } from './content.js';
import { validateCopy, emptyCopy } from './rich-text.mjs';

const root = dirname(fileURLToPath(import.meta.url));
const ids = new Set([...steps, ...details].map(x => x.id));
const repo = 'repos/AcastaPaloma/pi';
const contentPath = repo + '/contents/writeup/text.json';
export function github(endpoint, method = 'GET', body) {
  return new Promise((resolve, reject) => {
    const args = ['api', endpoint, '--method', method];
    if (body) args.push('--input', '-');
    const child = spawn('gh', args, { stdio: ['pipe', 'pipe', 'pipe'] });
    let output = '', error = '';
    child.stdout.on('data', data => { output += data; });
    child.stderr.on('data', data => { error += data; });
    child.on('error', () => reject(Object.assign(new Error('GitHub CLI is unavailable. Install gh and run gh auth login.'), { status: 503 })));
    child.on('close', code => {
      if (!code) {
        try { resolve(JSON.parse(output)); } catch { reject(new Error('Unexpected GitHub response.')); }
      } else {
        const status = /HTTP 404/.test(error) ? 404 : /HTTP (409|422)/.test(error) ? 409 : 502;
        reject(Object.assign(new Error(status === 409 ? 'The text changed on GitHub. Download your draft, then reload the latest version before saving.' : 'GitHub could not complete the request. Check gh auth status and try again.'), { status }));
      }
    });
    child.stdin.end(body ? JSON.stringify(body) : undefined);
  });
}
export function createEditorServer({ api = github, port = 4174 } = {}) {
  const token = randomBytes(32).toString('hex');
  let saving = false;
  async function state() {
    try {
      const file = await api(contentPath + '?ref=main');
      const doc = JSON.parse(Buffer.from(file.content, 'base64').toString('utf8'));
      validateCopy(doc, ids); return { sha: file.sha, doc };
    } catch (error) { if (error.status === 404) return { sha: null, doc: emptyCopy() }; throw error; }
  }
  function json(res, status, value) {
    res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    res.end(JSON.stringify(value));
  }
  return http.createServer(async (req, res) => {
    try {
      const host = '127.0.0.1:' + req.socket.localPort, origin = 'http://' + host;
      // Loopback binding, exact Host, Origin, and session token prevent a foreign
      // website from using the authenticated local publisher.
      if (req.headers.host !== host) return json(res, 403, { error: 'Invalid host.' });
      const path = new URL(req.url, origin).pathname;
      if (path.startsWith('/api/')) {
        if (req.headers['sec-fetch-site'] === 'cross-site') return json(res, 403, { error: 'Same-origin requests only.' });
        if (path === '/api/state' && req.method === 'GET') return json(res, 200, { ...await state(), token });
        if (path === '/api/publish' && req.method === 'POST') {
          if (req.headers.origin !== origin || req.headers['x-editor-token'] !== token ||
              !req.headers['content-type']?.startsWith('application/json')) return json(res, 403, { error: 'Invalid editor session. Reload the editor.' });
          if (saving) return json(res, 409, { error: 'Another save is in progress. Try again shortly.' });
          let raw = '';
          for await (const chunk of req) { raw += chunk; if (Buffer.byteLength(raw) > 2000000) return json(res, 413, { error: 'Text document is too large.' }); }
          let body;
          try { body = JSON.parse(raw); validateCopy(body.doc, ids); }
          catch (error) { return json(res, 400, { error: error.message }); }
          saving = true;
          try {
            const current = await state();
            if (body.sha !== current.sha) return json(res, 409, { error: 'A newer version is already saved. Download your draft, then reload to edit the latest version.' });
            if (JSON.stringify(current.doc) === JSON.stringify(body.doc)) return json(res, 200, { sha: current.sha, unchanged: true });
            const update = {
              message: 'Update Pi write-up text',
              content: Buffer.from(JSON.stringify(body.doc, null, 2) + '\n').toString('base64'), branch: 'main',
              ...(current.sha ? { sha: current.sha } : {})
            };
            const result = await api(contentPath, 'PUT', update);
            return json(res, 200, { sha: result.content.sha, commit: result.commit.sha });
          } finally { saving = false; }
        }
        if (path.startsWith('/api/deployment/') && req.method === 'GET') {
          const commit = path.split('/').pop();
          if (!/^[a-f0-9]{40}$/.test(commit)) return json(res, 400, { error: 'Invalid revision.' });
          const result = await api(repo + '/actions/workflows/pages.yml/runs?head_sha=' + commit + '&per_page=1');
          const run = result.workflow_runs[0];
          return json(res, 200, { status: run?.status ?? 'queued', conclusion: run?.conclusion ?? null });
        }
        return json(res, 404, { error: 'Not found.' });
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') return json(res, 405, { error: 'Method not allowed.' });
      const name = path === '/' || path === '/editor' ? 'training.html' : decodeURIComponent(path.slice(1));
      const allowed = new Set(['training.html', 'style.css', 'app.js', 'content.js', 'article.js', 'camera.mjs', 'reading.mjs', 'rich-text.mjs', 'math-text.mjs', 'text.json', 'editor.mjs', 'editor.css']);
      if (!allowed.has(name) && !/^assets\/[a-z0-9-]+\.svg$/.test(name)) return json(res, 404, { error: 'Not found.' });
      let data = await readFile(resolve(root, name));
      if (name === 'training.html' && path !== '/') data = Buffer.from(data.toString().replace('</head>', '<link rel="stylesheet" href="./editor.css"><script type="module" src="./editor.mjs"></script></head>'));
      const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml' };
      res.writeHead(200, { 'Content-Type': types[extname(name)], 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "frame-ancestors 'none'" });
      res.end(req.method === 'HEAD' ? undefined : data);
    } catch (error) { json(res, error.code === 'ENOENT' ? 404 : error.status ?? 500, { error: error.message }); }
  });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  createEditorServer().listen(4174, '127.0.0.1', () => console.log('Pi text editor: http://127.0.0.1:4174/editor'));
}
