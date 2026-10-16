"""Live, interactive two-arm demo: drag the objects and the destination on a table map, press Run, and
watch the policy act in MuJoCo through all three cameras.

    MUJOCO_GL=egl python -m pi_model.live_demo --checkpoint exports/demo/pi05_v1_step_0020000 --port 8765
    # then, from your laptop:  ssh -N -L 8765:localhost:8765 ubuntu@<gpu-host>  and open http://localhost:8765

The server binds to 127.0.0.1 only (reach it through the SSH tunnel). One simulator and one policy are
shared by everyone who opens the page; the last placement or Run wins. Policy inference and all MuJoCo
rendering happen on one worker thread (EGL contexts are per thread); HTTP handlers only enqueue commands
and read the latest frame. The episode loop is the same as pi_model.eval_sim.run_episode: oracle subtask
prompts from an observe-only scripted teacher, 25 actions executed per chunk, no inference delay.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pi_sim.tasks import ARMS, CENTER_BIN, PARK, REGION, RELAY, TaskEnv, TaskTeacher
from .eval_sim import _captioned_frame
from .policy import HistoryBuffer, Pi07Policy
from .prompt import PromptContext

TASKS = ("pick_place_any", "bin", "handover", "two_bins")
COLORS = ("red", "green", "blue", "yellow")
EXECUTE = 25
MAX_STREAMS = 40   # concurrent MJPEG streams (each open page holds two)
MAX_BODY = 16384   # bytes; placement and camera requests are tiny
FRAME_EVERY = 4  # render the views every 4 control steps (12.5 frames per simulated second)
VIEW_W, VIEW_H = 1152, 648  # the free 3D camera
DEFAULT_VIEW = {"azimuth": 245.0, "elevation": -24.0, "distance": 0.85, "lookat": [0.0, -0.07, 0.05]}


class Demo:
    def __init__(self, checkpoint, device, dtype, denoise_steps, episode_seconds):
        self.policy = Pi07Policy(checkpoint, device=device, dtype=dtype, denoise_steps=denoise_steps)
        self.episode_seconds = episode_seconds
        self.envs: dict = {}
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.frame_jpeg = b""   # the three robot cameras with captions
        self.view_jpeg = b""    # the free 3D camera
        self.frame_id = 0
        self.view = dict(DEFAULT_VIEW)
        self.view_version = 0
        self._view_renderers: dict = {}
        self._last_env = None
        self._last_caption = None
        self.status = {"state": "loading", "task": "two_bins", "message": "loading the policy"}
        self.placement: dict = {}
        self.stop_flag = threading.Event()
        self.realtime = True

    # ----------------------------------------------------------------- environment
    def env_for(self, task, color):
        second = next(c for c in COLORS if c != color)
        key = (task, color)
        if key not in self.envs:
            self.envs[key] = TaskEnv(task, control_mode="joint", object_color=color, second_color=second,
                                     control_hz=self.policy.hz, episode_seconds=self.episode_seconds)
        return self.envs[key]

    def place(self, task, color, cube, dest, cube2=None, seed=0):
        """Reset the task, then move the objects and the destination to the requested table positions."""
        env = self.env_for(task, color)
        env.reset(seed=seed, options={"randomize": True})
        rng = np.random.default_rng(seed)
        yaw = lambda: float(rng.uniform(-np.pi / 4, np.pi / 4))
        cube = self._clamp(cube, "left" if cube[0] < 0 else "right")
        if task == "two_bins":
            cube = self._clamp(cube, "left", strict_side=True)
            cube2 = self._clamp(cube2 if cube2 is not None else [0.18, 0.0], "right", strict_side=True)
            env._set_object("cube2", cube2, yaw())
            env.side = "both"
        else:
            env.side = "left" if cube[0] < 0 else "right"
        env._set_object("cube", cube, yaw())
        dest = np.clip(np.asarray(dest, float), [-0.30, -0.08], [0.30, 0.13])
        if task == "pick_place_any":
            env.data.mocap_pos[env._mocap["goal"]] = [dest[0], dest[1], 0.001]
        else:
            env.data.mocap_pos[env._mocap["bin"]] = [dest[0], dest[1], 0.0]
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        for _ in range(60):  # let the objects settle exactly as reset does
            mujoco.mj_step(env.model, env.data)
        env.data.time = 0
        mujoco.mj_forward(env.model, env.data)
        self.placement = {"task": task, "color": color, "cube": [float(v) for v in cube[:2]],
                          "dest": [float(v) for v in dest[:2]],
                          "cube2": [float(v) for v in cube2[:2]] if task == "two_bins" else None}
        return env

    @staticmethod
    def _clamp(xy, arm, strict_side=False):
        lo = {"left": [-0.30, -0.08], "right": [-0.06, -0.08]}[arm]
        hi = {"left": [0.06, 0.13], "right": [0.30, 0.13]}[arm]
        if strict_side:  # two_bins: each arm's object stays on its own half
            lo, hi = ([-0.30, -0.08], [-0.03, 0.13]) if arm == "left" else ([0.03, -0.08], [0.30, 0.13])
        return np.clip(np.asarray(xy, float), lo, hi)

    def randomize(self, task, color, seed):
        """A placement from the training distribution (what the policy has seen)."""
        env = self.env_for(task, color)
        env.reset(seed=seed, options={"randomize": True})
        cube = env.object_position("cube")[:2]
        if task == "pick_place_any":
            dest = env.data.mocap_pos[env._mocap["goal"]][:2]
        else:
            dest = env.bin_position[:2]
        cube2 = env.object_position("cube2")[:2] if task == "two_bins" else None
        return self.place(task, color, cube, dest, cube2, seed=seed)

    # ----------------------------------------------------------------- frames
    def render_view(self, env):
        """The whole scene from a free perspective camera the viewer orbits (not a robot camera)."""
        import mujoco
        r = self._view_renderers.get(id(env))
        if r is None:
            env.model.vis.global_.offwidth = max(VIEW_W, env.model.vis.global_.offwidth)
            env.model.vis.global_.offheight = max(VIEW_H, env.model.vis.global_.offheight)
            r = self._view_renderers[id(env)] = mujoco.Renderer(env.model, height=VIEW_H, width=VIEW_W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        with self.lock:
            v = dict(self.view)
        cam.azimuth, cam.elevation, cam.distance = v["azimuth"], v["elevation"], v["distance"]
        cam.lookat[:] = v["lookat"]
        r.update_scene(env.data, camera=cam)
        return r.render()

    def publish(self, env, task, subtask, t, info):
        self._last_env, self._last_caption = env, (task, subtask, t, info)
        frame = _captioned_frame(env, self.policy.cameras, task, subtask, t, info)
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=80)
        vbuf = io.BytesIO()
        Image.fromarray(self.render_view(env)).save(vbuf, format="JPEG", quality=85)
        with self.lock:
            self.frame_jpeg = buf.getvalue()
            self.view_jpeg = vbuf.getvalue()
            self.frame_id += 1

    def set_status(self, **kw):
        with self.lock:
            self.status = {**self.status, **kw}

    # ----------------------------------------------------------------- episode (mirrors eval_sim.run_episode)
    def run_episode(self, env, seed):
        policy = self.policy
        task_name = self.placement["task"]
        obs = env._observation()
        info = env._info(False)
        task = info["instruction"]
        history = HistoryBuffer(policy.cameras, hz=env.control_hz)
        shadow = TaskTeacher(env, rng=np.random.default_rng(seed), observe_only=True)
        generator = torch.Generator(device=policy.device).manual_seed(seed)

        def context():
            return PromptContext(task=task, subtask=shadow.subtask, speed_steps=policy.speed_for(task), quality=5,
                                 mistake=False, control_mode="joint", speed_bin=500)

        def observe(step):
            images = {cam: policy.prepare_image(env.render_camera(cam, native=True)) for cam in policy.cameras}
            history.append(step, images, obs["state"])

        observe_every = math.gcd(EXECUTE, policy.stride)
        t, cursor, calls = 0, 0, 1
        observe(t)
        chunk = policy.act(history, context(), generator=generator)
        self.set_status(state="running", task=task_name, instruction=task, message="running", success=False, t=0.0)
        start = time.monotonic()
        while not self.stop_flag.is_set():
            if cursor >= EXECUTE:
                observe(t)
                chunk = policy.act(history, context(), generator=generator)
                cursor = 0
                calls += 1
            obs, _, terminated, truncated, info = env.step(chunk[min(cursor, len(chunk) - 1)])
            shadow.observe()
            cursor += 1
            t += 1
            if t % observe_every == 0 and cursor < EXECUTE:
                observe(t)
            if t % FRAME_EVERY == 0 or terminated or truncated:
                self.publish(env, task, shadow.subtask, t, info)
                self.set_status(t=round(t / env.control_hz, 2), subtask=shadow.subtask)
                if self.realtime:  # pace to real time so the motion looks natural
                    ahead = t / env.control_hz - (time.monotonic() - start)
                    if ahead > 0:
                        time.sleep(ahead)
            if terminated or truncated:
                break
        success = bool(info.get("is_success"))
        msg = "stopped" if self.stop_flag.is_set() else ("SUCCESS" if success else
                                                         ("failed: object fell" if info.get("failure") else "time limit reached"))
        self.set_status(state="done", success=success, message=msg, t=round(t / env.control_hz, 2), calls=calls)

    # ----------------------------------------------------------------- worker thread
    def worker(self):
        seed = 0
        self.set_status(state="idle", message="ready")
        env = self.randomize("two_bins", "red", seed)
        self.publish(env, *self._preview_args(env))
        seen_view = self.view_version
        while True:
            try:
                cmd = self.commands.get(timeout=0.05)
            except queue.Empty:
                if self.view_version != seen_view and self._last_env is not None:  # orbiting while idle
                    seen_view = self.view_version
                    self.publish(self._last_env, *self._last_caption)
                continue
            self.stop_flag.clear()
            try:
                if cmd["op"] == "place":
                    p = cmd["placement"]
                    env = self.place(p["task"], p.get("color", "red"), p["cube"], p["dest"], p.get("cube2"), seed=seed)
                    self.publish(env, *self._preview_args(env))
                    self.set_status(state="idle", task=p["task"], message="placed; press Run", success=None, t=0.0)
                elif cmd["op"] == "randomize":
                    seed += 1
                    env = self.randomize(cmd["task"], cmd.get("color", "red"), seed)
                    self.publish(env, *self._preview_args(env))
                    self.set_status(state="idle", task=cmd["task"], message="training-distribution placement; press Run",
                                    success=None, t=0.0)
                elif cmd["op"] == "run":
                    p = self.placement
                    seed += 1
                    env = self.place(p["task"], p["color"], p["cube"], p["dest"], p.get("cube2"), seed=seed)
                    self.run_episode(env, seed)
            except Exception as exc:  # keep serving after a bad request
                self.set_status(state="error", message=f"{type(exc).__name__}: {exc}")

    def _preview_args(self, env):
        info = env._info(False)
        return info["instruction"], "-", 0, info


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SO-101 two-arm live demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#d9d7d2;--accent:#2f6fde}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececee;--muted:#9a9aa2;--card:#1d1d21;--line:#34343a;--accent:#6ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px} p.sub{color:var(--muted);margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select,button{font:inherit;padding:6px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
canvas{width:100%;max-width:640px;height:auto;touch-action:none;border-radius:8px;display:block}
img#live{width:100%;border-radius:8px;background:#000;display:block;margin-top:6px}
img#view{width:100%;border-radius:8px;background:#000;display:block;cursor:grab;user-select:none;touch-action:none}
img#view.dragging{cursor:grabbing}
#status{font-weight:600} .muted{color:var(--muted);font-size:13px}
.ok{color:#1f9d55}.bad{color:#d64545}
</style></head><body><main>
<h1>SO-101 two-arm policy, live in simulation</h1>
<p class="sub">π0.7 recipe on the π0.5 backbone (pi05_v1 step 20000). Drag the objects and the destination on the map, then press Run.</p>
<div class="card view3d"><div class="row" style="justify-content:space-between;margin-bottom:8px">
<div class="row"><span id="status">connecting…</span><span class="muted" id="sub"></span></div>
<div class="row"><button data-v="front">Front</button><button data-v="behind">Behind</button><button data-v="left">Side</button><button data-v="top">Top</button><button data-v="reset">Reset view</button></div></div>
<img id="view" src="/stream3d" alt="3D view of both arms and the table" draggable="false">
<p class="muted">Drag to orbit, scroll to zoom. This is a free camera around the scene; the policy itself only sees the three cameras below.</p>
<img id="live" src="/stream" alt="the policy's cameras: overhead, left wrist, right wrist"></div>
<div class="card"><div class="row">
<label>Task <select id="task">
<option value="two_bins">Two cubes into the bin, one per arm</option>
<option value="handover">Hand the cube over, then bin it</option>
<option value="bin">Put the cube in the bin</option>
<option value="pick_place_any">Place the cube on the green disk</option></select></label>
<label>Cube colour <select id="color"><option>red</option><option>green</option><option>blue</option><option>yellow</option></select></label>
<button id="rand">Random placement</button>
<button id="run" class="primary">Run</button><button id="stop">Stop</button>
</div>
<p class="muted" id="hint"></p>
<canvas id="map" width="640" height="400"></canvas>
<p class="muted">Shaded boxes are where training placed the objects. Anything else in reach works as a test of generalisation; expect lower success far outside them.</p>
</div>
</main><script>
const X0=-0.36,X1=0.36,Y0=-0.26,Y1=0.18, W=640,H=400;
const cv=document.getElementById('map'), g=cv.getContext('2d');
const px=x=>(x-X0)/(X1-X0)*W, py=y=>H-(y-Y0)/(Y1-Y0)*H;
const wx=u=>X0+u/W*(X1-X0), wy=v=>Y0+(H-v)/H*(Y1-Y0);
const REGION=__REGION__, CENTER_BIN=__CENTER_BIN__, RELAY=__RELAY__;
const COL={red:'#d33',green:'#2a2',blue:'#26d',yellow:'#db2'};
let P={task:'two_bins',color:'red',cube:[-0.18,0],cube2:[0.18,0],dest:[CENTER_BIN[0],CENTER_BIN[1]]}, drag=null;
function second(c){return ['red','green','blue','yellow'].find(k=>k!==c)}
function hint(){const t=P.task;document.getElementById('hint').textContent=
 t==='two_bins'?'Left cube for the left arm, right cube for the right arm, bin anywhere both can reach (near the middle).':
 t==='handover'?'Put the cube on one side and the bin on the other side; the arms pass it over the grey pad.':
 t==='bin'?'Cube and bin on the same side work best (one arm does it).':'Cube and green disk on the same side work best.'}
function draw(){
 const css=getComputedStyle(document.documentElement);
 g.clearRect(0,0,W,H); g.fillStyle=css.getPropertyValue('--bg'); g.fillRect(0,0,W,H);
 g.fillStyle='#c9b48f'; g.fillRect(px(-0.5),py(0.4),px(0.5)-px(-0.5),py(-0.4)-py(0.4));
 g.fillStyle='rgba(47,111,222,0.16)';
 for(const k of ['left','right']){const [[a,b],[c,d]]=REGION[k]; g.fillRect(px(a),py(d),px(b)-px(a),py(c)-py(d));}
 g.strokeStyle='rgba(0,0,0,0.35)'; g.setLineDash([5,4]); g.strokeRect(px(-0.30),py(0.13),px(0.30)-px(-0.30),py(-0.08)-py(0.13)); g.setLineDash([]);
 g.fillStyle='#444'; for(const x of [-0.18,0.18]){g.beginPath();g.arc(px(x),py(-0.22),16,0,7);g.fill();}
 g.fillStyle='#fff'; g.font='12px system-ui'; g.fillText('left arm',px(-0.18)-22,py(-0.22)+4); g.fillText('right arm',px(0.18)-25,py(-0.22)+4);
 if(P.task==='handover'){g.fillStyle='#999'; g.fillRect(px(RELAY[0]-0.03),py(RELAY[1]+0.03),px(0.06)-px(0),py(0)-py(0.06));}
 const d=P.dest;
 if(P.task==='pick_place_any'){g.fillStyle='#3b3';g.beginPath();g.arc(px(d[0]),py(d[1]),px(0.03)-px(0),0,7);g.fill();}
 else{g.fillStyle='#888';const s=px(0.048)-px(0);g.fillRect(px(d[0])-s,py(d[1])-s,2*s,2*s);g.fillStyle='#bbb';g.fillRect(px(d[0])-s+4,py(d[1])-s+4,2*s-8,2*s-8);}
 const cube=(c,col)=>{const s=px(0.015)-px(0);g.fillStyle=col;g.fillRect(px(c[0])-s,py(c[1])-s,2*s,2*s);g.strokeStyle='#000';g.strokeRect(px(c[0])-s,py(c[1])-s,2*s,2*s);};
 cube(P.cube,COL[P.color]); if(P.task==='two_bins') cube(P.cube2,COL[second(P.color)]);
 g.fillStyle=css.getPropertyValue('--fg'); g.font='12px system-ui'; g.fillText('dashed: reach limit · shaded: training placements · drag the cube(s) and the '+(P.task==='pick_place_any'?'green disk':'bin'),8,16);
}
function pick(u,v){const items=[['cube',P.cube],['dest',P.dest]]; if(P.task==='two_bins') items.push(['cube2',P.cube2]);
 let best=null,bd=1e9; for(const [k,c] of items){const dd=Math.hypot(px(c[0])-u,py(c[1])-v); if(dd<bd){bd=dd;best=k;}} return bd<40?best:null;}
function evpos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height];}
cv.addEventListener('pointerdown',e=>{const [u,v]=evpos(e);drag=pick(u,v);if(drag)cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(!drag)return;const [u,v]=evpos(e);P[drag]=[+wx(u).toFixed(3),+wy(v).toFixed(3)];draw();});
cv.addEventListener('pointerup',()=>{if(drag){drag=null;send('/place',{placement:P});}});
async function send(path,body){const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body||{})});return r.json();}
document.getElementById('task').onchange=e=>{P.task=e.target.value;
 if(P.task==='two_bins'){P.cube=[-0.18,0];P.cube2=[0.18,0];P.dest=[...CENTER_BIN];}
 else if(P.task==='handover'){P.cube=[-0.18,0];P.dest=[0.18,0.0];}
 else{P.cube=[-0.20,-0.02];P.dest=[-0.12,0.04];}
 hint();draw();send('/place',{placement:P});};
document.getElementById('color').onchange=e=>{P.color=e.target.value;draw();send('/place',{placement:P});};
document.getElementById('rand').onclick=async()=>{const s=await send('/randomize',{task:P.task,color:P.color});};
document.getElementById('run').onclick=()=>send('/run',{placement:P});
document.getElementById('stop').onclick=()=>send('/stop');
async function poll(){try{const s=await (await fetch('/state')).json();
 const el=document.getElementById('status'); el.textContent=s.status.message+(s.status.t?`  ·  t=${s.status.t}s`:'');
 el.className=s.status.success===true?'ok':(s.status.state==='done'||s.status.state==='error')?'bad':'';
 document.getElementById('sub').textContent=(s.status.subtask?('prompted: '+s.status.subtask+'  ·  '):'')+(s.viewers>1?`${s.viewers} people watching (one shared simulator: Run restarts it for everyone)`:'');
 if(s.placement&&s.placement.cube&&!drag&&s.status.state!=='running'&&s.placement.task===P.task){P.cube=s.placement.cube;P.dest=s.placement.dest;if(s.placement.cube2)P.cube2=s.placement.cube2;draw();}
}catch(e){} setTimeout(poll,500);}
// ---- orbit the free 3D camera
let V={azimuth:245,elevation:-24,distance:0.85}, vdrag=null, pending=null, inflight=false;
async function pushView(extra){pending={...V,...(extra||{})}; if(inflight)return; inflight=true;
 while(pending){const body=pending;pending=null; try{const r=await send('/view',{view:body}); if(r.view){V.azimuth=r.view.azimuth;V.elevation=r.view.elevation;V.distance=r.view.distance;}}catch(e){}}
 inflight=false;}
const vi=document.getElementById('view');
vi.addEventListener('pointerdown',e=>{vdrag=[e.clientX,e.clientY];vi.setPointerCapture(e.pointerId);vi.classList.add('dragging');});
vi.addEventListener('pointermove',e=>{if(!vdrag)return;const dx=e.clientX-vdrag[0],dy=e.clientY-vdrag[1];vdrag=[e.clientX,e.clientY];
 V.azimuth=(V.azimuth-dx*0.35+360)%360; V.elevation=Math.max(-89,Math.min(-3,V.elevation-dy*0.3)); pushView();});
vi.addEventListener('pointerup',()=>{vdrag=null;vi.classList.remove('dragging');});
vi.addEventListener('wheel',e=>{e.preventDefault();V.distance=Math.max(0.35,Math.min(2.5,V.distance*(e.deltaY>0?1.08:1/1.08)));pushView();},{passive:false});
const PRESETS={front:{azimuth:270,elevation:-22,distance:0.85},behind:{azimuth:90,elevation:-28,distance:0.95},left:{azimuth:180,elevation:-20,distance:1.0},top:{azimuth:90,elevation:-89,distance:1.0}};
document.querySelectorAll('button[data-v]').forEach(b=>b.onclick=()=>{const k=b.dataset.v; if(k==='reset'){send('/view',{view:{reset:true}}).then(r=>{if(r.view)Object.assign(V,r.view)});} else {Object.assign(V,PRESETS[k]);pushView();}});
hint();draw();poll();
</script></body></html>"""


def make_handler(demo: Demo):
    page = (PAGE.replace("__REGION__", json.dumps({k: [list(v[0]), list(v[1])] for k, v in REGION.items()}))
                .replace("__CENTER_BIN__", json.dumps([float(v) for v in CENTER_BIN]))
                .replace("__RELAY__", json.dumps([float(v) for v in RELAY]))).encode()

    streams = threading.BoundedSemaphore(MAX_STREAMS)
    viewers = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        timeout = 30  # seconds to receive a request; a stalled client cannot hold a thread forever

        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/state":
                with demo.lock:
                    self._json({"status": demo.status, "placement": demo.placement, "view": demo.view,
                                "viewers": viewers["n"]})
            elif self.path.startswith("/stream"):
                which = "view" if self.path.startswith("/stream3d") else "frame"
                if not streams.acquire(blocking=False):
                    return self._json({"error": "too many viewers right now, try again shortly"}, 503)
                if which == "view":
                    viewers["n"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with demo.lock:
                            fid, jpg = demo.frame_id, (demo.view_jpeg if which == "view" else demo.frame_jpeg)
                        if fid != last and jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            last = fid
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    return
                finally:
                    streams.release()
                    if which == "view":
                        viewers["n"] -= 1
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._json({"error": "request too large"}, 413)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                if self.path in ("/place",) and not isinstance(body.get("placement"), dict):
                    raise ValueError("missing placement")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": f"bad request: {exc}"}, 400)
            if self.path in ("/place", "/randomize", "/run") and demo.commands.qsize() > 6:
                return self._json({"error": "busy, try again in a moment"}, 429)
            if self.path == "/place":
                demo.stop_flag.set()
                demo.commands.put({"op": "place", "placement": _validated(body["placement"])})
            elif self.path == "/randomize":
                demo.stop_flag.set()
                demo.commands.put({"op": "randomize", "task": _task(body.get("task")), "color": _color(body.get("color"))})
            elif self.path == "/run":
                demo.stop_flag.set()
                if isinstance(body.get("placement"), dict):
                    demo.commands.put({"op": "place", "placement": _validated(body["placement"])})
                demo.commands.put({"op": "run"})
            elif self.path == "/stop":
                demo.stop_flag.set()
            elif self.path == "/view":
                v = body.get("view") or {}
                if not isinstance(v, dict):
                    return self._json({"error": "bad request: view must be an object"}, 400)
                try:
                    _ = [float(v[k]) for k in ("azimuth", "elevation", "distance") if k in v]
                    if "lookat" in v:
                        _ = [float(x) for x in v["lookat"]]
                except (TypeError, ValueError):
                    return self._json({"error": "bad request: view values must be numbers"}, 400)
                with demo.lock:
                    if v.get("reset"):
                        demo.view = dict(DEFAULT_VIEW)
                    else:
                        cur = demo.view
                        cur["azimuth"] = float(v.get("azimuth", cur["azimuth"])) % 360
                        cur["elevation"] = float(np.clip(float(v.get("elevation", cur["elevation"])), -89, -3))
                        cur["distance"] = float(np.clip(float(v.get("distance", cur["distance"])), 0.35, 2.5))
                        if isinstance(v.get("lookat"), list) and len(v["lookat"]) == 3:
                            cur["lookat"] = [float(np.clip(float(x), -0.6, 0.6)) for x in v["lookat"]]
                    demo.view_version += 1
                    view = dict(demo.view)
                return self._json({"ok": True, "view": view})
            else:
                return self._json({"error": "not found"}, 404)
            self._json({"ok": True})

    return Handler


def _task(t):
    return t if t in TASKS else "two_bins"


def _color(c):
    return c if c in COLORS else "red"


def _xy(v, default):
    try:
        x, y = float(v[0]), float(v[1])
        return [x, y] if all(map(math.isfinite, (x, y))) else default
    except (TypeError, ValueError, IndexError):
        return default


def _validated(p):
    return {"task": _task(p.get("task")), "color": _color(p.get("color")), "cube": _xy(p.get("cube"), [-0.18, 0.0]),
            "dest": _xy(p.get("dest"), [float(CENTER_BIN[0]), float(CENTER_BIN[1])]), "cube2": _xy(p.get("cube2"), [0.18, 0.0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = tunnel only; 0.0.0.0 = anyone who can reach this machine (no authentication)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--denoise-steps", type=int, default=5)
    ap.add_argument("--episode-seconds", type=float, default=40.0)
    args = ap.parse_args(argv)
    demo = Demo(args.checkpoint, args.device, args.dtype, args.denoise_steps, args.episode_seconds)
    threading.Thread(target=demo.worker, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    server.daemon_threads = True
    print(f"live demo on http://{args.host}:{args.port}  (tunnel: ssh -N -L {args.port}:localhost:{args.port} <host>)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
