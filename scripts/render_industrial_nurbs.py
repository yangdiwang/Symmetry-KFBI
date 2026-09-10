#!/usr/bin/env python3
"""Render native industrial NURBS evaluator samples without reconstructing geometry.

Example (from the repository root)::

    python scripts/render_industrial_nurbs.py

The JSON export contains a ``models`` list and a rectangular ``points`` grid for
each patch. This script only triangulates those supplied samples. Control points,
weights and knot vectors remain in the source JSON; no CAD reader or alternative
surface evaluator is used. PNGs are orthographic scientific visualizations. The
HTML viewer is self-contained, uses WebGL 2, and works without a web server.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


STYLES = {
    "sleeve": ("Flanged bushing", "Shoulder fillets · retaining groove · bore chamfers", "#81929e", 27, -55),
    "u_bracket": ("Bored clevis bracket", "Upright ears · aligned pin bores · rounded ends", "#8a9ca4", 29, -46),
    "flange": ("Raised-face flange", "Eight bolt holes · neck · sealing face · chamfers", "#87939e", 39, -60),
    "impeller": ("Open impeller", "Eight curved vanes · raised hub · circular shroud", "#758a9d", 43, -52),
}
BACKGROUND = "#f5f7fa"
INK = "#142b3b"
MUTED = "#627482"


@dataclass
class RenderModel:
    source: dict[str, Any]
    points: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    grid_lines: np.ndarray
    seam_lines: np.ndarray
    patch_ids: np.ndarray

    @property
    def name(self) -> str:
        return self.source["name"]

    @property
    def style(self) -> tuple:
        return STYLES.get(self.name, (self.name.replace("_", " ").title(),
                                      "Native NURBS surface", "#60879d", 35, -55))

    @property
    def center(self) -> np.ndarray:
        return (self.points.max(axis=0) + self.points.min(axis=0)) / 2

    @property
    def extent(self) -> float:
        return float(np.ptp(self.points, axis=0).max())


def load_models(path: Path) -> list[RenderModel]:
    """Validate exported samples and preserve patch orientation while triangulating."""
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    sources = document.get("models")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Input must contain a nonempty 'models' list")
    models = []
    seen = set()
    for source in sources:
        name = source.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Each model needs a unique nonempty name")
        if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Unsafe model filename: {name!r}")
        seen.add(name)
        patches = source.get("patches")
        if not isinstance(patches, list) or not patches:
            raise ValueError(f"{name}: expected nonempty patch list")
        vertices, normals, triangles, grid_lines, seam_lines, patch_ids = [], [], [], [], [], []
        offset = 0
        for patch_id, patch in enumerate(patches):
            grid = np.asarray(patch.get("points"), dtype=np.float64)
            if grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 2:
                raise ValueError(f"{name}/{patch.get('name', patch_id)}: points must be an M×N×3 grid")
            if not np.isfinite(grid).all():
                raise ValueError(f"{name}/{patch_id}: samples contain nonfinite coordinates")
            nu, nv = grid.shape[:2]
            ids = np.arange(nu * nv, dtype=np.uint32).reshape(nu, nv) + offset
            a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]
            triangles.extend((np.stack((a, b, c), axis=-1).reshape(-1, 3),
                              np.stack((a, c, d), axis=-1).reshape(-1, 3)))
            patch_ids.extend((np.full(a.size, patch_id), np.full(a.size, patch_id)))
            du, dv = np.gradient(grid, axis=(0, 1))
            normal = np.cross(du, dv)
            lengths = np.linalg.norm(normal, axis=-1)
            if np.any(lengths <= np.finfo(float).tiny):
                raise ValueError(f"{name}/{patch_id}: a sampled normal is degenerate")
            normal /= lengths[..., None]
            vertices.append(grid.reshape(-1, 3))
            normals.append(normal.reshape(-1, 3))
            grid_lines.extend((np.stack((ids[:-1, :], ids[1:, :]), axis=-1).reshape(-1, 2),
                               np.stack((ids[:, :-1], ids[:, 1:]), axis=-1).reshape(-1, 2)))
            for edge in (ids[0, :], ids[-1, :], ids[:, 0], ids[:, -1]):
                seam_lines.append(np.stack((edge[:-1], edge[1:]), axis=-1))
            offset += nu * nv
        model = RenderModel(source, np.concatenate(vertices), np.concatenate(normals),
                            np.concatenate(triangles), np.concatenate(grid_lines),
                            np.concatenate(seam_lines), np.concatenate(patch_ids))
        if model.extent <= 0:
            raise ValueError(f"{name}: zero model extent")
        models.append(model)
    return models


def shaded_colors(model: RenderModel) -> np.ndarray:
    """A restrained material with view-relative key and fill lights."""
    triangles = model.points[model.triangles]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(lengths[:, None], np.finfo(float).tiny)
    elev, azim = np.deg2rad(model.style[3:5])
    eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0])
    up = np.cross(eye, right)
    key = eye * .8 - right * .45 + up * .8
    key /= np.linalg.norm(key)
    fill = eye * .6 + right * .7 + up * .2
    fill /= np.linalg.norm(fill)
    luminance = .48 + .46 * np.maximum(0, face_normals @ key) + .11 * np.maximum(0, face_normals @ fill)
    # Tiny tonal differences reveal the patch layout without a distracting wireframe.
    patch_tint = np.array([1., .985, 1.012, .995])[model.patch_ids % 4]
    base = np.asarray(to_rgb(model.style[2]))
    colors = base[None, :] * luminance[:, None] * patch_tint[:, None]
    half_vector = key + eye
    half_vector /= np.linalg.norm(half_vector)
    specular = .13 * np.maximum(0, face_normals @ half_vector) ** 34
    return np.clip(colors + specular[:, None], 0, 1)


def plot_model(ax, model: RenderModel) -> None:
    collection = Poly3DCollection(model.points[model.triangles],
                                  facecolors=shaded_colors(model),
                                  edgecolors="none", linewidths=0,
                                  antialiased=False, zsort="average")
    ax.add_collection3d(collection)
    half = model.extent * .53
    center = model.center
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    view_zoom = 1.35 if model.name in {"flange", "impeller"} else 1.14
    ax.set_box_aspect((1, 1, 1), zoom=view_zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=model.style[3], azim=model.style[4])
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND)


def metric_caption(model: RenderModel) -> str:
    diagnostics = model.source.get("diagnostics", {})
    count = len(model.source["patches"])
    genus = model.source.get("expected_genus")
    parts = [f"{count} NURBS patches"]
    if genus is not None:
        parts.append(f"genus {genus}")
    if "volume" in diagnostics:
        parts.append(f"volume {float(diagnostics['volume']):.5g}")
    return "   /   ".join(parts)


def render_individual(model: RenderModel, output: Path, dpi: int) -> Path:
    fig = plt.figure(figsize=(10.5, 8.7), facecolor=BACKGROUND)
    fig.text(.075, .939, "NATIVE NURBS  /  INDUSTRIAL GEOMETRY", color=MUTED,
             fontsize=10, weight="semibold", family="DejaVu Sans")
    fig.text(.075, .881, model.style[0], color=INK, fontsize=31, weight="bold")
    fig.text(.075, .843, model.style[1], color=MUTED, fontsize=13)
    ax = fig.add_axes([.04, .075, .92, .79], projection="3d")
    plot_model(ax, model)
    fig.text(.075, .055, metric_caption(model), color=MUTED, fontsize=10)
    fig.text(.075, .026, "Native evaluator samples · orthographic view · dimensions in model units",
             color="#86939d", fontsize=8)
    path = output / f"{model.name}.png"
    fig.savefig(path, dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    return path


def render_overview(models: list[RenderModel], output: Path, dpi: int) -> Path:
    columns = 2
    rows = (len(models) + columns - 1) // columns
    fig = plt.figure(figsize=(15, 1.6 + rows * 5.8), facecolor=BACKGROUND)
    fig.text(.058, .955, "INDUSTRIAL NURBS  /  REFINED MODELS", color=INK, fontsize=25, weight="bold")
    fig.text(.058, .921, "Machined features and curved vanes constructed directly in the native geometry code",
             color=MUTED, fontsize=12)
    region_top, region_bottom = .885, .075
    height = (region_top - region_bottom) / rows
    for index, model in enumerate(models):
        col, row = index % columns, index // columns
        left = .042 + col * .49
        bottom = region_top - (row + 1) * height
        ax = fig.add_axes([left, bottom + .025, .45, height - .027], projection="3d")
        plot_model(ax, model)
        fig.text(left + .02, bottom + height - .008, f"0{index + 1}  {model.style[0]}",
                 color=INK, fontsize=18, weight="bold")
        fig.text(left + .02, bottom + .035, metric_caption(model), color=MUTED, fontsize=9)
    fig.text(.058, .035, "Actual native surface samples  /  no imported CAD  /  orthographic views",
             color=MUTED, fontsize=10)
    path = output / "overview.png"
    fig.savefig(path, dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    return path


def binary_array(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def render_interactive(models: list[RenderModel], output: Path) -> Path:
    exports = []
    for model in models:
        exports.append({
            "name": model.name, "title": model.style[0], "subtitle": model.style[1],
            "description": model.source.get("description", ""),
            "color": list(to_rgb(model.style[2])), "elev": model.style[3], "azim": model.style[4],
            "center": model.center.tolist(), "extent": model.extent,
            "patches": len(model.source["patches"]), "genus": model.source.get("expected_genus"),
            "diagnostics": model.source.get("diagnostics", {}),
            "positions": binary_array(model.points, "<f4"),
            "normals": binary_array(model.normals, "<f4"),
            "triangles": binary_array(model.triangles, "<u4"),
            "grid": binary_array(model.grid_lines, "<u4"),
            "seams": binary_array(model.seam_lines, "<u4"),
        })
    data = json.dumps(exports, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    path = output / "interactive.html"
    path.write_text(HTML_TEMPLATE.replace("__MODEL_DATA__", data), encoding="utf-8")
    return path


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native NURBS · Industrial Geometry</title>
<style>
:root{font-family:Inter,"Segoe UI",Arial,sans-serif;color:#142b3b;background:#f5f7fa;font-synthesis:none}
*{box-sizing:border-box}body{margin:0}main{position:relative;min-height:760px;height:100svh;overflow:hidden}
canvas{display:block;position:absolute;inset:0;width:100%;height:100%;touch-action:none;outline:none}
canvas:focus-visible{box-shadow:inset 0 0 0 3px #60879d}header{position:absolute;left:44px;top:35px;pointer-events:none}
.eyebrow{font-size:11px;letter-spacing:2.2px;font-weight:650;color:#627482;margin-bottom:14px}
h1{font-size:clamp(32px,4vw,53px);line-height:1.06;margin:0 0 12px;letter-spacing:-1.5px;font-weight:650}
.subtitle{font-size:15px;color:#627482;margin:0;max-width:470px}aside{position:absolute;left:44px;top:205px;width:224px}
.models{display:grid;gap:8px}button,a.button{font:inherit;cursor:pointer;border:1px solid #dbe2e8;background:#fff;color:#274254;border-radius:10px}
.model{padding:13px 15px;text-align:left;display:flex;align-items:center;gap:13px;box-shadow:0 2px 8px #142b3b03}
.model .number{font-size:10px;color:#8b99a3}.model.active{border-color:#819dab;background:#e8eff4;box-shadow:0 0 0 1px #819dab33}
.model:hover{background:#edf2f6}.model .label{font-size:14px;font-weight:600}.metrics{margin-top:24px;border-top:1px solid #dbe2e8;padding-top:17px;display:grid;grid-template-columns:1fr 1fr;gap:17px 10px}
.metric small{display:block;font-size:10px;letter-spacing:.5px;color:#7d8c98;margin-bottom:5px}.metric strong{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.details{font-size:11px;color:#7a8c98;line-height:1.7;margin-top:17px}.controls{position:absolute;right:34px;top:35px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;max-width:310px;justify-content:flex-end}
.controls label,.controls button,.controls a{padding:10px 12px;font-size:12px;border:1px solid #dbe2e8;border-radius:9px;background:#ffffffd9;color:#355062;backdrop-filter:blur(12px);text-decoration:none}
.controls label{cursor:pointer;user-select:none}.controls input{accent-color:#547e96;vertical-align:middle;margin-right:5px}
.controls button:hover,.controls a:hover{background:#e9f0f5}.hint{position:absolute;bottom:76px;left:50%;transform:translateX(-50%);color:#7e8c97;font-size:12px;white-space:nowrap;pointer-events:none}
footer{position:absolute;bottom:0;left:0;right:0;padding:21px 44px;border-top:1px solid #dfe5ea;background:#ffffff79;display:flex;justify-content:space-between;gap:20px;font-size:11px;color:#7a8c98;backdrop-filter:blur(12px)}
.status{color:#537f74}.status:before{content:"";display:inline-block;background:#719e90;width:6px;height:6px;border-radius:50%;margin-right:7px}
#error{position:absolute;left:30%;right:10%;top:40%;padding:24px;border-radius:12px;background:white;color:#8b4239;display:none}
@media(max-width:800px){main{min-height:850px}header{left:22px;top:24px}h1{font-size:37px}.subtitle{max-width:310px;font-size:13px}aside{left:22px;right:22px;top:155px;width:auto}.models{grid-template-columns:repeat(4,1fr);gap:5px}.model{padding:11px 7px;display:block}.model .number{display:none}.model .label{font-size:11px}.metrics{position:absolute;top:425px;left:0;right:0;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:0}.metric strong{font-size:16px}.details{position:absolute;top:505px}.controls{top:auto;right:22px;bottom:115px;max-width:none}.hint{bottom:165px;left:22px;transform:none;font-size:11px}footer{padding:17px 22px;font-size:10px;display:block;line-height:1.8}#error{left:22px;right:22px}.eyebrow{font-size:9px}}
</style></head><body><main>
<canvas id="view" tabindex="0" role="img" aria-label="Interactive NURBS model. Drag or use arrow keys to rotate; scroll or use plus and minus to zoom."></canvas>
<header><div class="eyebrow">NATIVE NURBS / INDUSTRIAL GEOMETRY</div><h1 id="title"></h1><p class="subtitle" id="subtitle"></p></header>
<aside><nav class="models" id="models" aria-label="Select model"></nav><div class="metrics" id="metrics"></div><p class="details" id="details"></p></aside>
<div class="controls"><label title="Boundaries between native patches"><input type="checkbox" id="seams" checked>Patch seams</label><label title="Parameter grid of native evaluator samples"><input type="checkbox" id="mesh">Sample mesh</label><button id="reset">Reset view</button><a class="button" id="png" download>PNG ↗</a></div>
<div class="hint">Drag to rotate · scroll to zoom · double-click to reset</div>
<footer><span>Sampled directly from native rational tensor-product patches. No CAD import.</span><span class="status">Self-contained viewer · model units</span></footer>
<div id="error" role="alert"></div>
</main><script>
"use strict";
const models=__MODEL_DATA__;
const canvas=document.getElementById('view');
const gl=canvas.getContext('webgl2',{antialias:true,alpha:false});
const error=document.getElementById('error');
if(!gl){error.style.display='block';error.textContent='This browser does not support WebGL 2. Open one of the PNG previews, or enable hardware acceleration.';}
else { try { start(); } catch(e) {error.style.display='block';error.textContent='Unable to render the model: '+e.message;console.error(e);} }
function start(){
 const vertex=`#version 300 es
 in vec3 a_position;in vec3 a_normal;uniform mat4 u_matrix;uniform mat3 u_rotation;uniform vec3 u_center;out vec3 v_normal;
 void main(){gl_Position=u_matrix*vec4(a_position-u_center,1.);v_normal=u_rotation*a_normal;}`;
 const fragment=`#version 300 es
 precision highp float;in vec3 v_normal;uniform vec3 u_color;uniform bool u_line;out vec4 outColor;
 void main(){if(u_line){outColor=vec4(u_color,1.);return;}vec3 n=normalize(v_normal);if(!gl_FrontFacing)n=-n;
 vec3 key=normalize(vec3(-.45,.8,.8));vec3 fill=normalize(vec3(.7,.2,.6));
 float d=.48+.46*max(dot(n,key),0.)+.11*max(dot(n,fill),0.);vec3 h=normalize(key+vec3(0.,0.,1.));
 float s=.13*pow(max(dot(n,h),0.),34.);outColor=vec4(clamp(u_color*d+vec3(s),0.,1.),1.);}`;
 function shader(type,source){const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
 if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const uniform=Object.fromEntries(['matrix','rotation','center','color','line'].map(n=>[n,gl.getUniformLocation(program,'u_'+n)]));
 const position=gl.getAttribLocation(program,'a_position'), normal=gl.getAttribLocation(program,'a_normal');
 function decode(encoded,Type){const binary=atob(encoded),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);return new Type(bytes.buffer);}
 function buffer(data,target){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b;}
 function upload(m){if(m.gpu)return;const tris=decode(m.triangles,Uint32Array),grid=decode(m.grid,Uint32Array),seams=decode(m.seams,Uint32Array);
 m.gpu={positions:buffer(decode(m.positions,Float32Array),gl.ARRAY_BUFFER),normals:buffer(decode(m.normals,Float32Array),gl.ARRAY_BUFFER),
 triangles:buffer(tris,gl.ELEMENT_ARRAY_BUFFER),grid:buffer(grid,gl.ELEMENT_ARRAY_BUFFER),seams:buffer(seams,gl.ELEMENT_ARRAY_BUFFER),counts:{triangles:tris.length,grid:grid.length,seams:seams.length}};
 for(const key of ['positions','normals','triangles','grid','seams'])delete m[key];}
 const nav=document.getElementById('models');let current=0,yaw=0,elevation=0,zoom=1,pointer=null,scheduled=false;
 const mesh=document.getElementById('mesh'),seams=document.getElementById('seams');
 models.forEach((m,i)=>{const button=document.createElement('button');button.className='model';button.innerHTML='<span class="number">0'+(i+1)+'</span><span class="label"></span>';button.querySelector('.label').textContent=m.title;button.addEventListener('click',()=>select(i));nav.appendChild(button);});
 function reset(){yaw=models[current].azim*Math.PI/180;elevation=models[current].elev*Math.PI/180;zoom=1;schedule();}
 function select(index){current=index;const m=models[index];upload(m);document.getElementById('title').textContent=m.title;document.getElementById('subtitle').textContent=m.subtitle;
 document.getElementById('png').href=m.name+'.png';[...nav.children].forEach((b,i)=>{b.classList.toggle('active',i===index);b.setAttribute('aria-pressed',i===index?'true':'false');});
 const d=m.diagnostics,metrics=[['NURBS PATCHES',m.patches],['GENUS',m.genus??'—'],['VOLUME',Number.isFinite(d.volume)?d.volume.toFixed(4):'—'],['SEAM GAP',Number.isFinite(d.max_seam_gap)?d.max_seam_gap.toExponential(1):'—']];
 const box=document.getElementById('metrics');box.replaceChildren();for(const [label,value] of metrics){const item=document.createElement('div');item.className='metric';const small=document.createElement('small');small.textContent=label;const strong=document.createElement('strong');strong.textContent=value;item.append(small,strong);box.appendChild(item);}
 document.getElementById('details').textContent=m.description;canvas.setAttribute('aria-label',m.title+'. Drag or use arrow keys to rotate; scroll or use plus and minus to zoom.');reset();}
 function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(draw);}}
 function draw(){scheduled=false;const scale=Math.min(window.devicePixelRatio||1,2),w=Math.round(canvas.clientWidth*scale),h=Math.round(canvas.clientHeight*scale);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 gl.viewport(0,0,w,h);gl.clearColor(245/255,247/255,250/255,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
 const m=models[current],gpu=m.gpu,c=Math.cos(yaw),s=Math.sin(yaw),ce=Math.cos(elevation),se=Math.sin(elevation);
 const right=[-s,c,0],up=[-se*c,-se*s,ce],eye=[ce*c,ce*s,se];const aspect=w/h;
 const mobile=canvas.clientWidth<=800;const size=(mobile?1.62:1.38)*zoom/m.extent;
 const tx=mobile?0:Math.min(.23,250/canvas.clientWidth),ty=mobile?.04:-.025;
 const matrix=new Float32Array([right[0]*size/aspect,up[0]*size,-eye[0]/(m.extent*3),0,right[1]*size/aspect,up[1]*size,-eye[1]/(m.extent*3),0,right[2]*size/aspect,up[2]*size,-eye[2]/(m.extent*3),0,tx,ty,0,1]);
 const rotation=new Float32Array([right[0],up[0],eye[0],right[1],up[1],eye[1],right[2],up[2],eye[2]]);
 gl.uniformMatrix4fv(uniform.matrix,false,matrix);gl.uniformMatrix3fv(uniform.rotation,false,rotation);gl.uniform3fv(uniform.center,m.center);
 gl.bindBuffer(gl.ARRAY_BUFFER,gpu.positions);gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,gpu.normals);gl.enableVertexAttribArray(normal);gl.vertexAttribPointer(normal,3,gl.FLOAT,false,0,0);
 gl.uniform1i(uniform.line,0);gl.uniform3fv(uniform.color,m.color);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,gpu.triangles);gl.drawElements(gl.TRIANGLES,gpu.counts.triangles,gl.UNSIGNED_INT,0);gl.disable(gl.POLYGON_OFFSET_FILL);
 gl.uniform1i(uniform.line,1);if(mesh.checked){gl.uniform3fv(uniform.color,m.color.map(v=>v*.69));gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,gpu.grid);gl.drawElements(gl.LINES,gpu.counts.grid,gl.UNSIGNED_INT,0);}if(seams.checked){gl.uniform3fv(uniform.color,m.color.map(v=>v*.62));gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,gpu.seams);gl.drawElements(gl.LINES,gpu.counts.seams,gl.UNSIGNED_INT,0);}
 }
 canvas.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY,id:e.pointerId};canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!pointer||pointer.id!==e.pointerId)return;yaw-=(e.clientX-pointer.x)*.009;elevation=Math.max(-1.5,Math.min(1.5,elevation+(e.clientY-pointer.y)*.009));pointer.x=e.clientX;pointer.y=e.clientY;schedule();});
 canvas.addEventListener('pointerup',()=>pointer=null);canvas.addEventListener('pointercancel',()=>pointer=null);
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(3.5,zoom*Math.exp(-e.deltaY*.001)));schedule();},{passive:false});
 canvas.addEventListener('dblclick',reset);canvas.addEventListener('keydown',e=>{let used=true;switch(e.key){case'ArrowLeft':yaw-=.12;break;case'ArrowRight':yaw+=.12;break;case'ArrowUp':elevation=Math.min(1.5,elevation+.1);break;case'ArrowDown':elevation=Math.max(-1.5,elevation-.1);break;case'+':case'=':zoom=Math.min(3.5,zoom*1.1);break;case'-':zoom=Math.max(.35,zoom/1.1);break;case'0':reset();break;default:used=false;}if(used){e.preventDefault();schedule();}});
 document.getElementById('reset').addEventListener('click',reset);mesh.addEventListener('change',schedule);seams.addEventListener('change',schedule);new ResizeObserver(schedule).observe(canvas);select(0);
}
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("output/industrial_nurbs/models.json"))
    parser.add_argument("--output", type=Path, default=Path("output/industrial_nurbs"))
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180)")
    parser.add_argument("--no-html", action="store_true", help="Skip the self-contained WebGL viewer")
    parser.add_argument("--only-html", action="store_true", help="Update only the self-contained WebGL viewer")
    args = parser.parse_args()
    if args.dpi < 72 or args.dpi > 600:
        parser.error("--dpi must be between 72 and 600")
    if args.no_html and args.only_html:
        parser.error("--no-html and --only-html cannot be combined")
    models = load_models(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    for model in models:
        print(f"{model.name}: {len(model.source['patches'])} native patches, "
              f"{len(model.points):,} sample vertices, {len(model.triangles):,} display triangles", flush=True)
        if not args.only_html:
            print(render_individual(model, args.output, args.dpi), flush=True)
    if not args.only_html:
        print(render_overview(models, args.output, args.dpi), flush=True)
    if not args.no_html:
        print(render_interactive(models, args.output), flush=True)


if __name__ == "__main__":
    main()
