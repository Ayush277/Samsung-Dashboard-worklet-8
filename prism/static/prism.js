/* PRISM Worklet 8 — ambient layer + live inference client.
 *
 * Two jobs:
 *   1. Ambient WebGL field behind the hero. Raw WebGL rather than a Three.js
 *      CDN: one shader is a few KB against ~600 KB, and the effect stays
 *      secondary to the interface, as the design reference requires.
 *   2. Live inference. Inputs are debounced and POSTed to the real model
 *      endpoints; the readout renders whatever the server returns. There is no
 *      client-side estimate or placeholder — if the request fails the UI says
 *      so rather than showing a number nobody computed.
 */

(() => {
  "use strict";

  /* ------------------------------------------------------------ ambient */
  const VERT = `attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}`;

  // Domain-warped fbm: slow, low-contrast colour drift in the brand indigo and
  // cyan over white. Deliberately quiet — it must never compete with the text.
  const FRAG = `
precision mediump float;
uniform vec2 r;uniform float t;
float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float n(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);
  return mix(mix(h(i),h(i+vec2(1,0)),u.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*n(p);p*=2.02;a*=.5;}return v;}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*r)/r.y;
  float tt=t*.045;
  vec2 q=vec2(fbm(uv*1.3+tt),fbm(uv*1.3+vec2(3.2,1.7)-tt));
  float f=fbm(uv*1.6+q*1.5+vec2(0.,tt*.7));
  vec3 indigo=vec3(.310,.275,.898);
  vec3 cyan=vec3(.024,.714,.831);
  vec3 col=mix(indigo,cyan,clamp(q.x*1.25,0.,1.));
  // Radial falloff keeps the centre clear for the headline.
  float m=smoothstep(.85,.05,length(uv*vec2(.72,1.15)));
  float a=pow(f,2.4)*m*.34;
  gl_FragColor=vec4(col,a);
}`;

  function ambient(canvas) {
    const gl = canvas.getContext("webgl", { alpha: true, antialias: false,
      premultipliedAlpha: false });
    if (!gl) return; // no WebGL: the page is complete without it

    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
    };
    const vs = sh(gl.VERTEX_SHADER, VERT), fs = sh(gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const uR = gl.getUniformLocation(prog, "r"), uT = gl.getUniformLocation(prog, "t");

    const resize = () => {
      // Capped DPR: past 2x this costs fill rate and buys nothing visible.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        gl.viewport(0, 0, w, h);
      }
      gl.uniform2f(uR, canvas.width, canvas.height);
    };
    window.addEventListener("resize", resize, { passive: true });
    resize();

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, running = true;
    const draw = (ms) => {
      gl.uniform1f(uT, reduce ? 0 : ms / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (reduce) return; // paint once, then stop
      if (running) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    // Stop animating when scrolled away or the tab is hidden.
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([e]) => {
        running = e.isIntersecting && !document.hidden;
        if (running && !reduce) raf = requestAnimationFrame(draw);
        else cancelAnimationFrame(raf);
      }, { threshold: 0 }).observe(canvas);
    }
    document.addEventListener("visibilitychange", () => {
      running = !document.hidden;
      if (running && !reduce) raf = requestAnimationFrame(draw);
      else cancelAnimationFrame(raf);
    });
  }

  /* ------------------------------------------------------ live inference */
  const debounce = (fn, ms) => {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  const fmt = (n, d = 2) =>
    (n === null || n === undefined || Number.isNaN(n))
      ? "—"
      : Number(n).toLocaleString(undefined,
          { minimumFractionDigits: d, maximumFractionDigits: d });

  /** Wire a form to an endpoint so every change re-scores against the model. */
  function live(form, endpoint, render) {
    const status = document.querySelector("[data-status]");
    const latency = document.querySelector("[data-latency]");
    let seq = 0;

    const send = async () => {
      const mine = ++seq;
      status && (status.textContent = "scoring");
      const body = new FormData(form);
      const t0 = performance.now();
      try {
        const res = await fetch(endpoint, { method: "POST", body });
        const data = await res.json();
        // A slower earlier request must not overwrite a newer answer.
        if (mine !== seq) return;
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        render(data);
        const rt = Math.round(performance.now() - t0);
        if (latency) {
          latency.textContent = `${data.inference_ms ?? "?"} ms model · ${rt} ms round trip`;
        }
        status && (status.textContent = "live");
      } catch (err) {
        if (mine !== seq) return;
        status && (status.textContent = "error");
        const out = document.querySelector("[data-error]");
        if (out) out.innerHTML = `<div class="err">${err.message}</div>`;
      }
    };

    const kick = debounce(send, 180);
    form.addEventListener("input", (e) => {
      const t = e.target;
      // Mirror range inputs into their numeric twin and vice versa.
      if (t.dataset.mirror) {
        const other = form.querySelector(`[name="${t.dataset.mirror}"]`);
        if (other && other.value !== t.value) other.value = t.value;
      }
      const out = form.querySelector(`[data-val="${t.name}"]`);
      if (out) out.textContent = t.value;
      kick();
    });
    form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
    send();
    return send;
  }

  window.PRISM = { ambient, live, fmt, debounce };

  document.addEventListener("DOMContentLoaded", () => {
    const c = document.getElementById("ambient");
    if (c) ambient(c);
  });
})();
