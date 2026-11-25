import React, { useMemo } from "react";
import { headState, signalName } from "./signal.js";

// Time-space diagram: time runs left to right (last WINDOW seconds), signals are stacked by their real distance from
// ITO. Each row shows what the eastbound head showed at that moment, so a coordinated "green wave" is visible as
// green blocks stepping to the right as you go up the corridor. An ambulance trip is drawn as its trajectory.

const WINDOW = 360;       // seconds shown
const W = 1200;
const H = 190;
const L = 92;             // left gutter for signal names
const R = 14;
const TOP = 14;
const BOT = 26;

export default function TimeSpace({ network, frames, index }) {
  const geo = useMemo(() => {
    if (!network) return null;
    const js = [...network.junctions].sort((a, b) => a.x - b.x);
    const min = js[0].x;
    const span = Math.max(1, js[js.length - 1].x - min);
    const y = Object.fromEntries(js.map((j) => [j.id, H - BOT - ((j.x - min) / span) * (H - TOP - BOT)]));
    const byId = Object.fromEntries(js.map((j) => [j.id, j]));
    return { js, y, byId, min, span };
  }, [network]);

  if (!geo || !frames.length) return null;
  const now = frames[index]?.t ?? 0;
  const t0 = Math.max(0, now - WINDOW);
  const sx = (t) => L + ((t - t0) / WINDOW) * (W - L - R);
  const visible = frames.slice(0, index + 1).filter((f) => f.t >= t0);
  const dt = frames.length > 1 ? frames[1].t - frames[0].t : 2;

  // merge consecutive frames with the same colour into one rect per row
  const rows = geo.js.map((j) => {
    const segs = [];
    visible.forEach((f) => {
      const c = headState(j, f.signals[j.id], "W").ball;
      const last = segs[segs.length - 1];
      if (last && last.c === c && Math.abs(last.t1 - f.t) < dt + 0.01) last.t1 = f.t + dt;
      else segs.push({ c, t0: f.t, t1: f.t + dt });
    });
    return { j, segs };
  });

  // ambulance trajectory: distance along the corridor over time
  const pts = [];
  visible.forEach((f) => {
    const a = f.ambulance;
    if (!a) return;
    const l = network.links.find((k) => k.id === a.link);
    if (!l) return;
    const fromX = l.from === "src" ? geo.min - 150 : geo.byId[l.from]?.x;
    const toX = l.to === "sink" ? geo.js[geo.js.length - 1].x + 150 : geo.byId[l.to]?.x;
    if (fromX == null || toX == null) return;
    const m = fromX + (toX - fromX) * a.progress;
    const yy = H - BOT - ((m - geo.min) / geo.span) * (H - TOP - BOT);
    pts.push(`${sx(f.t).toFixed(1)},${Math.max(TOP - 6, Math.min(H - BOT + 6, yy)).toFixed(1)}`);
  });

  const ticks = [];
  for (let t = Math.ceil(t0 / 60) * 60; t <= now; t += 60) ticks.push(t);

  return (
    <figure className="ts">
      <figcaption>
        <b>Eastbound signal timing, last 6 minutes</b>
        <span>Signals are spaced by their real distance. Green blocks stepping up and to the right mean traffic leaving one signal on green arrives at the next on green.</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Time-space diagram of eastbound signal states">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={TOP - 6} y2={H - BOT + 6} className="ts-grid" />
            <text x={sx(t)} y={H - 6} textAnchor="middle" className="ts-tick">{`${Math.floor(t / 60)}:00`}</text>
          </g>
        ))}
        {rows.map(({ j, segs }) => (
          <g key={j.id}>
            <text x={L - 12} y={geo.y[j.id]} textAnchor="end" dominantBaseline="middle" className="ts-label">{signalName(j.id)}</text>
            {segs.map((s, k) => (
              <rect key={k} x={sx(s.t0)} y={geo.y[j.id] - 5} width={Math.max(0.5, sx(s.t1) - sx(s.t0))} height={10} className={`ts-${s.c}`} />
            ))}
          </g>
        ))}
        {pts.length > 1 && <polyline points={pts.join(" ")} className="ts-amb" />}
        <line x1={sx(now)} x2={sx(now)} y1={TOP - 8} y2={H - BOT + 8} className="ts-now" />
      </svg>
    </figure>
  );
}
