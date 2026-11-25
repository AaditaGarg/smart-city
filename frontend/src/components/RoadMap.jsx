import React, { useMemo } from "react";

// Screen direction a road leaves a junction towards (y grows downwards).
const DIR = { N: [0, -1], E: [1, 0], S: [0, 1], W: [-1, 0] };
const STUB = 120;
const PAD = 150;

function queueColor(n) {
  if (n >= 60) return "var(--stop)";
  if (n >= 20) return "var(--amber)";
  return "var(--go)";
}

// Geometry for every link: start point, stop-line point and the side it enters from.
export function useGeometry(network, width = 1000) {
  return useMemo(() => {
    if (!network) return null;
    const js = network.junctions;
    const xs = js.map((j) => j.x);
    const ys = js.map((j) => j.y);
    const spanX = Math.max(1, Math.max(...xs) - Math.min(...xs));
    const spanY = Math.max(...ys) - Math.min(...ys);
    const scale = (width - 2 * PAD) / Math.max(spanX, spanY || 1);
    const height = spanY * scale + 2 * PAD;
    const pos = {};
    js.forEach((j) => {
      pos[j.id] = [PAD + (j.x - Math.min(...xs)) * scale, PAD + (j.y - Math.min(...ys)) * scale];
    });
    const links = {};
    network.links.forEach((l) => {
      if (l.to === "sink") return;
      const end = pos[l.to];
      const d = DIR[l.approach];
      const start = l.from === "src" ? [end[0] + d[0] * STUB, end[1] + d[1] * STUB] : pos[l.from];
      links[l.id] = { ...l, start, end, d };
    });
    return { pos, links, width, height, scale };
  }, [network, width]);
}

function SignalLamp({ x, y, color, label }) {
  return (
    <g>
      <circle cx={x} cy={y} r={6.5} fill={color} stroke="var(--asphalt)" strokeWidth={2} />
      {label && (
        <text x={x} y={y - 10} className="map-tiny" textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

// Which colour each approach of a junction shows right now.
function approachColor(junction, sig, approach) {
  if (!sig) return "var(--stop)";
  const phase = junction.phases[sig.phase];
  const inPhase = phase && phase.movements.some((m) => m[0] === approach);
  if (!inPhase) return "var(--stop)";
  if (sig.state === "green") return "var(--go)";
  if (sig.state === "yellow") return "var(--amber)";
  return "var(--stop)";
}

export default function RoadMap({ network, frame, highlight, width = 1000, compact = false }) {
  const g = useGeometry(network, width);
  if (!g) return <div className="map-empty">Loading road network…</div>;
  const roads = new Set();
  const roadSegs = [];
  Object.values(g.links).forEach((l) => {
    const key = [l.start.join(","), l.end.join(",")].sort().join("|");
    if (roads.has(key)) return;
    roads.add(key);
    roadSegs.push(l);
  });

  return (
    <svg viewBox={`0 0 ${g.width} ${g.height}`} className="road-map" role="img" aria-label="Live road network map">
      <defs>
        <pattern id="zebra" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="#2d353e" />
          <rect width="3" height="6" fill="#ecebe6" />
        </pattern>
      </defs>
      {roadSegs.map((l) => (
        <g key={`r-${l.id}`}>
          <line x1={l.start[0]} y1={l.start[1]} x2={l.end[0]} y2={l.end[1]} className="road" strokeWidth={l.lanes >= 3 ? 30 : 22} />
          <line x1={l.start[0]} y1={l.start[1]} x2={l.end[0]} y2={l.end[1]} className="lane-paint" />
        </g>
      ))}

      {Object.values(g.links).map((l) => {
        const n = frame?.queues?.[l.id] ?? 0;
        const len = Math.hypot(l.end[0] - l.start[0], l.end[1] - l.start[1]) - 18;
        const px = Math.min(len, n * (compact ? 1.4 : 1.1));
        const t = [-l.d[0], -l.d[1]]; // travel direction (towards the junction)
        const left = [t[1], -t[0]]; // left-hand traffic keeps left
        const off = l.lanes >= 3 ? 8 : 6;
        const sx = l.end[0] + l.d[0] * 16 + left[0] * off;
        const sy = l.end[1] + l.d[1] * 16 + left[1] * off;
        const ex = sx + l.d[0] * px;
        const ey = sy + l.d[1] * px;
        const overflow = n * (compact ? 1.4 : 1.1) > len;
        const lx = sx + l.d[0] * (px + 14) + left[0] * 12;
        const ly = sy + l.d[1] * (px + 14) + left[1] * 12;
        const hot = highlight && highlight === l.id;
        return (
          <g key={`q-${l.id}`}>
            {n > 0 && (
              <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={queueColor(n)} strokeWidth={hot ? 9 : 7} strokeLinecap="round" opacity={0.95} />
            )}
            {n > 0 && !compact && (
              <text x={lx} y={ly} className="map-count" textAnchor="middle" dominantBaseline="middle">
                {overflow ? `${n}+` : n}
              </text>
            )}
          </g>
        );
      })}

      {network.junctions.map((j) => {
        const [x, y] = g.pos[j.id];
        const sig = frame?.signals?.[j.id];
        const ped = sig && j.phases[sig.phase]?.pedestrian && sig.state === "green";
        const fault = sig?.mode === "fallback";
        const emerg = sig?.mode === "emergency";
        return (
          <g key={j.id}>
            <rect x={x - 17} y={y - 17} width={34} height={34} rx={4} className={`junction-box ${ped ? "is-ped" : ""}`} />
            {["N", "E", "S", "W"].map((a) => {
              const l = Object.values(g.links).find((k) => k.to === j.id && k.approach === a);
              if (!l) return null;
              const t = [-l.d[0], -l.d[1]];
              const left = [t[1], -t[0]];
              return (
                <SignalLamp
                  key={a}
                  x={x + l.d[0] * 25 + left[0] * 9}
                  y={y + l.d[1] * 25 + left[1] * 9}
                  color={approachColor(j, sig, a)}
                />
              );
            })}
            <text x={x} y={y + (compact ? 40 : 52)} className="map-label" textAnchor="middle">
              {j.id}
            </text>
            {!compact && (
              <text x={x} y={y + 68} className="map-tiny" textAnchor="middle">
                {ped ? "pedestrians crossing" : fault ? "camera fault: fixed plan" : emerg ? "green corridor" : sig ? j.phases[sig.phase]?.name : ""}
              </text>
            )}
            {fault && <circle cx={x + 17} cy={y - 17} r={6} fill="var(--amber)" />}
          </g>
        );
      })}

      {frame?.ambulance && g.links[frame.ambulance.link] && (() => {
        const l = g.links[frame.ambulance.link];
        const p = frame.ambulance.progress;
        const x = l.start[0] + (l.end[0] - l.start[0]) * p;
        const y = l.start[1] + (l.end[1] - l.start[1]) * p;
        return (
          <g className="ambulance" transform={`translate(${x},${y})`}>
            <rect x={-13} y={-8} width={26} height={16} rx={3} fill="#fff" stroke="var(--stop)" strokeWidth={2} />
            <path d="M-4 0h8M0 -4v8" stroke="var(--stop)" strokeWidth={2.5} />
          </g>
        );
      })()}
    </svg>
  );
}
