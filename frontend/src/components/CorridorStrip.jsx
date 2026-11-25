import React, { useMemo } from "react";
import { greenLeft, headState, queueOf, SIDE, signalName } from "./signal.js";

// The corridor as a control-room wall shows it. Junctions sit at their real distances along Vikas Marg, queues are
// drawn to the same metre scale (queued PCU x 7 m / lanes), and every approach has its own signal head at its stop
// line. Left-hand traffic: eastbound uses the upper carriageway, westbound the lower one.

const W = 1200;
const H = 362;
const Y = 210;            // main road centre line
const HALF = 30;
const X0 = 160;
const X1 = 1040;
const END = 20;           // drawn ends of the corridor
const N_TOP = 76;         // top of the north stubs
const S_BOT = 330;        // bottom of the south stubs

function Head({ x, y, s, title }) {
  const lamp = (c, i) => (
    <circle key={c} cx={x + 7} cy={y + 7 + i * 12} r={4.6} className={`lamp lamp-${c} ${s.ball === c ? "is-on" : ""}`} />
  );
  return (
    <g>
      <title>{title}</title>
      <rect x={x} y={y} width={14} height={38} rx={3} className="head" />
      {["red", "amber", "green"].map(lamp)}
      <rect x={x + 15} y={y + 25} width={12} height={12} rx={2} className="head" />
      <path d={`M${x + 17} ${y + 31} h4 v-3 l4 3 -4 3 v-3`} className={`arrow ${s.arrow ? "is-on" : s.arrowAmber ? "is-amber" : ""}`} />
    </g>
  );
}

export default function CorridorStrip({ network, frame, selected, onSelect }) {
  const pos = useMemo(() => {
    if (!network) return null;
    const xs = network.junctions.map((j) => j.x);
    const min = Math.min(...xs);
    const span = Math.max(1, Math.max(...xs) - min);
    return Object.fromEntries(network.junctions.map((j) => [j.id, X0 + ((j.x - min) / span) * (X1 - X0)]));
  }, [network]);

  if (!pos) return <div className="strip-empty">Loading the corridor…</div>;
  const links = network.links.filter((l) => l.to !== "sink");

  const queueBar = (l) => {
    const q = queueOf(l, frame);
    if (q.veh <= 0) return null;
    const x = pos[l.to];
    let room;
    if (l.approach === "W") room = (l.from === "src" ? x - END : x - pos[l.from]) - 48;
    else if (l.approach === "E") room = (l.from === "src" ? W - END - x : pos[l.from] - x) - 48;
    else if (l.approach === "N") room = Y - HALF - N_TOP - 4;
    else room = S_BOT - (Y + HALF) - 4;
    const len = Math.max(4, Math.min(1, q.share) * room);
    let x1, y1, x2, y2, tx, ty, anchor;
    if (l.approach === "W") { x1 = x - 24; y1 = Y - 15; x2 = x1 - len; y2 = y1; tx = x2 - 6; ty = y1; anchor = "end"; }
    else if (l.approach === "E") { x1 = x + 24; y1 = Y + 15; x2 = x1 + len; y2 = y1; tx = x2 + 6; ty = y1; anchor = "start"; }
    else if (l.approach === "N") { x1 = x + 9; y1 = Y - HALF - 3; x2 = x1; y2 = y1 - len; tx = x1 - 12; ty = y2 + 6; anchor = "end"; }
    else { x1 = x - 9; y1 = Y + HALF + 3; x2 = x1; y2 = y1 + len; tx = x1 + 12; ty = y2 - 6; anchor = "start"; }
    const label = q.spill
      ? (l.from === "src" ? `${q.veh} (backed up beyond view)` : `${q.veh}, backs up to ${signalName(l.from)}`)
      : `${q.veh}`;
    return (
      <g key={`q-${l.id}`}>
        <title>{`${q.veh} vehicles queued at ${signalName(l.to)} from the ${SIDE[l.approach]}: about ${Math.round(q.metres)} m of road, ${Math.round(q.share * 100)}% of this road's space`}</title>
        <line x1={x1} y1={y1} x2={x2} y2={y2} className={`queue ${q.spill ? "is-spill" : ""}`} />
        <text x={tx} y={ty} className={`queue-n ${q.spill ? "is-spill" : ""}`} textAnchor={anchor} dominantBaseline="middle">{label}</text>
      </g>
    );
  };

  let amb = null;
  if (frame?.ambulance) {
    const l = network.links.find((k) => k.id === frame.ambulance.link);
    if (l) {
      const a = l.from === "src" ? (l.approach === "W" ? END : W - END) : pos[l.from];
      const east = l.to === "sink" ? a > W / 2 : pos[l.to] >= a;
      const b = l.to === "sink" ? (east ? W - END : END) : pos[l.to];
      amb = { x: a + (b - a) * frame.ambulance.progress, y: east ? Y - 15 : Y + 15 };
    }
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="strip" role="group" aria-label="Vikas Marg corridor: signals and queues">
      <defs>
        <pattern id="zebra" width="8" height="8" patternUnits="userSpaceOnUse">
          <rect width="8" height="8" fill="#343d46" />
          <rect width="4" height="8" fill="#ecebe6" />
        </pattern>
      </defs>
      <rect x={END - 10} y={Y - HALF} width={W - 2 * END + 20} height={HALF * 2} className="road" />
      <line x1={END - 10} x2={W - END + 10} y1={Y} y2={Y} className="median" />
      <text x={END - 6} y={Y + HALF + 66} className="end-label">ITO</text>
      <text x={W - END + 6} y={Y + HALF + 66} className="end-label" textAnchor="end">Laxmi Nagar</text>
      <text x={END - 6} y={Y - HALF - 10} className="dir-label">eastbound</text>
      <text x={W - END + 6} y={Y + HALF + 40} className="dir-label" textAnchor="end">westbound</text>

      {network.junctions.map((j) => (
        <rect key={`x-${j.id}`} x={pos[j.id] - 20} y={N_TOP} width={40} height={S_BOT - N_TOP} className="road" />
      ))}

      {links.map(queueBar)}

      {network.junctions.map((j) => {
        const x = pos[j.id];
        const sig = frame?.signals?.[j.id];
        const phase = sig ? j.phases[sig.phase] : null;
        const left = greenLeft(sig);
        const ped = phase?.pedestrian && sig?.state === "green";
        const isSel = selected === j.id;
        let timer = null;
        let tcls = "";
        if (sig?.state === "green") {
          if (left) { timer = left.firm ? `${left.left}` : `${left.left}+`; tcls = "t-green"; }
          else { timer = `${Math.round(sig.green_elapsed)} s on`; tcls = "t-dim"; }
        } else if (sig?.state === "yellow") { timer = "changing"; tcls = "t-amber"; }
        else if (sig?.state === "allred") { timer = "all red"; tcls = "t-dim"; }
        const heads = [
          ["W", x - 46, Y - HALF - 44, "eastbound, from the ITO side"],
          ["N", x + 24, Y - HALF - 44, "southbound, from the north"],
          ["E", x + 24, Y + HALF + 6, "westbound, from the Laxmi Nagar side"],
          ["S", x - 46, Y + HALF + 6, "northbound, from the south"],
        ].filter(([a]) => network.links.some((l) => l.to === j.id && l.approach === a));
        return (
          <g
            key={j.id} className={`jn ${isSel ? "is-sel" : ""}`} role="button" tabIndex={0} aria-pressed={isSel}
            aria-label={`${j.name}: ${phase ? `${phase.name}, ${sig.state}` : "no data"}`}
            onClick={() => onSelect(j.id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(j.id); } }}
          >
            <rect x={x - 78} y={2} width={156} height={H - 4} rx={10} className="jn-hit" />
            <text x={x} y={20} textAnchor="middle" className="jn-id">{signalName(j.id)}</text>
            <text x={x} y={37} textAnchor="middle" className="jn-phase">{phase ? phase.name : "–"}</text>
            {timer && <text x={x} y={62} textAnchor="middle" className={`jn-timer ${tcls}`}>{timer}</text>}
            <rect x={x - 20} y={Y - HALF} width={40} height={HALF * 2} className="box" fill={ped ? "url(#zebra)" : undefined} />
            {heads.map(([a, hx, hy, t]) => <Head key={a} x={hx} y={hy} s={headState(j, sig, a)} title={`Signal for ${t}`} />)}
            {sig?.mode && sig.mode !== "normal" && (
              <text x={x} y={H - 10} textAnchor="middle" className={`jn-mode m-${sig.mode}`}>
                {{ fallback: "camera fault: fixed plan", emergency: "green corridor", fairness: "fairness rule", pedestrian: "pedestrian phase" }[sig.mode] || sig.mode}
              </text>
            )}
          </g>
        );
      })}

      {amb && (
        <g transform={`translate(${amb.x},${amb.y})`} className="amb">
          <title>Ambulance</title>
          <rect x={-14} y={-8} width={28} height={16} rx={3} />
          <path d="M-4 0h8M0 -4v8" />
        </g>
      )}
    </svg>
  );
}
