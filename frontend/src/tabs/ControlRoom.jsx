import React, { useEffect, useMemo, useState } from "react";
import { Pause, Play, RotateCcw, Siren } from "lucide-react";
import CorridorStrip from "../components/CorridorStrip.jsx";
import TimeSpace from "../components/TimeSpace.jsx";
import { greenLeft, MAX_GREEN_S, queueOf, SIDE, signalName } from "../components/signal.js";
import { usePlayer } from "../components/Player.jsx";
import { postJSON, simulate } from "../api.js";

// The operator's main screen. The backend simulator stands in for the live camera feed (clearly labelled),
// so the corridor is always running: signal states, queues, decisions and events update on their own.

const MODE = {
  normal: { label: "Adaptive", cls: "ok" },
  fairness: { label: "Fairness rule", cls: "info" },
  pedestrian: { label: "Pedestrian phase", cls: "info" },
  emergency: { label: "Green corridor", cls: "crit" },
  fallback: { label: "Camera fault: fixed plan", cls: "warn" },
};
const STRATEGIES = [
  { id: "max_pressure", label: "Adaptive (network)" },
  { id: "fixed", label: "Fixed timer (today)" },
  { id: "webster", label: "Adaptive (single junction)" },
];
const SPEEDS = [1, 5, 15, 30];          // frames per second; one frame = 2 simulated seconds
const clock = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
const mins = (s) => (s == null ? "–" : `${(s / 60).toFixed(1)} min`);

// The decision engine's reasons, in an operator's words ("phase 3" becomes the phase's name).
export function humanize(reason, junction) {
  if (!reason) return "";
  const name = (i) => junction?.phases?.[Number(i)]?.name ?? `phase ${i}`;
  return reason
    .replace(/^fairness: phase (\d+) red for (\d+)s$/, (_, i, sec) => `${name(i)} had been red for ${sec} s, so it goes next`)
    .replace(/^minimum green \((\d+)\/(\d+)s\)$/, (_, a, b) => `Minimum green time (${a} of ${b} s)`)
    .replace(/^clearing queue \((\d+)\/(\d+)s\)$/, (_, a, b) => `Clearing the queue (${a} of ${b} s planned)`)
    .replace(/^./, (c) => c.toUpperCase());
}

function buildEvents(run, network) {
  if (!run || !network) return [];
  const names = Object.fromEntries(network.junctions.map((j) => [j.id, j]));
  const ev = (run.alerts || []).map((a) => ({ t: a.t, sev: a.severity, text: a.message, j: null }));
  const fr = run.frames;
  for (let i = 1; i < fr.length; i += 1) {
    const a = fr[i - 1].signals;
    const b = fr[i].signals;
    Object.keys(b).forEach((jid) => {
      if (!a[jid]) return;
      if (b[jid].mode !== a[jid].mode && b[jid].mode !== "normal" && b[jid].mode !== "fallback") {
        ev.push({ t: fr[i].t, sev: b[jid].mode === "emergency" ? "critical" : "local", j: jid,
          text: `${MODE[b[jid].mode]?.label ?? b[jid].mode}. ${humanize(b[jid].reason, names[jid])}` });
      }
      if (b[jid].phase !== a[jid].phase) {
        const p = names[jid].phases[b[jid].phase];
        ev.push({ t: fr[i].t, sev: "phase", j: jid, text: `${p.name} green. ${humanize(b[jid].reason, names[jid])}` });
      }
    });
  }
  const internal = network.links.filter((l) => l.from !== "src" && l.to !== "sink");
  const spilling = {};
  fr.forEach((f) => {
    internal.forEach((l) => {
      const now = queueOf(l, f).spill;
      if (now && !spilling[l.id]) {
        ev.push({ t: f.t, sev: "warning", j: null,
          text: `Queue at ${signalName(l.to)} from the ${SIDE[l.approach]} has reached ${signalName(l.from)}: traffic there cannot get through even on green.` });
      } else if (!now && spilling[l.id]) {
        ev.push({ t: f.t, sev: "info", j: null, text: `Queue between ${signalName(l.from)} and ${signalName(l.to)} has cleared back from ${signalName(l.from)}.` });
      }
      spilling[l.id] = now;
    });
  });
  return ev.sort((x, y) => x.t - y.t);
}

function Trend({ frames, index }) {
  const pts = frames.slice(0, index + 1);
  if (pts.length < 2) return null;
  const vals = pts.map((f) => Object.values(f.queues).reduce((a, b) => a + b, 0));
  const max = Math.max(10, ...frames.map((f) => Object.values(f.queues).reduce((a, b) => a + b, 0)));
  const n = frames.length - 1;
  const d = vals.map((v, i) => `${i ? "L" : "M"}${((i / n) * 200).toFixed(1)} ${(36 - (v / max) * 34).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 200 38" className="trend" aria-hidden="true" preserveAspectRatio="none">
      <path d={d} />
    </svg>
  );
}

export default function ControlRoom({ scenarios, network }) {
  const [scenario, setScenario] = useState("morning_rush");
  const [strategy, setStrategy] = useState("max_pressure");
  const [run, setRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState("J2");
  const [corridor, setCorridor] = useState(null);   // {plan, without}
  const frames = run?.frames ?? [];
  const player = usePlayer(frames.length);
  const frame = frames[player.i];
  const t = frame?.t ?? 0;

  useEffect(() => { player.setSpeed(5); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const load = async (scn, ctl, extra = {}) => {
    setBusy(true);
    setError(null);
    try {
      setRun(await simulate(scn, ctl, extra));
    } catch (e) {
      setError(`Could not start the feed: ${e.message}. Check that the backend is running on port 8000.`);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => { load(scenario, strategy); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const changeScenario = (id) => { setScenario(id); setCorridor(null); load(id, strategy); };
  const changeStrategy = (id) => { setStrategy(id); setCorridor(null); load(scenario, id, scenario === "ambulance" ? { corridor: !!corridor } : {}); };

  const sendCorridor = async () => {
    setBusy(true);
    setError(null);
    try {
      const [plan, withC, without] = await Promise.all([
        postJSON("/network/emergency", { source: "operator" }),
        simulate("ambulance", strategy, { corridor: true }),
        simulate("ambulance", strategy, { corridor: false }),
      ]);
      setScenario("ambulance");
      setCorridor({ plan, without: without.metrics.ambulance_travel_s });
      setRun(withC);
    } catch (e) {
      setError(`Could not start the green corridor: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const events = useMemo(() => buildEvents(run, network), [run, network]);
  const shown = events.filter((e) => e.t <= t && ((e.sev !== "phase" && e.sev !== "local") || e.j === selected)).slice(-40).reverse();

  const list = (scenarios || []).filter((s) => s.network === "vikas_marg");
  const scnTitle = list.find((s) => s.id === scenario)?.title ?? scenario;
  const done = frames.length > 0 && player.i >= frames.length - 1;

  // live numbers for this moment
  const queues = frame?.queues ?? {};
  const waiting = Object.values(queues).reduce((a, b) => a + b, 0);
  const longest = Object.entries(queues).sort((a, b) => b[1] - a[1])[0];
  const adaptive = frame ? Object.values(frame.signals).filter((s) => s.mode !== "fallback").length : 0;
  const nJ = network?.junctions.length ?? 4;
  const longestLabel = (id) => {
    if (!id) return "";
    const l = network?.links.find((k) => k.id === id);
    if (!l) return "";
    const side = { W: "ITO side", E: "Laxmi Nagar side", N: "north", S: "south" }[l.approach];
    return `at ${l.to.replace("J", "signal ")}, ${side}`;
  };

  // ambulance timeline (corridor run)
  const ambFrames = frames.filter((f) => f.ambulance);
  const ambStart = ambFrames[0]?.t;
  const ambEnd = ambFrames[ambFrames.length - 1]?.t;
  const ambNow = frame?.ambulance;
  const ambArrived = corridor && ambEnd != null && t > ambEnd;

  const J = network?.junctions.find((j) => j.id === selected);
  const sig = frame?.signals?.[selected];
  const phase = J && sig ? J.phases[sig.phase] : null;
  const mode = MODE[sig?.mode] ?? MODE.normal;
  const approaches = J ? ["W", "E", "N", "S"].map((a) => {
    const l = network.links.find((k) => k.to === selected && k.approach === a);
    return l ? { a, q: queueOf(l, frame), label: { W: "From ITO side", E: "From Laxmi Nagar side", N: "From north", S: "From south" }[a] } : null;
  }).filter(Boolean) : [];
  const left = greenLeft(sig);
  const timeLine = !sig ? "–"
    : sig.state === "yellow" ? "Changing now (yellow, then all red)"
      : sig.state === "allred" ? "All red: clearing the junction"
        : left?.firm ? `${left.left} s left (planned)`
          : left ? `At least ${left.left} s more`
            : `Running ${Math.round(sig.green_elapsed)} s; continues while this road has the heaviest queue, up to ${MAX_GREEN_S} s`;
  const allLinks = network?.links.filter((l) => l.to !== "sink") ?? [];
  const backedUp = frame ? allLinks.filter((l) => queueOf(l, frame).spill).length : 0;

  return (
    <section className="cr">
      <div className="cr-bar">
        <div className="feed">
          <span className={`feed-dot ${player.playing ? "is-live" : ""}`} aria-hidden="true" />
          <div>
            <b>Simulated feed</b>
            <span>{scnTitle}</span>
          </div>
          <span className="feed-clock" aria-label="Time into the scenario">{clock(t)}</span>
        </div>
        <div className="cr-controls">
          <label className="field">Traffic situation
            <select value={scenario} onChange={(e) => changeScenario(e.target.value)} disabled={busy}>
              {list.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
            </select>
          </label>
          <label className="field">Signal control
            <select value={strategy} onChange={(e) => changeStrategy(e.target.value)} disabled={busy}>
              {STRATEGIES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </label>
          <div className="transport" role="group" aria-label="Playback">
            <button className="icon-btn" onClick={() => player.setPlaying(!player.playing)} disabled={!frames.length}
                    aria-label={player.playing ? "Pause feed" : "Resume feed"}>
              {player.playing ? <Pause size={17} /> : <Play size={17} />}
            </button>
            <button className="icon-btn" onClick={() => { player.setI(0); player.setPlaying(true); }} disabled={!frames.length} aria-label="Restart from the beginning">
              <RotateCcw size={16} />
            </button>
            <div className="seg" aria-label="Speed">
              {SPEEDS.map((s) => (
                <button key={s} className={player.speed === s ? "on" : ""} onClick={() => player.setSpeed(s)}>{s * 2}×</button>
              ))}
            </div>
          </div>
          <button className="btn btn-siren" onClick={sendCorridor} disabled={busy}>
            <Siren size={17} /> Send green corridor
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="strip-wrap">
        {busy && <div className="strip-busy">Starting the feed…</div>}
        <CorridorStrip network={network} frame={frame} selected={selected} onSelect={setSelected} />
        <p className="strip-hint">Select a signal to see what it is doing and why. Yellow bars are queued vehicles.</p>
      </div>

      <TimeSpace network={network} frames={frames} index={player.i} />

      {corridor && (
        <div className={`corridor-note ${ambArrived ? "is-done" : ""}`} role="status">
          <Siren size={18} aria-hidden="true" />
          {ambArrived ? (
            <p>Ambulance reached Laxmi Nagar in <b>{mins(run.metrics.ambulance_travel_s)}</b>. Without the corridor the same trip took <b>{mins(corridor.without)}</b>.</p>
          ) : ambNow ? (
            <p>Ambulance on the corridor ({clock(t - ambStart)} so far). Signals ahead turn green before it arrives: {corridor.plan.junctions.map((p) => `${p.junction.replace("J", "signal ")} at +${Math.round(p.green_from_s)} s`).join(", ")}.</p>
          ) : (
            <p>Green corridor requested. The ambulance enters at the ITO end at {clock(ambStart ?? 0)}; every switch still goes through yellow and all-red.</p>
          )}
        </div>
      )}

      <div className="cr-grid">
        <div className="detail" aria-live="polite">
          {J && (
            <>
              <div className="detail-head">
                <h2>{J.name}</h2>
                <span className={`mode mode-${mode.cls}`}>{mode.label}</span>
              </div>
              <div className="now">
                <div>
                  <span className="now-k">{sig?.state === "green" ? "Green now" : "Ending phase"}</span>
                  <span className="now-v">{phase ? phase.name : "–"}</span>
                </div>
                <div className="now-time">
                  <span className="now-k">Time left</span>
                  <span className="now-t">{timeLine}</span>
                </div>
              </div>
              <p className="why"><span className="now-k">Why</span>{sig ? humanize(sig.reason, J) : "Waiting for the first frame."}</p>
              <div>
                <span className="now-k">Queues waiting at this signal (length of road they fill)</span>
                <ul className="appr">
                  {approaches.map((x) => (
                    <li key={x.a} className={x.q.spill ? "is-spill" : ""}>
                      <span>{x.label}</span>
                      <span className="appr-bar" title={`${Math.round(x.q.share * 100)}% of this road's space`}>
                        <i style={{ width: `${Math.min(100, x.q.share * 100)}%` }} />
                      </span>
                      <b className="num">{x.q.veh} veh</b>
                      <span className="num appr-m">{x.q.veh ? `${Math.round(x.q.metres)} m` : ""}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <ol className="phases" aria-label="Phase order">
                {J.phases.map((p, i) => (
                  <li key={p.name} className={sig?.phase === i ? "on" : ""}>{p.name}</li>
                ))}
              </ol>
            </>
          )}
        </div>

        <div className="log">
          <h2>Event log</h2>
          {shown.length ? (
            <ul>
              {shown.map((e, k) => (
                <li key={`${e.t}-${k}`} className={`ev ev-${e.sev}`}>
                  <time className="num">{clock(e.t)}</time>
                  <span>{e.j && <b>{e.j.replace("J", "Signal ")} </b>}{e.text}</span>
                </li>
              ))}
            </ul>
          ) : <p className="muted">Phase changes at the selected signal and corridor-wide events appear here as they happen.</p>}
        </div>
      </div>

      <dl className="kpis">
        <div><dt>Vehicles waiting now</dt><dd className="num">{waiting}</dd><Trend frames={frames} index={player.i} /></div>
        <div><dt>Through so far</dt><dd className="num">{frame?.exited ?? 0}</dd></div>
        <div><dt>Longest queue</dt><dd><span className="num">{longest?.[1] ?? 0}</span><small>{longest && longest[1] > 0 ? longestLabel(longest[0]) : ""}</small></dd></div>
        <div><dt>Roads backed up to the signal behind</dt><dd className="num">{backedUp}</dd></div>
        <div><dt>Signals with working cameras</dt><dd className="num">{adaptive}/{nJ}</dd></div>
        {done && run && (
          <div className="kpi-final"><dt>Average wait, whole run</dt><dd className="num">{Math.round(run.metrics.avg_wait_s)} s</dd></div>
        )}
      </dl>
    </section>
  );
}
