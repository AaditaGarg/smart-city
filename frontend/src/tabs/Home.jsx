import React, { useEffect, useState } from "react";
import { BarChart3, Camera, Construction, Cpu, MonitorDot } from "lucide-react";
import CorridorStrip from "../components/CorridorStrip.jsx";
import { usePlayer } from "../components/Player.jsx";
import { getJSON, simulate } from "../api.js";

// Landing page: what the system is, the corridor running live, what it achieved (from the saved report), what state
// the real system is in right now, and a way into every tool.

const TOOLS = [
  { id: "control", Icon: MonitorDot, title: "Control room", text: "Watch all four signals, the queues on every road and the reason behind each decision. Send a green corridor for an ambulance." },
  { id: "camera", Icon: Camera, title: "Camera check", text: "Upload a photo or short video for each road at Signal 2. The detector counts vehicles and the signal plan updates." },
  { id: "impact", Icon: BarChart3, title: "Impact report", text: "Play the same traffic under the fixed timer and under smart control, side by side, and compare waiting times." },
  { id: "repairs", Icon: Construction, title: "Road repairs", text: "Rank city roads for repair by measured damage and how much traffic they carry, and create crew jobs." },
  { id: "model", Icon: Cpu, title: "Model and data", text: "Detector accuracy on held-out images, data sources and licences, and the limits of this prototype." },
];

const STEPS = [
  ["Cameras look at each road", "One camera per approach at every signal sends a frame every few seconds."],
  ["Vehicles are counted", "A YOLO detector finds cars, bikes, buses, trucks and people in each frame."],
  ["Counts become road load", "Each vehicle is weighted by the Indian standard IRC:106 (bike 0.5, car 1, bus 2.2)."],
  ["The four signals decide together", "Max-Pressure gives green where the queue is longest compared with the space ahead, so one signal does not push traffic into a jam at the next."],
  ["Safety rules always apply", "Minimum green, yellow and all-red are never skipped. A camera fault switches that signal to a safe fixed plan."],
];

const min = (s) => (s == null ? "–" : `${(s / 60).toFixed(1)} min`);
const pct = (a, b) => (a && b ? Math.round((1 - b / a) * 100) : null);
const ago = (iso) => {
  const s = (Date.now() - new Date(`${iso}Z`).getTime()) / 1000;
  if (!Number.isFinite(s)) return "";
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};

function LiveCorridor({ network }) {
  const [run, setRun] = useState(null);
  const frames = run?.frames ?? [];
  const player = usePlayer(frames.length);
  useEffect(() => { simulate("morning_rush", "max_pressure").then(setRun).catch(() => setRun(null)); }, []);
  useEffect(() => { player.setSpeed(5); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {   // loop like a wall screen
    if (frames.length && !player.playing && player.i >= frames.length - 1) { player.setI(0); player.setPlaying(true); }
  }, [player.playing, player.i, frames.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const t = frames[player.i]?.t ?? 0;
  return (
    <figure className="home-live">
      <CorridorStrip network={network} frame={frames[player.i]} selected={null} onSelect={() => { window.location.hash = "/control"; }} />
      <figcaption>
        <span className="feed-dot is-live" aria-hidden="true" />
        Simulated morning rush, playing at 10× ({String(Math.floor(t / 60)).padStart(2, "0")}:{String(t % 60).padStart(2, "0")} in).
        Yellow bars are queued vehicles; click a signal to open it in the control room.
      </figcaption>
    </figure>
  );
}

export default function Home({ network }) {
  const [report, setReport] = useState(null);
  const [health, setHealth] = useState(null);
  const [cams, setCams] = useState(null);
  const [rl, setRl] = useState(null);
  const [alerts, setAlerts] = useState(null);

  useEffect(() => {
    getJSON("/network/report").then(setReport).catch(() => setReport(false));
    getJSON("/health").then(setHealth).catch(() => setHealth(false));
    getJSON("/network/health").then(setCams).catch(() => setCams(false));
    getJSON("/network/rl-status").then(setRl).catch(() => setRl(false));
    getJSON("/alerts?limit=5").then(setAlerts).catch(() => setAlerts(false));
  }, []);

  const mr = report?.results?.morning_rush;
  const oc = report?.results?.over_capacity;
  const amb = report?.ambulance;
  const liveCams = cams ? cams.junctions.filter((j) => j.last_frame_age_s != null && j.last_frame_age_s <= cams.stale_after_s).length : 0;

  const status = [
    ["Server and database", health ? (health.database ? "Running" : "Server up, database not reachable") : health === false ? "Not reachable" : "Checking…", health && health.database ? "ok" : "bad"],
    ["Cameras sending frames", cams ? `${liveCams} of ${cams.junctions.length}${liveCams < cams.junctions.length ? ": signals without frames run the safe fixed plan" : ""}` : "Checking…", cams && liveCams === cams.junctions.length ? "ok" : "warn"],
    ["Vehicle detector", health ? (health.models.vehicle ? "Ready (YOLOv8n, COCO classes)" : "No weights found") : "Checking…", health?.models?.vehicle ? "ok" : "bad"],
    ["Road-damage detector", health ? (health.models.road_damage ? "Ready" : "Not trained on this machine yet") : "Checking…", health?.models?.road_damage ? "ok" : "warn"],
    ["Signal controller", rl ? (rl.deploy_rl ? "AI agent (beat Max-Pressure)" : `Max-Pressure. The AI agent averaged ${Math.round(rl.mean_avg_wait_s.rl)} s waits against ${Math.round(rl.mean_avg_wait_s.max_pressure)} s, so it stays off`) : "Max-Pressure", "ok"],
  ];

  return (
    <div className="home">
      <section className="hero">
        <div className="hero-text">
          <h1>Traffic signals on Vikas Marg that respond to the traffic in front of them</h1>
          <p>Four signals between ITO and Laxmi Nagar count vehicles with cameras and agree among themselves who gets green next, instead of each running a fixed timer.</p>
          <div className="hero-cta">
            <a className="btn" href="#/control">Open the control room</a>
            <a className="btn ghost" href="#/camera">Try it with a photo</a>
          </div>
        </div>
        <LiveCorridor network={network} />
      </section>

      <section className="results" aria-labelledby="res-h">
        <div className="sec-head">
          <h2 id="res-h">What changed, in simulation</h2>
          <p>Same traffic, five random days each, real signal spacing from OpenStreetMap. Fixed timer is how the signals run today.</p>
        </div>
        {report ? (
          <div className="res-grid">
            <div className="res">
              <h3>Average wait in the morning rush</h3>
              <p className="res-v"><span className="was">{Math.round(mr.fixed.avg_wait_s.mean)} s</span><span className="now">{Math.round(mr.max_pressure.avg_wait_s.mean)} s</span></p>
              <p className="res-k">{pct(mr.fixed.avg_wait_s.mean, mr.max_pressure.avg_wait_s.mean)}% less waiting per vehicle</p>
            </div>
            <div className="res">
              <h3>Longest queue at any signal</h3>
              <p className="res-v"><span className="was">{Math.round(mr.fixed.max_queue_veh.mean)}</span><span className="now">{Math.round(mr.max_pressure.max_queue_veh.mean)}</span></p>
              <p className="res-k">vehicles, morning rush</p>
            </div>
            <div className="res">
              <h3>Ambulance, ITO to Laxmi Nagar</h3>
              <p className="res-v"><span className="was">{min(amb.fixed_no_corridor_s)}</span><span className="now">{min(amb.max_pressure_with_corridor_s)}</span></p>
              <p className="res-k">fixed timer against smart control with a green corridor ({(amb.route_length_m / 1000).toFixed(1)} km)</p>
            </div>
            <div className="res res-limit">
              <h3>Where timing cannot help</h3>
              <p>When more vehicles arrive than the road can carry (the 500 against 499 test), smarter timing cut waiting by only {pct(oc.fixed.avg_wait_s.mean, oc.max_pressure.avg_wait_s.mean)}%. That needs diversion or more road, not a better signal.</p>
            </div>
          </div>
        ) : report === false ? (
          <p className="muted">No report yet. Run <code>python scripts/make_network_report.py</code> to compute these numbers.</p>
        ) : <p className="muted">Loading results…</p>}
        <a className="text-link" href="#/impact">See it play out in the impact report</a>
      </section>

      <section className="home-split">
        <div className="status-box" aria-labelledby="st-h">
          <h2 id="st-h">System status right now</h2>
          <dl>
            {status.map(([k, v, s]) => (
              <div key={k} className={`st st-${s}`}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
        </div>
        <div className="alerts-box" aria-labelledby="al-h">
          <h2 id="al-h">Latest alerts</h2>
          {alerts && alerts.length ? (
            <ul>
              {alerts.slice(0, 5).map((a) => (
                <li key={a.id} className={`al al-${a.severity}`}>
                  <span>{a.message}</span>
                  <small>{a.simulated ? "Simulated, " : ""}{ago(a.created_at)}</small>
                </li>
              ))}
            </ul>
          ) : <p className="muted">{alerts === false ? "Alerts could not be loaded." : alerts ? "No alerts. Corridor requests, camera faults and severe road damage appear here." : "Loading…"}</p>}
        </div>
      </section>

      <section className="how" aria-labelledby="how-h">
        <h2 id="how-h">How a signal decides</h2>
        <ol>
          {STEPS.map(([h, p]) => (
            <li key={h}><h3>{h}</h3><p>{p}</p></li>
          ))}
        </ol>
      </section>

      <section className="tools" aria-labelledby="tools-h">
        <h2 id="tools-h">Tools</h2>
        <ul>
          {TOOLS.map(({ id, Icon, title, text }) => (
            <li key={id}>
              <a href={`#/${id}`}>
                <Icon size={22} aria-hidden="true" />
                <span><b>{title}</b><span>{text}</span></span>
              </a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
