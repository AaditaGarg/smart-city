import React, { useEffect, useState } from "react";
import { Camera, Film, Footprints, Upload } from "lucide-react";
import { getJSON, postForm, postJSON } from "../api.js";

const SIDES = [
  { id: "N", key: "north", label: "North road", sample: "/samples/north_chowk_delhi.jpg" },
  { id: "E", key: "east", label: "East road", sample: "/samples/east_chowk_ambulance.jpg" },
  { id: "S", key: "south", label: "South road", sample: "/samples/south_chowk_bangalore.jpg" },
  { id: "W", key: "west", label: "West road", sample: "/samples/west_chowk_mumbai.jpg" },
];

function PressureBars({ phases, best }) {
  const max = Math.max(1, ...phases.map((p) => p.pressure));
  return (
    <ul className="pressure">
      {phases.map((p) => (
        <li key={p.index} className={p.index === best ? "best" : ""}>
          <span className="pname">{p.name}</span>
          <span className="pbar"><i style={{ width: `${(p.pressure / max) * 100}%` }} /></span>
          <span className="pval">{p.queue_pcu} PCU</span>
        </li>
      ))}
    </ul>
  );
}

function JunctionGlyph({ decision, approaches }) {
  // which approaches get green in the recommended phase
  const green = new Set((decision?.green_approaches) || []);
  const pos = { N: [100, 28], E: [172, 100], S: [100, 172], W: [28, 100] };
  return (
    <svg viewBox="0 0 200 200" className="glyph" aria-label="Junction with recommended green">
      <rect x="80" y="0" width="40" height="200" className="road" />
      <rect x="0" y="80" width="200" height="40" className="road" />
      {Object.entries(pos).map(([a, [x, y]]) => (
        <g key={a}>
          <circle cx={x} cy={y} r="11" fill={green.has(a) ? "var(--go)" : "var(--stop)"} />
          <text x={x} y={y + 4} textAnchor="middle" className="glyph-t">{approaches?.[a]?.n_vehicles ?? "–"}</text>
        </g>
      ))}
    </svg>
  );
}

export default function LiveJunction({ network }) {
  const [files, setFiles] = useState({});
  const [previews, setPreviews] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [video, setVideo] = useState(null);
  const [health, setHealth] = useState(null);
  const [ped, setPed] = useState(null);
  const junction = network?.junctions.find((j) => j.id === "J2");

  const refreshHealth = () => getJSON("/network/health").then(setHealth).catch(() => {});
  useEffect(() => { refreshHealth(); const id = setInterval(refreshHealth, 5000); return () => clearInterval(id); }, []);

  const pick = (side, file) => {
    setFiles((f) => ({ ...f, [side]: file }));
    setPreviews((p) => ({ ...p, [side]: URL.createObjectURL(file) }));
  };

  const loadSamples = async () => {
    setError(null);
    for (const s of SIDES) {
      const blob = await fetch(s.sample).then((r) => r.blob());
      pick(s.id, new File([blob], s.sample.split("/").pop(), { type: blob.type || "image/jpeg" }));
    }
  };

  const analyse = async () => {
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      SIDES.forEach((s) => files[s.id] && form.append(s.key, files[s.id]));
      const r = await postForm("/network/decide-photos?junction_id=J2", form);
      const ph = junction?.phases[r.decision.recommended_phase];
      r.decision.green_approaches = ph ? [...new Set(ph.movements.map((m) => m[0]))] : [];
      setResult(r);
      refreshHealth();
    } catch (e) {
      setError(`Could not analyse the photos: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const analyseVideo = async (side, file) => {
    setError(null);
    setVideo({ side, busy: true });
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await postForm(`/network/decide-video?junction_id=J2&approach=${side}`, form);
      setVideo({ side, ...r });
    } catch (e) {
      setVideo(null);
      setError(`Could not read the video: ${e.message}`);
    }
  };

  const pressPed = async () => setPed(await postJSON("/network/pedestrian-request", { junction_id: "J2" }));

  const d = result?.decision;
  const vmax = video?.frames ? Math.max(1, ...video.frames.map((f) => f.estimate_pcu)) : 1;

  return (
    <section className="tab-body">

      <div className="cams">
        {SIDES.map((s) => {
          const a = result?.approaches?.[s.id];
          return (
            <figure key={s.id} className="cam">
              <div className="cam-img">
                {a?.image ? <img src={`data:image/jpeg;base64,${a.image}`} alt={`${s.label} with detections`} />
                  : previews[s.id] ? <img src={previews[s.id]} alt={s.label} />
                  : <div className="cam-empty"><Camera size={22} /> No photo</div>}
              </div>
              <figcaption>
                <b>{s.label}</b>
                {a ? <span>{a.n_vehicles} {a.n_vehicles === 1 ? "vehicle" : "vehicles"} = {a.pcu} PCU{a.beyond_camera ? ", queue goes beyond camera" : ""}</span> : <span className="muted">waiting for photo</span>}
                {a?.ambulance?.confidence && (
                  <span className={a.ambulance.confirmed ? "tag-crit" : "tag-warn"}>
                    {a.ambulance.confirmed ? "Ambulance confirmed (2 frames)" : "Ambulance seen once: needs a 2nd frame"}
                  </span>
                )}
                <span className="cam-actions">
                  <label className="link-btn"><Upload size={14} /> Photo
                    <input type="file" accept="image/*" hidden onChange={(e) => e.target.files[0] && pick(s.id, e.target.files[0])} />
                  </label>
                  <label className="link-btn"><Film size={14} /> Video
                    <input type="file" accept="video/*" hidden onChange={(e) => e.target.files[0] && analyseVideo(s.id, e.target.files[0])} />
                  </label>
                </span>
              </figcaption>
            </figure>
          );
        })}
      </div>

      <div className="toolbar">
        <button className="btn ghost" onClick={loadSamples}>Use sample photos</button>
        <button className="btn" onClick={analyse} disabled={busy || Object.keys(files).length === 0}>
          {busy ? "Counting vehicles…" : "Decide signal timing"}
        </button>
        <button className="btn ghost" onClick={pressPed}><Footprints size={16} /> Press pedestrian button</button>
      </div>
      {error && <p className="error">{error}</p>}
      {ped && <p className="muted">Pedestrian request at J2 recorded. The walk phase comes at the next phase change, within {ped.served_within_s} s at most.</p>}

      {d && (
        <div className="decision">
          <JunctionGlyph decision={d} approaches={result.approaches} />
          <div className="panel why">
            <h3>Why this decision?</h3>
            <p className="why-text">{d.explanation}</p>
            <div className="timing">
              <div><span className="big">{d.green_s}</span><span className="unit">s green at most</span></div>
              <div><span className="big">{d.yellow_s}</span><span className="unit">s yellow</span></div>
              <div><span className="big">{d.all_red_s}</span><span className="unit">s all-red</span></div>
            </div>
            <div className="maxbar" aria-label="Green time budget"><i style={{ width: `${(d.green_s / 60) * 100}%` }} /><span>ends early if the queue clears twice in a row</span></div>
            <h4>Pressure of each phase</h4>
            <PressureBars phases={d.phases} best={d.recommended_phase} />
            <p className="muted">Model: {result.model}</p>
          </div>
        </div>
      )}

      {video && (
        <div className="panel">
          <h3>Video, {SIDES.find((s) => s.id === video.side)?.label}</h3>
          {video.busy ? <p className="muted">Reading one frame per second…</p> : (
            <>
              <div className="spark">
                {video.frames.map((f) => (
                  <i key={f.t} title={`${f.t}s: ${f.estimate_pcu} PCU`} style={{ height: `${(f.estimate_pcu / vmax) * 100}%` }} className={f.empty_confirmed ? "empty" : ""} />
                ))}
              </div>
              <p className="muted">Queue estimate per second (PCU). {video.early_switch_ready ? "Stop line empty in 2 checks: the green can end early." : `Final queue ${video.final_queue_pcu} PCU: keep green while it clears.`}</p>
            </>
          )}
        </div>
      )}

      {health && (
        <div className="panel">
          <h3>Camera watchdog</h3>
          <ul className="health">
            {health.junctions.map((j) => (
              <li key={j.junction} className={j.mode === "adaptive" ? "ok" : "fallback"}>
                <b>{j.junction}</b> {j.mode === "adaptive" ? `adaptive, last frame ${j.last_frame_age_s}s ago` : "no recent camera frame: runs the safe fixed-time plan"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
