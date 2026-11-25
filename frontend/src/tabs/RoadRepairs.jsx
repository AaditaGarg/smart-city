import React, { useEffect, useState } from "react";
import { Upload } from "lucide-react";
import { getJSON, postForm, postJSON } from "../api.js";

// Road repairs: real data only. Segments come from /segments, damage from uploaded photos (/predict/road-damage),
// ranking from /segments/prioritize-repairs, and crew jobs are stored via /maintenance.
export default function RoadRepairs() {
  const [segments, setSegments] = useState([]);
  const [ranked, setRanked] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [segId, setSegId] = useState("");
  const [scan, setScan] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [newName, setNewName] = useState("");
  const [newImp, setNewImp] = useState(1);
  const [hasModel, setHasModel] = useState(null);

  const refresh = async () => {
    try {
      const [s, j] = await Promise.all([getJSON("/segments"), getJSON("/maintenance?status=open")]);
      setSegments(s);
      setJobs(j);
      setRanked(s.length ? await postJSON("/segments/prioritize-repairs", {}) : []);
      if (!segId && s.length) setSegId(String(s[0].id));
    } catch (e) {
      setError(`Could not load repairs: ${e.message}`);
    }
  };

  useEffect(() => {
    refresh();
    getJSON("/health").then((h) => setHasModel(Boolean(h.models?.road_damage))).catch(() => setHasModel(null));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addSegment = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    try {
      const s = await postJSON("/segments", { name: newName.trim(), traffic_importance: Number(newImp) });
      setNewName("");
      setSegId(String(s.id));
      await refresh();
    } catch (err) {
      setError(`Could not add the road: ${err.message}`);
    }
  };

  const upload = async (file) => {
    if (!file || !segId) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      setScan(await postForm(`/predict/road-damage?segment_id=${segId}&annotate_image=true`, form));
      await refresh();
    } catch (e) {
      setError(`Scan failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const dispatch = async (row) => {
    try {
      await postJSON("/maintenance", {
        segment_id: row.segment_id,
        description: row.recommended_action,
        priority_score: row.priority_score,
      });
      await refresh();
    } catch (e) {
      setError(`Could not create the job: ${e.message}`);
    }
  };

  const openFor = new Set(jobs.map((j) => j.segment_id));

  return (
    <section className="tab-body">
      {error && <p className="error">{error}</p>}

      <div className="grid-2">
        <div className="panel">
          <h3>Scan a road</h3>
          {segments.length ? (
            <>
              <label className="select">Road
                <select value={segId} onChange={(e) => setSegId(e.target.value)}>
                  {segments.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
              </label>
              {hasModel === false ? (
                <p className="tag-warn">No road-damage model on this machine yet. Train it with
                  {" "}<code>python -m ml.training.train_yolo --task road_damage</code>, then scans work here.</p>
              ) : (
                <label className="btn" style={{ alignSelf: "flex-start" }}>
                  <Upload size={16} /> {busy ? "Scanning…" : "Upload road photo"}
                  <input type="file" accept="image/*" hidden disabled={busy} onChange={(e) => upload(e.target.files[0])} />
                </label>
              )}
            </>
          ) : <p className="muted">No roads yet. Add one below.</p>}
          {scan && (
            <>
              <img src={`data:image/jpeg;base64,${scan.annotated_image_b64}`} alt="Scanned road with detected defects"
                   style={{ width: "100%", borderRadius: "var(--r-md)" }} />
              <div className="stat-row">
                <div><span className="big">{scan.road_condition?.score ?? 0}</span><span className="unit">damage score</span></div>
                <div><span className="big">{scan.n_detections}</span><span className="unit">defects</span></div>
              </div>
              <p className="muted">{Object.entries(scan.counts).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ") || "No defects found"}</p>
            </>
          )}
        </div>

        <div className="panel">
          <h3>Add a road</h3>
          <form onSubmit={addSegment} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <label className="select">Name
              <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Vikas Marg, ITO to Laxmi Nagar"
                     style={{ background: "var(--kerb)", border: "1px solid var(--lane)", borderRadius: "var(--r-sm)", padding: "7px 10px" }} />
            </label>
            <label className="select">Traffic importance (0.1 quiet lane to 5 arterial)
              <input type="number" min="0.1" max="5" step="0.1" value={newImp} onChange={(e) => setNewImp(e.target.value)}
                     style={{ background: "var(--kerb)", border: "1px solid var(--lane)", borderRadius: "var(--r-sm)", padding: "7px 10px", width: 120 }} />
            </label>
            <button className="btn ghost" type="submit" style={{ alignSelf: "flex-start" }}>Add road</button>
          </form>
        </div>
      </div>

      <div className="panel">
        <h3>Repair priority</h3>
        {ranked.length ? (
          <table className="cmp">
            <thead><tr><th>Road</th><th>Damage</th><th>Importance</th><th>Priority</th><th>Urgency</th><th>Action</th><th /></tr></thead>
            <tbody>
              {ranked.map((r) => (
                <tr key={r.segment_id}>
                  <th>{r.segment_name}</th>
                  <td>{r.damage_score}</td>
                  <td>{r.traffic_importance}×</td>
                  <td><b>{r.priority_score}</b></td>
                  <td><span className={/CRITICAL|HIGH/.test(r.priority_level) ? "tag-crit" : "tag-warn"}>{r.priority_level}</span></td>
                  <td className="muted">{r.recommended_action}</td>
                  <td>
                    {openFor.has(r.segment_id)
                      ? <span className="muted">Crew job open</span>
                      : <button className="btn ghost" onClick={() => dispatch(r)} disabled={!r.damage_score}>Create crew job</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">Add a road and scan a photo to see the ranking.</p>}
        <p className="muted">Roads that have not been scanned yet show a damage score of 0.</p>
      </div>
    </section>
  );
}
