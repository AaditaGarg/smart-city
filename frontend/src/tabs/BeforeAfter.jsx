import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import RoadMap from "../components/RoadMap.jsx";
import PlayerBar, { usePlayer } from "../components/Player.jsx";
import { CONTROLLER_LABEL, fmtS, getJSON, simulate } from "../api.js";

const COLORS = { fixed: "#8b96a1", webster: "#e0a43a", max_pressure: "#2dbe73", rl: "#6f8cff" };

function Side({ title, run, frame, tone }) {
  const waiting = frame ? Object.values(frame.queues).reduce((a, b) => a + b, 0) : 0;
  return (
    <div className={`side side-${tone}`}>
      <div className="side-head">
        <h3>{title}</h3>
        <div className="side-stats">
          <span><b>{waiting}</b> waiting</span>
          <span><b>{frame?.exited ?? 0}</b> through</span>
        </div>
      </div>
      {run && <RoadMap network={run.network} frame={frame} compact width={1000} />}
    </div>
  );
}

function pct(a, b) {
  if (a == null || b == null || a === 0) return null;
  return Math.round(((a - b) / a) * 100);
}

export default function BeforeAfter({ scenarios }) {
  const [scenario, setScenario] = useState("morning_rush");
  const [smart, setSmart] = useState("max_pressure");
  const [runs, setRuns] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const len = runs ? Math.min(runs.fixed.frames.length, runs.smart.frames.length) : 0;
  const player = usePlayer(len);
  const t = runs?.fixed.frames[player.i]?.t ?? 0;

  const go = async (scn = scenario, ctl = smart) => {
    setError(null);
    setRuns(null);
    try {
      const [fixed, s] = await Promise.all([simulate(scn, "fixed"), simulate(scn, ctl)]);
      setRuns({ fixed, smart: s });
    } catch (e) {
      setError(`Could not run the comparison: ${e.message}`);
    }
  };
  useEffect(() => { go(); getJSON("/network/report").then(setReport).catch(() => setReport(null)); }, []); // eslint-disable-line

  const m0 = runs?.fixed.metrics;
  const m1 = runs?.smart.metrics;
  const done = runs && player.i >= len - 1;

  const chartRows = report
    ? Object.entries(report.results).map(([scn, row]) => {
        const r = { scenario: scn.replace(/_/g, " ") };
        Object.entries(row).forEach(([c, m]) => {
          r[c] = m.avg_wait_s.mean;
          r[`${c}_sd`] = m.avg_wait_s.std;
        });
        return r;
      })
    : [];

  return (
    <section className="tab-body">
      <div className="toolbar">
        <label className="select">Scenario
          <select value={scenario} onChange={(e) => { setScenario(e.target.value); go(e.target.value, smart); }}>
            {(scenarios || []).filter((s) => s.network === "vikas_marg").map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
          </select>
        </label>
        <label className="select">Compare fixed timer with
          <select value={smart} onChange={(e) => { setSmart(e.target.value); go(scenario, e.target.value); }}>
            {["webster", "max_pressure", "rl"].map((c) => <option key={c} value={c}>{CONTROLLER_LABEL[c]}</option>)}
          </select>
        </label>
      </div>
      {error && <p className="error">{error}</p>}
      {!runs && !error && <p className="muted">Running both controllers on the same traffic (same random seed)…</p>}
      {runs && (
        <>
          <div className="versus">
            <Side title="Fixed timer (today)" run={runs.fixed} frame={runs.fixed.frames[player.i]} tone="before" />
            <Side title={CONTROLLER_LABEL[smart]} run={runs.smart} frame={runs.smart.frames[player.i]} tone="after" />
          </div>
          <PlayerBar player={player} length={len} t={t} />
          <div className={`panel verdict ${done ? "" : "pending"}`}>
            <h3>{done ? "Measured result for this run" : "Final numbers appear when the run ends"}</h3>
            <table className="cmp">
              <thead><tr><th /><th>Fixed timer</th><th>{CONTROLLER_LABEL[smart]}</th><th>Change</th></tr></thead>
              <tbody>
                {[
                  ["Average wait per vehicle", "avg_wait_s", fmtS],
                  ["Worst road's average wait", "worst_approach_avg_wait_s", fmtS],
                  ["95% of vehicles waited less than", "p95_wait_s", fmtS],
                  ["Longest queue", "max_queue_veh", (v) => `${v} veh`],
                  ["Pedestrian wait", "ped_avg_wait_s", fmtS],
                ].map(([label, k, f]) => {
                  const p = pct(m0[k], m1[k]);
                  return (
                    <tr key={k}>
                      <th>{label}</th><td>{done ? f(m0[k]) : "…"}</td><td>{done ? f(m1[k]) : "…"}</td>
                      <td className={p > 0 ? "good" : p < 0 ? "bad" : ""}>{done && p != null ? `${p > 0 ? "−" : "+"}${Math.abs(p)}%` : ""}</td>
                    </tr>
                  );
                })}
                <tr><th>Vehicles that got through</th><td>{done ? m0.vehicles_exited : "…"}</td><td>{done ? m1.vehicles_exited : "…"}</td><td /></tr>
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="panel">
        <h3>All scenarios, averaged over several random days</h3>
        {report ? (
          <>
            <p className="muted">Average wait per vehicle; whiskers show the spread over {report.seeds} seeds. Simulation of Vikas Marg (4 real signals) and a 2×2 grid.</p>
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={chartRows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
                  <CartesianGrid stroke="var(--lane)" vertical={false} />
                  <XAxis dataKey="scenario" tick={{ fill: "var(--dust)", fontSize: 12 }} interval={0} />
                  <YAxis tick={{ fill: "var(--dust)", fontSize: 12 }} unit=" s" />
                  <Tooltip contentStyle={{ background: "var(--kerb)", border: "1px solid var(--lane)" }} formatter={(v) => `${v} s`} />
                  <Legend />
                  {report.controllers.map((c) => (
                    <Bar key={c} dataKey={c} name={CONTROLLER_LABEL[c]} fill={COLORS[c]}>
                      <ErrorBar dataKey={`${c}_sd`} stroke="var(--paint)" width={3} />
                    </Bar>
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <p className="muted">No saved report yet. Run <code>python scripts/make_network_report.py</code> to measure every scenario.</p>
        )}
      </div>
    </section>
  );
}
