import React, { useEffect, useState } from "react";
import { getJSON } from "../api.js";

const LIMITS = [
  ["Simulation, not street trial", "Results come from a simulator of the real Vikas Marg signals with typical Indian peak-hour volumes, not from live field counts."],
  ["Road hardware", "Connecting to real signal controllers needs the vendor protocol and approval from Delhi Traffic Police. The code has a plug-in point for it."],
  ["Bigger dataset", "The Indian Driving Dataset (22 GB) would give a stronger detector; this laptop had room for a 456 MB subset."],
  ["Ambulance detection", "Camera detection is a backup. A GPS or RFID unit on ambulances is the reliable trigger."],
  ["Over-capacity", "When more cars arrive than a road can carry, timing can only share delay fairly. Diversion and road capacity fix the rest."],
  ["Lane discipline", "Indian mixed traffic varies a lot; results are shown as a range over many random days, including a chaotic-traffic test."],
];

export default function ModelData() {
  const [m, setM] = useState(null);
  const [rl, setRl] = useState(null);
  const [report, setReport] = useState(null);
  useEffect(() => {
    getJSON("/model-metrics").then(setM).catch(() => setM({}));
    getJSON("/network/report").then(setReport).catch(() => {});
    fetch("/api/network/rl-status").then((r) => (r.ok ? r.json() : null)).then(setRl).catch(() => {});
  }, []);

  const task = m?.indian_vehicle ? "indian_vehicle" : m?.vehicle ? "vehicle" : null;
  const per = task ? m[task].per_class : [];

  return (
    <section className="tab-body">
      <div className="grid-2">
        <div className="panel">
          <h3>Vehicle detector</h3>
          {task ? (
            <>
              <p className="muted">{task === "indian_vehicle" ? "Indian-traffic model we trained (bikes, autos, buses, ambulances), scored on held-out test images. It is not used live yet: on real junction photos it missed too many vehicles, so live detection uses the standard COCO YOLOv8n (car, bike, bicycle, bus, truck, person)." : "Earlier KITTI model (cars, vans, trucks), scored on held-out test images."}</p>
              <table className="cmp">
                <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>mAP50</th></tr></thead>
                <tbody>
                  {per.map((r) => (
                    <tr key={r["class"]}><th>{r["class"]}</th><td>{r.precision?.toFixed(2)}</td><td>{r.recall?.toFixed(2)}</td><td>{r.mAP50?.toFixed(2)}</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : <p className="muted">No evaluation found yet.</p>}
        </div>
        <div className="panel">
          <h3>Data sources</h3>
          <ul className="sources">
            <li><b>Indian Vehicle Dataset</b> (DataCluster Labs, Kaggle): photos from Indian cities with auto, bike, bus, truck, car, ambulance labels. Academic use.</li>
            <li><b>AmbuAlert</b> (Kaggle): labelled ambulance images, CC-BY-SA-4.0.</li>
            <li><b>OpenStreetMap</b>: positions of the 4 signals on Vikas Marg, Delhi, and lane counts.</li>
            <li><b>IRC:106-1990</b>: PCU values (bike 0.5, car 1.0, auto 1.2, bus/truck 2.2).</li>
            <li><b>Saturation flow</b> 525 × road width PCU/h, the standard for Indian mixed traffic.</li>
          </ul>
          {rl && <p className="muted">AI agent: {rl.deploy_rl ? "beats Max-Pressure on held-out days and is deployed." : "did not beat Max-Pressure on held-out days, so the system stays on Max-Pressure."}</p>}
          {report?.sensitivity_morning_rush_avg_wait_s && (
            <>
              <h4>Robustness check (morning rush, average wait)</h4>
              <table className="cmp">
                <thead><tr><th>Condition</th><th>Fixed</th><th>Max-Pressure</th></tr></thead>
                <tbody>
                  {Object.entries(report.sensitivity_morning_rush_avg_wait_s).map(([k, v]) => (
                    <tr key={k}><th>{k}</th><td>{Math.round(v.fixed)} s</td><td>{Math.round(v.max_pressure)} s</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>
      <div className="panel">
        <h3>Limits we are honest about</h3>
        <dl className="limits">
          {LIMITS.map(([k, v]) => (<div key={k}><dt>{k}</dt><dd>{v}</dd></div>))}
        </dl>
      </div>
    </section>
  );
}
