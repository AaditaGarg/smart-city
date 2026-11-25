import React, { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { getJSON } from "./api.js";
import Home from "./tabs/Home.jsx";
import ControlRoom from "./tabs/ControlRoom.jsx";
import LiveJunction from "./tabs/LiveJunction.jsx";
import BeforeAfter from "./tabs/BeforeAfter.jsx";
import RoadRepairs from "./tabs/RoadRepairs.jsx";
import ModelData from "./tabs/ModelData.jsx";
import "./styles.css";

export const PAGES = [
  { id: "home", label: "Home" },
  { id: "control", label: "Control room", title: "Control room",
    lede: "The whole corridor at a glance: every signal, every queue, and why each signal is doing what it does." },
  { id: "camera", label: "Camera check", title: "Camera check",
    lede: "Give Signal 2 one photo or video per road and see the vehicle count and the timing it would choose." },
  { id: "impact", label: "Impact report", title: "Impact report",
    lede: "The same traffic played twice, side by side: today's fixed timer against camera-driven control." },
  { id: "repairs", label: "Road repairs", title: "Road repairs",
    lede: "Score road damage from photos and rank repairs by damage and how busy each road is." },
  { id: "model", label: "Model and data", title: "Model and data",
    lede: "How accurate the detector is, where the data comes from, and what this prototype cannot do yet." },
];

const readRoute = () => {
  const id = window.location.hash.replace(/^#\/?/, "");
  return PAGES.some((p) => p.id === id) ? id : "home";
};

export const go = (id) => { window.location.hash = `/${id}`; };

export default function App() {
  const [page, setPage] = useState(readRoute);
  const [meta, setMeta] = useState(null);
  const [online, setOnline] = useState(null);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    const on = () => { setPage(readRoute()); setMenu(false); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    getJSON("/network/scenarios").then((m) => { setMeta(m); setOnline(true); }).catch(() => setOnline(false));
  }, []);

  const current = PAGES.find((p) => p.id === page);
  useEffect(() => {
    document.title = page === "home" ? "Vikas Marg traffic control" : `${current.title}: Vikas Marg traffic control`;
  }, [page, current]);

  return (
    <div className="sc">
      <header className="topbar">
        <a className="brand" href="#/home" aria-label="Vikas Marg traffic control, home">
          <svg width="18" height="40" viewBox="0 0 18 40" aria-hidden="true">
            <rect x="1" y="1" width="16" height="38" rx="5" fill="var(--head)" stroke="var(--line)" />
            <circle cx="9" cy="9.5" r="4" fill="var(--stop)" />
            <circle cx="9" cy="20" r="4" fill="var(--amber)" opacity=".3" />
            <circle cx="9" cy="30.5" r="4" fill="var(--go)" opacity=".3" />
          </svg>
          <span>
            <b>Vikas Marg traffic control</b>
            <small>ITO to Laxmi Nagar, Delhi</small>
          </span>
        </a>
        <button className="menu-btn" onClick={() => setMenu(!menu)} aria-expanded={menu} aria-controls="site-nav"
                aria-label={menu ? "Close menu" : "Open menu"}>
          {menu ? <X size={20} /> : <Menu size={20} />}
        </button>
        <nav id="site-nav" className={`sitenav ${menu ? "is-open" : ""}`} aria-label="Main">
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={page === p.id ? "on" : ""} aria-current={page === p.id ? "page" : undefined}>
              {p.label}
            </a>
          ))}
        </nav>
        <span className={`status ${online ? "on" : online === false ? "off" : ""}`} role="status">
          {online == null ? "Connecting…" : online ? "Server connected" : "Server offline"}
        </span>
      </header>

      {online === false && (
        <p className="offline">The server is not answering. Start it with <code>python -m uvicorn backend.app.main:app --port 8000</code> from the project folder, then reload this page.</p>
      )}

      <main className="sc-main">
        {current.title && (
          <div className="page-head">
            <h1>{current.title}</h1>
            <p>{current.lede}</p>
          </div>
        )}
        {page === "home" && <Home network={meta?.network} />}
        {/* the control room keeps running while other pages are open, like the wall screen in a real control room */}
        <div hidden={page !== "control"}><ControlRoom scenarios={meta?.scenarios} network={meta?.network} /></div>
        {page === "camera" && <LiveJunction network={meta?.network} />}
        {page === "impact" && <BeforeAfter scenarios={meta?.scenarios} />}
        {page === "repairs" && <RoadRepairs />}
        {page === "model" && <ModelData />}
      </main>

      <footer className="foot">
        <p>Smart India Hackathon 2025 prototype. Traffic shown in the control room is simulated from typical Indian peak-hour volumes; signal positions come from OpenStreetMap.</p>
        <nav aria-label="Footer">
          {PAGES.map((p) => <a key={p.id} href={`#/${p.id}`}>{p.label}</a>)}
        </nav>
      </footer>
    </div>
  );
}
