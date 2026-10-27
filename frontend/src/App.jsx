import { useEffect, useState } from "react";

export default function App() {
  const [status, setStatus] = useState("checking...");
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setStatus(d.status)).catch(() => setStatus("backend unreachable"));
  }, []);
  return (
    <div style={{ fontFamily: "sans-serif", padding: 24 }}>
      <h1>Smart City Intelligence System</h1>
      <p>Backend status: <b>{status}</b></p>
    </div>
  );
}
