import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  Camera,
  CheckCircle2,
  Clock,
  Compass,
  Download,
  FileText,
  Flame,
  Gauge,
  Image as ImageIcon,
  Layers,
  Leaf,
  Play,
  RotateCcw,
  Send,
  Shield,
  Siren,
  Sliders,
  TrendingDown,
  Truck,
  Upload,
  Zap,
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("traffic_signal");
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [backendHealth, setBackendHealth] = useState({ status: "checking", database: false });

  // 4 Lanes Upload & Detection State
  const [lanes, setLanes] = useState([
    {
      id: "north",
      name: "Northbound (Chandni Chowk)",
      lanes: 2,
      file: null,
      previewUrl: null,
      annotatedB64: null,
      detections: 0,
      counts: {},
      pcu: 0,
      isEmergency: false,
      isDetecting: false,
      isUploaded: false,
      sampleFileUrl: "/samples/north_chowk_delhi.jpg",
      sampleFileName: "north_chowk_delhi.jpg",
      sampleLabel: "North (Chandni Chowk Traffic)",
    },
    {
      id: "south",
      name: "Southbound (Bangalore Chowk)",
      lanes: 2,
      file: null,
      previewUrl: null,
      annotatedB64: null,
      detections: 0,
      counts: {},
      pcu: 0,
      isEmergency: false,
      isDetecting: false,
      isUploaded: false,
      sampleFileUrl: "/samples/south_chowk_bangalore.jpg",
      sampleFileName: "south_chowk_bangalore.jpg",
      sampleLabel: "South (Bangalore Junction Chowk)",
    },
    {
      id: "east",
      name: "Eastbound (Ambulance Chowk)",
      lanes: 2,
      file: null,
      previewUrl: null,
      annotatedB64: null,
      detections: 0,
      counts: {},
      pcu: 0,
      isEmergency: false,
      isDetecting: false,
      isUploaded: false,
      sampleFileUrl: "/samples/east_chowk_ambulance.jpg",
      sampleFileName: "east_chowk_ambulance.jpg",
      sampleLabel: "East (Ambulance Corridor Chowk)",
    },
    {
      id: "west",
      name: "Westbound (Mumbai Chowk)",
      lanes: 1,
      file: null,
      previewUrl: null,
      annotatedB64: null,
      detections: 0,
      counts: {},
      pcu: 0,
      isEmergency: false,
      isDetecting: false,
      isUploaded: false,
      sampleFileUrl: "/samples/west_chowk_mumbai.jpg",
      sampleFileName: "west_chowk_mumbai.jpg",
      sampleLabel: "West (Mumbai Intersection Chowk)",
    },
  ]);

  // Traffic Light Optimization & Simulation State
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [isSystemActive, setIsSystemActive] = useState(false);
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);
  const [phaseSecondsRemaining, setPhaseSecondsRemaining] = useState(0);
  const [phaseLightState, setPhaseLightState] = useState("yellow");
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);

  // Road Damage Priority Table
  const [rankedRepairs, setRankedRepairs] = useState([
    { id: 1, name: "Outer Ring Road (NH-44 Expressway)", traffic_importance: 4.8, damage_score: 78.5, defect_counts: { pothole: 5, alligator_crack: 3 }, priority_score: 94.2, priority_level: "CRITICAL", action: "Emergency patching within 24h" },
    { id: 2, name: "Mahatma Gandhi Marg (Central Arterial)", traffic_importance: 4.2, damage_score: 62.0, defect_counts: { pothole: 2, alligator_crack: 4 }, priority_score: 72.8, priority_level: "HIGH", action: "Repair crew dispatch within 48h" },
    { id: 3, name: "Tech Zone Blvd (IT Corridor)", traffic_importance: 3.5, damage_score: 34.0, defect_counts: { longitudinal_crack: 3 }, priority_score: 32.5, priority_level: "MEDIUM", action: "Weekly municipal batch" },
    { id: 4, name: "Subhash Nagar Link Road", traffic_importance: 2.1, damage_score: 84.0, defect_counts: { pothole: 6, alligator_crack: 6 }, priority_score: 82.0, priority_level: "CRITICAL", action: "Emergency asphalt milling" },
  ]);

  // Clock
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Health check
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setBackendHealth(d))
      .catch(() => setBackendHealth({ status: "offline", database: false }));
  }, []);

  // API Call to detect vehicles on an uploaded file
  const runDetectionOnImage = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/predict/vehicles?annotate_image=true", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Detection failed");
    return await res.json();
  };

  // Handle uploading picture for a specific lane (0 = North, 1 = South, 2 = East, 3 = West)
  const handleFileUploadForLane = async (laneIndex, file) => {
    if (!file) return;

    const updated = [...lanes];
    updated[laneIndex].file = file;
    updated[laneIndex].previewUrl = URL.createObjectURL(file);
    updated[laneIndex].isDetecting = true;
    setLanes([...updated]);

    try {
      const data = await runDetectionOnImage(file);
      updated[laneIndex].isDetecting = false;
      updated[laneIndex].annotatedB64 = data.annotated_image_b64;
      updated[laneIndex].detections = data.n_detections;
      updated[laneIndex].counts = data.counts || {};
      updated[laneIndex].isUploaded = true;

      // Check if ambulance detected
      const isEm = Boolean(data.counts && (data.counts.ambulance > 0 || data.counts.emergency > 0));
      updated[laneIndex].isEmergency = isEm;

      // Calculate PCU
      let pcu = 0;
      Object.entries(data.counts || {}).forEach(([k, v]) => {
        if (k === "truck" || k === "tram" || k === "bus") pcu += v * 3.0;
        else if (k === "van") pcu += v * 1.5;
        else if (k === "bike") pcu += v * 0.5;
        else pcu += v * 1.0;
      });
      updated[laneIndex].pcu = Math.max(1.0, Math.round(pcu * 10) / 10);

      setLanes([...updated]);
    } catch (err) {
      console.error(err);
      updated[laneIndex].isDetecting = false;
      setLanes([...updated]);
      alert(`Detection failed on lane ${updated[laneIndex].name}: ${err.message}`);
    }
  };

  // 1-Click: Automatically load all 4 provided sample pictures into the 4 lanes & detect
  const handleAutoLoadAndDetectAll4 = async () => {
    const updated = [...lanes];

    for (let i = 0; i < updated.length; i++) {
      updated[i].isDetecting = true;
      setLanes([...updated]);

      try {
        const resp = await fetch(updated[i].sampleFileUrl);
        const blob = await resp.blob();
        const file = new File([blob], updated[i].sampleFileName, { type: "image/jpeg" });
        const data = await runDetectionOnImage(file);

        updated[i].file = file;
        updated[i].previewUrl = URL.createObjectURL(blob);
        updated[i].annotatedB64 = data.annotated_image_b64;
        updated[i].detections = data.n_detections;
        updated[i].counts = data.counts || {};
        updated[i].isUploaded = true;
        updated[i].isDetecting = false;

        const isEm = Boolean(data.counts && (data.counts.ambulance > 0 || data.counts.emergency > 0));
        updated[i].isEmergency = isEm;

        let pcu = 0;
        Object.entries(data.counts || {}).forEach(([k, v]) => {
          if (k === "truck" || k === "tram") pcu += v * 3.0;
          else if (k === "van") pcu += v * 1.5;
          else if (k === "bike") pcu += v * 0.5;
          else pcu += v * 1.0;
        });
        updated[i].pcu = Math.max(1.0, Math.round(pcu * 10) / 10);
      } catch (err) {
        console.error(err);
        updated[i].isDetecting = false;
      }
    }

    setLanes([...updated]);
  };

  // 🚀 User clicks this button to calculate Webster and START signals based on their uploaded pictures!
  const handleStartSignalsFromUploads = async () => {
    const uploadedLanes = lanes.filter((l) => l.isUploaded);
    if (uploadedLanes.length < 2) {
      alert("Please upload pictures for at least 2 lanes (or click 'Auto-Load All 4 Sample Pictures') before starting signals!");
      return;
    }

    const approachesPayload = lanes.map((l) => ({
      name: l.name,
      lanes: l.lanes,
      counts: l.isUploaded ? l.counts : { car: 2 },
      has_emergency: l.isEmergency,
    }));

    try {
      const res = await fetch("/api/signals/optimize-dynamic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approaches: approachesPayload }),
      });

      if (!res.ok) throw new Error("Failed to calculate signal timing");
      const result = await res.json();
      setOptimizationResult(result);
      setIsSystemActive(true);
      setActivePhaseIndex(0);

      if (result.phase_timings && result.phase_timings.length > 0) {
        setPhaseSecondsRemaining(Math.round(result.phase_timings[0].green_time));
        setPhaseLightState("green");
        setIsSimulationRunning(true);
      }
    } catch (err) {
      console.error(err);
      alert(`Error starting traffic signals: ${err.message}`);
    }
  };

  // Real-time Traffic Light Countdown Runner
  useEffect(() => {
    if (!isSimulationRunning || !optimizationResult || !optimizationResult.phase_timings) return;
    const interval = setInterval(() => {
      setPhaseSecondsRemaining((prev) => {
        if (prev > 1) {
          if (prev <= 4 && phaseLightState === "green") {
            setPhaseLightState("yellow");
          }
          return prev - 1;
        } else {
          const timings = optimizationResult.phase_timings;
          const nextIndex = (activePhaseIndex + 1) % timings.length;
          setActivePhaseIndex(nextIndex);
          setPhaseLightState("green");
          return Math.round(timings[nextIndex].green_time || 20);
        }
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [isSimulationRunning, optimizationResult, activePhaseIndex, phaseLightState]);

  const isEmergencyActive = optimizationResult?.mode === "EMERGENCY_OVERRIDE";

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top Header Bar */}
      <header
        style={{
          borderBottom: "1px solid var(--border-color)",
          background: "rgba(10, 14, 23, 0.9)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 50,
          padding: "14px 24px",
        }}
      >
        <div style={{ maxWidth: 1400, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 42,
                height: 42,
                borderRadius: 10,
                background: "linear-gradient(135deg, #0284c7, #06b6d4)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 0 15px rgba(6, 182, 212, 0.4)",
              }}
            >
              <Zap size={24} color="#fff" />
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.5px" }}>SMART CITY INTELLIGENCE</h1>
                <span className="badge badge-cyan">SIH 2025</span>
                <span className="badge badge-emerald">User Picture Upload Driven</span>
              </div>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Upload 4 Approach Road Photos $\to$ YOLOv8 Detects Vehicle Counts $\to$ Webster Optimizes Traffic Signals
              </p>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)", background: "rgba(255,255,255,0.04)", padding: "6px 12px", borderRadius: 20 }}>
              <Shield size={14} color="#10b981" />
              <span>Security: <b>OWASP Protected</b></span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)", background: "rgba(255,255,255,0.04)", padding: "6px 12px", borderRadius: 20 }}>
              <Activity size={14} color={backendHealth.status === "ok" ? "#10b981" : "#f59e0b"} />
              <span>Backend: <b>{backendHealth.status}</b></span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#38bdf8", background: "rgba(6,182,212,0.1)", padding: "6px 12px", borderRadius: 20 }}>
              <Clock size={14} />
              <span>{currentTime}</span>
            </div>
          </div>
        </div>

        {/* Emergency Alert Banner */}
        {isEmergencyActive && (
          <div
            style={{
              maxWidth: 1400,
              margin: "12px auto 0",
              background: "linear-gradient(90deg, #991b1b, #dc2626)",
              padding: "10px 16px",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              boxShadow: "0 0 20px rgba(220, 38, 38, 0.6)",
              animation: "siren-pulse 1.5s infinite",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Siren size={22} color="#fff" />
              <span style={{ fontWeight: 700, fontSize: 14 }}>
                🚨 AMBULANCE DETECTED IN UPLOADED PHOTO! Green Corridor Active on {optimizationResult.emergency_approach}! All conflicting signals set to safe all-red clearance.
              </span>
            </div>
          </div>
        )}
      </header>

      {/* Main Tabs Navigation */}
      <nav style={{ background: "rgba(17, 24, 39, 0.8)", borderBottom: "1px solid var(--border-color)", padding: "6px 24px" }}>
        <div style={{ maxWidth: 1400, margin: "0 auto", display: "flex", gap: 8 }}>
          <button className={`tab-btn ${activeTab === "traffic_signal" ? "active" : ""}`} onClick={() => setActiveTab("traffic_signal")}>
            <Sliders size={16} /> 4-Lane Photo Ingestion & Dynamic Signals
          </button>
          <button className={`tab-btn ${activeTab === "road_damage" ? "active" : ""}`} onClick={() => setActiveTab("road_damage")}>
            <Layers size={16} /> Road Damage Prioritization
          </button>
          <button className={`tab-btn ${activeTab === "metrics" ? "active" : ""}`} onClick={() => setActiveTab("metrics")}>
            <Gauge size={16} /> AI Benchmarks
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ maxWidth: 1400, margin: "24px auto", padding: "0 24px", flex: 1, width: "100%" }}>
        {activeTab === "traffic_signal" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* 📥 STEP 1: DOWNLOAD TEST IMAGES CARD */}
            <div className="glass-card" style={{ background: "rgba(15, 23, 42, 0.95)", border: "1px solid rgba(6, 182, 212, 0.4)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 14, marginBottom: 12 }}>
                <div>
                  <h3 style={{ fontSize: 16, fontWeight: 700, color: "#38bdf8", display: "flex", alignItems: "center", gap: 8 }}>
                    <Download size={18} color="#06b6d4" /> Step 1: Real Traffic Chowk Photos (Click Ki Hui Pictures)
                  </h3>
                  <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
                    Asli Traffic Chowks (Chandni Chowk, Bangalore Junction, Mumbai Chowk, Ambulance) ki photos — download karke upload karein ya seedha <b>"⚡ 1-Click: Auto-Load All 4 Chowk Pictures"</b> dabayein:
                  </p>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleAutoLoadAndDetectAll4}
                  style={{ fontSize: 13, padding: "8px 16px" }}
                >
                  <Zap size={15} /> ⚡ 1-Click: Auto-Load & Detect All 4 Chowk Images
                </button>
              </div>

              {/* 4 Download Pills */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 10 }}>
                {lanes.map((l) => (
                  <a
                    key={l.id}
                    href={l.sampleFileUrl}
                    download={l.sampleFileName}
                    style={{
                      background: "rgba(30, 41, 59, 0.8)",
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: 8,
                      padding: "10px 14px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      color: "#fff",
                      textDecoration: "none",
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <ImageIcon size={16} color="#38bdf8" />
                      <span>{l.sampleLabel}</span>
                    </div>
                    <ArrowDownToLine size={15} color="#10b981" />
                  </a>
                ))}
              </div>
            </div>

            {/* 📤 STEP 2: 4 UPLOAD DROPZONES (NORTH, SOUTH, EAST, WEST) */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                  <Upload size={18} color="#06b6d4" /> Step 2: Upload Pictures For Each Lane (AI Counts Vehicles)
                </h3>
                <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  Uploaded: <b>{lanes.filter((l) => l.isUploaded).length} of 4 Lanes</b>
                </span>
              </div>

              {/* 4 Lanes Upload Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
                {lanes.map((lane, idx) => {
                  const inputId = `lane-file-input-${lane.id}`;

                  return (
                    <div
                      key={lane.id}
                      style={{
                        background: lane.isEmergency ? "rgba(239, 68, 68, 0.12)" : "rgba(19, 27, 46, 0.8)",
                        border: lane.isEmergency ? "1px solid #ef4444" : lane.isUploaded ? "1px solid #10b981" : "1px dashed rgba(255,255,255,0.2)",
                        borderRadius: 12,
                        padding: 14,
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontWeight: 700, fontSize: 14, color: "#fff" }}>{lane.name}</span>
                        {lane.isEmergency ? (
                          <span className="badge badge-rose siren-active">🚑 AMBULANCE</span>
                        ) : lane.isUploaded ? (
                          <span className="badge badge-emerald">✓ DETECTED</span>
                        ) : (
                          <span className="badge badge-amber">WAITING PHOTO</span>
                        )}
                      </div>

                      {/* Photo / Detection Display Frame */}
                      <div
                        style={{
                          height: 180,
                          background: "#090d16",
                          borderRadius: 8,
                          position: "relative",
                          overflow: "hidden",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          border: "1px solid rgba(255,255,255,0.06)",
                        }}
                      >
                        {lane.isDetecting ? (
                          <div style={{ color: "#38bdf8", fontSize: 13, fontWeight: 600 }}>YOLOv8 Scanning Vehicles...</div>
                        ) : lane.annotatedB64 ? (
                          <img
                            src={`data:image/jpeg;base64,${lane.annotatedB64}`}
                            alt={lane.name}
                            style={{ width: "100%", height: "100%", objectFit: "contain" }}
                          />
                        ) : lane.previewUrl ? (
                          <img
                            src={lane.previewUrl}
                            alt={lane.name}
                            style={{ width: "100%", height: "100%", objectFit: "contain" }}
                          />
                        ) : (
                          <div
                            onClick={() => document.getElementById(inputId).click()}
                            style={{ textAlign: "center", cursor: "pointer", color: "var(--text-muted)", padding: 10 }}
                          >
                            <Upload size={30} style={{ opacity: 0.4, margin: "0 auto 6px" }} />
                            <div style={{ fontSize: 12, fontWeight: 600 }}>Click to Upload Photo</div>
                            <div style={{ fontSize: 10 }}>or drag picture here</div>
                          </div>
                        )}
                      </div>

                      {/* Detection Breakdown Pill */}
                      <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: "8px 10px", fontSize: 12 }}>
                        {lane.isUploaded ? (
                          <div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 700, marginBottom: 2 }}>
                              <span style={{ color: "#34d399" }}>AI Vehicle Count: {lane.detections}</span>
                              <span style={{ color: "#38bdf8" }}>{lane.pcu} PCU</span>
                            </div>
                            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                              {Object.entries(lane.counts).map(([k, v]) => `${k}: ${v}`).join(", ") || "No vehicles"}
                            </div>
                          </div>
                        ) : (
                          <span style={{ color: "var(--text-muted)", fontSize: 11 }}>No photo uploaded yet.</span>
                        )}
                      </div>

                      {/* File Upload Trigger */}
                      <input
                        type="file"
                        id={inputId}
                        accept="image/*"
                        style={{ display: "none" }}
                        onChange={(e) => handleFileUploadForLane(idx, e.target.files[0])}
                      />
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: 12, width: "100%" }}
                        onClick={() => document.getElementById(inputId).click()}
                      >
                        <Upload size={13} /> {lane.isUploaded ? "Change Photo" : `Upload ${lane.name} Photo`}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 🚀 STEP 3: MASTER TRIGGER BUTTON */}
            <div className="glass-card" style={{ textAlign: "center", padding: 24, background: "rgba(15, 23, 42, 0.95)", border: "1px solid #10b981" }}>
              <h3 style={{ fontSize: 18, fontWeight: 800, marginBottom: 6 }}>
                Step 3: Run Webster Optimization on Uploaded Vehicle Counts
              </h3>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 18, maxWidth: 650, margin: "0 auto 16px" }}>
                Click below to feed the detected vehicle counts into Webster's formula and start the live dynamic traffic signals!
              </p>
              <button
                className="btn btn-primary"
                onClick={handleStartSignalsFromUploads}
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  padding: "14px 32px",
                  boxShadow: "0 0 25px rgba(16, 185, 129, 0.4)",
                  background: "linear-gradient(135deg, #059669, #10b981)",
                }}
              >
                <Zap size={18} /> 🚀 Optimize Signals Using Uploaded Images & Start Traffic Lights
              </button>
            </div>

            {/* 🚦 STEP 4: 4-WAY INTERSECTION TRAFFIC LIGHTS SIMULATION */}
            <div className="glass-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
                <div>
                  <h3 style={{ fontSize: 18, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                    <Compass size={20} color="#06b6d4" /> Live 4-Way Traffic Signal Status
                  </h3>
                  <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
                    {isSystemActive
                      ? "Traffic lights are dynamically running based on your uploaded photos!"
                      : "Standby: Upload images above and click 'Optimize Signals' to start."}
                  </p>
                </div>
                {isSystemActive && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setIsSimulationRunning(!isSimulationRunning)}
                      style={{ fontSize: 12 }}
                    >
                      {isSimulationRunning ? "⏸ Pause Lights" : "▶ Resume Lights"}
                    </button>
                  </div>
                )}
              </div>

              {/* 4 Traffic Light Displays */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20 }}>
                {lanes.map((lane, idx) => {
                  const timing = optimizationResult?.phase_timings?.find((p) => p.approach === lane.name);
                  const isCurrentActive = activePhaseIndex === idx;
                  const currentLight = !isSystemActive
                    ? "yellow"
                    : isEmergencyActive
                    ? lane.isEmergency
                      ? "green"
                      : "red"
                    : isCurrentActive
                    ? phaseLightState
                    : "red";

                  return (
                    <div
                      key={lane.id}
                      style={{
                        background: isCurrentActive && isSystemActive ? "rgba(6, 182, 212, 0.08)" : "rgba(17, 24, 39, 0.6)",
                        border: isCurrentActive && isSystemActive ? "1px solid rgba(6, 182, 212, 0.5)" : "1px solid rgba(255,255,255,0.06)",
                        borderRadius: 12,
                        padding: 16,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                        <div>
                          <h4 style={{ fontSize: 15, fontWeight: 700 }}>{lane.name}</h4>
                          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{lane.lanes} Lanes</span>
                        </div>
                        {/* Traffic Light Housing */}
                        <div className="traffic-light-housing">
                          <div className={`light-lens red ${currentLight === "red" ? "active" : ""}`} />
                          <div className={`light-lens yellow ${currentLight === "yellow" ? "active" : ""}`} />
                          <div className={`light-lens green ${currentLight === "green" ? "active" : ""}`} />
                        </div>
                      </div>

                      {/* Live Timer Countdown */}
                      <div
                        style={{
                          background: "rgba(0,0,0,0.3)",
                          borderRadius: 8,
                          padding: "10px 14px",
                          marginBottom: 12,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                        }}
                      >
                        <div>
                          <div style={{ fontSize: 10, textTransform: "uppercase", color: "var(--text-muted)" }}>Signal State</div>
                          <div style={{ fontSize: 14, fontWeight: 700, color: currentLight === "green" ? "#34d399" : currentLight === "yellow" ? "#fbbf24" : "#f87171" }}>
                            {currentLight.toUpperCase()}
                          </div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 10, textTransform: "uppercase", color: "var(--text-muted)" }}>Remaining</div>
                          <div style={{ fontSize: 20, fontWeight: 800, fontFamily: "monospace" }}>
                            {isSystemActive
                              ? isCurrentActive
                                ? `${phaseSecondsRemaining}s`
                                : `${Math.round(timing?.green_time || 0)}s allocated`
                              : "--"}
                          </div>
                        </div>
                      </div>

                      {/* Detections from Uploaded Picture */}
                      <div style={{ fontSize: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)", marginBottom: 4 }}>
                          <span>From Uploaded Photo:</span>
                          <span style={{ color: "#fff", fontWeight: 600 }}>{lane.detections} Vehicles ({lane.pcu} PCU)</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Mathematical Formulation Breakdown Table */}
              {optimizationResult?.phase_timings && (
                <div style={{ marginTop: 24, borderTop: "1px solid var(--border-color)", paddingTop: 18 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                    <FileText size={16} color="#06b6d4" /> Webster's Mathematical Calculation (Derived From Your Uploaded Photos)
                  </h4>
                  <div style={{ overflowX: "auto" }}>
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Approach</th>
                          <th>Photo PCU</th>
                          <th>Flow q (PCU/h)</th>
                          <th>Flow Ratio (y = q/s)</th>
                          <th>Allocated Green Time (G₁)</th>
                          <th>Amber Clearance</th>
                          <th>Cycle Ratio</th>
                        </tr>
                      </thead>
                      <tbody>
                        {optimizationResult.phase_timings.map((p) => (
                          <tr key={p.approach}>
                            <td style={{ fontWeight: 600 }}>{p.approach}</td>
                            <td>{p.pcu}</td>
                            <td>{p.flow_q_pcu_hr || "N/A"}</td>
                            <td style={{ fontFamily: "monospace", color: "#38bdf8" }}>{p.flow_ratio_y || "–"}</td>
                            <td style={{ fontWeight: 700, color: "#34d399", fontSize: 14 }}>{p.green_time}s</td>
                            <td>{p.yellow_time}s</td>
                            <td>{p.green_ratio ? `${(p.green_ratio * 100).toFixed(1)}%` : "–"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 2: ROAD DAMAGE PRIORITIZATION */}
        {/* ============================================================== */}
        {activeTab === "road_damage" && (
          <div className="glass-card">
            <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
              <Layers size={20} color="#f59e0b" /> Municipal Road Infrastructure Pothole & Defect Prioritization
            </h3>
            <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>
              Prioritizes city roads based on AI defect detection (potholes/cracks) and segment traffic importance.
            </p>

            <div style={{ overflowX: "auto" }}>
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Segment Name</th>
                    <th>Traffic Importance</th>
                    <th>Damage Score</th>
                    <th>Defects</th>
                    <th>Priority Score</th>
                    <th>Urgency</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedRepairs.map((seg) => (
                    <tr key={seg.id}>
                      <td style={{ fontWeight: 600 }}>{seg.name}</td>
                      <td><span className="badge badge-purple">{seg.traffic_importance}x</span></td>
                      <td>{seg.damage_score}</td>
                      <td>{seg.defect_counts.pothole || 0} Potholes</td>
                      <td style={{ fontWeight: 800, fontFamily: "monospace" }}>{seg.priority_score}</td>
                      <td>
                        <span className={`badge ${seg.priority_level === "CRITICAL" ? "badge-rose" : "badge-amber"}`}>
                          {seg.priority_level}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn btn-primary"
                          style={{ fontSize: 11, padding: "4px 10px" }}
                          onClick={() => alert(`Maintenance Dispatched for ${seg.name}!`)}
                        >
                          Dispatch Crew
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 3: BENCHMARKS */}
        {/* ============================================================== */}
        {activeTab === "metrics" && (
          <div className="glass-card">
            <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
              <Gauge size={20} color="#10b981" /> YOLOv8 & ML Model Benchmarks
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: 14 }}>
                <h4 style={{ color: "#38bdf8", fontWeight: 700, marginBottom: 6 }}>Vehicle Detector (KITTI YOLOv8)</h4>
                <div>mAP@50: <b>80.1%</b></div>
                <div>GPU Latency: <b>7.7 ms (~130 FPS)</b></div>
                <div>CPU Latency: <b>13.5 ms (~74 FPS)</b></div>
              </div>
              <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: 14 }}>
                <h4 style={{ color: "#fbbf24", fontWeight: 700, marginBottom: 6 }}>Road Damage (RDD2022)</h4>
                <div>Overall mAP50: <b>36.2%</b></div>
                <div>Pothole mAP50: <b>39.4%</b></div>
              </div>
              <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: 14 }}>
                <h4 style={{ color: "#a855f7", fontWeight: 700, marginBottom: 6 }}>Congestion Estimator (Random Forest)</h4>
                <div>Accuracy: <b>88.7%</b></div>
                <div>Inference Time: <b>&lt; 1 ms</b></div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-color)", padding: "16px 24px", textAlign: "center", fontSize: 12, color: "var(--text-muted)" }}>
        Smart India Hackathon (SIH 2025) • Dynamic Traffic Light Management & Urban Mobility
      </footer>
    </div>
  );
}
