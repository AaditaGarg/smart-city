import React, { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

// Plays simulation frames (one frame = 2 simulated seconds) at a chosen speed.
export function usePlayer(length, { autoplay = true } = {}) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(15);
  const last = useRef(0);

  useEffect(() => {
    setI(0);
    setPlaying(autoplay && length > 0);
  }, [length, autoplay]);

  useEffect(() => {
    if (!playing || !length) return undefined;
    // A timer, not requestAnimationFrame: animation frames stop when the tab is in the background, and a
    // control-room feed must keep its clock. Elapsed wall time is used, so throttled timers catch up.
    const step = 1000 / speed;
    last.current = performance.now();
    const tick = () => {
      const now = performance.now();
      const n = Math.floor((now - last.current) / step);
      if (n > 0) {
        last.current += n * step;
        setI((v) => {
          if (v + n >= length - 1) {
            setPlaying(false);
            return length - 1;
          }
          return v + n;
        });
      }
    };
    const id = setInterval(tick, Math.min(100, step));
    return () => clearInterval(id);
  }, [playing, speed, length]);

  return { i, setI, playing, setPlaying, speed, setSpeed };
}

export default function PlayerBar({ player, length, t }) {
  const { i, setI, playing, setPlaying, speed, setSpeed } = player;
  return (
    <div className="player">
      <button className="icon-btn" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"} disabled={!length}>
        {playing ? <Pause size={18} /> : <Play size={18} />}
      </button>
      <button className="icon-btn" onClick={() => { setI(0); setPlaying(true); }} aria-label="Restart" disabled={!length}>
        <RotateCcw size={16} />
      </button>
      <input
        type="range" min={0} max={Math.max(0, length - 1)} value={i}
        onChange={(e) => setI(Number(e.target.value))} aria-label="Simulation time" className="scrub"
      />
      <button className="sc-jump" onClick={() => { setI(Math.max(0, length - 1)); setPlaying(false); }} disabled={!length}>
        Jump to end
      </button>
      <span className="clock">{String(Math.floor(t / 60)).padStart(2, "0")}:{String(t % 60).padStart(2, "0")}</span>
      <label className="speed">
        Speed
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          {[4, 8, 15, 30, 60].map((s) => (
            <option key={s} value={s}>{s * 2}×</option>
          ))}
        </select>
      </label>
    </div>
  );
}
