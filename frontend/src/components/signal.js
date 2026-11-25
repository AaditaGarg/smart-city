// Shared, data-faithful helpers for drawing the corridor.

export const JAM_SPACING_M = 7.0;   // decision_engine/network_model.py: metres of lane per queued PCU
export const MAX_GREEN_S = 60;      // decision_engine/network_control.py

// What one approach's signal head shows right now: the round lamp and the right-turn arrow.
export function headState(junction, sig, approach) {
  if (!sig) return { ball: "red", arrow: false };
  const phase = junction.phases[sig.phase];
  const moves = phase ? phase.movements.filter((m) => m[0] === approach).map((m) => m[1]) : [];
  const through = moves.includes("S") || moves.includes("L");
  const right = moves.includes("R");
  if (sig.state === "allred" || !moves.length) return { ball: "red", arrow: false };
  if (sig.state === "yellow") return { ball: through ? "amber" : "red", arrow: false, arrowAmber: right && !through };
  return { ball: through ? "green" : "red", arrow: right };
}

// Seconds left in the current green, only when the controller has committed to a time.
// Returns {left, firm}: firm=false means "at least this long" (minimum green may be extended).
export function greenLeft(sig) {
  if (!sig || sig.state !== "green") return null;
  let m = /^clearing queue \((\d+)\/(\d+)s\)$/.exec(sig.reason || "");
  if (m) return { left: Math.max(0, Number(m[2]) - Number(m[1])), firm: true };
  m = /^minimum green \((\d+)\/(\d+)s\)$/.exec(sig.reason || "");
  if (m) return { left: Math.max(0, Number(m[2]) - Number(m[1])), firm: false };
  return null;
}

// Physical queue on a link: metres of road it occupies, share of the link, and whether it spills back.
export function queueOf(link, frame) {
  const veh = frame?.queues?.[link.id] ?? 0;
  const pcu = frame?.queue_pcu?.[link.id] ?? veh;   // older backends: assume ~1 PCU per vehicle
  const metres = (pcu * JAM_SPACING_M) / Math.max(1, link.lanes);
  const share = link.storage_pcu ? pcu / link.storage_pcu : 0;
  return { veh, pcu, metres, share, spill: share >= 0.9 };
}

export const SIDE = { W: "ITO side", E: "Laxmi Nagar side", N: "north", S: "south" };
export const signalName = (jid) => jid.replace("J", "Signal ");
