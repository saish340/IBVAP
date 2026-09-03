import { useEffect, useState } from "react";

const API_BASE = "/api";

async function getJSON(path, fallback) {
  try {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch {
    return fallback;
  }
}

const CAPABILITY_INFO = {
  detection: ["🎯", "Object detection", "YOLO (Ultralytics)"],
  tracking: ["🧭", "Multi-object tracking", "YOLO + ByteTrack"],
  face_recognition: ["🧑", "Face recognition", "DeepFace"],
  ocr: ["🔤", "Text recognition", "PaddleOCR"],
  pose: ["🤸", "Pose estimation", "MediaPipe"],
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [streams, setStreams] = useState(null);
  const [capabilities, setCapabilities] = useState([]);

  useEffect(() => {
    getJSON("/health", null).then(setHealth);
    getJSON("/streams", []).then(setStreams);
    getJSON("/analytics/capabilities", { capabilities: [] }).then((data) =>
      setCapabilities(data.capabilities ?? []),
    );
  }, []);

  const online = health?.status === "ok";
  const degraded = health?.status === "degraded";
  const badge = online
    ? "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30"
    : degraded
      ? "bg-amber-500/10 text-amber-400 ring-amber-500/30"
      : "bg-rose-500/10 text-rose-400 ring-rose-500/30";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800/60 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">
            IBVAP{" "}
            <span className="ml-2 hidden text-sm font-normal text-slate-400 sm:inline">
              Real-Time Video Analytics Platform
            </span>
          </h1>
          <span className={`rounded-full px-3 py-1 text-xs font-medium ring-1 ${badge}`}>
            {health ? `backend: ${health.status}` : "backend: offline"}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-8 p-6">
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Streams
            </h2>
            <code className="rounded bg-slate-900 px-2 py-1 text-xs text-slate-400">
              POST /api/streams
            </code>
          </div>
          {streams === null ? (
            <p className="text-sm text-slate-500">Loading…</p>
          ) : streams.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">
              <p>No streams yet.</p>
              <p className="mt-2">
                Add one via <code className="text-slate-300">POST /api/streams</code> — see the
                README for an example curl command.
              </p>
            </div>
          ) : (
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {streams.map((stream) => (
                <li
                  key={stream.id}
                  className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{stream.name}</span>
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        stream.running ? "bg-emerald-400" : "bg-slate-600"
                      }`}
                      title={stream.running ? "analyzing" : "stopped"}
                    />
                  </div>
                  <p className="mt-1 truncate text-xs text-slate-500" title={stream.source_url}>
                    {stream.source_url}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-1">
                    {stream.capabilities.map((cap) => (
                      <span
                        key={cap}
                        className="rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300"
                      >
                        {CAPABILITY_INFO[cap]?.[0] ?? "•"} {cap}
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">
            Inference capabilities
          </h2>
          {capabilities.length === 0 ? (
            <p className="text-sm text-slate-500">Backend unreachable — is it running on :8000?</p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {capabilities.map((cap) => {
                const [icon, title, stack] = CAPABILITY_INFO[cap] ?? ["•", cap, ""];
                return (
                  <div
                    key={cap}
                    className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"
                  >
                    <div className="text-2xl">{icon}</div>
                    <div className="mt-2 font-medium">{title}</div>
                    <div className="text-xs text-slate-500">{stack}</div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <p className="pt-4 text-center text-xs text-slate-600">
          Placeholder dashboard — wire up live video (WebSocket /ws/streams/&#123;id&#125;) and
          result rendering next.
        </p>
      </main>
    </div>
  );
}
