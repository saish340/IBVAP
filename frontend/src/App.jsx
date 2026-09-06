import { useEffect, useMemo, useState } from "react";

const API = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function severityStyle(severity) {
  if (severity === "critical" || severity === "intrusion") return "border-red-500/40 bg-red-500/10 text-red-200";
  if (severity === "warning" || severity === "face_match") return "border-amber-400/40 bg-amber-400/10 text-amber-100";
  return "border-slate-700 bg-slate-900 text-slate-300";
}

function reliabilityText(alert) {
  if (alert.detection_reliability_score == null) return null;
  const score = `${Math.round(alert.detection_reliability_score * 100)}% confidence`;
  const condition = alert.detection_condition && alert.detection_condition !== "CLEAR"
    ? `, ${alert.detection_condition.toLowerCase().replaceAll("_", " ")}`
    : "";
  return `${score}${condition}`;
}

function Panel({ title, action, children, className = "" }) {
  return (
    <section className={`border border-slate-800 bg-slate-950/70 ${className}`}>
      <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
        <h2 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-400">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [streams, setStreams] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState({ items: [], total: 0 });
  const [watchlist, setWatchlist] = useState([]);
  const [name, setName] = useState("");
  const [image, setImage] = useState("");
  const [message, setMessage] = useState("");
  const [enrolling, setEnrolling] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [degradation, setDegradation] = useState(null);
  const [adaptive, setAdaptive] = useState(null);
  const selected = useMemo(() => streams.find((stream) => stream.id === selectedId), [streams, selectedId]);

  async function refresh() {
    try {
      const [service, streamData, eventData, people] = await Promise.all([
        request("/health"), request("/streams"), request("/events/history?page_size=12"), request("/watchlist"),
      ]);
      setHealth(service);
      setStreams(streamData);
      setHistory(eventData);
      setWatchlist(people);
      setSelectedId((current) => current ?? streamData.find((stream) => stream.running)?.id ?? streamData[0]?.id ?? null);
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedId) return undefined;
    let active = true;
    const poll = async () => {
      try {
        const result = await request(`/analytics/streams/${selectedId}/results`);
        if (active) {
          setAnalysis(result.results ?? null);
          setDegradation(result.degradation ?? null);
          setAdaptive(result.adaptive ?? null);
        }
      } catch {
        if (active) setAnalysis(null);
      }
    };
    poll();
    const timer = setInterval(poll, 2000);
    return () => { active = false; clearInterval(timer); };
  }, [selectedId]);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    let socket;
    let retryTimer;
    let active = true;
    const connect = () => {
      if (!active) return;
      socket = new WebSocket(`${protocol}://${window.location.host}/ws/alerts`);
      socket.onmessage = (event) => {
        const alert = JSON.parse(event.data);
        setAlerts((current) => [alert, ...current].slice(0, 30));
        setHistory((current) => ({ ...current, items: [alert, ...current.items].slice(0, 12), total: current.total + 1 }));
      };
      socket.onclose = () => { if (active) retryTimer = setTimeout(connect, 2000); };
    };
    connect();
    return () => { active = false; clearTimeout(retryTimer); socket?.close(); };
  }, []);

  async function enroll(event) {
    event.preventDefault();
    if (!name.trim()) {
      setMessage("Enter a person name first.");
      return;
    }
    if (!image) {
      setMessage("Choose a face photo first.");
      return;
    }
    setEnrolling(true);
    setMessage("Processing face with RetinaFace and ArcFace...");
    try {
      await request("/watchlist/enroll", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: name.trim(), image_base64: image }) });
      setMessage(`${name} added to watchlist`);
      setName(""); setImage(""); refresh();
    } catch (error) {
      setMessage(`Enrollment failed: ${error.message}`);
    } finally {
      setEnrolling(false);
    }
  }

  function selectImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImage(reader.result);
    reader.readAsDataURL(file);
  }

  const frameUrl = selected ? `${API}/streams/${selected.id}/mjpeg` : "";
  const online = health?.status === "ok";

  return (
    <div className="min-h-screen bg-[#071014] text-slate-100">
      <header className="border-b border-cyan-950/70 bg-[#0a171c] px-5 py-4 md:px-8">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between">
          <div><p className="text-xs font-bold tracking-[0.35em] text-cyan-400">IBVAP / CONTROL ROOM</p><h1 className="mt-1 text-2xl font-semibold tracking-tight">Live video intelligence</h1></div>
          <div className="flex items-center gap-3 text-xs uppercase tracking-widest text-slate-400"><span className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-red-400"}`} />{online ? "System online" : "Backend offline"}</div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1500px] gap-5 p-5 md:p-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.75fr)]">
        <div className="space-y-5">
          <Panel title="Live camera" action={selected && <span className="text-xs text-slate-500">{selected.name} / {selected.running ? "streaming" : "stopped"}</span>}>
            <div className="aspect-video bg-black">
              {frameUrl ? <img key={selected.id} src={frameUrl} className="h-full w-full object-contain" alt="Live camera stream" /> : <div className="flex h-full items-center justify-center text-sm text-slate-600">Register a stream to begin</div>}
            </div>
            <div className="flex flex-wrap gap-2 border-t border-slate-800 p-4">{streams.map((stream) => <button key={stream.id} onClick={() => setSelectedId(stream.id)} className={`border px-3 py-2 text-left text-xs ${selectedId === stream.id ? "border-cyan-400 bg-cyan-400/10 text-cyan-200" : "border-slate-700 text-slate-400"}`}><span className="font-semibold">{stream.name}</span><span className="ml-2 text-slate-600">{stream.running ? "LIVE" : "OFFLINE"}</span></button>)}</div>
            {(analysis || degradation) && <div className="grid gap-2 border-t border-slate-800 p-4 text-xs sm:grid-cols-4">{degradation && <div className="border border-emerald-900 bg-emerald-400/5 p-3"><p className="uppercase tracking-widest text-emerald-500">Condition monitor</p><p className="mt-2 text-lg text-emerald-200">{degradation.condition}</p><p className="text-slate-400">severity {(degradation.severity * 100).toFixed(0)}%</p></div>}{adaptive && <div className="border border-violet-900 bg-violet-400/5 p-3"><p className="uppercase tracking-widest text-violet-400">Adaptive layer</p><p className="mt-2 text-lg text-violet-200">{adaptive.mode.replaceAll("_", " ")}</p><p className="text-slate-400">reliability {(adaptive.reliability_score * 100).toFixed(0)}% / fence {adaptive.fence_consensus_frames} frames</p></div>}{analysis?.tracking && <div className="border border-cyan-900 bg-cyan-400/5 p-3"><p className="uppercase tracking-widest text-cyan-500">Tracking</p><p className="mt-2 text-lg text-cyan-200">{analysis.tracking.count} active</p>{analysis.tracking.tracks?.map((track) => <p key={`${track.track_id}-${track.bbox?.join("-")}`} className="text-slate-400">{track.class} / ID {track.track_id} / {(track.confidence * 100).toFixed(0)}%</p>)}</div>}{analysis?.face_verification && <div className="border border-amber-900 bg-amber-400/5 p-3"><p className="uppercase tracking-widest text-amber-500">Faces</p><p className="mt-2 text-lg text-amber-200">{analysis.face_verification.count} detected</p>{analysis.face_verification.faces?.map((face, index) => <p key={`${face.name}-${index}`} className="text-slate-400">{face.name} / {(face.confidence * 100).toFixed(0)}%</p>)}</div>}{analysis?.anpr && <div className="border border-slate-700 bg-slate-900 p-3"><p className="uppercase tracking-widest text-slate-500">ANPR</p><p className="mt-2 text-lg text-slate-200">{analysis.anpr.count} plates</p></div>}</div>}
          </Panel>

          <Panel title="Event history" action={<span className="text-xs text-slate-500">{history.total} recorded</span>}>
            <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-xs uppercase tracking-wider text-slate-600"><tr><th className="px-5 py-3">Time</th><th className="px-5 py-3">Module</th><th className="px-5 py-3">Event</th><th className="px-5 py-3">Reliability</th><th className="px-5 py-3">Camera</th></tr></thead><tbody>{history.items.map((event) => <tr key={event.id} className="border-t border-slate-900"><td className="whitespace-nowrap px-5 py-3 text-xs text-slate-500">{event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : "-"}</td><td className="px-5 py-3"><span className={`border px-2 py-1 text-xs ${severityStyle(event.severity)}`}>{event.module}</span></td><td className="max-w-[420px] truncate px-5 py-3 text-slate-300">{event.message}</td><td className="whitespace-nowrap px-5 py-3 text-xs text-slate-400">{reliabilityText(event) || "-"}</td><td className="px-5 py-3 text-xs text-slate-500">{event.camera_id || "-"}</td></tr>)}</tbody></table>{history.items.length === 0 && <p className="px-5 py-8 text-sm text-slate-600">No events recorded.</p>}</div>
          </Panel>
        </div>

        <aside className="space-y-5">
          <Panel title="Realtime alerts" action={<span className="text-xs text-cyan-500">WS /alerts</span>}>
            <div className="max-h-[360px] space-y-2 overflow-y-auto p-4">{alerts.length === 0 ? <p className="py-8 text-sm text-slate-600">Listening for alerts...</p> : alerts.map((alert) => <article key={alert.id} className={`border p-3 ${severityStyle(alert.severity)}`}><div className="flex justify-between gap-3 text-xs uppercase tracking-wider"><span>{alert.module}</span><time>{alert.timestamp ? new Date(alert.timestamp * 1000).toLocaleTimeString() : "now"}</time></div><p className="mt-2 text-sm">{alert.message}</p>{reliabilityText(alert) && <p className="mt-2 text-xs font-medium uppercase tracking-wider text-cyan-300">{reliabilityText(alert)}</p>}</article>)}</div>
          </Panel>

          <Panel title="Watchlist enrollment">
            <form onSubmit={enroll} className="space-y-3 p-5"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Person name" className="w-full border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400" /><input type="file" accept="image/*" onChange={selectImage} className="w-full text-xs text-slate-400 file:mr-3 file:border-0 file:bg-cyan-500 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-slate-950" /><button disabled={enrolling} className="w-full bg-cyan-400 px-3 py-2 text-sm font-bold text-slate-950 hover:bg-cyan-300 disabled:cursor-wait disabled:opacity-50">{enrolling ? "Processing..." : "Enroll face"}</button>{message && <p role="status" className="text-xs text-cyan-300">{message}</p>}</form>
            <div className="border-t border-slate-800 px-5 py-4">{watchlist.length === 0 ? <p className="text-sm text-slate-600">No enrolled people.</p> : <ul className="space-y-2">{watchlist.map((person) => <li key={person.id} className="flex justify-between text-sm"><span>{person.name}</span><span className="text-xs text-slate-600">{person.model}</span></li>)}</ul>}</div>
          </Panel>

          <Panel title="Streams"><div className="divide-y divide-slate-900">{streams.length === 0 ? <p className="p-5 text-sm text-slate-600">No cameras registered.</p> : streams.map((stream) => <div key={stream.id} className="flex items-center justify-between p-4"><div><p className="text-sm font-medium">{stream.name}</p><p className="mt-1 max-w-[230px] truncate text-xs text-slate-600">{stream.source_url}</p></div><span className={`text-xs ${stream.running ? "text-emerald-400" : "text-slate-600"}`}>{stream.running ? "RUNNING" : "STOPPED"}</span></div>)}</div></Panel>
        </aside>
      </main>
    </div>
  );
}
