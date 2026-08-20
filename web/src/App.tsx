import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  Check,
  ArrowLeftRight,
  ArrowDownUp,
  Bot,
  Box,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  CircleDot,
  Clock3,
  Command,
  Copy,
  Gauge,
  Image as ImageIcon,
  LoaderCircle,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Search,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";
import { AuditEventDrawer } from "./components/AuditEventDrawer";
import { StageTimeline, type TimelineGroup } from "./components/StageTimeline";
import { eventNeedsAttention } from "./audit";
import type {
  Action,
  AuditEvent,
  Diff,
  Render,
  Run,
  Scene,
  SceneDetail,
  TimedEvent,
} from "./types";

const API_REFRESH_MS = 5000;
type SortOrder = "asc" | "desc";
type View = "review" | "diff" | "actions";
const EVENT_FILTER_OPTIONS = ["all", "attention", "contract", "llm", "benchmark", "tool", "repair", "orchestration", "system"] as const;
type EventFilter = (typeof EVENT_FILTER_OPTIONS)[number];

function queryValue(name: string): string {
  return new URLSearchParams(window.location.search).get(name) ?? "";
}

function initialView(): View {
  const value = queryValue("view");
  return value === "diff" || value === "actions" ? value : "review";
}

function initialSortOrder(): SortOrder {
  return queryValue("order") === "desc" ? "desc" : "asc";
}

function initialEventFilter(): EventFilter {
  const value = queryValue("kind");
  return EVENT_FILTER_OPTIONS.includes(value as EventFilter) ? value as EventFilter : "all";
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json() as Promise<T>;
}

function imageUrl(path: string | null): string | null {
  return path ? `/api/image?path=${encodeURIComponent(path)}` : null;
}

function formatIdentifier(value: string): string {
  return value.replace(/^_+|_impl$/g, "").replaceAll("_", " ");
}

function eventTime(value?: string): number {
  const timestamp = new Date(value ?? "").valueOf();
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function eventAtCheckpoint(
  event: AuditEvent,
  previousCheckpointTime: number,
  checkpointTime: number,
): AuditEvent | null {
  const completedTime = eventTime(event.created_at);
  const canBeActiveAtCheckpoint = event.kind === "llm" && Boolean(event.started_at);
  const startedTime = canBeActiveAtCheckpoint
    ? eventTime(event.started_at)
    : completedTime;
  if (completedTime <= previousCheckpointTime || startedTime > checkpointTime) return null;
  return canBeActiveAtCheckpoint && completedTime > checkpointTime
    ? { ...event, checkpoint_state: "active" }
    : event;
}

function sortCheckpointEvents(events: AuditEvent[], order: SortOrder): AuditEvent[] {
  const direction = order === "asc" ? 1 : -1;
  return [...events].sort((left, right) => {
    const leftTime = eventTime(left.checkpoint_state === "active" ? left.started_at : left.created_at);
    const rightTime = eventTime(right.checkpoint_state === "active" ? right.started_at : right.created_at);
    return direction * (leftTime - rightTime);
  });
}

function sortByTime<T extends { created_at?: string }>(values: T[], order: SortOrder): T[] {
  const direction = order === "asc" ? 1 : -1;
  return [...values].sort((left, right) => direction * (eventTime(left.created_at) - eventTime(right.created_at)));
}

function renderSequence(render: Render): number | null {
  const match = /\/renders_(\d+)$/.exec(render.id);
  return match ? Number(match[1]) : null;
}

function sortRenders(values: Render[], order: SortOrder): Render[] {
  const stages = new Map<string, Render[]>();
  for (const render of values) {
    const stageRenders = stages.get(render.stage) ?? [];
    stageRenders.push(render);
    stages.set(render.stage, stageRenders);
  }

  const orderedStages = [...stages.values()].sort(
    (left, right) => Math.min(...left.map((render) => eventTime(render.created_at)))
      - Math.min(...right.map((render) => eventTime(render.created_at))),
  );
  const chronological = orderedStages.flatMap((stageRenders) => [...stageRenders].sort((left, right) => {
    const leftSequence = renderSequence(left);
    const rightSequence = renderSequence(right);
    if (leftSequence !== null && rightSequence !== null && leftSequence !== rightSequence) {
      return leftSequence - rightSequence;
    }
    return eventTime(left.created_at) - eventTime(right.created_at);
  }));
  return order === "asc" ? chronological : chronological.reverse();
}

function actionStage(action: Action): string {
  const name = action.tool_name.toLowerCase();
  if (name.includes("manipuland")) return "manipuland";
  if (name.includes("ceiling")) return "ceiling_mounted";
  if (name.includes("wall")) return "wall_mounted";
  return "furniture";
}

function actionAuditEvents(actions: Action[]): AuditEvent[] {
  return actions.map((action) => ({
    id: `tool-action:${action.step_number}`,
    kind: "tool",
    source: "action_log",
    created_at: action.timestamp,
    stage: actionStage(action),
    actor: "designer",
    function: action.tool_name,
    title: formatIdentifier(action.tool_name),
    audit_status: "tool_action",
    detail: { step_number: action.step_number, arguments: action.arguments },
  }));
}

function formatTime(value?: string): string {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value.replace("T", " ").slice(0, 19) : date.toLocaleString();
}

function legacyAuditEvents(detail: SceneDetail | null): AuditEvent[] {
  if (!detail) return [];
  const systemEvents = detail.timings.map((event: TimedEvent, index) => ({
    id: `legacy-timing:${index}`,
    kind: "system" as const,
    source: "timing",
    created_at: event.created_at,
    stage: String(event.stage ?? "system"),
    actor: String(event.module ?? "system"),
    function: String(event.event ?? "pipeline_event"),
    title: String(event.event ?? "pipeline event").replaceAll("_", " "),
    elapsed_sec: event.elapsed_sec,
    audit_status: "timing_only",
    detail: event.detail,
  }));
  const llmEvents = detail.llm_calls.map((event: TimedEvent, index) => ({
    id: `legacy-llm:${index}`,
    kind: "llm" as const,
    source: "llm",
    created_at: event.created_at,
    stage: String(event.stage ?? "unknown"),
    actor: String(event.agent_role ?? "LLM"),
    function: String(event.event ?? "llm_call"),
    title: String(event.event ?? "llm call").replaceAll("_", " "),
    elapsed_sec: event.elapsed_sec,
    audit_status: "excerpt_only",
    token_usage: typeof event.token_usage === "object" && event.token_usage !== null
      ? event.token_usage as Record<string, number>
      : undefined,
    prompt_chars: typeof event.prompt_chars === "number" ? event.prompt_chars : undefined,
    output_chars: typeof event.output_chars === "number" ? event.output_chars : undefined,
    detail: {
      prompt_excerpt: event.prompt_excerpt,
      output_excerpt: event.output_excerpt,
    },
  }));
  return [...systemEvents, ...llmEvents];
}

function eventSearchText(event: AuditEvent): string {
  return [
    event.title,
    event.function,
    event.actor,
    event.stage,
    event.kind,
    event.audit_status,
    event.detail ? JSON.stringify(event.detail) : "",
    event.metrics ? JSON.stringify(event.metrics) : "",
    event.evaluation ? JSON.stringify(event.evaluation) : "",
    event.repair ? JSON.stringify(event.repair) : "",
    event.contract ? JSON.stringify(event.contract) : "",
  ].join(" ").toLowerCase();
}

function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [selectedRun, setSelectedRun] = useState(() => queryValue("run"));
  const [selectedScene, setSelectedScene] = useState(() => queryValue("scene"));
  const [detail, setDetail] = useState<SceneDetail | null>(null);
  const [selectedRender, setSelectedRender] = useState(() => queryValue("render"));
  const [comparisonRender, setComparisonRender] = useState(() => queryValue("compare"));
  const [diff, setDiff] = useState<Diff | null>(null);
  const [drawerEvent, setDrawerEvent] = useState<AuditEvent | Action | null>(null);
  const [search, setSearch] = useState("");
  const [eventSearch, setEventSearch] = useState(() => queryValue("q"));
  const deferredEventSearch = useDeferredValue(eventSearch);
  const [stageFilter, setStageFilter] = useState(() => queryValue("stage") || "all");
  const [eventFilter, setEventFilter] = useState<EventFilter>(initialEventFilter);
  const [sortOrder, setSortOrder] = useState<SortOrder>(initialSortOrder);
  const [view, setView] = useState<View>(initialView);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [linkCopied, setLinkCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const loadRuns = async () => {
      try {
        const payload = await getJson<{ runs: Run[] }>("/api/runs");
        if (cancelled) return;
        setRuns(payload.runs);
        setSelectedRun((current) => payload.runs.some((run) => run.id === current) ? current : payload.runs[0]?.id || "");
        setError("");
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load probe runs");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void loadRuns();
    const timer = window.setInterval(() => void loadRuns(), API_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    let cancelled = false;
    const loadScenes = async () => {
      try {
        const payload = await getJson<{ scenes: Scene[] }>(`/api/runs/${encodeURIComponent(selectedRun)}/scenes`);
        if (cancelled) return;
        setScenes(payload.scenes);
        setSelectedScene((current) => payload.scenes.some((scene) => scene.path === current) ? current : payload.scenes[0]?.path || "");
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load scenes");
      }
    };
    void loadScenes();
    const timer = window.setInterval(() => void loadScenes(), API_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedScene) return;
    let cancelled = false;
    const loadDetail = async () => {
      try {
        const payload = await getJson<SceneDetail>(`/api/scene?path=${encodeURIComponent(selectedScene)}`);
        if (cancelled) return;
        setDetail(payload);
        const chronological = sortRenders(payload.renders, "asc");
        setSelectedRender((current) => payload.renders.some((render) => render.id === current) ? current : chronological.at(-1)?.id || "");
        setComparisonRender((current) => payload.renders.some((render) => render.id === current) ? current : chronological.at(-2)?.id || "");
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load scene details");
      }
    };
    void loadDetail();
    const timer = window.setInterval(() => void loadDetail(), API_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedScene]);

  const activeScene = scenes.find((scene) => scene.path === selectedScene);
  const chronologicalRenders = useMemo(() => sortRenders(detail?.renders ?? [], "asc"), [detail]);
  const renders = useMemo(() => sortRenders(chronologicalRenders, sortOrder), [chronologicalRenders, sortOrder]);
  const currentRender = renders.find((render) => render.id === selectedRender) ?? renders[0];
  const beforeRender = renders.find((render) => render.id === comparisonRender);
  const allEvents = useMemo(() => {
    const auditEvents = detail?.audit_events?.length ? detail.audit_events : legacyAuditEvents(detail);
    const values = [...auditEvents, ...actionAuditEvents(detail?.actions ?? [])];
    return sortByTime(values, "asc");
  }, [detail]);
  const stages = useMemo(() => ["all", ...new Set(allEvents.map((event) => event.stage || "system"))], [allEvents]);
  const attentionCount = useMemo(() => allEvents.filter(eventNeedsAttention).length, [allEvents]);
  const visibleEvents = useMemo(
    () => sortByTime(allEvents.filter((event) => {
      const matchesStage = stageFilter === "all" || event.stage === stageFilter;
      const matchesType = eventFilter === "all"
        || (eventFilter === "attention" ? eventNeedsAttention(event) : event.kind === eventFilter);
      const matchesSearch = !deferredEventSearch.trim() || eventSearchText(event).includes(deferredEventSearch.trim().toLowerCase());
      return matchesStage && matchesType && matchesSearch;
    }), sortOrder),
    [allEvents, deferredEventSearch, eventFilter, sortOrder, stageFilter],
  );
  const visibleScenes = scenes.filter((scene) => `${scene.room} ${scene.batch} ${scene.scene}`.toLowerCase().includes(search.trim().toLowerCase()));
  const visibleActions = useMemo(() => {
    const visibleActionIds = new Set(visibleEvents
      .filter((event) => event.id.startsWith("tool-action:"))
      .map((event) => event.id));
    return (detail?.actions ?? []).filter((action) => visibleActionIds.has(`tool-action:${action.step_number}`));
  }, [detail?.actions, visibleEvents]);
  const timelineGroups = useMemo<TimelineGroup[]>(() => {
    if (!chronologicalRenders.length) return [{ id: "unassigned", events: visibleEvents }];
    const groups: TimelineGroup[] = chronologicalRenders.map((render, index) => {
      const checkpointTime = eventTime(render.created_at);
      const previousTime = index > 0
        ? eventTime(chronologicalRenders[index - 1].created_at)
        : Number.NEGATIVE_INFINITY;
      const checkpointEvents = visibleEvents.flatMap((event) => {
        const checkpointEvent = eventAtCheckpoint(event, previousTime, checkpointTime);
        return checkpointEvent ? [checkpointEvent] : [];
      });
      return {
        id: render.id,
        render,
        events: sortCheckpointEvents(checkpointEvents, sortOrder),
      };
    });
    const newestCheckpointTime = eventTime(chronologicalRenders.at(-1)?.created_at);
    const trailingEvents = visibleEvents.filter(
      (event) => eventTime(event.created_at) > newestCheckpointTime,
    );
    if (trailingEvents.length) groups.push({ id: "after-latest", events: trailingEvents });
    return sortOrder === "asc" ? groups : groups.reverse();
  }, [chronologicalRenders, sortOrder, visibleEvents]);

  const timelineEvents = visibleEvents;

  useEffect(() => {
    setStageFilter((current) => stages.includes(current) ? current : "all");
  }, [stages]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        event.key !== "/"
        || event.metaKey
        || event.ctrlKey
        || event.altKey
        || target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
      ) return;
      event.preventDefault();
      document.getElementById("event-search")?.focus();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedRun) params.set("run", selectedRun);
    if (selectedScene) params.set("scene", selectedScene);
    if (selectedRender) params.set("render", selectedRender);
    if (comparisonRender) params.set("compare", comparisonRender);
    if (view !== "review") params.set("view", view);
    if (stageFilter !== "all") params.set("stage", stageFilter);
    if (eventFilter !== "all") params.set("kind", eventFilter);
    if (eventSearch.trim()) params.set("q", eventSearch.trim());
    if (sortOrder !== "asc") params.set("order", sortOrder);
    const nextLocation = `${window.location.pathname}${params.size ? `?${params.toString()}` : ""}`;
    if (nextLocation !== `${window.location.pathname}${window.location.search}`) {
      window.history.replaceState(null, "", nextLocation);
    }
  }, [comparisonRender, eventFilter, eventSearch, selectedRender, selectedRun, selectedScene, sortOrder, stageFilter, view]);

  useEffect(() => {
    if (view !== "review" || !selectedRender) return;
    const frame = window.requestAnimationFrame(() => {
      const group = document.getElementById(`timeline-snapshot-${selectedRender}`);
      const timeline = group?.closest(".timeline");
      if (group && timeline) {
        timeline.scrollTo({ top: (group as HTMLElement).offsetTop - (timeline as HTMLElement).offsetTop - 8, behavior: "smooth" });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [selectedRender, view]);

  useEffect(() => {
    if (!beforeRender || !currentRender || !beforeRender.state_path || !currentRender.state_path || beforeRender.state_path === currentRender.state_path) {
      setDiff(null);
      return;
    }
    let cancelled = false;
    void getJson<Diff>(`/api/diff?before=${encodeURIComponent(beforeRender.state_path)}&after=${encodeURIComponent(currentRender.state_path)}`)
      .then((payload) => !cancelled && setDiff(payload))
      .catch(() => !cancelled && setDiff(null));
    return () => { cancelled = true; };
  }, [beforeRender, currentRender]);

  const refresh = () => {
    setSelectedRun("");
    setLoading(true);
    window.location.reload();
  };

  const copyAuditLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1800);
    } catch {
      setError("Unable to copy the current audit link.");
    }
  };

  return (
    <main className="app-shell">
      <a className="skip-link" href="#audit-workspace">Skip to audit workspace</a>
      <aside className={`sidebar ${sidebarOpen ? "is-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={17} /></div>
          <div><strong>Critic Probe</strong><span>SceneSmith observability</span></div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <div className="sidebar-label">Probe runs</div>
        <div className="run-list">
          {runs.map((run) => (
            <button key={run.id} className={`run-item ${run.id === selectedRun ? "active" : ""}`} onClick={() => setSelectedRun(run.id)}>
              <span className={`status-dot ${run.status}`} />
              <span className="run-name">{run.id}</span>
              <span className="run-count">{run.scene_count}</span>
            </button>
          ))}
        </div>
        <div className="scene-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter scenes" /></div>
        <div className="sidebar-label scene-label">Scenes <span>{visibleScenes.length}</span></div>
        <div className="scene-list">
          {visibleScenes.map((scene) => (
            <button key={scene.path} className={`scene-item ${scene.path === selectedScene ? "active" : ""}`} onClick={() => setSelectedScene(scene.path)}>
              <span className="scene-room">{scene.room}</span>
              <span className="scene-meta">{scene.batch.replace("batch_", "B")} <span className={`status-inline ${scene.status}`}>{scene.status}</span></span>
            </button>
          ))}
          {!visibleScenes.length && <div className="sidebar-empty">No scenes match this filter.</div>}
        </div>
        <div className="sidebar-footer"><CircleDot size={14} /><span>Polling every 5s</span></div>
      </aside>

      <section className="workspace" id="audit-workspace">
        <header className="topbar">
          <div className="context"><button className="icon-button nav-toggle" onClick={() => setSidebarOpen((value) => !value)} aria-label="Toggle navigation">{sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}</button><div><div className="crumb">{selectedRun || "Loading run"} <ChevronRight size={14} /> {activeScene?.batch ?? ""}</div><h1>{activeScene ? `${activeScene.room} / ${activeScene.scene}` : "Critic review"}</h1></div></div>
          <div className="top-actions"><span className={`status-pill ${activeScene?.status ?? "running"}`}><span className="status-dot" />{activeScene?.status ?? "loading"}</span><button className="icon-button" onClick={() => void copyAuditLink()} title="Copy a link to this audit view" aria-label="Copy a link to this audit view">{linkCopied ? <Check size={17} /> : <Copy size={17} />}</button><button className="icon-button" onClick={refresh} title="Refresh data" aria-label="Refresh data"><RefreshCw size={17} /></button><span className="sr-only" aria-live="polite">{linkCopied ? "Audit link copied" : ""}</span></div>
        </header>

        {error && <div className="error-banner">{error}</div>}
        {loading && !detail ? <div className="loading"><LoaderCircle className="spin" size={24} /> Loading critic probe data</div> : <>
          <section className="metrics-strip">
            <Metric icon={<Clock3 size={17} />} label="Events" value={String(allEvents.length)} subtext={`${detail?.actions.length ?? 0} tool actions`} />
            <Metric icon={<Bot size={17} />} label="LLM calls" value={String(allEvents.filter((event) => event.kind === "llm").length)} subtext="task compiler, planner, designer and critic" />
            <Metric icon={<Gauge size={17} />} label="Peak input context" value={tokenCount(detail?.audit_summary?.max_input_context_tokens)} subtext={peakContextCaption(detail)} />
            <Metric icon={<ImageIcon size={17} />} label="Snapshots" value={String(renders.length)} subtext="rendered scene states" />
            <Metric icon={<Command size={17} />} label="Quality" value={qualityValue(detail)} subtext={qualityCaption(detail)} emphasis />
          </section>

          <nav className="view-tabs" aria-label="Review views">
            <button className={view === "review" ? "active" : ""} onClick={() => setView("review")}><Activity size={16} />Review</button>
            <button className={view === "diff" ? "active" : ""} onClick={() => setView("diff")}><ArrowLeftRight size={16} />Scene diff</button>
            <button className={view === "actions" ? "active" : ""} onClick={() => setView("actions")}><Wrench size={16} />Tool log</button>
          </nav>

          {view === "review" && <ReviewView prompt={detail?.prompt} currentRender={currentRender} renders={renders} selectedRender={selectedRender} setSelectedRender={setSelectedRender} sortOrder={sortOrder} setSortOrder={setSortOrder} groups={timelineGroups} stages={stages} stageFilter={stageFilter} setStageFilter={setStageFilter} eventSearch={eventSearch} setEventSearch={setEventSearch} eventFilter={eventFilter} setEventFilter={setEventFilter} visibleEventCount={visibleEvents.length} totalEventCount={allEvents.length} attentionCount={attentionCount} onOpenEvent={setDrawerEvent} />}
          {view === "diff" && <DiffView beforeRender={beforeRender} currentRender={currentRender} renders={renders} comparisonRender={comparisonRender} selectedRender={selectedRender} setComparisonRender={setComparisonRender} setSelectedRender={setSelectedRender} diff={diff} />}
          {view === "actions" && <ActionView actions={visibleActions} totalActions={detail?.actions.length ?? 0} onOpenAction={(action) => setDrawerEvent(actionAuditEvents([action])[0])} />}
        </>}
      </section>
      {drawerEvent && ("tool_name" in drawerEvent
        ? <ActionDrawer value={drawerEvent} onClose={() => setDrawerEvent(null)} />
        : <AuditEventDrawer event={drawerEvent} scenePath={selectedScene} events={timelineEvents} onNavigate={setDrawerEvent} onClose={() => setDrawerEvent(null)} />)}
    </main>
  );
}

function Metric({ icon, label, value, subtext, emphasis = false }: { icon: ReactNode; label: string; value: string; subtext: string; emphasis?: boolean }) {
  return <div className={`metric ${emphasis ? "emphasis" : ""}`}><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{subtext}</small></div></div>;
}

function RenderSelect({ label, renders, value, onChange }: { label: string; renders: Render[]; value: string; onChange: (value: string) => void }) {
  return <label className="render-select"><span>{label}</span><div><select value={value} onChange={(event) => onChange(event.target.value)}>{renders.map((render) => <option value={render.id} key={render.id}>{render.label}</option>)}</select><ChevronDown size={15} /></div></label>;
}

function SnapshotPicker({ renders, value, onChange, sortOrder, setSortOrder }: { renders: Render[]; value: string; onChange: (value: string) => void; sortOrder: SortOrder; setSortOrder: (value: SortOrder) => void }) {
  const selectedIndex = renders.findIndex((render) => render.id === value);
  const activeIndex = selectedIndex < 0 ? 0 : selectedIndex;
  const changeBy = (offset: number) => {
    const next = renders[activeIndex + offset];
    if (next) onChange(next.id);
  };
  return <div className="snapshot-picker"><label className="time-order"><span>Time order</span><div><ArrowDownUp size={14} /><select value={sortOrder} onChange={(event) => setSortOrder(event.target.value as SortOrder)} aria-label="Time order"><option value="desc">Newest first</option><option value="asc">Oldest first</option></select></div></label><div className="snapshot-stepper"><button className="icon-button" type="button" onClick={() => changeBy(-1)} disabled={activeIndex === 0} title="Load previous snapshot" aria-label="Load previous snapshot"><ChevronUp size={16} /></button><button className="icon-button" type="button" onClick={() => changeBy(1)} disabled={activeIndex >= renders.length - 1} title="Load next snapshot" aria-label="Load next snapshot"><ChevronDown size={16} /></button></div><RenderSelect label="Snapshot" renders={renders} value={value} onChange={onChange} /></div>;
}

function SceneFrame({ render, side = false }: { render?: Render; side?: boolean }) {
  const path = side ? render?.side_image : render?.top_image;
  return <div className="scene-frame">{path ? <img src={imageUrl(path)!} alt={`${render?.label} ${side ? "side" : "top"} render`} /> : <div className="empty-frame"><ImageIcon size={22} /><span>{side ? "Side render unavailable" : "Top render unavailable"}</span></div>}<span className="frame-label">{side ? "Side" : "Top"}</span></div>;
}

function ReviewView({ prompt, currentRender, renders, selectedRender, setSelectedRender, sortOrder, setSortOrder, groups, stages, stageFilter, setStageFilter, eventSearch, setEventSearch, eventFilter, setEventFilter, visibleEventCount, totalEventCount, attentionCount, onOpenEvent }: { prompt?: string; currentRender?: Render; renders: Render[]; selectedRender: string; setSelectedRender: (value: string) => void; sortOrder: SortOrder; setSortOrder: (value: SortOrder) => void; groups: TimelineGroup[]; stages: string[]; stageFilter: string; setStageFilter: (value: string) => void; eventSearch: string; setEventSearch: (value: string) => void; eventFilter: EventFilter; setEventFilter: (value: EventFilter) => void; visibleEventCount: number; totalEventCount: number; attentionCount: number; onOpenEvent: (event: AuditEvent) => void }) {
  return (
    <div className="review-grid">
      <section className="render-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Selected checkpoint</span><h2>Scene render</h2></div>
          <SnapshotPicker renders={renders} value={selectedRender} onChange={setSelectedRender} sortOrder={sortOrder} setSortOrder={setSortOrder} />
        </div>
        <div className="render-grid"><SceneFrame render={currentRender} /><SceneFrame render={currentRender} side /></div>
        {prompt?.trim() && <div className="scene-prompt"><span>Scene prompt</span><p>{prompt}</p></div>}
      </section>
      <StageTimeline groups={groups} selectedRender={selectedRender} stages={stages} stageFilter={stageFilter} setStageFilter={setStageFilter} eventSearch={eventSearch} setEventSearch={setEventSearch} eventFilter={eventFilter} setEventFilter={setEventFilter} visibleEventCount={visibleEventCount} totalEventCount={totalEventCount} attentionCount={attentionCount} onOpenEvent={onOpenEvent} />
    </div>
  );
}

function DiffView({ beforeRender, currentRender, renders, comparisonRender, selectedRender, setComparisonRender, setSelectedRender, diff }: { beforeRender?: Render; currentRender?: Render; renders: Render[]; comparisonRender: string; selectedRender: string; setComparisonRender: (value: string) => void; setSelectedRender: (value: string) => void; diff: Diff | null }) {
  const hasComparableStates = Boolean(beforeRender?.state_path && currentRender?.state_path);
  return <section className="diff-panel"><div className="panel-heading"><div><span className="eyebrow">Checkpoint comparison</span><h2>Scene diff</h2></div><div className="diff-selects"><RenderSelect label="Before" renders={renders} value={comparisonRender} onChange={setComparisonRender} /><ArrowLeftRight size={18} /><RenderSelect label="After" renders={renders} value={selectedRender} onChange={setSelectedRender} /></div></div>{!hasComparableStates && <div className="excerpt-note">Final views are rendered from the completed Blender scene and have no JSON checkpoint for object-level comparison.</div>}<div className="diff-images"><SceneFrame render={beforeRender} /><SceneFrame render={currentRender} /></div><div className="delta-grid"><DeltaList title="Added" values={diff?.added.map((item) => item.object_id) ?? []} tone="positive" /><DeltaList title="Moved or rotated" values={diff?.changed.map((item) => item.object_id) ?? []} tone="warning" /><DeltaList title="Removed" values={diff?.removed.map((item) => item.object_id) ?? []} tone="negative" /></div></section>;
}

function DeltaList({ title, values, tone }: { title: string; values: string[]; tone: string }) { return <div className={`delta-list ${tone}`}><span>{title}</span><strong>{values.length}</strong><div>{values.length ? values.map((value) => <code key={value}>{value}</code>) : <small>No changes</small>}</div></div>; }

function ActionView({ actions, totalActions, onOpenAction }: { actions: Action[]; totalActions: number; onOpenAction: (action: Action) => void }) { return <section className="action-panel"><div className="panel-heading"><div><span className="eyebrow">Designer tool calls</span><h2>Action log</h2></div><span className="count-label">{actions.length === totalActions ? `${actions.length} steps` : `${actions.length} of ${totalActions} steps`}</span></div><div className="action-table"><div className="table-head"><span>#</span><span>Tool</span><span>Arguments</span><span>Time</span></div>{actions.map((action) => <button key={action.step_number} className="table-row" onClick={() => onOpenAction(action)}><span>{String(action.step_number).padStart(2, "0")}</span><span><Box size={15} />{action.tool_name.replaceAll("_", " ")}</span><code>{JSON.stringify(action.arguments)}</code><time>{formatTime(action.timestamp).slice(-8)}</time></button>)}{!actions.length && <div className="empty-list">No tool actions match the current audit filters.</div>}</div></section>; }

function ActionDrawer({ value, onClose }: { value: Action; onClose: () => void }) { return <div className="drawer-backdrop" onMouseDown={onClose}><aside className="event-drawer" onMouseDown={(event) => event.stopPropagation()}><header><div><span className="eyebrow">Tool action</span><h2>{value.tool_name}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close detail"><X size={19} /></button></header><div className="drawer-body"><DetailBlock label="Timestamp" value={formatTime(value.timestamp)} /><DetailBlock label="Arguments" value={JSON.stringify(value.arguments, null, 2)} code /><DetailBlock label="Raw action" value={JSON.stringify(value, null, 2)} code /></div></aside></div>; }

function DetailBlock({ label, value, code = false }: { label: string; value: string; code?: boolean }) { return <section className="detail-block"><span>{label}</span>{code ? <pre>{value}</pre> : <p>{value}</p>}</section>; }

function qualityValue(detail: SceneDetail | null): string { const grades = Object.values(detail?.score_summary.grades ?? {}); if (!grades.length) return "--"; return `${(grades.reduce((total, grade) => total + grade, 0) / grades.length).toFixed(1)}/10`; }
function qualityCaption(detail: SceneDetail | null): string { const grades = detail?.score_summary.grades ?? {}; return Object.keys(grades).length ? `${Object.keys(grades).length} critic dimensions` : "scores pending"; }
function tokenCount(value?: number | null): string { return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "--"; }
function peakContextCaption(detail: SceneDetail | null): string { const events = detail?.audit_summary?.max_input_context_events ?? []; if (!events.length) return "not recorded for this scene"; return events.map((event) => `${event.actor} / ${event.function.replaceAll("_", " ")}`).join(", "); }

export default App;
