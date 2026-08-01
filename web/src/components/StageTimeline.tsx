import { Activity, AlertTriangle, ArrowRight, Bot, CircleDot, FileCheck2, Image as ImageIcon, Search, Workflow, Wrench, X } from "lucide-react";
import { eventNeedsAttention } from "../audit";
import type { AuditEvent, Render } from "../types";

export type TimelineGroup = {
  id: string;
  render?: Render;
  events: AuditEvent[];
};

type AgentRole = "compiler" | "designer" | "planner" | "critic" | "renderer" | "system";

type TimelineSegment = {
  role: AgentRole;
  events: AuditEvent[];
};

type FlowStep = {
  role: AgentRole;
  count: number;
  inferred: boolean;
};

type EventFilter = AuditEvent["kind"] | "attention" | "all";

const AGENT_LABELS: Record<AgentRole, string> = {
  compiler: "Task Compiler",
  designer: "Designer",
  planner: "Planner",
  critic: "Critic",
  renderer: "Renderer",
  system: "System",
};

function formatStage(stage: string): string {
  return stage.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatTime(value?: string): string {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value.replace("T", " ").slice(0, 19) : date.toLocaleString();
}

function formatDuration(seconds?: number | null): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

function eventTitle(event: AuditEvent): string {
  return event.title || event.function.replaceAll("_", " ");
}

function formatEventFilter(value: EventFilter): string {
  if (value === "all") return "All event types";
  if (value === "attention") return "Needs review";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function classifyAgent(event: AuditEvent): AgentRole {
  const actor = event.actor.toLowerCase();
  const stage = event.stage.toLowerCase();
  const functionName = event.function.toLowerCase();
  if (actor.includes("task_compiler")) return "compiler";
  if (actor.includes("designer")) return "designer";
  if (actor.includes("planner")) return "planner";
  if (actor.includes("critic") || stage.includes("critic") || functionName.includes("critique")) return "critic";
  if (actor.includes("render") || functionName.includes("render")) return "renderer";
  return "system";
}

function eventIcon(event: AuditEvent) {
  const name = event.function;
  if (event.kind === "contract") return <FileCheck2 size={15} />;
  if (event.kind === "orchestration") return <Workflow size={15} />;
  if (event.kind === "repair") return <Wrench size={15} />;
  if (event.kind === "tool" || name.includes("add_") || name.includes("move_") || name.includes("snap_")) return <Wrench size={15} />;
  if (event.kind === "llm" || name.includes("critic")) return <Bot size={15} />;
  if (event.kind === "benchmark") return <CircleDot size={15} />;
  if (name.includes("render")) return <ImageIcon size={15} />;
  if (name.includes("repair")) return <Wrench size={15} />;
  return <Activity size={15} />;
}

function isWorker(role: AgentRole): boolean {
  return role === "designer" || role === "critic";
}

function hasRecordedPlanner(segments: TimelineSegment[]): boolean {
  // Orchestration events are the authoritative record of Planner delegation.
  // A Planner segment can be separated from workers by renders or checkpoints.
  return segments.some((segment) => segment.role === "planner");
}

function coordinationFlow(segments: TimelineSegment[], plannerRecorded: boolean): FlowStep[] {
  const observed = segments
    .filter((segment) => segment.role !== "system")
    .map((segment) => ({ role: segment.role, count: segment.events.length, inferred: false }));
  if (!observed.length) return [];
  const flow: FlowStep[] = [];
  observed.forEach((step, index) => {
    const previous = flow.at(-1);
    if (
      isWorker(step.role)
      && (index === 0 || previous?.role !== "planner")
      && !plannerRecorded
    ) {
      flow.push({ role: "planner", count: 0, inferred: true });
    }
    flow.push(step);
  });
  return flow;
}

function segmentEvents(events: AuditEvent[]): TimelineSegment[] {
  return events.reduce<TimelineSegment[]>((segments, event) => {
    const role = classifyAgent(event);
    const last = segments.at(-1);
    if (last?.role === role) {
      last.events.push(event);
    } else {
      segments.push({ role, events: [event] });
    }
    return segments;
  }, []);
}

function AgentHandoff({ segments }: { segments: TimelineSegment[] }) {
  const agents = coordinationFlow(segments, hasRecordedPlanner(segments));
  if (!agents.length) return null;
  return <div className="agent-handoff" aria-label="Agent coordination sequence"><span>Coordination flow</span><div>{agents.map((step, index) => <span className={`agent-handoff-step${step.inferred ? " inferred" : ""}`} key={`${step.role}-${index}`}><b className={`agent-badge ${step.role}`}>{AGENT_LABELS[step.role]}</b><small>{step.inferred ? "not recorded" : step.count}</small>{index < agents.length - 1 && <ArrowRight size={12} />}</span>)}</div></div>;
}

function AgentTransition({ from, to, plannerRecorded }: { from: AgentRole; to: AgentRole; plannerRecorded: boolean }) {
  const needsPlanner = isWorker(from) && isWorker(to) && !plannerRecorded;
  return <div className="agent-transition"><span>{AGENT_LABELS[from]}</span><ArrowRight size={12} />{needsPlanner && <><b className="agent-badge planner">Planner</b><small>not recorded</small><ArrowRight size={12} /></>}<span>{AGENT_LABELS[to]}</span></div>;
}

function TimelineEventRow({ event, onOpenEvent }: { event: AuditEvent; onOpenEvent: (event: AuditEvent) => void }) {
  const role = classifyAgent(event);
  const activeAtCheckpoint = event.checkpoint_state === "active";
  const needsAttention = eventNeedsAttention(event);
  const displayedTime = activeAtCheckpoint ? event.started_at : event.created_at;
  return <button className={`timeline-event${activeAtCheckpoint ? " active-at-checkpoint" : ""}${needsAttention ? " needs-attention" : ""}`} key={event.id} onClick={() => onOpenEvent(event)}><span className={`timeline-icon ${event.kind}`}>{eventIcon(event)}</span><span className="event-main"><strong><span className="event-title">{eventTitle(event)}</span><em className={`event-kind ${event.kind}`}>{event.kind}</em>{needsAttention && <em className="attention-flag"><AlertTriangle size={11} />Needs review</em>}{activeAtCheckpoint && <em className="checkpoint-state">active at snapshot</em>}</strong><small>{event.actor || AGENT_LABELS[role]} <i>in</i> {formatStage(event.stage)} <i>/</i> {event.function}</small></span><span className="event-meta"><time>{formatTime(displayedTime).slice(-8)}</time><b>{activeAtCheckpoint ? "In progress" : formatDuration(event.elapsed_sec)}</b></span></button>;
}

function AgentSegment({ segment, onOpenEvent }: { segment: TimelineSegment; onOpenEvent: (event: AuditEvent) => void }) {
  return <div className={`agent-segment ${segment.role}`}><div className="agent-segment-heading"><b className={`agent-badge ${segment.role}`}>{AGENT_LABELS[segment.role]}</b><span>{segment.events.length} {segment.events.length === 1 ? "event" : "events"}</span></div>{segment.events.map((event) => <TimelineEventRow key={event.id} event={event} onOpenEvent={onOpenEvent} />)}</div>;
}

export function StageTimeline({ groups, selectedRender, stages, stageFilter, setStageFilter, eventSearch, setEventSearch, eventFilter, setEventFilter, visibleEventCount, totalEventCount, attentionCount, onOpenEvent }: { groups: TimelineGroup[]; selectedRender: string; stages: string[]; stageFilter: string; setStageFilter: (value: string) => void; eventSearch: string; setEventSearch: (value: string) => void; eventFilter: EventFilter; setEventFilter: (value: EventFilter) => void; visibleEventCount: number; totalEventCount: number; attentionCount: number; onOpenEvent: (event: AuditEvent) => void }) {
  const hasMatchingEvents = groups.some((group) => group.events.length > 0);
  const displayedGroups = groups.filter((group) => group.events.length > 0 || group.id === selectedRender);
  const filtersActive = Boolean(eventSearch.trim() || eventFilter !== "all" || stageFilter !== "all");
  const clearFilters = () => {
    setEventSearch("");
    setEventFilter("all");
    setStageFilter("all");
  };
  return <section className="timeline-panel">
    <div className="panel-heading">
      <div><span className="eyebrow">Execution trace</span><h2>Stage timeline</h2><small className="checkpoint-note">Grouped by checkpoint and agent handoff</small></div>
      <div className="stage-filter">{stages.map((stage) => <button key={stage} className={stageFilter === stage ? "active" : ""} onClick={() => setStageFilter(stage)}>{stage === "all" ? "All" : formatStage(stage)}</button>)}</div>
    </div>
    <div className="timeline-audit-toolbar">
      <label className="event-search"><Search size={15} /><span className="sr-only">Search execution trace</span><input id="event-search" type="search" value={eventSearch} onChange={(event) => setEventSearch(event.target.value)} placeholder="Search trace" /></label>
      <label className="event-kind-select"><span>Event type</span><select value={eventFilter} onChange={(event) => setEventFilter(event.target.value as EventFilter)}>{(["all", "attention", "contract", "llm", "benchmark", "tool", "repair", "orchestration", "system"] as EventFilter[]).map((filter) => <option key={filter} value={filter}>{formatEventFilter(filter)}</option>)}</select></label>
      <button className={`attention-filter${eventFilter === "attention" ? " active" : ""}`} type="button" onClick={() => setEventFilter((eventFilter === "attention" ? "all" : "attention") as EventFilter)} aria-pressed={eventFilter === "attention"}><AlertTriangle size={14} />Needs review <span>{attentionCount}</span></button>
      <span className="event-count" aria-live="polite">{visibleEventCount} of {totalEventCount}</span>
      {filtersActive && <button className="icon-button clear-audit-filters" type="button" onClick={clearFilters} title="Clear audit filters" aria-label="Clear audit filters"><X size={16} /></button>}
    </div>
    <div className="timeline">
      {!hasMatchingEvents && <div className="empty-list"><strong>No trace events match these filters.</strong><span>Try clearing a filter or search for a different term.</span></div>}
      {hasMatchingEvents && displayedGroups.map((group) => {
        const segments = segmentEvents(group.events);
        const plannerRecorded = hasRecordedPlanner(segments);
        return <section className={`timeline-group ${group.id === selectedRender ? "selected" : ""}`} id={`timeline-snapshot-${group.id}`} key={group.id}><div className="timeline-group-heading"><div><strong>{group.render ? group.render.label : "After latest checkpoint"}</strong><small>{group.render ? formatTime(group.render.created_at) : "Events completed after the newest render"}</small></div><span>{group.events.length} events</span></div><AgentHandoff segments={segments} />{segments.map((segment, index) => <div key={`${segment.role}-${index}`}>{index > 0 && <AgentTransition from={segments[index - 1].role} to={segment.role} plannerRecorded={plannerRecorded} />}<AgentSegment segment={segment} onOpenEvent={onOpenEvent} /></div>)}{!group.events.length && <div className="empty-group">No events in this checkpoint range.</div>}</section>;
      })}
    </div>
  </section>;
}
