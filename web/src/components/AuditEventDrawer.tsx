import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, LoaderCircle, X } from "lucide-react";
import { isAuthoritativeResult } from "../audit";
import type { AssetEvidenceView, AuditDetail, AuditEvent, BenchmarkEvaluation, IntentConstraint, IntentContractExecution, ObjectSelector, RepairAudit, TaskCompilerAudit } from "../types";

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

function formatStage(stage: string): string {
  return stage.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function valueText(value: unknown): string {
  if (value === undefined || value === null || value === "") return "Not recorded";
  if (typeof value !== "string") return JSON.stringify(value, null, 2);
  const trimmed = value.trim();
  const looksLikeJson = (trimmed.startsWith("{") && trimmed.endsWith("}"))
    || (trimmed.startsWith("[") && trimmed.endsWith("]"));
  if (!looksLikeJson) return value;
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return value;
  }
}

function DetailBlock({ label, value, code = false }: { label: string; value: unknown; code?: boolean }) {
  const text = valueText(value);
  return <section className="detail-block"><span>{label}</span>{code ? <pre>{text}</pre> : <p>{text}</p>}</section>;
}

type ConversationMessage = NonNullable<AuditDetail["messages"]>[number];

type ApiRequestGroup = {
  input: ConversationMessage[];
  response: ConversationMessage[];
};

function groupApiRequests(messages: ConversationMessage[]): ApiRequestGroup[] {
  const groups: ApiRequestGroup[] = [];
  let current: ApiRequestGroup = { input: [], response: [] };
  messages.forEach((message) => {
    const isInput = message.direction === "input" || message.direction === "tool_output";
    if (isInput && current.response.length > 0) {
      groups.push(current);
      current = { input: [], response: [] };
    }
    (isInput ? current.input : current.response).push(message);
  });
  if (current.input.length > 0 || current.response.length > 0) groups.push(current);
  return groups;
}

function ConversationMessageRow({ message, index }: { message: ConversationMessage; index: number }) {
  return <div className={`conversation-message ${message.direction}`} key={`${message.database}-${message.message_id}-${index}`}><header><b>{message.agent || "Agent"}</b><small>{message.direction.replace("_", " ")} / {message.database || "audit record"}</small><time>{formatTime(message.created_at)}</time></header><pre>{valueText(message.content)}</pre></div>;
}

function ConversationTrace({ messages }: { messages: NonNullable<AuditDetail["messages"]> }) {
  if (!messages.length) return null;
  const requests = groupApiRequests(messages);
  return <section className="detail-block conversation-trace"><div className="conversation-trace-heading"><span>Conversation trace</span><small>{requests.length} API {requests.length === 1 ? "request" : "requests"}</small></div>{requests.map((request, requestIndex) => <article className="api-request-group" key={`request-${requestIndex}`}><div className="api-request-heading"><b>API request {requestIndex + 1}</b><small>{request.input.length} input {request.input.length === 1 ? "item" : "items"} / {request.response.length} response {request.response.length === 1 ? "item" : "items"}</small></div><section className="api-request-part input"><span>New input</span>{request.input.length > 0 ? request.input.map((message, index) => <ConversationMessageRow message={message} index={index} key={`${message.database}-${message.message_id}-${index}`} />) : <p>Input not recorded in this trace.</p>}</section><section className="api-request-part response"><span>Model response</span>{request.response.length > 0 ? request.response.map((message, index) => <ConversationMessageRow message={message} index={index} key={`${message.database}-${message.message_id}-${index}`} />) : <p>Response not recorded in this trace.</p>}</section></article>)}</section>;
}

function assetThumbnailUrl(path: string, width = 260): string {
  return `/api/asset-thumbnail?path=${encodeURIComponent(path)}&width=${width}`;
}

function formatDimensions(value?: number[]): string {
  if (!value?.length) return "Dimensions not recorded";
  return `${value.map((item) => `${item.toFixed(2)}m`).join(" x ")}`;
}

function CandidateEvidence({ candidateId, views }: { candidateId: string; views?: AssetEvidenceView[] }) {
  const evidence = Array.isArray(views) ? views.filter((view): view is { label?: string; path: string } => Boolean(view?.path)) : [];
  const primary = evidence.find((view) => view.label?.toLowerCase().includes("iso")) ?? evidence[0];
  const additional = evidence.filter((view) => view !== primary);
  if (!primary) return <div className="candidate-image-missing">No render evidence</div>;
  return <div className="candidate-evidence"><img src={assetThumbnailUrl(primary.path)} alt={`${candidateId} ${primary.label || "render"}`} loading="lazy" decoding="async" />{additional.length > 0 && <details><summary>{additional.length} additional view{additional.length === 1 ? "" : "s"}</summary><div className="additional-evidence">{additional.map((view) => <figure key={`${candidateId}-${view.path}`}><img src={assetThumbnailUrl(view.path, 200)} alt={`${candidateId} ${view.label || "render"}`} loading="lazy" decoding="async" /><figcaption>{view.label || "Evidence view"}</figcaption></figure>)}</div></details>}</div>;
}

function SelectionTrace({ trace }: { trace: NonNullable<AuditDetail["selection_trace"]> }) {
  if (trace.status === "not_applicable") return null;
  if (trace.status === "not_recorded") return <section className="detail-block"><span>Asset retrieval and VLM selection</span><div className="excerpt-note">{trace.note || "This action has no recorded retrieval/VLM selection trace."}</div>{trace.asset && <DetailBlock label="Asset" value={trace.asset} code />}</section>;
  const selectedId = typeof trace.vlm_selection?.selected_hssd_id === "string" ? trace.vlm_selection.selected_hssd_id : "";
  return <section className="detail-block selection-trace"><span>Asset retrieval and VLM selection</span>{trace.asset && <DetailBlock label="Placed asset" value={trace.asset} code />}<div className="candidate-trace-heading"><b>Candidate retrieval</b><small>{trace.retrieval?.backend || "unknown"}</small></div><div className="candidate-grid">{trace.retrieval?.candidates?.map((candidate, index) => { const candidateId = candidate.hssd_id || `candidate-${index + 1}`; const selected = candidateId === selectedId; return <article className={`candidate-card${selected ? " selected" : ""}`} key={candidateId}><CandidateEvidence candidateId={candidateId} views={candidate.evidence_views} /><div className="candidate-card-body"><div><strong>Candidate {candidate.original_index || index + 1}</strong>{selected && <em>VLM selected</em>}</div><p>{candidate.object_name || candidateId}</p><dl><div><dt>Similarity</dt><dd>{typeof candidate.similarity_score === "number" ? candidate.similarity_score.toFixed(3) : "--"}</dd></div><div><dt>Size</dt><dd>{formatDimensions(candidate.size)}</dd></div></dl></div></article>; })}</div><DetailBlock label="Requested dimensions" value={trace.retrieval?.requested_dimensions} code /><DetailBlock label="VLM filtering and final selection" value={trace.vlm_selection} code /></section>;
}

function selectorText(selector?: ObjectSelector | null): string {
  if (!selector) return "none";
  const count = selector.count ? `${selector.quantifier || "exactly"} ${selector.count} ` : "";
  const role = selector.role ? ` (${selector.role})` : "";
  const secondary = selector.secondary_category ? ` + ${selector.secondary_category}` : "";
  return `${count}${selector.category}${secondary}${role}`;
}

function constraintRationale(constraint: IntentConstraint): string {
  return constraint.evidence_span || constraint.inference_reason || "No rationale recorded";
}

function inventoryText(value: unknown): string {
  return Array.isArray(value) && value.length > 0 ? value.map(String).join(", ") : "None";
}

function CompilerContractAudit({ contract }: { contract?: TaskCompilerAudit }) {
  if (!contract) return <div className="excerpt-note">The typed compiler contract was not recorded for this run.</div>;
  const constraints = contract.constraints ?? [];
  const taskSpec = contract.task_spec ?? {};
  return <section className="structured-audit compiler-contract"><div className="structured-audit-heading"><span>Typed scene contract</span><small className={`status-label ${contract.status}`}>{contract.status}</small></div>{contract.failure_reason && <div className="contract-warning" role="alert"><b>Deterministic fallback used</b><span>{contract.failure_reason}</span></div>}<div className="audit-facts"><DetailBlock label="Contract schema" value={contract.spec_version} /><DetailBlock label="Room / style" value={[taskSpec.room_type, taskSpec.style].filter(Boolean).join(" / ")} /></div><div className="contract-inventory"><DetailBlock label="Furniture" value={inventoryText(taskSpec.required_large_objects)} /><DetailBlock label="Wall mounted" value={inventoryText(taskSpec.required_wall_objects)} /><DetailBlock label="Ceiling mounted" value={inventoryText(taskSpec.required_ceiling_objects)} /><DetailBlock label="Manipulands" value={inventoryText(taskSpec.required_small_objects)} /></div><div className="contract-list">{constraints.length > 0 ? constraints.map((constraint, index) => <article className="contract-row" key={`${constraint.relation}-${index}`}><header><b>{constraint.relation.replaceAll("_", " ")}</b><span className="contract-source">{constraint.source.replaceAll("_", " ")}</span></header><p><strong>{selectorText(constraint.subjects)}</strong><span> to </span><strong>{selectorText(constraint.targets)}</strong></p><small>{constraintRationale(constraint)}</small></article>) : <div className="excerpt-note">No executable intent constraints were compiled.</div>}</div><details className="raw-contract"><summary>Full task specification</summary><DetailBlock label="Task specification" value={taskSpec} code /></details></section>;
}

function ContractExecutionAudit({ rows, resolutionRate }: { rows: IntentContractExecution[]; resolutionRate?: number }) {
  if (!rows.length) return null;
  const counts = rows.reduce<Record<string, number>>((values, row) => {
    values[row.state] = (values[row.state] ?? 0) + 1;
    return values;
  }, {});
  const percentage = typeof resolutionRate === "number" ? `${Math.round(resolutionRate * 100)}% resolved` : `${counts.passed ?? 0} passed`;
  return <section className="contract-execution"><div className="contract-execution-summary"><strong>Intent contract execution</strong><span>{percentage}</span></div><div className="contract-state-counts" aria-label="Contract execution status">{["failed", "blocked", "pending", "passed"].filter((state) => counts[state]).map((state) => <span className={`contract-state ${state}`} key={state}><b>{counts[state]}</b> {state}</span>)}</div><div className="contract-list">{rows.map((row) => <article className={`contract-row execution ${row.state}`} key={row.constraint_id}><header><div><b>{row.relation.replaceAll("_", " ")}</b><small>{row.constraint_id}</small></div><span className={`contract-state ${row.state}`}>{row.state}</span></header><dl><div><dt>Bound subjects</dt><dd>{row.subject_ids?.join(", ") || "unbound"}</dd></div><div><dt>Bound targets</dt><dd>{row.target_ids?.join(", ") || "none"}</dd></div><div><dt>Dependencies</dt><dd>{row.dependency_constraint_ids?.join(", ") || "none"}</dd></div><div><dt>Repair strategy</dt><dd>{row.repair_strategy || "none"}</dd></div></dl><p className="contract-provenance"><span>{row.source.replaceAll("_", " ")}</span>{row.evidence_span || row.inference_reason || "No rationale recorded"}</p></article>)}</div></section>;
}

function BenchmarkAudit({ evaluation }: { evaluation?: BenchmarkEvaluation }) {
  if (!evaluation || (!evaluation.results?.length && !evaluation.summary && !evaluation.gate)) {
    return <div className="excerpt-note">This historical benchmark event contains timing only; per-check evaluation was not recorded.</div>;
  }
  const results = evaluation.results ?? [];
  const labelOrder: Record<string, number> = { fail: 0, degraded: 1, unknown: 2, pass: 3 };
  const orderedResults = [...results].sort((left, right) => {
    const tierDelta = Number(isAuthoritativeResult(right)) - Number(isAuthoritativeResult(left));
    return tierDelta || (labelOrder[left.label || "unknown"] ?? 2) - (labelOrder[right.label || "unknown"] ?? 2);
  });
  const sceneSummary = evaluation.summary?.scene_summary ?? evaluation.summary;
  const metricSummary = evaluation.summary?.metric_summary;
  const intentContract = evaluation.case_pack?.intent_contract;
  const execution = intentContract?.execution ?? [];
  return <section className="structured-audit benchmark-audit"><div className="structured-audit-heading"><span>SceneBenchmark evaluation</span><small>{evaluation.schema_version || `${results.length} checks`}</small></div><div className="audit-facts"><DetailBlock label="Gate" value={evaluation.gate} code /><DetailBlock label="Scene summary" value={sceneSummary} code /></div><ContractExecutionAudit rows={execution} resolutionRate={intentContract?.resolution_rate} />{orderedResults.length > 0 && <><div className="check-list-heading"><strong>Rule checks</strong><span>{results.filter(isAuthoritativeResult).length} scored / {results.length} total</span></div><div className="benchmark-results">{orderedResults.map((result, index) => <article className={`benchmark-result ${result.label || "unknown"}${isAuthoritativeResult(result) ? "" : " non-authoritative"}`} key={result.check_id || index}><header><div><b>{result.check_id || `Check ${index + 1}`}</b><small>{result.metric || "Metric not recorded"}</small></div><div className="result-labels"><em>{result.label || "unknown"}</em><em className="tier-label">{result.scoring_tier || "core"}</em>{result.contract_state && <em className={`contract-state ${result.contract_state}`}>{result.contract_state}</em>}</div></header><dl><div><dt>Subject</dt><dd>{result.primary_object || "--"}</dd></div><div><dt>Related</dt><dd>{result.related_objects?.join(", ") || "--"}</dd></div></dl><DetailBlock label="Reason" value={result.reason} />{result.repair_advice && <DetailBlock label="Repair advice" value={result.repair_advice} />}{result.evidence !== undefined && <details className="raw-contract"><summary>Evidence</summary><DetailBlock label="Evidence" value={result.evidence} code /></details>}</article>)}</div></>}{metricSummary !== undefined && <DetailBlock label="Metric summary" value={metricSummary} code />}</section>;
}

function RepairAuditView({ repair }: { repair?: RepairAudit }) {
  if (!repair) return <div className="excerpt-note">This historical repair event contains timing only; structured repair details were not recorded.</div>;
  const affectedObjects = repair.affected_objects ?? [];
  return <section className="structured-audit repair-audit"><div className="structured-audit-heading"><span>Automatic repair</span><small>{repair.status || "unknown"}</small></div><div className="audit-facts"><DetailBlock label="Strategy" value={repair.strategy} /><DetailBlock label="Source" value={repair.source} /><DetailBlock label="Attempt" value={repair.attempt} /><DetailBlock label="Trigger" value={repair.trigger_reasons?.join("\n")} /></div><DetailBlock label="Actions" value={repair.actions?.join("\n")} />{affectedObjects.length > 0 && <div className="repair-objects">{affectedObjects.map((object, index) => <article key={`${String(object.object_id || "object")}-${index}`}><b>{String(object.object_id || `Object ${index + 1}`)}</b>{object.relation_type !== undefined && <small>{String(object.relation_type)}</small>}<DetailBlock label="Pose change" value={object} code /></article>)}</div>}{repair.detail && Object.keys(repair.detail).length > 0 && <DetailBlock label="Repair result" value={repair.detail} code />}</section>;
}

function PhysicsContextAudit({ event }: { event: AuditEvent }) {
  const context = event.detail?.physics_context;
  if (typeof context !== "string" || !context.trim()) {
    return <div className="excerpt-note">This historical physics check contains timing only; its detailed feedback was not persisted.</div>;
  }
  const totalChars = event.detail?.physics_context_chars;
  const truncated = event.detail?.physics_context_truncated === true;
  const caption = typeof totalChars === "number"
    ? `${totalChars} characters${truncated ? "; stored excerpt is truncated" : ""}`
    : truncated ? "Stored excerpt is truncated" : undefined;
  return <section className="structured-audit"><div className="structured-audit-heading"><span>Physics and geometry feedback</span>{caption && <small>{caption}</small>}</div><DetailBlock label="Check result" value={context} /></section>;
}

export function AuditEventDrawer({ event, scenePath, events, onNavigate, onClose }: { event: AuditEvent; scenePath: string; events: AuditEvent[]; onNavigate: (event: AuditEvent) => void; onClose: () => void }) {
  const [detail, setDetail] = useState<AuditDetail | null>(null);
  const [error, setError] = useState("");
  const drawerRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  const isLegacyLlm = event.kind === "llm" && event.id.startsWith("legacy-llm:");
  const fallbackInput = event.detail?.prompt_excerpt;
  const fallbackOutput = event.detail?.output_excerpt;
  const hasLocalExcerpt = fallbackInput !== undefined || fallbackOutput !== undefined;
  const shouldLoadDetail = (event.kind === "llm" || event.kind === "tool") && !isLegacyLlm;

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusDrawerControl = () => closeButtonRef.current?.focus();
    const handleKeyDown = (keyboardEvent: KeyboardEvent) => {
      if (keyboardEvent.key === "Escape") {
        keyboardEvent.preventDefault();
        onCloseRef.current();
        return;
      }
      if (keyboardEvent.key !== "Tab" || !drawerRef.current) return;
      const focusable = Array.from(drawerRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"))
        .filter((element) => !element.hasAttribute("disabled"));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable.at(-1)!;
      if (keyboardEvent.shiftKey && document.activeElement === first) {
        keyboardEvent.preventDefault();
        last.focus();
      } else if (!keyboardEvent.shiftKey && document.activeElement === last) {
        keyboardEvent.preventDefault();
        first.focus();
      }
    };
    focusDrawerControl();
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      opener?.focus();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError("");
    if (!shouldLoadDetail) return () => { cancelled = true; };
    void fetch(`/api/audit-event?path=${encodeURIComponent(scenePath)}&event_id=${encodeURIComponent(event.id)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Request failed (${response.status})`);
        return response.json() as Promise<AuditDetail>;
      })
      .then((payload) => !cancelled && setDetail(payload))
      .catch((reason) => !cancelled && setError(reason instanceof Error ? reason.message : "Unable to load audit detail"));
    return () => { cancelled = true; };
  }, [event.id, scenePath, shouldLoadDetail]);

  const audit = detail?.event ?? event;
  const isLlm = audit.kind === "llm";
  const provenance = detail?.provenance ?? audit.audit_status;
  const eventIndex = events.findIndex((item) => item.id === event.id);
  const previousEvent = eventIndex > 0 ? events[eventIndex - 1] : undefined;
  const nextEvent = eventIndex >= 0 ? events[eventIndex + 1] : undefined;

  return <div className="drawer-backdrop" onMouseDown={onClose}><aside className="event-drawer" ref={drawerRef} role="dialog" aria-modal="true" aria-labelledby="audit-event-title" onMouseDown={(mouseEvent) => mouseEvent.stopPropagation()}><header><div><span className="eyebrow">{audit.kind} audit event</span><h2 id="audit-event-title">{audit.title}</h2></div><div className="drawer-actions"><button className="icon-button" onClick={() => previousEvent && onNavigate(previousEvent)} disabled={!previousEvent} title="Open previous event" aria-label="Open previous event"><ChevronUp size={18} /></button><button className="icon-button" onClick={() => nextEvent && onNavigate(nextEvent)} disabled={!nextEvent} title="Open next event" aria-label="Open next event"><ChevronDown size={18} /></button><button className="icon-button" ref={closeButtonRef} onClick={onClose} aria-label="Close detail"><X size={19} /></button></div></header><div className="drawer-body"><div className="audit-summary"><span>{provenance.replaceAll("_", " ")}</span>{event.checkpoint_state === "active" && <span>active at selected snapshot</span>}{isLlm && <span>{detail?.has_full_input ? "full input" : "excerpt fallback"}</span>}{isLlm && <span>{detail?.has_full_output ? "full output" : "excerpt fallback"}</span>}</div><div className="audit-facts"><DetailBlock label="Completed" value={formatTime(audit.created_at)} /><DetailBlock label="Started" value={formatTime(audit.started_at)} /><DetailBlock label="Duration" value={formatDuration(audit.elapsed_sec)} /><DetailBlock label="Stage" value={formatStage(audit.stage)} /><DetailBlock label="Agent / role" value={audit.actor} /><DetailBlock label="Function / event" value={audit.function} /></div>{audit.token_usage && Object.keys(audit.token_usage).length > 0 && <DetailBlock label="Token usage" value={audit.token_usage} code />}{audit.function === "physics_context" && <PhysicsContextAudit event={audit} />}{audit.kind !== "llm" && audit.kind !== "benchmark" && audit.kind !== "repair" && audit.kind !== "contract" && audit.function !== "physics_context" && audit.detail && Object.keys(audit.detail).length > 0 && <DetailBlock label="Recorded detail" value={audit.detail} code />}{audit.kind === "contract" && <CompilerContractAudit contract={audit.contract} />}{audit.kind === "benchmark" && <BenchmarkAudit evaluation={audit.evaluation} />}{audit.kind === "repair" && <RepairAuditView repair={audit.repair} />}{!isLlm && audit.metrics && Object.keys(audit.metrics).length > 0 && <DetailBlock label="Deterministic metrics" value={audit.metrics} code />}{error && <div className="error-banner">{error}</div>}{isLlm && hasLocalExcerpt && !detail && <><div className="excerpt-note">Full audit record is unavailable for this older run; showing captured excerpts.</div><DetailBlock label="LLM input excerpt" value={fallbackInput} code /><DetailBlock label="LLM output excerpt" value={fallbackOutput} code /></>}{shouldLoadDetail && !detail && !error && !(isLlm && hasLocalExcerpt) && <div className="audit-loading"><LoaderCircle className="spin" size={18} /> Loading full audit record</div>}{isLlm && detail && <><ConversationTrace messages={detail.messages ?? []} /><DetailBlock label={detail.has_full_input ? "Full LLM input" : "LLM input excerpt"} value={detail.input} code /><DetailBlock label={detail.has_full_output ? "Full LLM output" : "LLM output excerpt"} value={detail.output} code />{detail.raw_response !== undefined && <DetailBlock label="Raw model response" value={detail.raw_response} code />}{detail.reasoning.length > 0 && <DetailBlock label="Reasoning" value={detail.reasoning} code />}{detail.tool_calls.length > 0 && <section className="detail-block"><span>Function calls</span>{detail.tool_calls.map((call, index) => <div className="tool-call" key={`${call.name}-${index}`}><strong>{call.name}</strong>{call.database && <small>{call.database}</small>}<DetailBlock label="Arguments" value={call.arguments} code /><DetailBlock label="Result" value={call.output} code /></div>)}</section>}{detail.metrics && Object.keys(detail.metrics).length > 0 && <DetailBlock label="Deterministic metrics" value={detail.metrics} code />}{detail.session_databases && <DetailBlock label="Audit source databases" value={detail.session_databases.join(", ")} />}</>}{audit.kind === "tool" && detail?.selection_trace && <SelectionTrace trace={detail.selection_trace} />}</div></aside></div>;
}
