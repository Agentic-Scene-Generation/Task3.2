export type Run = {
  id: string;
  scene_count: number;
  updated_at: string;
  status: "complete" | "running";
  modes: string[];
};

export type Scene = {
  id: string;
  path: string;
  batch: string;
  scene: string;
  room: string;
  mode: string;
  status: "complete" | "running";
  stages: string[];
  event_count: number;
  updated_at: string;
  score_summary: Record<string, number>;
};

export type Render = {
  id: string;
  stage: string;
  label: string;
  state_path: string | null;
  top_image: string | null;
  side_image: string | null;
  has_scores: boolean;
  created_at: string;
};

export type TimedEvent = {
  created_at?: string;
  stage?: string;
  module?: string;
  agent_role?: string;
  event?: string;
  elapsed_sec?: number | null;
  prompt_excerpt?: string;
  output_excerpt?: string;
  detail?: Record<string, unknown>;
  source: "timing" | "llm";
  [key: string]: unknown;
};

export type AuditEvent = {
  id: string;
  kind: "llm" | "system" | "benchmark" | "tool" | "orchestration" | "repair";
  source: string;
  created_at?: string;
  started_at?: string;
  stage: string;
  actor: string;
  function: string;
  title: string;
  elapsed_sec?: number | null;
  audit_status: string;
  token_usage?: Record<string, number>;
  detail?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  evaluation?: BenchmarkEvaluation;
  repair?: RepairAudit;
  prompt_chars?: number;
  output_chars?: number;
  has_error?: boolean;
  checkpoint_state?: "active";
  orchestration?: {
    call_id: string;
    phase: "dispatch" | "resume";
    child_agent: string;
  };
};

export type BenchmarkResult = {
  check_id?: string;
  metric?: string;
  label?: string;
  primary_object?: string;
  related_objects?: string[];
  reason?: string;
  repair_advice?: string;
  evidence?: unknown;
  [key: string]: unknown;
};

export type BenchmarkEvaluation = {
  results?: BenchmarkResult[];
  summary?: Record<string, unknown>;
  gate?: Record<string, unknown>;
};

export type RepairAudit = {
  source?: string;
  strategy?: string;
  status?: string;
  attempt?: number | null;
  trigger_reasons?: string[];
  actions?: string[];
  affected_objects?: Array<Record<string, unknown>>;
  detail?: Record<string, unknown>;
};

export type AuditDetail = {
  event: AuditEvent;
  provenance: string;
  input: unknown;
  output: unknown;
  raw_response?: unknown;
  reasoning: unknown[];
  messages?: Array<{
    database?: string;
    agent?: string;
    message_id?: number;
    created_at?: string;
    direction: "input" | "output" | "reasoning" | "tool_call" | "tool_output";
    content: unknown;
  }>;
  tool_calls: Array<{
    database?: string;
    message_id?: number;
    name: string;
    arguments: unknown;
    output?: unknown;
  }>;
  session_databases?: string[];
  metrics?: Record<string, unknown>;
  action?: Action;
  selection_trace?: {
    status: "recorded" | "not_recorded" | "not_applicable";
    asset?: Record<string, unknown>;
    retrieval?: {
      backend?: string;
      requested_dimensions?: unknown;
      candidates?: AssetRetrievalCandidate[];
    };
    vlm_selection?: Record<string, unknown>;
    note?: string;
  };
  has_full_input?: boolean;
  has_full_output?: boolean;
};

export type AssetEvidenceView = {
  label?: string;
  path?: string;
};

export type AssetRetrievalCandidate = {
  original_index?: number;
  hssd_id?: string;
  object_name?: string;
  category?: string;
  size?: number[];
  similarity_score?: number;
  evidence_views?: AssetEvidenceView[];
};

export type Action = {
  step_number: number;
  timestamp: string;
  tool_name: string;
  arguments: Record<string, unknown>;
};

export type SceneDetail = {
  path: string;
  actions: Action[];
  timings: TimedEvent[];
  llm_calls: TimedEvent[];
  audit_events: AuditEvent[];
  renders: Render[];
  score_summary: { grades?: Record<string, number>; summary?: string };
  messages: Array<{ id: string; agent: string; created_at: string; content: string }>;
  event_counts: Record<string, number>;
};

export type Diff = {
  added: Array<{ object_id: string; description?: string }>;
  removed: Array<{ object_id: string; description?: string }>;
  changed: Array<{ object_id: string; before: unknown; after: unknown }>;
};
