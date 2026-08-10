import type { AuditEvent, BenchmarkResult } from "./types";

const NON_AUTHORITATIVE_TIERS = new Set(["auxiliary", "ignored"]);

export function isAuthoritativeResult(result: BenchmarkResult): boolean {
  return !NON_AUTHORITATIVE_TIERS.has(String(result.scoring_tier || "core").toLowerCase());
}

export function eventNeedsAttention(event: AuditEvent): boolean {
  if (event.has_error || event.kind === "repair") return true;
  if (event.kind === "contract" && event.contract?.status === "degraded") return true;
  const gate = event.evaluation?.gate;
  if (gate?.blocked === true) return true;
  const gateLabel = String(gate?.label || "").toLowerCase();
  if (gateLabel === "fail" || gateLabel === "degraded") return true;
  return Boolean(event.evaluation?.results?.some((result) => {
    const label = String(result.label || "").toLowerCase();
    return isAuthoritativeResult(result) && (label === "fail" || label === "degraded");
  }));
}
