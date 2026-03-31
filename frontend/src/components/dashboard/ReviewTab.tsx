import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type AnomalyObject, type LLMReview } from "@/lib/api";
import ScoreBadge from "./ScoreBadge";
import FeedbackButtons from "./FeedbackButtons";
import ObjectDetailModal from "./ObjectDetailModal";
import { Loader2, Brain, CheckCircle2 } from "lucide-react";

export default function ReviewTab() {
  const [selectedObject, setSelectedObject] = useState<AnomalyObject | null>(null);
  // Only tracks which cards are currently mid-fade animation (visual only, resets are fine)
  const [fadingOids, setFadingOids] = useState<Set<string>>(new Set());
  const queryClient = useQueryClient();

  const { data: objects, isLoading } = useQuery({
    queryKey: ["flagged"],
    queryFn: () => api.getFlagged(50),
    // Always treat cache as stale so switching tabs always fetches fresh data
    staleTime: 0,
  });

  const { data: llmReviews } = useQuery({
    queryKey: ["llm-reviews"],
    queryFn: api.getLLMReviews,
    staleTime: 60_000,
  });

  const llmMap = new Map<string, LLMReview>();
  llmReviews?.forEach(r => {
    if (!llmMap.has(r.oid)) llmMap.set(r.oid, r);
  });

  const handleFeedback = (oid: string) => {
    // Start fade-out animation
    setFadingOids(prev => new Set(prev).add(oid));
    // After animation, force a fresh refetch from the server
    // The server now excludes reviewed objects (human_feedback IS NOT NULL)
    setTimeout(() => {
      setFadingOids(prev => {
        const next = new Set(prev);
        next.delete(oid);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["flagged"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    }, 450);
  };

  // Filter out objects currently fading (mid-animation only — server drives the real list)
  const visibleObjects = (objects ?? []).filter(o => !fadingOids.has(o.oid));

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-indigo" />
      </div>
    );
  }

  if (visibleObjects.length === 0) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <h2 className="text-2xl font-bold text-foreground">🚨 Flagged for Review</h2>
          <span className="text-sm text-muted-foreground font-mono">0 pending</span>
        </div>
        <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
          <CheckCircle2 className="w-12 h-12 text-classified opacity-60" />
          <p className="text-lg font-semibold text-foreground">All caught up!</p>
          <p className="text-sm text-muted-foreground max-w-sm">
            No flagged objects pending review. New objects will appear here after the next session.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <h2 className="text-2xl font-bold text-foreground">🚨 Flagged for Review</h2>
        <span className="text-sm text-muted-foreground font-mono">
          {visibleObjects.length} pending
        </span>
      </div>

      <div className="grid gap-4">
        {visibleObjects.map((obj) => {
          const llm = llmMap.get(obj.oid);
          return (
            <div
              key={obj.oid}
              className={`card-space p-5 transition-all duration-400 ${fadingOids.has(obj.oid)
                  ? "opacity-0 scale-95 -translate-y-1"
                  : "opacity-100 scale-100"
                }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2 flex-wrap">
                    <button
                      onClick={() => setSelectedObject(obj)}
                      className="font-mono text-sm font-bold text-indigo-light hover:underline"
                    >
                      {obj.oid}
                    </button>
                    <ScoreBadge score={obj.score} />
                    {obj.simbad_match ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-classified/10 text-classified">
                        ✅ {obj.simbad_match}
                      </span>
                    ) : (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-flagged/10 text-flagged">
                        ❌ NO SIMBAD
                      </span>
                    )}
                    {(obj.flag_count ?? 0) >= 2 && (
                      <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${(obj.flag_count ?? 0) >= 3
                          ? "bg-red-500/20 text-red-400 animate-pulse"
                          : "bg-orange-500/15 text-orange-400"
                        }`}>
                        🔁 Seen {obj.flag_count}× — strong anomaly signal
                      </span>
                    )}
                    {llm && (
                      <span className={`text-[10px] px-2 py-0.5 rounded inline-flex items-center gap-1 ${llm.verdict === "interesting"
                          ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                          : llm.verdict === "known_type"
                            ? "bg-classified/15 text-classified border border-classified/30"
                            : "bg-muted text-muted-foreground border border-border"
                        }`}>
                        <Brain className="w-3 h-3" />
                        {llm.verdict === "known_type" ? "known" : llm.verdict} ({(llm.confidence * 100).toFixed(0)}%)
                        {llm.is_candidate === 1 && " ⭐"}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground font-mono">
                    <span>RA {obj.ra.toFixed(4)}</span>
                    <span>Dec {obj.dec.toFixed(4)}</span>
                    <span>{obj.n_detections} det</span>
                    {obj.mag_mean && <span>mag {obj.mag_mean.toFixed(2)}</span>}
                  </div>
                  {obj.triage_reason && (
                    <p className="text-xs text-muted-foreground mt-1.5 italic">{obj.triage_reason}</p>
                  )}
                  {llm && (
                    <p className="text-xs text-muted-foreground mt-1 pl-3 border-l-2 border-indigo/30">
                      🧠 {llm.reasoning.slice(0, 120)}{llm.reasoning.length > 120 ? "..." : ""}
                    </p>
                  )}
                </div>
                <div className="shrink-0">
                  <FeedbackButtons
                    oid={obj.oid}
                    onFeedbackGiven={() => handleFeedback(obj.oid)}
                    compact
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <ObjectDetailModal
        open={!!selectedObject}
        onOpenChange={(open) => !open && setSelectedObject(null)}
        object={selectedObject}
        llmReview={selectedObject ? llmMap.get(selectedObject.oid) : undefined}
      />
    </div>
  );
}
