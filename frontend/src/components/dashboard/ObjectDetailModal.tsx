import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ExternalLink, Brain, Star } from "lucide-react";
import ScoreBadge from "./ScoreBadge";
import FeedbackButtons from "./FeedbackButtons";
import type { LLMReview } from "@/lib/api";

interface ObjectDetailModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  object: {
    oid: string;
    score: number;
    ra: number;
    dec: number;
    triage?: string;
    triage_reason?: string;
    n_detections?: number;
    mag_mean?: number;
    rec_error?: number;
    knn_dist?: number;
    simbad_match?: string | null;
    simbad_otype?: string | null;
    auto_class?: string | null;
    class_distance?: number | null;
  } | null;
  llmReview?: LLMReview;
}

export default function ObjectDetailModal({ open, onOpenChange, object, llmReview }: ObjectDetailModalProps) {
  if (!object) return null;

  const alerceUrl = `https://alerce.online/object/${object.oid}`;
  const simbadUrl = `http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=${object.ra}+${object.dec}&Radius=5&Radius.unit=arcsec`;

  const metrics = [
    { label: "Anomaly Score", value: <ScoreBadge score={object.score} /> },
    { label: "Triage", value: object.triage ?? "—" },
    { label: "RA", value: object.ra?.toFixed(4) ?? "—" },
    { label: "Dec", value: object.dec?.toFixed(4) ?? "—" },
    { label: "Detections", value: object.n_detections ?? "—" },
    { label: "Mean Mag", value: object.mag_mean?.toFixed(2) ?? "—" },
    { label: "Rec Error", value: object.rec_error?.toFixed(4) ?? "—" },
    { label: "kNN Dist", value: object.knn_dist?.toFixed(4) ?? "—" },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl bg-card border-border">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <span className="font-mono text-xl text-foreground">{object.oid}</span>
            {object.simbad_match ? (
              <span className="text-xs px-2 py-0.5 rounded bg-classified/15 text-classified border border-classified/30">
                ✅ {object.simbad_match}
              </span>
            ) : (
              <span className="text-xs px-2 py-0.5 rounded bg-flagged/15 text-flagged border border-flagged/30">
                ❌ NO SIMBAD MATCH
              </span>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          {metrics.map((m) => (
            <div key={m.label} className="card-space-elevated p-3 rounded-md">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">{m.label}</div>
              <div className="font-mono text-sm text-foreground">{m.value}</div>
            </div>
          ))}
        </div>

        {object.auto_class && (
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-muted-foreground">AI Class:</span>
            <span className="text-xs px-2 py-0.5 rounded bg-indigo/10 text-indigo-light border border-indigo/20 font-mono">
              {object.auto_class} (d={object.class_distance?.toFixed(3)})
            </span>
          </div>
        )}

        {object.triage_reason && (
          <div className="mt-3 p-3 rounded-md bg-muted/50 border border-border">
            <div className="text-xs text-muted-foreground mb-1">Triage Reason</div>
            <p className="text-sm text-foreground">{object.triage_reason}</p>
          </div>
        )}

        {/* LLM Review Section */}
        {llmReview && (
          <div className="mt-4 p-4 rounded-lg bg-indigo/5 border border-indigo/20">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="w-4 h-4 text-indigo-light" />
              <span className="text-sm font-bold text-foreground">LLM Astronomer Review</span>
              {llmReview.is_candidate === 1 && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-discovery/15 border border-discovery/30 text-discovery inline-flex items-center gap-1">
                  <Star className="w-3 h-3" /> Candidate
                </span>
              )}
            </div>

            <div className="grid grid-cols-3 gap-3 mb-3">
              <div className="text-center">
                <div className={`text-sm font-bold ${llmReview.verdict === "interesting" ? "text-amber-400" :
                    llmReview.verdict === "known_type" ? "text-classified" : "text-muted-foreground"
                  }`}>
                  {llmReview.verdict === "known_type" ? "Known Type" : llmReview.verdict.charAt(0).toUpperCase() + llmReview.verdict.slice(1)}
                </div>
                <div className="text-[10px] text-muted-foreground uppercase">Verdict</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-foreground font-mono">
                  {(llmReview.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-muted-foreground uppercase">Confidence</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-foreground font-mono">
                  {llmReview.suggested_class ?? "—"}
                </div>
                <div className="text-[10px] text-muted-foreground uppercase">Suggested Class</div>
              </div>
            </div>

            {/* Confidence bar */}
            <div className="flex items-center gap-2 mb-3">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${llmReview.confidence >= 0.8 ? "bg-classified" :
                      llmReview.confidence >= 0.5 ? "bg-amber-500" : "bg-flagged"
                    }`}
                  style={{ width: `${llmReview.confidence * 100}%` }}
                />
              </div>
            </div>

            <div className="p-3 rounded bg-muted/30 border border-border">
              <p className="text-sm text-foreground leading-relaxed">{llmReview.reasoning}</p>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3 mt-4">
          <a href={alerceUrl} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-indigo/10 text-indigo-light border border-indigo/20 hover:bg-indigo/20 transition-colors">
            View on ALeRCE <ExternalLink className="w-3 h-3" />
          </a>
          <a href={simbadUrl} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-cosmic/10 text-cosmic-light border border-cosmic/20 hover:bg-cosmic/20 transition-colors">
            Query SIMBAD <ExternalLink className="w-3 h-3" />
          </a>
        </div>

        <div className="mt-4 pt-4 border-t border-border">
          <FeedbackButtons oid={object.oid} onFeedbackGiven={() => onOpenChange(false)} />
        </div>
      </DialogContent>
    </Dialog>
  );
}
