import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type LLMReview } from "@/lib/api";
import { Loader2, Star, ExternalLink, Filter } from "lucide-react";

const VERDICT_STYLES = {
    interesting: { bg: "bg-amber-500/15", border: "border-amber-500/30", text: "text-amber-400", label: "Interesting" },
    known_type: { bg: "bg-classified/15", border: "border-classified/30", text: "text-classified", label: "Known Type" },
    noise: { bg: "bg-dismissed/15", border: "border-dismissed/30", text: "text-dismissed", label: "Noise" },
} as const;

type VerdictFilter = "all" | "interesting" | "known_type" | "noise";

export default function LLMTab() {
    const [filter, setFilter] = useState<VerdictFilter>("all");

    const { data: reviews, isLoading } = useQuery({
        queryKey: ["llm-reviews"],
        queryFn: api.getLLMReviews,
    });

    const { data: llmStats } = useQuery({
        queryKey: ["llm-stats"],
        queryFn: api.getLLMStats,
    });

    if (isLoading) {
        return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 animate-spin text-indigo" /></div>;
    }

    const filtered = filter === "all" ? reviews : reviews?.filter(r => r.verdict === filter);
    const candidates = reviews?.filter(r => r.is_candidate) ?? [];

    return (
        <div>
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-foreground">🧠 LLM Astronomer</h2>
                <span className="text-sm text-muted-foreground font-mono">
                    Llama 3.3 70B (Groq)
                </span>
            </div>

            {/* Summary cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="card-space p-4 text-center">
                    <div className="text-2xl font-bold font-mono text-foreground">{llmStats?.total ?? 0}</div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wider">Total Reviewed</div>
                </div>
                <div className="card-space p-4 text-center">
                    <div className="text-2xl font-bold font-mono text-amber-400">{llmStats?.verdicts?.interesting ?? 0}</div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wider">Interesting</div>
                </div>
                <div className="card-space p-4 text-center">
                    <div className="text-2xl font-bold font-mono text-discovery">{llmStats?.candidates ?? 0}</div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wider">Candidates</div>
                </div>
                <div className="card-space p-4 text-center">
                    <div className="text-2xl font-bold font-mono text-classified">{llmStats?.verdicts?.known_type ?? 0}</div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wider">Known Types</div>
                </div>
            </div>

            {/* Filter buttons */}
            <div className="flex items-center gap-2 mb-6">
                <Filter className="w-4 h-4 text-muted-foreground" />
                {(["all", "interesting", "known_type", "noise"] as const).map(f => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${filter === f
                                ? "bg-indigo/20 text-indigo-light border border-indigo/30"
                                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                            }`}
                    >
                        {f === "all" ? "All" : f === "known_type" ? "Known Type" : f.charAt(0).toUpperCase() + f.slice(1)}
                    </button>
                ))}
                <span className="text-xs text-muted-foreground ml-auto font-mono">
                    {filtered?.length ?? 0} reviews
                </span>
            </div>

            {/* Candidate highlight */}
            {candidates.length > 0 && filter === "all" && (
                <div className="mb-6 p-4 rounded-lg bg-discovery/10 border border-discovery/20 glow-discovery">
                    <h3 className="text-sm font-bold text-discovery mb-2">⭐ Discovery Candidates</h3>
                    <div className="space-y-2">
                        {candidates.map(c => (
                            <div key={c.oid + c.timestamp} className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="font-mono text-sm text-foreground">{c.oid}</span>
                                    <span className="text-xs text-discovery">{(c.confidence * 100).toFixed(0)}% confident</span>
                                </div>
                                <span className="text-xs text-muted-foreground">{c.reasoning.slice(0, 60)}...</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Review cards */}
            <div className="space-y-3">
                {filtered?.map((review, idx) => {
                    const style = VERDICT_STYLES[review.verdict] ?? VERDICT_STYLES.noise;
                    return (
                        <div key={review.oid + review.timestamp + idx} className="card-space p-5">
                            <div className="flex flex-col md:flex-row md:items-start justify-between gap-3">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-3 mb-2 flex-wrap">
                                        <a
                                            href={`https://alerce.online/object/${review.oid}`}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="font-mono text-sm font-bold text-indigo-light hover:underline inline-flex items-center gap-1"
                                        >
                                            {review.oid} <ExternalLink className="w-3 h-3" />
                                        </a>
                                        <span className={`text-xs px-2 py-0.5 rounded-full ${style.bg} ${style.border} ${style.text} border font-medium`}>
                                            {style.label}
                                        </span>
                                        {review.is_candidate === 1 && (
                                            <span className="text-xs px-2 py-0.5 rounded-full bg-discovery/15 border border-discovery/30 text-discovery font-medium inline-flex items-center gap-1">
                                                <Star className="w-3 h-3" /> Candidate
                                            </span>
                                        )}
                                        {review.suggested_class && (
                                            <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                                                {review.suggested_class}
                                            </span>
                                        )}
                                    </div>

                                    {/* Confidence bar */}
                                    <div className="flex items-center gap-3 mb-3">
                                        <span className="text-xs text-muted-foreground w-20">Confidence</span>
                                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden max-w-xs">
                                            <div
                                                className={`h-full rounded-full transition-all ${review.confidence >= 0.8 ? "bg-classified" :
                                                        review.confidence >= 0.5 ? "bg-amber-500" : "bg-flagged"
                                                    }`}
                                                style={{ width: `${review.confidence * 100}%` }}
                                            />
                                        </div>
                                        <span className="font-mono text-xs text-foreground w-10">
                                            {(review.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>

                                    {/* Reasoning */}
                                    <div className="p-3 rounded-md bg-muted/30 border border-border">
                                        <p className="text-sm text-foreground leading-relaxed">{review.reasoning}</p>
                                    </div>
                                </div>

                                <div className="text-xs text-muted-foreground font-mono whitespace-nowrap">
                                    {new Date(review.timestamp).toLocaleDateString()}
                                </div>
                            </div>
                        </div>
                    );
                })}

                {(!filtered || filtered.length === 0) && (
                    <div className="text-center py-16 text-muted-foreground">
                        <p className="text-lg mb-2">No LLM reviews yet</p>
                        <p className="text-sm">Run a session to let the AI astronomer review flagged objects</p>
                    </div>
                )}
            </div>
        </div>
    );
}
