import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { AnimatedCounter } from "@/components/AnimatedCounter";
import { Brain, Loader2, CheckCircle, XCircle } from "lucide-react";
import { toast } from "sonner";
import { useState } from "react";
import type { RetrainResult } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const CHART_COLORS = {
  flagged: "#ef4444",
  classified: "#22c55e",
  dismissed: "#6b7280",
  interesting: "#f59e0b",
  known_type: "#22c55e",
  noise: "#6b7280",
  bar: "#6366f1",
};

export default function StatsTab() {
  const [retrainResult, setRetrainResult] = useState<RetrainResult | null>(null);
  const queryClient = useQueryClient();

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["stats"],
    queryFn: api.getStats,
  });

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
  });

  const { data: scoreDist } = useQuery({
    queryKey: ["score-distribution"],
    queryFn: api.getScoreDistribution,
  });

  const { data: llmStats } = useQuery({
    queryKey: ["llm-stats"],
    queryFn: api.getLLMStats,
  });

  const retrainMutation = useMutation({
    mutationFn: api.retrain,
    onSuccess: (res) => {
      setRetrainResult(res);
      toast.success("AI retrained successfully!");
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
    onError: () => toast.error("Retrain failed"),
  });

  const statCards = [
    { label: "Total Scored", value: stats?.total ?? 0, color: "text-foreground" },
    { label: "Flagged", value: stats?.flagged ?? 0, color: "text-flagged" },
    { label: "Classified", value: stats?.classified ?? 0, color: "text-classified" },
    { label: "Dismissed", value: stats?.dismissed ?? 0, color: "text-dismissed" },
    { label: "Candidates", value: stats?.discoveries ?? 0, color: "text-discovery" },
    { label: "Feedback Given", value: stats?.feedback ?? 0, color: "text-indigo" },
  ];

  const triagePieData = [
    { name: "Flagged", value: stats?.flagged ?? 0, color: CHART_COLORS.flagged },
    { name: "Classified", value: stats?.classified ?? 0, color: CHART_COLORS.classified },
    { name: "Dismissed", value: stats?.dismissed ?? 0, color: CHART_COLORS.dismissed },
  ].filter(d => d.value > 0);

  const llmBarData = llmStats?.verdicts
    ? Object.entries(llmStats.verdicts).map(([verdict, count]) => ({
      name: verdict === "known_type" ? "Known Type" : verdict.charAt(0).toUpperCase() + verdict.slice(1),
      count,
      fill: CHART_COLORS[verdict as keyof typeof CHART_COLORS] || CHART_COLORS.bar,
    }))
    : [];

  const StatusBadge = ({ active }: { active: boolean }) => (
    active
      ? <span className="inline-flex items-center gap-1 text-xs text-classified"><CheckCircle className="w-3.5 h-3.5" /> Active</span>
      : <span className="inline-flex items-center gap-1 text-xs text-flagged"><XCircle className="w-3.5 h-3.5" /> Inactive</span>
  );

  if (statsLoading || healthLoading) {
    return <div className="flex justify-center py-20"><Loader2 className="w-6 h-6 animate-spin text-indigo" /></div>;
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-foreground mb-6">📊 Stats & AI Status</h2>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
        {statCards.map((s, i) => (
          <div key={i} className="card-space p-5 text-center">
            <div className={`text-3xl font-bold font-mono ${s.color} mb-1`}>
              <AnimatedCounter target={s.value} />
            </div>
            <div className="text-xs text-muted-foreground uppercase tracking-wider">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Score Distribution */}
        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Score Distribution</h3>
          {scoreDist && scoreDist.some(d => d.count > 0) ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={scoreDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 20% 16%)" />
                <XAxis dataKey="range" tick={{ fill: "hsl(220 15% 55%)", fontSize: 11 }} />
                <YAxis tick={{ fill: "hsl(220 15% 55%)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "hsl(222 40% 8%)", border: "1px solid hsl(222 20% 16%)", borderRadius: 8 }}
                  labelStyle={{ color: "hsl(220 20% 90%)" }}
                  itemStyle={{ color: "#6366f1" }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-muted-foreground text-sm">
              No score data yet — run a session first
            </div>
          )}
        </div>

        {/* Triage Pie */}
        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-4">Triage Breakdown</h3>
          {triagePieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={triagePieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={{ stroke: "hsl(220 15% 55%)" }}
                >
                  {triagePieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: "hsl(222 40% 8%)", border: "1px solid hsl(222 20% 16%)", borderRadius: 8 }}
                  itemStyle={{ color: "hsl(220 20% 90%)" }}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "hsl(220 15% 55%)" }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-muted-foreground text-sm">
              No triage data yet
            </div>
          )}
        </div>
      </div>

      {/* LLM verdict chart */}
      {llmBarData.length > 0 && (
        <div className="card-space p-6 mb-8">
          <h3 className="text-lg font-bold text-foreground mb-4">
            🧠 LLM Verdict Distribution
            <span className="text-sm font-normal text-muted-foreground ml-2">
              ({llmStats?.total ?? 0} reviews, {llmStats?.candidates ?? 0} candidates)
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={llmBarData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 20% 16%)" />
              <XAxis type="number" tick={{ fill: "hsl(220 15% 55%)", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: "hsl(220 15% 55%)", fontSize: 12 }} width={90} />
              <Tooltip
                contentStyle={{ backgroundColor: "hsl(222 40% 8%)", border: "1px solid hsl(222 20% 16%)", borderRadius: 8 }}
                itemStyle={{ color: "hsl(220 20% 90%)" }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {llmBarData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* AI Model Status */}
      <div className="card-space p-6 mb-6">
        <h3 className="text-lg font-bold text-foreground mb-4">AI Model Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card-space-elevated p-4 flex items-center justify-between gap-2">
            <span className="text-sm text-foreground">TransformerAE</span>
            <StatusBadge active={true} />
          </div>
          <div className="card-space-elevated p-4 flex items-center justify-between gap-2">
            <span className="text-sm text-foreground">Isolation Forest</span>
            <StatusBadge active={health?.isolation_forest ?? false} />
          </div>
          <div className="card-space-elevated p-4 flex items-center justify-between gap-2">
            <span className="text-sm text-foreground">Feedback CLF</span>
            <StatusBadge active={health?.feedback_clf ?? false} />
          </div>
          <div className="card-space-elevated p-4 flex items-center justify-between gap-2">
            <span className="text-sm text-foreground">LLM Astronomer</span>
            <StatusBadge active={true} />
          </div>
        </div>
      </div>

      {/* Retrain section */}
      <div className="card-space p-6">
        <h3 className="text-lg font-bold text-foreground mb-2">🔄 AI Retraining</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Retraining happens <span className="text-foreground font-medium">automatically at the start of each session</span> using all your feedback.
          Just review flagged objects daily — no manual action needed.
          Use the button below only to retrain immediately without waiting for the next session.
        </p>
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <button
            onClick={() => retrainMutation.mutate()}
            disabled={retrainMutation.isPending}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium bg-gradient-to-r from-indigo to-cosmic text-primary-foreground glow-indigo hover:scale-105 transition-transform disabled:opacity-50 text-sm"
          >
            <Brain className={`w-4 h-4 ${retrainMutation.isPending ? "animate-spin" : ""}`} />
            {retrainMutation.isPending ? "Retraining..." : "Retrain AI Now"}
          </button>
          <p className="text-xs text-muted-foreground">
            Feedback labels collected: <span className="font-mono text-foreground">{stats?.feedback ?? 0}</span>
            {(stats?.feedback ?? 0) < 10 && " — give more feedback to improve model quality"}
          </p>
        </div>

        {retrainResult && (
          <div className="mt-4 p-4 rounded-lg bg-classified/10 border border-classified/20 text-sm">
            <p className="text-classified font-medium mb-1">
              ✅ Retrained! Feedback CLF {retrainResult.feedback_clf_trained ? "✅" : "❌"} | Isolation Forest {retrainResult.iso_forest_trained ? "✅" : "❌"}
            </p>
            <p className="text-muted-foreground font-mono text-xs">
              {retrainResult.stats.n_samples} training samples ({retrainResult.stats.n_positive} positive, {retrainResult.stats.n_negative} negative)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

