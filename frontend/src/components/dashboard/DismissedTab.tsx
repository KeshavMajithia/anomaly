import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import ScoreBadge from "./ScoreBadge";
import { Loader2, RotateCw, ArrowUp } from "lucide-react";
import { toast } from "sonner";

export default function DismissedTab() {
  const [rescanResult, setRescanResult] = useState<number | null>(null);
  const queryClient = useQueryClient();

  const { data: objects, isLoading } = useQuery({
    queryKey: ["dismissed"],
    queryFn: () => api.getDismissed(200),
  });

  const rescanMutation = useMutation({
    mutationFn: api.rescan,
    onSuccess: (res) => {
      setRescanResult(res.promoted);
      toast.success(`✅ ${res.promoted} objects promoted from dismissed to flagged`);
      queryClient.invalidateQueries({ queryKey: ["dismissed"] });
      queryClient.invalidateQueries({ queryKey: ["flagged"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
    onError: () => toast.error("Rescan failed"),
  });

  if (isLoading) {
    return <div className="flex justify-center py-20"><Loader2 className="w-6 h-6 animate-spin text-indigo" /></div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-foreground">📦 Dismissed</h2>
        <button
          onClick={() => rescanMutation.mutate()}
          disabled={rescanMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-indigo/15 text-indigo-light border border-indigo/20 hover:bg-indigo/25 transition-colors disabled:opacity-50"
        >
          <RotateCw className={`w-4 h-4 ${rescanMutation.isPending ? "animate-spin" : ""}`} />
          Run Retrospective Rescan
        </button>
      </div>

      {rescanResult !== null && (
        <div className="mb-4 p-3 rounded-lg bg-classified/10 border border-classified/20 text-classified text-sm">
          ✅ {rescanResult} objects promoted from dismissed to flagged
        </div>
      )}

      <div className="card-space overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                {["OID", "Score", "Detections", "RA", "Dec"].map((h) => (
                  <th key={h} className="text-left p-3 text-[10px] uppercase tracking-wider text-muted-foreground">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {objects?.map((obj) => (
                <tr key={obj.oid} className="border-b border-border/50 hover:bg-muted/20 transition-colors">
                  <td className="p-3 font-mono text-sm text-indigo-light">{obj.oid}</td>
                  <td className="p-3"><ScoreBadge score={obj.score} /></td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.n_detections}</td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.ra.toFixed(4)}</td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.dec.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
