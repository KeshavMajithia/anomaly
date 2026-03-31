import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type AnomalyObject } from "@/lib/api";
import ScoreBadge from "./ScoreBadge";
import ObjectDetailModal from "./ObjectDetailModal";
import { Loader2, ArrowUpDown } from "lucide-react";

type SortKey = "score" | "auto_class" | "class_distance";

export default function ClassifiedTab() {
  const [selectedObject, setSelectedObject] = useState<AnomalyObject | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortAsc, setSortAsc] = useState(false);

  const { data: objects, isLoading } = useQuery({
    queryKey: ["classified"],
    queryFn: () => api.getClassified(200),
  });

  const sorted = objects?.slice().sort((a, b) => {
    const av = a[sortKey] ?? 0;
    const bv = b[sortKey] ?? 0;
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  if (isLoading) {
    return <div className="flex justify-center py-20"><Loader2 className="w-6 h-6 animate-spin text-indigo" /></div>;
  }

  const SortHeader = ({ label, k }: { label: string; k: SortKey }) => (
    <button onClick={() => toggleSort(k)} className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground">
      {label} <ArrowUpDown className="w-3 h-3" />
    </button>
  );

  return (
    <div>
      <h2 className="text-2xl font-bold text-foreground mb-6">🏷️ Auto-Classified</h2>
      <div className="card-space overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-3 text-[10px] uppercase tracking-wider text-muted-foreground">OID</th>
                <th className="text-left p-3"><SortHeader label="Auto Class" k="auto_class" /></th>
                <th className="text-left p-3"><SortHeader label="Score" k="score" /></th>
                <th className="text-left p-3"><SortHeader label="Class Dist" k="class_distance" /></th>
                <th className="text-left p-3 text-[10px] uppercase tracking-wider text-muted-foreground">RA</th>
                <th className="text-left p-3 text-[10px] uppercase tracking-wider text-muted-foreground">Dec</th>
              </tr>
            </thead>
            <tbody>
              {sorted?.map((obj) => (
                <tr key={obj.oid} className="border-b border-border/50 hover:bg-muted/20 transition-colors">
                  <td className="p-3">
                    <button onClick={() => setSelectedObject(obj)} className="font-mono text-sm text-indigo-light hover:underline">
                      {obj.oid}
                    </button>
                  </td>
                  <td className="p-3 font-mono text-sm text-classified">{obj.auto_class ?? "—"}</td>
                  <td className="p-3"><ScoreBadge score={obj.score} /></td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.class_distance?.toFixed(4) ?? "—"}</td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.ra.toFixed(4)}</td>
                  <td className="p-3 font-mono text-xs text-muted-foreground">{obj.dec.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <ObjectDetailModal open={!!selectedObject} onOpenChange={(o) => !o && setSelectedObject(null)} object={selectedObject} />
    </div>
  );
}
