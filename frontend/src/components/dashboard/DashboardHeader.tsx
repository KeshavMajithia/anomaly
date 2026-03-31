import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Link } from "react-router-dom";
import { Activity } from "lucide-react";

export default function DashboardHeader() {
  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: api.getStats,
    refetchInterval: 10000,
  });

  const items = [
    { label: "Scanned", value: stats?.total ?? 0, color: "text-foreground" },
    { label: "Flagged", value: stats?.flagged ?? 0, color: "text-flagged" },
    { label: "Classified", value: stats?.classified ?? 0, color: "text-classified" },
    { label: "Dismissed", value: stats?.dismissed ?? 0, color: "text-dismissed" },
    { label: "Candidates", value: stats?.discoveries ?? 0, color: "text-discovery" },
  ];

  return (
    <header className="h-14 border-b border-border px-6 flex items-center justify-between bg-card/50 backdrop-blur-sm">
      <Link to="/" className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
        <Activity className="w-4 h-4 text-indigo" />
        <span className="font-mono text-xs">MISSION CONTROL</span>
      </Link>
      <div className="flex items-center gap-6">
        {items.map((item) => (
          <div key={item.label} className="text-center hidden sm:block">
            <div className={`font-mono text-sm font-bold ${item.color}`}>
              {item.value.toLocaleString()}
            </div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{item.label}</div>
          </div>
        ))}
      </div>
    </header>
  );
}
