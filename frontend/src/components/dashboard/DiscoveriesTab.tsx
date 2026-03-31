import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { ExternalLink, Loader2, Sparkles } from "lucide-react";

export default function DiscoveriesTab() {
  const { data: discoveries, isLoading } = useQuery({
    queryKey: ["discoveries"],
    queryFn: api.getDiscoveries,
  });

  if (isLoading) {
    return <div className="flex justify-center py-20"><Loader2 className="w-6 h-6 animate-spin text-indigo" /></div>;
  }

  return (
    <div>
      <h2 className="text-2xl font-bold text-foreground mb-6">🔭 Candidates</h2>
      {discoveries?.length === 0 ? (
        <div className="card-space p-12 text-center">
          <Sparkles className="w-12 h-12 mx-auto mb-4 text-discovery/30" />
          <p className="text-muted-foreground">No candidates yet. Mark objects as Interesting from the Review tab to track them here for follow-up.</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {discoveries?.map((d) => (
            <div key={d.oid} className="card-space p-5 glow-discovery border-discovery/20 hover:border-discovery/40 transition-colors">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-discovery" />
                <span className="font-mono text-sm font-bold text-foreground">{d.oid}</span>
                <span className="ml-auto font-mono text-xs text-discovery font-bold">{d.score.toFixed(4)}</span>
                {(d.flag_count ?? 0) >= 2 && (
                  <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${(d.flag_count ?? 0) >= 3
                      ? "bg-red-500/20 text-red-400"
                      : "bg-orange-500/15 text-orange-400"
                    }`}>
                    🔁 {d.flag_count}× flagged
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono text-muted-foreground mb-3">
                <span>RA {d.ra.toFixed(4)}</span>
                <span>Dec {d.dec.toFixed(4)}</span>
                <span>{d.simbad_match ? `SIMBAD: ${d.simbad_match}` : "No SIMBAD match"}</span>
                <span>By: {d.confirmed_by}</span>
              </div>
              <div className="flex gap-2">
                <a href={`https://alerce.online/object/${d.oid}`} target="_blank" rel="noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs bg-indigo/10 text-indigo-light hover:bg-indigo/20 transition-colors">
                  ALeRCE <ExternalLink className="w-3 h-3" />
                </a>
                <a href={`http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=${d.ra}+${d.dec}&Radius=5&Radius.unit=arcsec`}
                  target="_blank" rel="noreferrer"
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs bg-cosmic/10 text-cosmic-light hover:bg-cosmic/20 transition-colors">
                  SIMBAD <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
