import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type FeedbackRequest } from "@/lib/api";
import { Sparkles, Ban, Tag } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const CLASS_OPTIONS = [
  { value: "E", label: "E — Eclipsing Binary" },
  { value: "BYDra", label: "BYDra — BY Draconis" },
  { value: "RSCVN", label: "RSCVN — RS Canum Ven." },
  { value: "LPV", label: "LPV — Long Period Variable" },
  { value: "RRL", label: "RRL — RR Lyrae" },
  { value: "DSCT", label: "DSCT — Delta Scuti" },
  { value: "Other", label: "Other" },
];

interface FeedbackButtonsProps {
  oid: string;
  onFeedbackGiven?: () => void;
  compact?: boolean;
}

export default function FeedbackButtons({ oid, onFeedbackGiven, compact = false }: FeedbackButtonsProps) {
  const [showClassify, setShowClassify] = useState(false);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (data: FeedbackRequest) => api.sendFeedback(data),
    onSuccess: (res) => {
      toast.success(res.message);
      queryClient.invalidateQueries({ queryKey: ["flagged"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["discoveries"] });
      onFeedbackGiven?.();
    },
    onError: () => toast.error("Failed to send feedback"),
  });

  const btnClass = cn(
    "flex items-center gap-1.5 rounded-md font-medium transition-all duration-200",
    compact ? "px-2.5 py-1.5 text-xs" : "px-3 py-2 text-sm"
  );

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        onClick={() => mutation.mutate({ oid, action: "interesting" })}
        disabled={mutation.isPending}
        className={cn(btnClass, "bg-discovery/15 text-discovery hover:bg-discovery/25 border border-discovery/20")}
      >
        <Sparkles className="w-3.5 h-3.5" />
        Interesting
      </button>
      <button
        onClick={() => mutation.mutate({ oid, action: "noise" })}
        disabled={mutation.isPending}
        className={cn(btnClass, "bg-dismissed/15 text-dismissed hover:bg-dismissed/25 border border-dismissed/20")}
      >
        <Ban className="w-3.5 h-3.5" />
        Noise
      </button>
      <div className="relative">
        <button
          onClick={() => setShowClassify(!showClassify)}
          className={cn(btnClass, "bg-classified/15 text-classified hover:bg-classified/25 border border-classified/20")}
        >
          <Tag className="w-3.5 h-3.5" />
          Classify
        </button>
        {showClassify && (
          <div className="absolute top-full left-0 mt-1 z-50 bg-card border border-border rounded-lg shadow-xl p-1 min-w-[200px]">
            {CLASS_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => {
                  mutation.mutate({ oid, action: "classify", label: opt.value });
                  setShowClassify(false);
                }}
                className="w-full text-left px-3 py-1.5 text-sm text-foreground hover:bg-muted rounded-md transition-colors font-mono"
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
