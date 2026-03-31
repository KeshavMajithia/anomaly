import { cn } from "@/lib/utils";

interface ScoreBadgeProps {
  score: number;
  className?: string;
}

export default function ScoreBadge({ score, className }: ScoreBadgeProps) {
  const color = score >= 0.7
    ? "bg-flagged/15 text-flagged border-flagged/30"
    : score >= 0.4
      ? "bg-discovery/15 text-discovery border-discovery/30"
      : "bg-indigo/15 text-indigo border-indigo/30";

  return (
    <span className={cn(
      "inline-flex items-center px-2 py-0.5 rounded-md text-xs font-mono font-bold border",
      color,
      className
    )}>
      {score.toFixed(4)}
    </span>
  );
}
