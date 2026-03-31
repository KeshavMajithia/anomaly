import { useState, useEffect } from "react";
import { Star, Sparkles, Tag, Package, BarChart3, Info, Menu, Brain, X } from "lucide-react";
import { cn } from "@/lib/utils";
import ReviewTab from "@/components/dashboard/ReviewTab";
import ClassifiedTab from "@/components/dashboard/ClassifiedTab";
import DismissedTab from "@/components/dashboard/DismissedTab";
import DiscoveriesTab from "@/components/dashboard/DiscoveriesTab";
import StatsTab from "@/components/dashboard/StatsTab";
import AboutTab from "@/components/dashboard/AboutTab";
import LLMTab from "@/components/dashboard/LLMTab";
import DashboardHeader from "@/components/dashboard/DashboardHeader";

const tabs = [
  { id: "review", label: "Review", icon: Star, color: "text-flagged" },
  { id: "llm", label: "LLM Astronomer", icon: Brain, color: "text-amber-400" },
  { id: "classified", label: "Classified", icon: Tag, color: "text-classified" },
  { id: "dismissed", label: "Dismissed", icon: Package, color: "text-dismissed" },
  { id: "discoveries", label: "Candidates", icon: Sparkles, color: "text-discovery" },
  { id: "stats", label: "Stats", icon: BarChart3, color: "text-indigo" },
  { id: "about", label: "About", icon: Info, color: "text-muted-foreground" },
] as const;

type TabId = typeof tabs[number]["id"];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>("review");
  // On mobile (< md), sidebar starts closed; on desktop it starts open
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Detect screen size
  useEffect(() => {
    const check = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      // Auto-open sidebar on desktop, auto-close on mobile
      if (!mobile) setSidebarOpen(true);
      else setSidebarOpen(false);
    };
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const handleTabChange = (id: TabId) => {
    setActiveTab(id);
    // On mobile, close sidebar after selecting a tab
    if (isMobile) setSidebarOpen(false);
  };

  const renderTab = () => {
    switch (activeTab) {
      case "review": return <ReviewTab />;
      case "llm": return <LLMTab />;
      case "classified": return <ClassifiedTab />;
      case "dismissed": return <DismissedTab />;
      case "discoveries": return <DiscoveriesTab />;
      case "stats": return <StatsTab />;
      case "about": return <AboutTab />;
    }
  };

  return (
    <div className="min-h-screen bg-background flex relative">
      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 h-full z-30 bg-sidebar border-r border-sidebar-border transition-all duration-300 flex flex-col",
          // Desktop: collapse to icon-only; Mobile: slide in/out
          isMobile
            ? sidebarOpen ? "w-56 shadow-2xl" : "-translate-x-full w-56"
            : sidebarOpen ? "w-56" : "w-14"
        )}
      >
        <div className="p-3 flex items-center justify-between border-b border-sidebar-border h-14 shrink-0">
          {sidebarOpen && (
            <span className="font-bold text-sm gradient-text truncate">Stellar AH</span>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-md hover:bg-sidebar-accent transition-colors ml-auto"
            aria-label="Toggle sidebar"
          >
            {isMobile && sidebarOpen
              ? <X className="w-5 h-5 text-sidebar-foreground" />
              : <Menu className="w-5 h-5 text-sidebar-foreground" />
            }
          </button>
        </div>
        <nav className="flex-1 py-3 px-2 space-y-1 overflow-y-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={cn(
                "w-full flex items-center gap-3 px-2.5 py-2.5 rounded-md text-sm transition-all duration-200",
                activeTab === tab.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50"
              )}
            >
              <tab.icon className={cn("w-4 h-4 flex-shrink-0", activeTab === tab.id && tab.color)} />
              {sidebarOpen && <span className="truncate">{tab.label}</span>}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main content — shifts right on desktop when sidebar is open */}
      <div
        className={cn(
          "flex-1 min-w-0 transition-all duration-300",
          // Only push content on desktop; mobile sidebar overlays
          !isMobile && (sidebarOpen ? "ml-56" : "ml-14")
        )}
      >
        {/* Mobile top bar with menu button */}
        {isMobile && (
          <div className="fixed top-0 left-0 right-0 z-10 h-14 bg-card/80 backdrop-blur-sm border-b border-border flex items-center px-4 gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-md hover:bg-sidebar-accent transition-colors"
            >
              <Menu className="w-5 h-5 text-foreground" />
            </button>
            <span className="font-bold text-sm gradient-text">{tabs.find(t => t.id === activeTab)?.label}</span>
          </div>
        )}
        <div className={isMobile ? "pt-14" : ""}>
          <DashboardHeader />
          <main className="p-4 md:p-6">
            {renderTab()}
          </main>
        </div>
      </div>
    </div>
  );
}
