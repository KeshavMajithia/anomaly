import { Link } from "react-router-dom";
import { StarField } from "@/components/StarField";
import { AnimatedCounter } from "@/components/AnimatedCounter";
import { Telescope, Brain, Sparkles, ArrowRight, Bot } from "lucide-react";

export default function Landing() {
  return (
    <div className="min-h-screen bg-space-gradient relative overflow-hidden">
      <StarField />

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 text-center">
        <div className="animate-fade-in">
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight gradient-text text-glow mb-6">
            Stellar Anomaly Hunter
          </h1>
          <p className="text-lg md:text-xl max-w-2xl mx-auto text-muted-foreground mb-12 leading-relaxed">
            Self-improving AI that discovers the unknown in the night sky
          </p>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-lg font-semibold text-lg bg-gradient-to-r from-indigo to-cosmic text-primary-foreground glow-indigo hover:scale-105 transition-transform duration-200"
          >
            Launch Dashboard
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Stats */}
      <section className="relative z-10 py-20 px-6">
        <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            { value: 74831, label: "Objects Scanned" },
            { value: 4, label: "Source Self-Learning", suffix: "-source" },
            { value: 1, label: "Real-time ZTF Data", suffix: "" },
          ].map((stat, i) => (
            <div
              key={i}
              className="card-space p-8 text-center"
              style={{ animationDelay: `${i * 150}ms` }}
            >
              <div className="text-4xl md:text-5xl font-bold font-mono gradient-text mb-2">
                {stat.label === "Real-time ZTF Data" ? (
                  <span>Real-time</span>
                ) : (
                  <AnimatedCounter target={stat.value} suffix={stat.suffix} />
                )}
              </div>
              <div className="text-muted-foreground text-sm uppercase tracking-wider">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="relative z-10 py-20 px-6">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-16 gradient-text">
          How It Works
        </h2>
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            {
              icon: Telescope,
              title: "Scan",
              desc: "Ingest real-time ZTF light curves from the night sky",
            },
            {
              icon: Brain,
              title: "Score",
              desc: "TransformerAE + Isolation Forest + Feedback Classifier pipeline",
            },
            {
              icon: Bot,
              title: "LLM Review",
              desc: "AI astronomer (Llama 3.3) provides expert reasoning on flagged objects",
            },
            {
              icon: Sparkles,
              title: "Discover",
              desc: "Flag the unknown, learn from human & AI feedback, improve every session",
            },
          ].map((step, i) => (
            <div
              key={i}
              className="card-space p-8 text-center group hover:glow-indigo transition-shadow duration-500 animate-slide-up"
              style={{ animationDelay: `${i * 200}ms` }}
            >
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-indigo to-cosmic flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <step.icon className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-bold mb-3 text-foreground">{step.title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Description */}
      <section className="relative z-10 py-16 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <p className="text-muted-foreground leading-relaxed text-lg">
            Our 4-source self-improving pipeline — human feedback, self-validation,
            pseudo-labeling, and LLM astronomer — processes live telescope data,
            flags genuinely unknown objects, and gets smarter with every session.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-8 px-6 border-t border-border">
        <p className="text-center text-sm text-muted-foreground">
          Built for astronomical anomaly detection research
        </p>
      </footer>
    </div>
  );
}
