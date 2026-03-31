import { ExternalLink } from "lucide-react";

export default function AboutTab() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-2xl font-bold text-foreground mb-6">ℹ️ About Stellar Anomaly Hunter</h2>

      <div className="space-y-6">
        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-3">Architecture</h3>
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
            The system uses a three-source pipeline to detect genuine anomalies in ZTF light curve data:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="card-space-elevated p-4">
              <div className="text-sm font-bold text-indigo-light mb-1">TransformerAE</div>
              <p className="text-xs text-muted-foreground">Autoencoder reconstructs light curves. High reconstruction error = anomalous.</p>
            </div>
            <div className="card-space-elevated p-4">
              <div className="text-sm font-bold text-cosmic-light mb-1">Isolation Forest</div>
              <p className="text-xs text-muted-foreground">Unsupervised outlier detection on feature space. Retrained with new data.</p>
            </div>
            <div className="card-space-elevated p-4">
              <div className="text-sm font-bold text-discovery mb-1">Feedback Classifier</div>
              <p className="text-xs text-muted-foreground">Learns from human labels. Grows more accurate with each interaction.</p>
            </div>
          </div>
        </div>

        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-3">Scoring Formula</h3>
          <div className="bg-muted/30 rounded-lg p-4 font-mono text-sm text-center text-foreground">
            final_score = 0.4 × AE + 0.3 × IF + 0.3 × Feedback
          </div>
        </div>

        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-3">Self-Improvement</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Every human feedback action (interesting, noise, classify) trains the Feedback Classifier.
            Periodic retrospective rescans re-evaluate dismissed objects with updated models.
            The system learns from its own mistakes — objects initially dismissed may be promoted
            to flagged after retraining.
          </p>
        </div>

        <div className="card-space p-6">
          <h3 className="text-lg font-bold text-foreground mb-3">External Resources</h3>
          <div className="flex flex-wrap gap-3">
            {[
              { label: "ALeRCE", url: "https://alerce.online" },
              { label: "ZTF", url: "https://www.ztf.caltech.edu" },
              { label: "SIMBAD", url: "http://simbad.u-strasbg.fr/simbad/" },
            ].map((link) => (
              <a key={link.label} href={link.url} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-md text-sm bg-indigo/10 text-indigo-light border border-indigo/20 hover:bg-indigo/20 transition-colors">
                {link.label} <ExternalLink className="w-3 h-3" />
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
