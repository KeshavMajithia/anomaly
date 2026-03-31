import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  size: number;
  speed: number;
  opacity: number;
  flickerSpeed: number;
  flickerPhase: number;
  isAnomaly: boolean;
  anomalyTimer: number;
  anomalyFlash: number;
}

export function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let stars: Star[] = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initStars();
    };

    const initStars = () => {
      const count = Math.floor((canvas.width * canvas.height) / 3000);
      stars = Array.from({ length: count }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 1.5 + 0.3,
        speed: Math.random() * 0.15 + 0.02,
        opacity: Math.random() * 0.8 + 0.2,
        flickerSpeed: Math.random() * 0.02 + 0.005,
        flickerPhase: Math.random() * Math.PI * 2,
        isAnomaly: Math.random() < 0.008,
        anomalyTimer: 0,
        anomalyFlash: 0,
      }));
    };

    const render = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const star of stars) {
        star.y -= star.speed;
        star.x += Math.sin(time * 0.0003 + star.flickerPhase) * 0.03;

        if (star.y < -5) {
          star.y = canvas.height + 5;
          star.x = Math.random() * canvas.width;
        }

        const flicker = Math.sin(time * star.flickerSpeed + star.flickerPhase) * 0.3 + 0.7;
        let alpha = star.opacity * flicker;

        if (star.isAnomaly) {
          star.anomalyTimer += 0.016;
          if (star.anomalyTimer > 8 + Math.random() * 15) {
            star.anomalyFlash = 1;
            star.anomalyTimer = 0;
          }
          if (star.anomalyFlash > 0) {
            star.anomalyFlash -= 0.02;
            const flash = star.anomalyFlash;
            ctx.beginPath();
            const gradient = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, 15 * flash);
            gradient.addColorStop(0, `rgba(168, 85, 247, ${0.8 * flash})`);
            gradient.addColorStop(0.5, `rgba(99, 102, 241, ${0.3 * flash})`);
            gradient.addColorStop(1, "transparent");
            ctx.fillStyle = gradient;
            ctx.arc(star.x, star.y, 15 * flash, 0, Math.PI * 2);
            ctx.fill();
            alpha = 1;
          }
        }

        ctx.beginPath();
        ctx.fillStyle = star.isAnomaly && star.anomalyFlash > 0
          ? `rgba(168, 85, 247, ${alpha})`
          : `rgba(200, 210, 240, ${alpha})`;
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fill();
      }

      animId = requestAnimationFrame(render);
    };

    resize();
    window.addEventListener("resize", resize);
    animId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
