import { useEffect, useRef } from 'react';

export const NetworkBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    const GRID = 80;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const xOff = (canvas.width % GRID) / 2;
      const yOff = (canvas.height % GRID) / 2;

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;

      for (let x = xOff; x <= canvas.width; x += GRID) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }

      for (let y = yOff; y <= canvas.height; y += GRID) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      const time = Date.now() * 0.001;
      const cols = Math.ceil(canvas.width / GRID);
      const rows = Math.ceil(canvas.height / GRID);

      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const noise = Math.sin(c * 0.7 + time * 0.3) * Math.cos(r * 0.5 + time * 0.2);
          if (noise > 0.6) {
            const cx = xOff + c * GRID;
            const cy = yOff + r * GRID;
            const alpha = (noise - 0.6) * 0.15;
            ctx.fillStyle = `rgba(200, 255, 0, ${alpha})`;
            ctx.fillRect(cx, cy, GRID, GRID);
          }
        }
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none opacity-60"
    />
  );
};
