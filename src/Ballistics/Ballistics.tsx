import React, { useEffect, useRef, useState } from 'react';
import { useInteractiveTraining } from '../InteractiveLearning/useInteractiveTraining';
import { TrainingCharts } from '../Training/TrainingCharts';
import { decode, examples, idealLaunch, input, Launch, position } from './physics';

export function Ballistics() {
  const [target, setTarget] = useState({ distance: 60, height: 12, gravity: 9.81 });
  const [aim, setAim] = useState<Launch | null>(null);
  const [shot, setShot] = useState(0);
  const [showIdeal, setShowIdeal] = useState(true);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const { history, training, status, train, stop, infer, modelVersion } = useInteractiveTraining();
  useEffect(() => {
    const values = infer([input(target)]);
    setAim(values ? decode(values) : null);
    // Inference reads the current model; it is triggered by target or completed training.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, modelVersion, training]);
  useEffect(() => {
    const context = canvas.current?.getContext('2d'); if (!context) return;
    let frame = 0;
    const start = performance.now();
    const ideal = idealLaunch(target);
    const flightTime = aim ? 2 * aim.speed * Math.sin(aim.angle) / target.gravity : 0;
    function draw(now: number) {
      const ctx = context!;
      ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, 560, 320);
      ctx.strokeStyle = '#475569'; ctx.beginPath(); ctx.moveTo(0, 280); ctx.lineTo(560, 280); ctx.stroke();
      const path = (launch: Launch, color: string, dashed: boolean) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [6, 5] : []); ctx.beginPath();
        const end = 2 * launch.speed * Math.sin(launch.angle) / target.gravity;
        for (let i = 0; i <= 120; i += 1) {
          const p = position(launch, target.gravity, i / 120 * end);
          const x = 40 + p.x * 4, y = 280 - p.y * 5;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
      };
      if (showIdeal) path(ideal, '#94a3b8', true);
      if (aim) path(aim, '#38bdf8', false);
      ctx.fillStyle = '#fbbf24'; ctx.beginPath(); ctx.arc(40 + target.distance * 4, 280 - target.height * 5, 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(32, 272, 16, 8);
      ctx.font = '12px sans-serif'; ctx.fillStyle = '#a7b2c7'; ctx.fillText('Click to move the target · distance 10–110 m / height 0–35 m', 16, 306);
      if (shot && aim) {
        const time = Math.min(flightTime, (now - start) / 1000 * 1.5);
        const p = position(aim, target.gravity, time);
        ctx.fillStyle = '#fb7185'; ctx.beginPath(); ctx.arc(40 + p.x * 4, 280 - p.y * 5, 5, 0, Math.PI * 2); ctx.fill();
        if (time < flightTime) frame = requestAnimationFrame(draw);
      }
    }
    draw(start); return () => cancelAnimationFrame(frame);
  }, [target, aim, shot, showIdeal]);
  function startTraining() {
    setShot(0);
    const data = examples(1200);
    train({ inputs: data.map((row) => row.inputs), targets: data.map((row) => row.targets), preview: [input(target)], neurons: 32, classification: false, epochs: 180, batchSize: 128, learningRate: 0.005, retainModel: true });
  }
  const arrival = aim ? target.distance / (aim.speed * Math.cos(aim.angle)) : 0;
  const miss = aim ? Math.abs(position(aim, target.gravity, arrival).y - target.height) : null;
  return <div style={{ display: 'grid', gap: 16 }}>
    <h3 style={{ margin: 0 }}>Neural ballistics</h3>
    <div style={{ fontSize: 13, color: '#a7b2c7' }}>Distance, target height, and gravity → angle and launch speed. A 3 → 32 → 32 → 2 network learns from simulated examples; shots follow physical equations.</div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <button onClick={training ? stop : startTraining}>{training ? 'stop training' : 'train aiming'}</button>
      <button disabled={!aim || training} onClick={() => setShot((previous) => previous + 1)}>launch</button>
      <label>Gravity {target.gravity.toFixed(2)} m/s² <input type="range" min={3} max={17} step={0.1} disabled={training} value={target.gravity} onChange={(event) => { setShot(0); setTarget({ ...target, gravity: Number(event.target.value) }); }} /></label>
      <label><input type="checkbox" checked={showIdeal} onChange={(event) => setShowIdeal(event.target.checked)} /> Ideal trajectory</label>
    </div>
    <div style={{ fontSize: 13, color: '#a7b2c7' }} aria-live="polite">{status} · target {target.distance.toFixed(1)} m away / {target.height.toFixed(1)} m high</div>
    {aim && <div>Network: {(aim.angle * 180 / Math.PI).toFixed(1)}° · {aim.speed.toFixed(2)} m/s · vertical miss at target: {miss!.toFixed(2)} m</div>}
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div><canvas ref={canvas} width={560} height={320} aria-label="Target, neural and ideal ballistic trajectories" onPointerDown={(event) => {
        if (training) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width * 560, y = (event.clientY - rect.top) / rect.height * 320;
        setShot(0); setTarget({ ...target, distance: Math.max(10, Math.min(110, (x - 40) / 4)), height: Math.max(0, Math.min(35, (280 - y) / 5)) });
      }} style={{ width: '100%', borderRadius: 16, border: '1px solid #475569', cursor: 'crosshair' }} />
        <p style={{ fontSize: 12, color: '#a7b2c7' }}>Blue = neural aim; gray dashed = minimum-speed ideal solution. Constant gravity, no air resistance. Move the target after training: the network recalculates without retraining. Displayed angles and speeds are bounded to 5–85° and 1–60 m/s.</p></div>
      <TrainingCharts history={history} classification={false} lossLabel="Normalized aiming MSE" validationNote="Training uses 1,200 synthetic examples. The miss is evaluated with physics at your selected target, rather than taken from the training loss." />
    </div>
  </div>;
}
