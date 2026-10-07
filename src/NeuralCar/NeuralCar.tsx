import React, { useEffect, useRef, useState } from 'react';
import { breed, Car, createCar, fitness, Genome, randomGenome, SENSOR_ANGLES, SENSOR_RANGE, sensors, step, TRACK } from './simulation';

interface Generation { generation: number; best: number; average: number; scores: number[] }
interface Simulation { cars: Car[]; generation: number; champion: Genome | null; best: number; mode: 'evolve' | 'champion'; trail: [number, number][] }
const initial = (): Simulation => ({ cars: Array.from({ length: 40 }, () => createCar(randomGenome())), generation: 1, champion: null, best: 0, mode: 'evolve', trail: [] });

function EvolutionCharts({ history }: { history: Generation[] }) {
  const max = Math.max(0.1, ...history.map((row) => row.best));
  const x = (index: number) => 40 + index / Math.max(1, history.length - 1) * 300;
  const y = (value: number) => 150 - value / max * 120;
  return <div style={{ display: 'grid', gap: 12 }}>
    <div style={{ fontWeight: 700 }}>Distance per generation · laps</div>
    <svg viewBox="0 0 360 190" role="img" aria-label="Best and average laps per generation" style={{ width: '100%' }}>
      {[0, 0.5, 1].map((fraction) => <g key={fraction}>
        <line x1={40} x2={340} y1={y(max * fraction)} y2={y(max * fraction)} stroke="#334155" />
        <text x={34} y={y(max * fraction) + 4} textAnchor="end" fill="#a7b2c7" fontSize={10}>{(max * fraction).toFixed(2)}</text>
      </g>)}
      {(['best', 'average'] as const).map((key, series) => <g key={key}>
        <path d={history.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(row[key])}`).join(' ')} fill="none" stroke={series ? '#fbbf24' : '#38bdf8'} strokeWidth={2} />
        {history.map((row, index) => <circle key={index} cx={x(index)} cy={y(row[key])} r={3} fill={series ? '#fbbf24' : '#38bdf8'}><title>{`Generation ${row.generation}, ${key}: ${row[key].toFixed(3)} laps`}</title></circle>)}
      </g>)}
      <text x={40} y={168} fill="#a7b2c7" fontSize={10}>1</text><text x={340} y={168} textAnchor="end" fill="#a7b2c7" fontSize={10}>{history.length || 1}</text>
      <text x={190} y={185} textAnchor="middle" fill="#a7b2c7" fontSize={10}>Generation</text>
      {!history.length && <text x={190} y={95} textAnchor="middle" fill="#a7b2c7" fontSize={12}>Waiting for the first generation…</text>}
    </svg>
    <div style={{ fontSize: 12 }}><span style={{ color: '#38bdf8' }}>● Best</span>　<span style={{ color: '#fbbf24' }}>● Population average</span></div>
    <div style={{ fontWeight: 700 }}>Population heatmap</div>
    <div style={{ maxHeight: 170, overflow: 'auto' }}>
      {history.map((row) => <div key={row.generation} style={{ display: 'flex', height: 12, gap: 1, marginBottom: 2 }}>
        <span style={{ width: 30, flexShrink: 0, fontSize: 10 }}>{row.generation}</span>
        {row.scores.map((score, index) => <div key={index} title={`Generation ${row.generation}, rank ${index + 1}: ${score.toFixed(3)} laps`}
          style={{ flex: 1, minWidth: 3, background: `hsl(200, 85%, ${18 + Math.min(1, score / max) * 55}%)` }} />)}
      </div>)}
    </div>
    <div style={{ fontSize: 12, color: '#a7b2c7' }}>Rows = generations; columns = drivers ranked by distance. Lighter = farther.</div>
  </div>;
}

export function NeuralCar() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simulation = useRef<Simulation | null>(null);
  if (!simulation.current) simulation.current = initial();
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [showSensors, setShowSensors] = useState(true);
  const [history, setHistory] = useState<Generation[]>([]);
  const [stats, setStats] = useState({ generation: 1, alive: 40, best: 0, current: 0 });
  const [mode, setMode] = useState<'evolve' | 'champion'>('evolve');

  useEffect(() => {
    let frame = 0;
    let lastFrame = 0;
    let lastUpdate = 0;
    const draw = (time: number) => {
      const sim = simulation.current!;
      const context = canvasRef.current?.getContext('2d');
      if (!context) return;
      if (running) {
        const elapsed = lastFrame ? Math.min(50, time - lastFrame) : 1000 / 60;
        const steps = Math.max(1, Math.round(elapsed / (1000 / 60) * speed));
        for (let tick = 0; tick < steps; tick += 1) {
          sim.cars.forEach(step);
          const leader = sim.cars.reduce((a, b) => fitness(a) > fitness(b) ? a : b);
          if (sim.mode === 'champion') {
            sim.trail.push([leader.x, leader.y]);
            if (sim.trail.length > 2000) sim.trail.shift();
          }
          if (sim.cars.every((car) => !car.alive)) {
            if (sim.mode === 'champion') { setRunning(false); break; }
            const scores = sim.cars.map(fitness).sort((a, b) => b - a);
            if (!sim.champion || scores[0] > sim.best) { sim.champion = [...leader.genome]; sim.best = scores[0]; }
            const record = { generation: sim.generation, best: scores[0], average: scores.reduce((a, b) => a + b, 0) / scores.length, scores };
            setHistory((previous) => [...previous, record]);
            sim.cars = breed(sim.cars);
            sim.generation += 1;
          }
        }
      }
      lastFrame = time;
      context.fillStyle = '#0f172a'; context.fillRect(0, 0, TRACK.width, TRACK.height);
      context.fillStyle = '#334155'; context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx, TRACK.ry, 0, 0, Math.PI * 2); context.fill();
      context.fillStyle = '#0f172a'; context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx * TRACK.inner, TRACK.ry * TRACK.inner, 0, 0, Math.PI * 2); context.fill();
      context.strokeStyle = '#94a3b8'; context.lineWidth = 1; context.setLineDash([8, 10]);
      context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx * 0.82, TRACK.ry * 0.82, 0, 0, Math.PI * 2); context.stroke(); context.setLineDash([]);
      context.strokeStyle = '#f8fafc'; context.lineWidth = 4; context.beginPath(); context.moveTo(TRACK.cx + TRACK.rx * TRACK.inner, TRACK.cy); context.lineTo(TRACK.cx + TRACK.rx, TRACK.cy); context.stroke();
      if (sim.trail.length) {
        context.strokeStyle = '#38bdf8'; context.lineWidth = 2; context.beginPath(); sim.trail.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y)); context.stroke();
      }
      const leader = sim.cars.reduce((a, b) => fitness(a) > fitness(b) ? a : b);
      sim.cars.forEach((car) => {
        context.save(); context.translate(car.x, car.y); context.rotate(car.heading);
        context.fillStyle = car === leader ? '#38bdf8' : car.alive ? 'rgba(251,191,36,0.6)' : 'rgba(148,163,184,0.2)';
        context.fillRect(-7, -4, 14, 8); context.fillStyle = '#e2e8f0'; context.fillRect(3, -3, 3, 6); context.restore();
      });
      if (showSensors && leader.alive) {
        const readings = sensors(leader);
        readings.forEach((value, index) => {
          const angle = leader.heading + SENSOR_ANGLES[index];
          context.strokeStyle = 'rgba(74,222,128,0.7)'; context.lineWidth = 1;
          context.beginPath(); context.moveTo(leader.x, leader.y); context.lineTo(leader.x + Math.cos(angle) * value * SENSOR_RANGE, leader.y + Math.sin(angle) * value * SENSOR_RANGE); context.stroke();
        });
      }
      context.fillStyle = '#a7b2c7'; context.font = '16px sans-serif'; context.textAlign = 'center';
      context.fillText(sim.mode === 'evolve' ? '40 neural drivers · clockwise' : 'Champion replay', TRACK.cx, TRACK.cy);
      if (time - lastUpdate > 100) {
        lastUpdate = time;
        setStats({ generation: sim.generation, alive: sim.cars.filter((car) => car.alive).length, best: sim.best, current: fitness(leader) });
      }
      frame = window.requestAnimationFrame(draw);
    };
    frame = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(frame);
  }, [running, speed, showSensors]);

  function reset() {
    setRunning(false); simulation.current = initial(); setHistory([]); setMode('evolve');
    setStats({ generation: 1, alive: 40, best: 0, current: 0 });
  }
  function champion() {
    const sim = simulation.current!;
    if (!sim.champion) return;
    sim.cars = [createCar(sim.champion)]; sim.mode = 'champion'; sim.trail = [];
    setMode('champion'); setRunning(true);
  }
  function evolve() {
    const sim = simulation.current!;
    if (sim.mode === 'champion') {
      sim.cars = Array.from({ length: 40 }, () => createCar(sim.champion!));
      sim.cars = breed(sim.cars); sim.mode = 'evolve'; sim.trail = [];
    }
    setMode('evolve'); setRunning(true);
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
    <h3 style={{ margin: 0 }}>Neural racing</h3>
    <div style={{ color: '#a7b2c7', fontSize: 13 }}>5 distance sensors + speed → 8 neurons → steering and throttle. 40 drivers evolve through selection and mutation; no predefined driving rules.</div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <button onClick={running ? () => setRunning(false) : mode === 'champion' ? champion : evolve}>{running ? 'pause' : mode === 'champion' ? 'replay champion' : 'start / resume'}</button>
      <button onClick={champion} disabled={!simulation.current.champion}>watch champion</button>
      {mode === 'champion' && <button onClick={evolve}>continue evolution</button>}
      <button onClick={reset}>reset</button>
      <label>Speed{' '}<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={1}>1×</option><option value={4}>4×</option><option value={12}>12×</option></select></label>
      <label><input type="checkbox" checked={showSensors} onChange={(event) => setShowSensors(event.target.checked)} /> Sensors</label>
    </div>
    <div aria-live="polite" style={{ fontSize: 13 }}>Generation {stats.generation} · alive {stats.alive} · current {stats.current.toFixed(2)} laps · record {stats.best.toFixed(2)} laps</div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div><canvas ref={canvasRef} width={TRACK.width} height={TRACK.height} aria-label="Track with neural cars and distance sensor rays" style={{ width: '100%', borderRadius: 16, border: '1px solid #475569' }} />
        <p style={{ color: '#a7b2c7', fontSize: 12 }}>Blue = leading driver. A generation ends after all cars finish or crash (maximum 3 laps / 1,800 steps). The best 3 drivers survive; others mutate or start fresh. Let several generations run.</p>
      </div>
      <EvolutionCharts history={history} />
    </div>
  </div>;
}
