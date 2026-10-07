import { Localized, useLocale } from '../i18n/Locale';
import React, { useEffect, useRef, useState } from 'react';
import { breed, Car, createCar, fitness, laps, MAX_LAPS, MAX_TICKS, Genome, randomGenome, SENSOR_ANGLES, SENSOR_RANGE, sensors, stepRace, nextLiveHeat, POPULATION, TRACK } from './simulation';

interface Generation { generation: number; best: number; average: number; scores: number[]; finishRate: number; collisionRate: number }
interface Simulation { cars: Car[]; generation: number; champion: Genome[] | null; replay: Car[]; best: number; mode: 'evolve' | 'champion'; trail: [number, number][]; visibleHeat: number }
const initial = (): Simulation => ({ cars: Array.from({ length: POPULATION }, (_, index) => createCar(randomGenome(), index % 3)), generation: 1, champion: null, replay: [], best: 0, mode: 'evolve', trail: [], visibleHeat: 0 });

function EvolutionCharts({ history }: { history: Generation[] }) {
  const max = Math.max(0.1, ...history.map((row) => row.best));
  const x = (index: number) => 40 + index / Math.max(1, history.length - 1) * 300;
  const y = (value: number) => 150 - value / max * 120;
  return <div style={{ display: 'grid', gap: 12 }}>
    <div style={{ fontWeight: 700 }}><Localized>{"Driving score · distance + finish speed bonus"}</Localized></div>
    <svg viewBox="0 0 360 190" role="img" aria-label="Best and average driving score per generation" style={{ width: '100%' }}>
      <Localized>{[0, 0.5, 1].map((fraction) => <g key={fraction}>
        <line x1={40} x2={340} y1={y(max * fraction)} y2={y(max * fraction)} stroke="#404040" />
        <text x={34} y={y(max * fraction) + 4} textAnchor="end" fill="#a6a39c" fontSize={10}><Localized>{(max * fraction).toFixed(2)}</Localized></text>
      </g>)}</Localized>
      <Localized>{(['best', 'average'] as const).map((key, series) => <g key={key}>
        <path d={history.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(row[key])}`).join(' ')} fill="none" stroke={series ? '#dedbd2' : '#c4ab72'} strokeWidth={1} />
        <Localized>{history.map((row, index) => <circle key={index} cx={x(index)} cy={y(row[key])} r={3} fill={series ? '#dedbd2' : '#c4ab72'}><title><Localized>{`Generation ${row.generation}, ${key}: ${row[key].toFixed(3)} points`}</Localized></title></circle>)}</Localized>
      </g>)}</Localized>
      <text x={40} y={168} fill="#a6a39c" fontSize={10}><Localized>{"1"}</Localized></text><text x={340} y={168} textAnchor="end" fill="#a6a39c" fontSize={10}><Localized>{history.length || 1}</Localized></text>
      <text x={190} y={185} textAnchor="middle" fill="#a6a39c" fontSize={10}><Localized>{"Generation"}</Localized></text>
      <Localized>{!history.length && <text x={190} y={95} textAnchor="middle" fill="#a6a39c" fontSize={12}><Localized>{"Waiting for the first generation…"}</Localized></text>}</Localized>
    </svg>
    <div style={{ fontSize: 12 }}><span style={{ color: '#c4ab72' }}><Localized>{"● Best"}</Localized></span>　<span style={{ color: '#dedbd2' }}><Localized>{"● Population average"}</Localized></span></div>
    <div className="metric-summary">
      <div><small><Localized>Finish rate</Localized></small><strong>{history.length ? `${(history[history.length - 1].finishRate * 100).toFixed(0)}%` : '—'}</strong></div>
      <div><small><Localized>Collision rate</Localized></small><strong>{history.length ? `${(history[history.length - 1].collisionRate * 100).toFixed(0)}%` : '—'}</strong></div>
    </div>
    <div style={{ fontSize: 12 }}><Localized>Finish / collision rates by generation</Localized></div>
    <svg viewBox="0 0 360 110" role="img" aria-label="Finish and collision rates by generation" style={{ width: '100%' }}>
      <line x1={40} x2={340} y1={90} y2={90} stroke="#404040" strokeWidth={0.5} />
      {(['finishRate', 'collisionRate'] as const).map((key, series) => <path key={key} d={history.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${90 - row[key] * 70}`).join(' ')} fill="none" stroke={series ? '#a6a39c' : '#c4ab72'} strokeWidth={0.8} />)}
      <text x={10} y={24} fill="#a6a39c" fontSize={9}>100%</text><text x={18} y={94} fill="#a6a39c" fontSize={9}>0%</text>
    </svg>
    <div style={{ fontWeight: 700 }}><Localized>{"Population heatmap"}</Localized></div>
    <div style={{ maxHeight: 170, overflow: 'auto' }}>
      <Localized>{history.map((row) => <div key={row.generation} style={{ display: 'flex', height: 12, gap: 1, marginBottom: 2 }}>
        <span style={{ width: 30, flexShrink: 0, fontSize: 10 }}><Localized>{row.generation}</Localized></span>
        <Localized>{row.scores.map((score, index) => <div key={index} title={`Generation ${row.generation}, rank ${index + 1}: ${score.toFixed(3)} points`}
          style={{ flex: 1, minWidth: 3, background: `hsl(42, 25%, ${18 + Math.min(1, score / max) * 55}%)` }} />)}</Localized>
      </div>)}</Localized>
    </div>
    <div style={{ fontSize: 12, color: '#a6a39c' }}><Localized>{"Rows = generations; columns = drivers ranked by score. Lighter = better."}</Localized></div>
  </div>;
}

export function NeuralCar() {
  const language = useLocale();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simulation = useRef<Simulation | null>(null);
  if (!simulation.current) simulation.current = initial();
  const [heat, setHeat] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [showSensors, setShowSensors] = useState(true);
  const [history, setHistory] = useState<Generation[]>([]);
  const [stats, setStats] = useState({ generation: 1, alive: 3, totalAlive: POPULATION, best: 0, current: 0, bestLap: null as number | null, finished: 0, crashed: 0, timedOut: 0, collisions: 0, stalled: 0, reason: null as Car['stopReason'] });
  const [mode, setMode] = useState<'evolve' | 'champion'>('evolve');

  useEffect(() => {
    let frame = 0;
    let lastFrame = 0;
    let lastUpdate = 0;
    const draw = (time: number) => {
      const sim = simulation.current!;
      const context = canvasRef.current?.getContext('2d');
      if (!context) return;
      const currentCars = sim.mode === 'champion' ? sim.replay : sim.cars;
      if (running) {
        const elapsed = lastFrame ? Math.min(50, time - lastFrame) : 1000 / 60;
        const steps = Math.max(1, Math.round(elapsed / (1000 / 60) * speed));
        for (let tick = 0; tick < steps; tick += 1) {
          for (let group = 0; group < currentCars.length; group += 3) stepRace(currentCars.slice(group, group + 3), sim.mode === 'champion');
          const leader = currentCars.reduce((a, b) => fitness(a) > fitness(b) ? a : b);
          if (sim.mode === 'champion') {
            sim.trail.push([leader.x, leader.y]);
            if (sim.trail.length > 2000) sim.trail.shift();
          }
          if (currentCars.every((car) => !car.alive)) {
            if (sim.mode === 'champion') { setRunning(false); break; }
            const scores = sim.cars.map(fitness).sort((a, b) => b - a);
            if (!sim.champion || scores[0] > sim.best) { sim.champion = [...sim.cars].sort((a, b) => fitness(b) - fitness(a)).slice(0, 3).map((car) => [...car.genome]); sim.best = scores[0]; }
            const record = { finishRate: sim.cars.filter((car) => car.stopReason === 'finish').length / POPULATION, collisionRate: sim.cars.filter((car) => car.stopReason === 'collision').length / POPULATION, generation: sim.generation, best: scores[0], average: scores.reduce((a, b) => a + b, 0) / scores.length, scores };
            setHistory((previous) => [...previous, record]);
            sim.cars = breed(sim.cars);
            sim.generation += 1;
            sim.visibleHeat = 0;
            break;
          }
        }
      }
      if (sim.mode === 'evolve' && running) {
        sim.visibleHeat = nextLiveHeat(sim.cars, sim.visibleHeat);
        if (sim.visibleHeat !== heat) setHeat(sim.visibleHeat);
      }
      lastFrame = time;
      context.fillStyle = '#242424'; context.fillRect(0, 0, TRACK.width, TRACK.height);
      context.fillStyle = '#404040'; context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx, TRACK.ry, 0, 0, Math.PI * 2); context.fill();
      context.fillStyle = '#242424'; context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx * TRACK.inner, TRACK.ry * TRACK.inner, 0, 0, Math.PI * 2); context.fill();
      context.strokeStyle = '#a6a39c'; context.lineWidth = 1; context.setLineDash([8, 10]);
      context.beginPath(); context.ellipse(TRACK.cx, TRACK.cy, TRACK.rx * 0.82, TRACK.ry * 0.82, 0, 0, Math.PI * 2); context.stroke(); context.setLineDash([]);
      context.strokeStyle = '#eeece6'; context.lineWidth = 4; context.beginPath(); context.moveTo(TRACK.cx + TRACK.rx * TRACK.inner, TRACK.cy); context.lineTo(TRACK.cx + TRACK.rx, TRACK.cy); context.stroke();
      if (sim.trail.length) {
        context.strokeStyle = '#38bdf8'; context.lineWidth = 2; context.beginPath(); sim.trail.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y)); context.stroke();
      }
      const displayed = sim.mode === 'champion' ? sim.replay : sim.cars.slice(sim.visibleHeat * 3, sim.visibleHeat * 3 + 3);
      const leader = displayed.reduce((a, b) => fitness(a) > fitness(b) ? a : b);
      displayed.forEach((car, index) => {
        context.save(); context.translate(car.x, car.y); context.rotate(car.heading);
        context.fillStyle = car.stopReason === 'finish' ? '#4ade80' : car.alive ? ['#38bdf8', '#fbbf24', '#c084fc'][index] : 'rgba(166,163,156,0.2)';
        context.fillRect(-7, -4, 14, 8); context.fillStyle = '#e7e5df'; context.fillRect(3, -3, 3, 6); context.restore();
      });
      if (showSensors && leader.alive) {
        const readings = sensors(leader);
        readings.forEach((value, index) => {
          const angle = leader.heading + SENSOR_ANGLES[index];
          context.strokeStyle = 'rgba(74,222,128,0.85)'; context.lineWidth = 1;
          context.beginPath(); context.moveTo(leader.x, leader.y); context.lineTo(leader.x + Math.cos(angle) * value * SENSOR_RANGE, leader.y + Math.sin(angle) * value * SENSOR_RANGE); context.stroke();
        });
      }
      if (showSensors) {
        displayed.filter((car) => car.alive).forEach((car) => {
          displayed.filter((other) => other !== car && other.stopReason !== 'finish').forEach((other) => {
            if (Math.hypot(car.x - other.x, car.y - other.y) > SENSOR_RANGE) return;
            context.strokeStyle = 'rgba(244,114,182,0.8)'; context.beginPath(); context.moveTo(car.x, car.y); context.lineTo(other.x, other.y); context.stroke();
          });
        });
      }
      context.fillStyle = '#a6a39c'; context.font = '16px sans-serif'; context.textAlign = 'center';
      context.fillText(language === 'ru' ? (sim.mode === 'evolve' ? `Заезд ${sim.visibleHeat + 1} / ${POPULATION / 3} · три водителя` : 'Гонка трёх лучших') : (sim.mode === 'evolve' ? `Heat ${sim.visibleHeat + 1} / ${POPULATION / 3} · three drivers` : 'Top three race'), TRACK.cx, TRACK.cy);
      if (time - lastUpdate > 100) {
        lastUpdate = time;
        setStats({ generation: sim.generation, alive: displayed.filter((car) => car.alive).length, totalAlive: currentCars.filter((car) => car.alive).length, best: sim.best, current: laps(leader),
          bestLap: leader.bestLap, finished: displayed.filter((car) => car.stopReason === 'finish').length,
          crashed: displayed.filter((car) => car.stopReason === 'crash' || car.stopReason === 'wrong-way').length,
          timedOut: displayed.filter((car) => car.stopReason === 'timeout').length, collisions: displayed.filter((car) => car.stopReason === 'collision').length, stalled: displayed.filter((car) => car.stopReason === 'stalled').length, reason: leader.stopReason });
      }
      frame = window.requestAnimationFrame(draw);
    };
    frame = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(frame);
  }, [running, speed, showSensors, heat, language]);

  function reset() {
    setRunning(false); simulation.current = initial(); setHistory([]); setMode('evolve'); setHeat(0);
    setStats({ generation: 1, alive: 3, totalAlive: POPULATION, best: 0, current: 0, bestLap: null as number | null, finished: 0, crashed: 0, timedOut: 0, collisions: 0, stalled: 0, reason: null as Car['stopReason'] });
  }
  function champion() {
    const sim = simulation.current!;
    if (!sim.champion) return;
    sim.replay = sim.champion.map((genome, index) => createCar(genome, index)); sim.mode = 'champion'; sim.trail = [];
    setMode('champion'); setRunning(true);
  }
  function evolve() {
    const sim = simulation.current!;
    if (sim.mode === 'champion') {
      sim.mode = 'evolve'; sim.trail = [];
    }
    setMode('evolve'); setRunning(true);
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
    <h3 style={{ margin: 0 }}><Localized>{"Neural racing"}</Localized></h3>
    <div style={{ color: '#a6a39c', fontSize: 13 }}><Localized>{"42 drivers train in 14 independent heats of three. Networks sense walls, speed, and both rivals; they learn steering, braking, and collision avoidance through evolution."}</Localized></div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <button onClick={running ? () => setRunning(false) : mode === 'champion' ? champion : evolve}><Localized>{running ? 'pause' : mode === 'champion' ? 'replay race' : 'start / resume'}</Localized></button>
      <button onClick={champion} disabled={!simulation.current.champion}><Localized>{"race top three"}</Localized></button>
      <Localized>{mode === 'champion' && <button onClick={evolve}><Localized>{"continue evolution"}</Localized></button>}</Localized>
      <button onClick={reset}><Localized>{"reset"}</Localized></button>
      <Localized>{mode === 'evolve' && <label><Localized>{"Watch heat"}</Localized><Localized>{' '}</Localized><select value={heat} onChange={(event) => { simulation.current!.visibleHeat = Number(event.target.value); setHeat(Number(event.target.value)); }}>
        <Localized>{Array.from({ length: POPULATION / 3 }, (_, index) => <option key={index} value={index}><Localized>{index + 1}</Localized></option>)}</Localized>
      </select></label>}</Localized>
      <label><Localized>{"Speed"}</Localized><Localized>{' '}</Localized><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={1}><Localized>{"1×"}</Localized></option><option value={4}><Localized>{"4×"}</Localized></option><option value={12}><Localized>{"12×"}</Localized></option></select></label>
      <label><input type="checkbox" checked={showSensors} onChange={(event) => setShowSensors(event.target.checked)} /><Localized>{" Sensors"}</Localized></label>
    </div>
    <div aria-live="polite" style={{ fontSize: 13 }}><Localized>{"Generation "}</Localized><Localized>{stats.generation}</Localized><Localized>{" · alive in view "}</Localized><Localized>{stats.alive}</Localized><Localized>{"/3 · alive overall "}</Localized><Localized>{stats.totalAlive}</Localized><Localized>{"/"}</Localized><Localized>{mode === 'champion' ? 3 : POPULATION}</Localized><Localized>{" · current "}</Localized><Localized>{stats.current.toFixed(2)}</Localized><Localized>{" laps · record score "}</Localized><Localized>{stats.best.toFixed(2)}</Localized><Localized>{" · best lap "}</Localized><Localized>{stats.bestLap === null ? "—" : `${stats.bestLap.toFixed(2)} sec`}</Localized></div>
    <div style={{ fontSize: 13, color: '#a6a39c' }}><Localized>{mode === 'champion'
      ? stats.reason ? `Leading driver stopped: ${stats.reason}` : 'Top-three race has no lap or time limit. Crashed cars remain obstacles.'
      : `Finished: ${stats.finished} · crashed / wrong way: ${stats.crashed} · stuck: ${stats.stalled} · car collisions: ${stats.collisions} · timed out: ${stats.timedOut}`}</Localized></div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div><canvas ref={canvasRef} width={TRACK.width} height={TRACK.height} aria-label="Track with neural cars and distance sensor rays" style={{ width: '100%', borderRadius: 16, border: '1px solid #55524c' }} />
        <p style={{ color: '#a6a39c', fontSize: 12 }}><Localized>{"Blue, yellow, and purple = the three drivers; green = finished. Green rays show walls; pink lines show nearby rivals. Other heats train separately, with no collisions between heats. The view automatically switches when its three drivers stop; the next generation starts after all 42 finish. Training runs end at "}</Localized><Localized>{MAX_LAPS}</Localized><Localized>{" laps or "}</Localized><Localized>{MAX_TICKS / 60}</Localized><Localized>{" simulated seconds. Car collisions stop both drivers and reduce earned distance points by 30%. Drivers making no forward progress for 5 simulated seconds are stopped. Opponents and starting slots are shuffled each generation. Finishers earn a speed bonus, so evolution keeps improving after cars learn to stay on the road. Lap times use simulated time and do not change at 12× playback."}</Localized></p>
      </div>
      <EvolutionCharts history={history} />
    </div>
  </div>;
}
