import { Localized } from '../i18n/Locale';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import source from './teams.json';
import { futureWeek, HOLDOUT, LOOKBACK, mae, normalization, parseTeamData, TeamData, windows } from './data';
import { TrainingCharts, TrainingEpoch } from '../Training/TrainingCharts';

interface Forecast { backtest: number[]; future: number[]; error: number; persistence: number; average: number }

export function TeamForecast({ data = source }: { data?: TeamData }) {
  const parsed = useMemo(() => { try { return { value: parseTeamData(data), error: '' }; } catch (error) { return { value: null, error: (error as Error).message }; } }, [data]);
  const [teamId, setTeamId] = useState('');
  const team = parsed.value?.teams.find((item) => item.id === teamId) ?? parsed.value?.teams[0];
  const [horizon, setHorizon] = useState(4);
  const [training, setTraining] = useState(false);
  const [status, setStatus] = useState('Ready to forecast');
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [result, setResult] = useState<Forecast | null>(null);
  const active = useRef(true), running = useRef(false), stopping = useRef(false);
  const modelRef = useRef<tf.Sequential | null>(null);
  useEffect(() => { active.current = true; return () => { active.current = false; if (modelRef.current) modelRef.current.stopTraining = true; }; }, []);
  useEffect(() => { setResult(null); setHistory([]); setStatus('Ready to forecast'); }, [team, horizon]);

  async function train() {
    if (!team || running.current) return;
    running.current = true; stopping.current = false;
    setTraining(true); setResult(null); setHistory([]);
    const model = tf.sequential(); modelRef.current = model;
    const optimizer = tf.train.adam(0.005);
    const tensors: tf.Tensor[] = [];
    try {
      model.add(tf.layers.dense({ inputShape: [LOOKBACK], units: 16, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 8, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 1 }));
      model.compile({ optimizer, loss: 'meanSquaredError' });
      const all = team.weekly.map((row) => row.medianDays), prefix = all.slice(0, -HOLDOUT), actual = all.slice(-HOLDOUT);
      // Fit normalization exclusively on the prefix; hidden future weeks cannot affect it.
      const { mean, scale } = normalization(prefix);
      const start = performance.now();
      const fit = async (values: number[], phase: number) => {
        const rows = windows(values, mean, scale);
        const xs = tf.tensor2d(rows.inputs), ys = tf.tensor2d(rows.targets); tensors.push(xs, ys);
        await model.fit(xs, ys, { epochs: 120, batchSize: 16, shuffle: false, callbacks: {
          onBatchEnd: async () => { if (!active.current || stopping.current) model.stopTraining = true; },
          onEpochEnd: async (epoch, logs) => {
            if (!active.current) { model.stopTraining = true; return; }
            setHistory((previous) => [...previous, { epoch: phase * 120 + epoch + 1, elapsedMs: performance.now() - start, loss: logs?.loss }]);
            setStatus(`${phase === 0 ? 'Backtest: history before hidden weeks' : 'Refit: all observed weeks'} · epoch ${epoch + 1}/120`);
            await tf.nextFrame();
          },
        } });
      };
      const forecast = (values: number[], count: number): number[] => {
        const rolled = [...values], output: number[] = [];
        for (let index = 0; index < count; index += 1) {
          const value = tf.tidy(() => {
            const xs = tf.tensor2d([rolled.slice(-LOOKBACK).map((entry) => (entry - mean) / scale)]);
            const normalized = (model.predict(xs) as tf.Tensor).dataSync()[0];
            return Math.max(0, normalized * scale + mean);
          });
          if (!Number.isFinite(value)) throw new Error('Non-finite forecast');
          output.push(value); rolled.push(value);
        }
        return output;
      };
      await fit(prefix, 0);
      if (!active.current || stopping.current) return;
      const backtest = forecast(prefix, HOLDOUT);
      const error = mae(actual, backtest);
      const persistence = mae(actual, new Array(HOLDOUT).fill(prefix[prefix.length - 1]));
      const average = mae(actual, new Array(HOLDOUT).fill(prefix.slice(-4).reduce((sum, value) => sum + value, 0) / 4));
      await fit(all, 1);
      if (!active.current || stopping.current) return;
      const future = forecast(all, horizon);
      setResult({ backtest, future, error, persistence, average });
      setStatus('Forecast ready');
    } catch (error) { console.error(error); if (active.current) setStatus('Forecast failed. Try again.'); }
    finally {
      tensors.forEach((tensor) => tensor.dispose()); model.dispose(); optimizer.dispose(); modelRef.current = null;
      running.current = false;
      if (active.current) { setTraining(false); if (stopping.current) setStatus('Stopped'); }
    }
  }
  if (!team || !parsed.value) return <div role="alert"><Localized>{"Invalid team JSON: "}</Localized><Localized>{parsed.error}</Localized></div>;
  const actual = team.weekly.map((row) => row.medianDays);
  const total = actual.length + horizon;
  const bands = result?.future.map((value, index) => ({ low: Math.max(0, value - result.error * Math.sqrt(index + 1)), high: value + result.error * Math.sqrt(index + 1) })) ?? [];
  const max = Math.max(...actual, ...bands.map((row) => row.high)) * 1.15;
  const x = (index: number) => 45 + index / (total - 1) * 505, y = (value: number) => 220 - value / max * 190;
  const path = (values: number[], offset: number) => values.map((value, i) => `${i ? 'L' : 'M'} ${x(i + offset)} ${y(value)}`).join(' ');
  return <div style={{ display: 'grid', gap: 16 }}>
    <h3 style={{ margin: 0 }}><Localized>{"Team T2M forecast"}</Localized></h3>
    <div style={{ fontSize: 13, color: '#a6a39c' }}><Localized>{parsed.value.demo ? 'Synthetic demo data · ' : ''}<Localized></Localized>{parsed.value.definition}</Localized></div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <label><Localized>{"Team "}</Localized><select disabled={training} value={team.id} onChange={(event) => setTeamId(event.target.value)}><Localized>{parsed.value.teams.map((item) => <option key={item.id} value={item.id}><Localized>{item.name}</Localized></option>)}</Localized></select></label>
      <label><Localized>{"Horizon "}</Localized><select disabled={training} value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><Localized>{[2, 4, 8].map((value) => <option key={value} value={value}><Localized>{value}</Localized><Localized>{" weeks"}</Localized></option>)}</Localized></select></label>
      <button onClick={training ? () => { stopping.current = true; if (modelRef.current) modelRef.current.stopTraining = true; } : train}><Localized>{training ? 'stop' : 'train & forecast'}</Localized></button>
    </div>
    <div style={{ fontSize: 13, color: '#a6a39c' }} aria-live="polite"><Localized>{status}</Localized></div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div>
        <svg viewBox="0 0 580 270" style={{ width: '100%' }} role="img" aria-label="Weekly T2M history, hidden-period backtest, and future forecast">
          <Localized>{[0, 0.5, 1].map((fraction) => <g key={fraction}><line x1={45} x2={550} y1={y(max * fraction)} y2={y(max * fraction)} stroke="#404040" /><text x={39} y={y(max * fraction) + 4} fill="#a6a39c" textAnchor="end" fontSize={11}><Localized>{(max * fraction).toFixed(1)}</Localized></text></g>)}</Localized>
          <rect x={x(actual.length - HOLDOUT)} y={30} width={x(actual.length - 1) - x(actual.length - HOLDOUT)} height={190} fill="rgba(196,171,114,0.06)" />
          <text x={x(actual.length - HOLDOUT)} y={24} fill="#dedbd2" fontSize={10}><Localized>{"Hidden "}</Localized><Localized>{HOLDOUT}</Localized><Localized>{" weeks"}</Localized></text>
          <path d={path(actual, 0)} fill="none" stroke="#c4ab72" strokeWidth={1} />
          <Localized>{team.weekly.map((row, index) => <circle key={row.week} cx={x(index)} cy={y(row.medianDays)} r={2} fill="#c4ab72"><title><Localized>{row.week}</Localized><Localized>{": "}</Localized><Localized>{row.medianDays.toFixed(2)}</Localized><Localized>{" days"}</Localized><Localized>{row.n === undefined ? '' : ` · N=${row.n}`}</Localized></title></circle>)}</Localized>
          <Localized>{result && <>
            <path d={path(result.backtest, actual.length - HOLDOUT)} fill="none" stroke="#dedbd2" strokeWidth={1} strokeDasharray="5 4" />
            <polygon points={bands.map((row, index) => `${x(actual.length + index)},${y(row.high)}`).concat([...bands].reverse().map((row, index) => `${x(actual.length + bands.length - 1 - index)},${y(row.low)}`)).join(' ')} fill="rgba(196,171,114,0.12)" />
            <path d={path([actual[actual.length - 1], ...result.future], actual.length - 1)} fill="none" stroke="#b4b5a9" strokeWidth={1} />
            <Localized>{result.future.map((value, index) => <circle key={index} cx={x(actual.length + index)} cy={y(value)} r={3} fill="#b4b5a9"><title><Localized>{futureWeek(team.weekly[actual.length - 1].week, index + 1)}</Localized><Localized>{": "}</Localized><Localized>{value.toFixed(2)}</Localized><Localized>{" days"}</Localized></title></circle>)}</Localized>
          </>}</Localized>
          <line x1={x(actual.length - 1)} x2={x(actual.length - 1)} y1={30} y2={220} stroke="#a6a39c" strokeDasharray="3 4" />
          <text x={45} y={240} fill="#a6a39c" fontSize={10}><Localized>{team.weekly[0].week}</Localized></text><text x={550} y={240} textAnchor="end" fill="#a6a39c" fontSize={10}><Localized>{futureWeek(team.weekly[actual.length - 1].week, horizon)}</Localized></text>
          <text x={290} y={260} textAnchor="middle" fill="#a6a39c" fontSize={11}><Localized>{"Weekly metric · days"}</Localized></text>
        </svg>
        <div style={{ fontSize: 12 }}><span style={{ color: '#c4ab72' }}><Localized>{"● Actual"}</Localized></span>　<span style={{ color: '#dedbd2' }}><Localized>{"● Backtest"}</Localized></span>　<span style={{ color: '#b4b5a9' }}><Localized>{"● Forecast"}</Localized></span></div>
        <Localized>{result && <>
          <table style={{ width: '100%', marginTop: 16, fontSize: 13, textAlign: 'left' }}><thead><tr><th><Localized>{"8-week backtest"}</Localized></th><th><Localized>{"MAE · days"}</Localized></th></tr></thead><tbody>
            <tr><td><Localized>{"Micro network"}</Localized></td><td><Localized>{result.error.toFixed(2)}</Localized></td></tr><tr><td><Localized>{"Last observed week"}</Localized></td><td><Localized>{result.persistence.toFixed(2)}</Localized></td></tr><tr><td><Localized>{"Last 4-week average"}</Localized></td><td><Localized>{result.average.toFixed(2)}</Localized></td></tr>
          </tbody></table>
          <p style={{ fontSize: 13 }}><Localized>{result.error < Math.min(result.persistence, result.average) ? 'The network beats both baselines on this hidden period.' : 'A simple baseline matches or beats the network on this hidden period.'}</Localized></p>
          <table style={{ width: '100%', fontSize: 13, textAlign: 'left' }}><thead><tr><th><Localized>{"Week"}</Localized></th><th><Localized>{"Forecast · days"}</Localized></th></tr></thead><tbody><Localized>{result.future.map((value, index) => <tr key={index}><td><Localized>{futureWeek(team.weekly[actual.length - 1].week, index + 1)}</Localized></td><td><Localized>{value.toFixed(2)}</Localized></td></tr>)}</Localized></tbody></table>
        </>}</Localized>
        <p style={{ color: '#a6a39c', fontSize: 12 }}><Localized>{"The network uses the previous 6 weeks. Backtest predictions are recursive: hidden actuals are never fed back. After testing, it refits on all history. The shaded band is ± backtest MAE × √forecast week, an illustrative error scale, not a calibrated confidence interval. Curve-only forecasts cannot anticipate staffing or process changes."}</Localized></p>
      </div>
      <TrainingCharts history={history} classification={false} lossLabel="Normalized T2M MSE" validationNote="Epochs 1–120: prefix training; 121–240: refit on full history. Generalization is measured by the separate chronological backtest." />
    </div>
  </div>;
}
