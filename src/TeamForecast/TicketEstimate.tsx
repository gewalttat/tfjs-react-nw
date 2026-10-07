import { Localized } from '../i18n/Locale';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { TrainingCharts, TrainingEpoch } from '../Training/TrainingCharts';
import { mae, normalization } from './data';
import { featureSchema, hasTicketFeatures, ticketFeatures } from './ticketFeatures';
import { METRIC_LABELS, Metric, ticketDurations, TTtmDbTicketRow } from './tickets';

export function TicketEstimate({ tickets, metric }: { tickets: TTtmDbTicketRow[]; metric: Metric }) {
  const rows = useMemo(() => tickets.filter(hasTicketFeatures).map((ticket) => ({ ticket, labels: ticketDurations(ticket) }))
    .filter((row) => row.labels?.[metric] !== null && row.labels?.[metric] !== undefined)
    .sort((a, b) => Date.parse(a.ticket.released_at!) - Date.parse(b.ticket.released_at!)), [tickets, metric]);
  const candidates = useMemo(() => tickets.filter(hasTicketFeatures), [tickets]);
  const [selected, setSelected] = useState(candidates.find((ticket) => !ticket.released_at)?.kaiten_card_id ?? candidates[0]?.kaiten_card_id ?? '');
  const target = candidates.find((ticket) => ticket.kaiten_card_id === selected) ?? candidates[0];
  const [training, setTraining] = useState(false), [status, setStatus] = useState('Ready to train');
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [result, setResult] = useState<{ error: number; baseline: number; mean: number; scale: number; schema: ReturnType<typeof featureSchema> } | null>(null);
  const [estimate, setEstimate] = useState<number | null>(null);
  const active = useRef(true), running = useRef(false), stopping = useRef(false), modelRef = useRef<tf.Sequential | null>(null);
  useEffect(() => { active.current = true; return () => {
    active.current = false;
    if (running.current) { if (modelRef.current) modelRef.current.stopTraining = true; }
    else { modelRef.current?.dispose(); modelRef.current = null; }
  }; }, []);
  useEffect(() => {
    if (!result || !target || !modelRef.current || training) { setEstimate(null); return; }
    const value = tf.tidy(() => {
      const xs = tf.tensor2d([ticketFeatures(target, result.schema)]);
      return Math.max(0, (modelRef.current!.predict(xs) as tf.Tensor).dataSync()[0] * result.scale + result.mean);
    });
    setEstimate(Number.isFinite(value) ? value : null);
  }, [target, result, training]);
  async function train() {
    if (running.current || rows.length < 30) return;
    running.current = true; stopping.current = false; setTraining(true); setResult(null); setHistory([]);
    modelRef.current?.dispose();
    const model = tf.sequential(); modelRef.current = model;
    const optimizer = tf.train.adam(0.003), tensors: tf.Tensor[] = [];
    let kept = false;
    try {
      const count = Math.max(6, Math.ceil(rows.length * 0.2));
      const cutoff = Date.parse(rows[rows.length - count].ticket.released_at!);
      // All labels in the prefix were available before the hidden release period.
      const prefix = rows.filter((row) => Date.parse(row.ticket.released_at!) < cutoff);
      const hidden = rows.filter((row) => Date.parse(row.ticket.released_at!) >= cutoff);
      if (prefix.length < 20 || !hidden.length) throw new Error('Need at least 20 examples before the hidden release period.');
      const schema = featureSchema(prefix.map((row) => row.ticket));
      const { mean, scale } = normalization(prefix.map((row) => row.labels![metric]!));
      const encode = (items: typeof rows) => {
        const xs = tf.tensor2d(items.map((row) => ticketFeatures(row.ticket, schema)));
        const ys = tf.tensor2d(items.map((row) => [(row.labels![metric]! - mean) / scale]));
        tensors.push(xs, ys); return { xs, ys };
      };
      const trainRows = encode(prefix), testRows = encode(hidden);
      model.add(tf.layers.dense({ inputShape: [trainRows.xs.shape[1]], units: 16, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: 8, activation: 'tanh' })); model.add(tf.layers.dense({ units: 1 }));
      model.compile({ optimizer, loss: 'meanSquaredError' });
      const start = performance.now();
      const fit = async (xs: tf.Tensor2D, ys: tf.Tensor2D, phase: number) => {
        await model.fit(xs, ys, { epochs: 100, batchSize: 32, shuffle: true,
          ...(phase === 0 ? { validationData: [testRows.xs, testRows.ys] as [tf.Tensor2D, tf.Tensor2D] } : {}),
          callbacks: {
            onBatchEnd: async () => { if (!active.current || stopping.current) model.stopTraining = true; },
            onEpochEnd: async (epoch, logs) => {
              if (!active.current) { model.stopTraining = true; return; }
              setStatus(`${phase ? 'Refit on all completed tickets' : 'Chronological validation'} · ${epoch + 1}/100`);
              setHistory((previous) => [...previous, { epoch: phase * 100 + epoch + 1, elapsedMs: performance.now() - start, loss: logs?.loss, validationLoss: logs?.val_loss }]);
              await tf.nextFrame();
            },
          },
        });
      };
      await fit(trainRows.xs, trainRows.ys, 0);
      if (!active.current || stopping.current) return;
      const values = tf.tidy(() => Array.from((model.predict(testRows.xs) as tf.Tensor).dataSync()).map((value) => Math.max(0, value * scale + mean)));
      const actual = hidden.map((row) => row.labels![metric]!);
      const error = mae(actual, values), sorted = prefix.map((row) => row.labels![metric]!).sort((a, b) => a - b);
      const midpoint = (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.ceil((sorted.length - 1) / 2)]) / 2;
      const baseline = mae(actual, actual.map(() => midpoint));
      const full = encode(rows); await fit(full.xs, full.ys, 1);
      if (!active.current || stopping.current) return;
      kept = true; setResult({ error, baseline, mean, scale, schema }); setStatus(`Ready · ${prefix.length} training / ${hidden.length} hidden tickets`);
    } catch (error) { if (active.current) setStatus((error as Error).message); }
    finally {
      tensors.forEach((tensor) => tensor.dispose()); optimizer.dispose();
      if (!kept) { model.dispose(); modelRef.current = null; }
      running.current = false;
      if (active.current) { setTraining(false); if (stopping.current) setStatus('Stopped'); }
    }
  }
  return <div style={{ display: 'grid', gap: 16 }}>
    <h3 style={{ margin: 0 }}><Localized>{"Per-ticket estimate · "}</Localized><Localized>{METRIC_LABELS[metric]}</Localized></h3>
    <div style={{ fontSize: 13, color: '#a6a39c' }}><Localized>{rows.length}</Localized><Localized>{" labeled tickets · inputs: space, first-commit repository/source/date, title length. No merge/release dates or durations enter X. raw fields are unavailable."}</Localized></div>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
      <button disabled={!training && rows.length < 30} onClick={training ? () => { stopping.current = true; if (modelRef.current) modelRef.current.stopTraining = true; } : train}><Localized>{training ? 'stop' : 'train ticket predictor'}</Localized></button>
      <label><Localized>{"Ticket "}</Localized><select value={target?.kaiten_card_id ?? ''} disabled={training} onChange={(event) => setSelected(event.target.value)}><Localized>{candidates.map((ticket) => <option key={ticket.kaiten_card_id} value={ticket.kaiten_card_id}><Localized>{ticket.kaiten_card_id}</Localized><Localized>{" · "}</Localized><Localized>{ticket.released_at ? 'released' : 'in progress'}</Localized><Localized>{" · "}</Localized><Localized>{ticket.title?.slice(0, 60) ?? ''}</Localized></option>)}</Localized></select></label>
    </div>
    <div style={{ fontSize: 13, color: '#a6a39c' }} aria-live="polite"><Localized>{status}<Localized></Localized>{rows.length < 30 ? ' · at least 30 labeled tickets required' : ''}<Localized></Localized>{!candidates.length ? ' · no tickets with a valid first commit' : ''}</Localized></div>
    <Localized>{result && <div><Localized>{"Hidden-period MAE: "}</Localized><Localized>{result.error.toFixed(2)}</Localized><Localized>{" days · historical-median baseline: "}</Localized><Localized>{result.baseline.toFixed(2)}</Localized><Localized>{" days. "}</Localized><Localized>{result.error < result.baseline ? 'Network beats this baseline.' : 'Baseline matches or beats the network.'}</Localized></div>}</Localized>
    <Localized>{estimate !== null && <div><Localized>{"Estimated duration: "}</Localized><strong><Localized>{estimate.toFixed(2)}</Localized><Localized>{" days"}</Localized></strong><Localized>{" · "}</Localized><Localized>{METRIC_LABELS[metric]}<Localized></Localized>{target?.released_at && <span><Localized>{" · actual: "}</Localized><Localized>{ticketDurations(target)?.[metric]?.toFixed(2) ?? 'unknown'}</Localized><Localized>{" days"}</Localized></span>}</Localized></div>}</Localized>
    <TrainingCharts history={history} classification={false} lossLabel="Normalized duration MSE" validationNote="First 100 epochs: chronological holdout; next 100: refit on all completed tickets. Estimates are full-stage durations, not remaining time. Validation uses current stored ticket fields, not historical feature snapshots." />
  </div>;
}
