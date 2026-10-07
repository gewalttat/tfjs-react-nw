import React from 'react';
import { Localized } from '../i18n/Locale';
import type { TrainingEpoch } from './TrainingCharts';

export function LearningDiagnostics({ history }: { history: TrainingEpoch[] }) {
  const measured = history.filter((row) => row.loss !== undefined && Number.isFinite(row.loss));
  const first = measured[0], latest = measured[measured.length - 1];
  const last = history[history.length - 1];
  const lossChange = first?.loss && latest?.loss !== undefined ? (1 - latest.loss / first.loss) * 100 : null;
  const latestValidation = [...history].reverse().find((row) => row.validationLoss !== undefined);
  const latestAccuracy = [...history].reverse().find((row) => row.accuracy !== undefined);
  const stats = [
    ['Epoch', last ? last.epoch.toFixed(1) : '—'],
    ['Time', last ? `${(last.elapsedMs / 1000).toFixed(1)} s` : '—'],
    ['Loss', latest?.loss?.toFixed(4) ?? '—'],
    ['Loss reduction', lossChange === null ? '—' : `${lossChange >= 0 ? '−' : '+'}${Math.abs(lossChange).toFixed(1)}%`],
    ['Validation loss', latestValidation?.validationLoss?.toFixed(4) ?? '—'],
    ['Accuracy', latestAccuracy?.accuracy === undefined ? '—' : `${(latestAccuracy.accuracy * 100).toFixed(1)}%`],
  ];
  const epochs = Array.from(new Map(history.filter((row) => Number.isInteger(row.epoch) && row.epoch > 0).map((row) => [row.epoch, row])).values());
  const durations = epochs.map((row, index) => ({ value: row.elapsedMs - (epochs[index - 1]?.elapsedMs ?? 0), epoch: row.epoch }));
  const changes = epochs.slice(1).filter((row, index) => row.loss !== undefined && epochs[index].loss !== undefined).map((row) => {
    const previous = epochs[epochs.indexOf(row) - 1];
    return { value: (row.loss! - previous.loss!) / Math.max(0.000001, Math.abs(previous.loss!)) * 100, epoch: row.epoch };
  });
  function chart(values: { value: number; epoch: number }[], title: string, unit: string, signed: boolean) {
    const scale = Math.max(1, ...values.map((row) => Math.abs(row.value)));
    const x = (index: number) => 28 + index / Math.max(1, values.length - 1) * 286;
    const y = (value: number) => signed ? 76 - value / scale * 42 : 120 - value / scale * 85;
    return <div style={{ border: '1px solid #404040', borderRadius: 5, padding: 12, minWidth: 0 }}>
      <div className="chart-caption"><Localized>{title}</Localized><span>{unit}</span></div>
      <svg viewBox="0 0 330 150" role="img" aria-label={title} style={{ width: '100%', display: 'block' }}>
        <line x1={28} x2={314} y1={y(0)} y2={y(0)} stroke="#45433f" strokeWidth={0.5} />
        <path d={values.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(row.value)}`).join(' ')} fill="none" stroke="#c4ab72" strokeWidth={0.8} />
        {values.map((row, index) => <circle key={index} cx={x(index)} cy={y(row.value)} r={1.5} fill="#c4ab72"><title>{`Epoch ${row.epoch}: ${row.value.toFixed(2)} ${unit}`}</title></circle>)}
        {!values.length && <text x={165} y={80} textAnchor="middle" fill="#a6a39c" fontSize={11}><Localized>Waiting for training…</Localized></text>}
        <text x={28} y={142} fill="#a6a39c" fontSize={9}>{values[0]?.epoch ?? 0}</text><text x={314} y={142} textAnchor="end" fill="#a6a39c" fontSize={9}>{values[values.length - 1]?.epoch ?? 0}</text>
      </svg>
    </div>;
  }
  return <>
    <div className="metric-summary">{stats.map(([label, value]) => <div key={label}><small><Localized>{label}</Localized></small><strong>{value}</strong></div>)}</div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
      {chart(changes, 'Loss change per epoch', '%', true)}
      {chart(durations, 'Epoch duration', 'ms', false)}
    </div>
  </>;
}
