import React from 'react';

export interface TrainingEpoch {
  epoch: number;
  elapsedMs: number;
  loss?: number;
  validationLoss?: number;
  accuracy?: number;
  validationAccuracy?: number;
}

const COLORS = ['#38bdf8', '#fbbf24'];
const finite = (value: number | undefined): value is number => value !== undefined && Number.isFinite(value);
const card: React.CSSProperties = { minWidth: 0, border: '1px solid rgba(148,163,184,0.2)', borderRadius: 12, padding: 12 };

function MetricChart({ history, accuracy, lossLabel }: { history: TrainingEpoch[]; accuracy: boolean; lossLabel: string }) {
  const keys: (keyof Pick<TrainingEpoch, 'loss' | 'validationLoss' | 'accuracy' | 'validationAccuracy'>)[] = accuracy
    ? ['accuracy', 'validationAccuracy'] : ['loss', 'validationLoss'];
  const values = history.flatMap((row) => keys.map((key) => row[key])).filter(finite);
  const max = accuracy ? 1 : Math.max(0.001, ...values) * 1.1;
  const end = Math.max(1, history[history.length - 1]?.elapsedMs ?? 1);
  const x = (ms: number) => 58 + (ms / end) * 282;
  const y = (value: number) => 158 - (value / max) * 128;
  const format = (value: number) => accuracy ? `${(value * 100).toFixed(0)}%` : value.toLocaleString('en-US', { maximumSignificantDigits: 3 });
  const title = accuracy ? 'Accuracy · higher is better' : `${lossLabel} · lower is better`;

  return (
    <div style={card}>
      <div style={{ fontSize: 13, fontWeight: 700 }}>{title}</div>
      <svg viewBox="0 0 360 200" role="img" aria-label={`${title}, training and validation over time`} style={{ width: '100%', display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <g key={fraction}>
            <line x1={58} x2={340} y1={y(max * fraction)} y2={y(max * fraction)} stroke="#334155" />
            <text x={52} y={y(max * fraction) + 4} textAnchor="end" fill="#a7b2c7" fontSize={10}>{format(max * fraction)}</text>
            <text x={x(end * fraction)} y={176} textAnchor="middle" fill="#a7b2c7" fontSize={10}>{Math.round(end * fraction).toLocaleString('en-US')}</text>
          </g>
        ))}
        <text x={199} y={194} textAnchor="middle" fill="#a7b2c7" fontSize={10}>Training time (ms)</text>
        {keys.map((key, index) => {
          const points = history.filter((row) => finite(row[key]));
          return (
            <g key={key}>
              <path d={points.map((row, i) => `${i === 0 ? 'M' : 'L'} ${x(row.elapsedMs)} ${y(row[key]!)}`).join(' ')} fill="none" stroke={COLORS[index]} strokeWidth={2} />
              {points.map((row, i) => (
                <circle key={i} cx={x(row.elapsedMs)} cy={y(row[key]!)} r={2.5} fill={COLORS[index]}>
                  <title>{`${index === 0 ? 'Training' : 'Validation'}, epoch ${row.epoch.toFixed(2)}, ${Math.round(row.elapsedMs)} ms: ${accuracy ? `${(row[key]! * 100).toFixed(1)}%` : row[key]!.toFixed(4)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        {values.length === 0 && <text x={199} y={98} textAnchor="middle" fill="#a7b2c7" fontSize={12}>Waiting for training…</text>}
      </svg>
      <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
        <span style={{ color: COLORS[0] }}>● Training</span>
        <span style={{ color: COLORS[1] }}>● Validation</span>
      </div>
    </div>
  );
}

function TrainingHeatmap({ history, classification }: { history: TrainingEpoch[]; classification: boolean }) {
  const rows: { label: string; key: 'loss' | 'validationLoss' | 'accuracy' | 'validationAccuracy' }[] = [
    { label: 'Train loss', key: 'loss' }, { label: 'Val loss', key: 'validationLoss' },
    ...(classification ? [{ label: 'Train acc', key: 'accuracy' as const }, { label: 'Val acc', key: 'validationAccuracy' as const }] : []),
  ];
  // Keep all recorded samples available, including runs with hundreds of epochs.
  return (
    <div style={card}>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Training heatmap</div>
      {history.length === 0 ? <div style={{ color: '#a7b2c7', fontSize: 12 }}>Waiting for training…</div> : (
        <div style={{ display: 'flex' }}>
          <div style={{ flexShrink: 0, width: 70, fontSize: 11 }}>{rows.map((row) => <div key={row.key} style={{ height: 24, lineHeight: '24px' }}>{row.label}</div>)}</div>
          <div style={{ overflowX: 'auto', flex: 1, minWidth: 0 }}>
            <div style={{ minWidth: Math.max(200, history.length * 5) }}>
              {rows.map(({ label, key }) => {
                const values = history.map((row) => row[key]).filter(finite);
                const min = Math.min(...values);
                const max = Math.max(...values);
                return <div key={key} style={{ display: 'flex', height: 24, gap: 1 }}>
                  {history.map((row, i) => {
                    const value = row[key];
                    const intensity = finite(value) ? (max === min ? 0.5 : (value - min) / (max - min)) : 0;
                    return <div key={i} style={{ flex: 1, minWidth: 4, background: finite(value) ? `hsl(200, 85%, ${18 + intensity * 55}%)` : '#1e293b' }}
                      title={`${label}, epoch ${row.epoch.toFixed(2)}, ${Math.round(row.elapsedMs)} ms: ${finite(value) ? value.toFixed(4) : 'not measured'}`} />;
                  })}
                </div>;
              })}
            </div>
          </div>
        </div>
      )}
      <div style={{ fontSize: 11, color: '#a7b2c7', marginTop: 10 }}>Time → · darker = lower, lighter = higher. Each row has its own scale; gray = no measurement.</div>
    </div>
  );
}

export function TrainingCharts({ history, classification = true, lossLabel = 'Loss', validationNote = 'Validation uses 10% of the examples, excluded from training.' }: {
  history: TrainingEpoch[];
  classification?: boolean;
  lossLabel?: string;
  validationNote?: string;
}) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {classification && <MetricChart history={history} accuracy lossLabel={lossLabel} />}
      <MetricChart history={history} accuracy={false} lossLabel={lossLabel} />
      <TrainingHeatmap history={history} classification={classification} />
      <div style={{ fontSize: 12, color: '#a7b2c7' }}>{validationNote} Hover over points or heatmap cells for values.</div>
    </div>
  );
}
