import React, { useMemo, useState } from 'react';
import source from './dashboard.json';
import { TicketEstimate } from './TicketEstimate';
import { TeamForecast } from './TeamForecast';
import { buildBuckets, METRIC_LABELS, Metric, ticketForecastData, TTtmDbDashboardResponse, TTtmDbTicketRow } from './tickets';

interface Props {
  response?: TTtmDbDashboardResponse;
  // Optional space-to-team naming override; defaults to space_id.
  getTeam?: (ticket: TTtmDbTicketRow) => { id: string; name: string } | null;
}
export function TicketForecast({ response = source, getTeam }: Props) {
  const [mode, setMode] = useState('curve');
  const [metric, setMetric] = useState<Metric>('total');
  const [q, setQ] = useState(0.5);
  const [teamId, setTeamId] = useState('all');
  const parsed = useMemo(() => {
    try {
      if (!response || !Array.isArray(response.tickets)) throw new Error('Expected tickets array from /api/db.');
      const groups = new Map<string, { name: string; tickets: TTtmDbTicketRow[] }>();
      groups.set('all', { name: 'All spaces', tickets: response.tickets });
      response.tickets.forEach((ticket) => {
        const team = getTeam ? getTeam(ticket) : ticket.space_id == null ? null : { id: String(ticket.space_id), name: `Space ${ticket.space_id}` }; if (!team || team.id === 'all') return;
        if (!groups.has(team.id)) groups.set(team.id, { name: team.name, tickets: [] });
        groups.get(team.id)!.tickets.push(ticket);
      });
      const group = groups.get(teamId) ?? groups.get('all')!;
      return { groups, ...buildBuckets(group.tickets), name: group.name, tickets: group.tickets, error: '' };
    } catch (error) { return { groups: new Map(), buckets: [], samples: [], name: '', tickets: [], error: (error as Error).message }; }
  }, [response, getTeam, teamId]);
  const demo = response === source;
  const data = useMemo(() => ticketForecastData(parsed.buckets, metric, q, parsed.name, demo), [parsed.buckets, metric, q, parsed.name, demo]);
  const latest = parsed.buckets[parsed.buckets.length - 1];
  return <div style={{ display: 'grid', gap: 16 }}>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <label>Mode <select value={mode} onChange={(event) => setMode(event.target.value)}><option value="curve">Weekly curve</option><option value="ticket">Per-ticket estimate</option></select></label>
      {parsed.groups.size > 1 && <label>Team <select value={teamId} onChange={(event) => setTeamId(event.target.value)}>{Array.from(parsed.groups.entries()).map(([id, group]) => <option key={id} value={id}>{group.name}</option>)}</select></label>}
      <label>Metric <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>{(Object.keys(METRIC_LABELS) as Metric[]).map((key) => <option key={key} value={key}>{METRIC_LABELS[key]}</option>)}</select></label>
      {mode === 'curve' && <label>Quantile <select value={q} onChange={(event) => setQ(Number(event.target.value))}>{[0.5, 0.85, 0.95].map((value) => <option key={value} value={value}>P{value * 100}</option>)}</select></label>}
    </div>
    {parsed.error ? <div role="alert">{parsed.error}</div> : <>
      <div style={{ fontSize: 13, color: '#a7b2c7' }}>{parsed.samples.length} released tickets · latest week {latest?.week ?? '—'} · N: commit→merge {latest?.commitToMerge.length ?? 0}, merge→release {latest?.mergeToRelease.length ?? 0}, total {latest?.total.length ?? 0}</div>
      {mode === 'curve' && <div style={{ fontSize: 12, color: '#a7b2c7' }}>Forecast uses the latest consecutive weeks with measured values ({data.teams[0].weekly.length} available; at least 36 required). Missing values are skipped, not replaced by zero. {teamId === 'all' ? 'All spaces combined; choose a space to model one team.' : ''}</div>}
      {mode === 'curve' ? <TeamForecast key={`${teamId}:${metric}:${q}`} data={data} /> : <TicketEstimate key={`${teamId}:${metric}`} tickets={parsed.tickets} metric={metric} />}
    </>}
  </div>;
}
