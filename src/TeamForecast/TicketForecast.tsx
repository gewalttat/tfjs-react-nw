import { Localized } from '../i18n/Locale';
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
      <label><Localized>{"Mode "}</Localized><select value={mode} onChange={(event) => setMode(event.target.value)}><option value="curve"><Localized>{"Weekly curve"}</Localized></option><option value="ticket"><Localized>{"Per-ticket estimate"}</Localized></option></select></label>
      <Localized>{parsed.groups.size > 1 && <label><Localized>{"Team "}</Localized><select value={teamId} onChange={(event) => setTeamId(event.target.value)}><Localized>{Array.from(parsed.groups.entries()).map(([id, group]) => <option key={id} value={id}><Localized>{group.name}</Localized></option>)}</Localized></select></label>}</Localized>
      <label><Localized>{"Metric "}</Localized><select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}><Localized>{(Object.keys(METRIC_LABELS) as Metric[]).map((key) => <option key={key} value={key}><Localized>{METRIC_LABELS[key]}</Localized></option>)}</Localized></select></label>
      <Localized>{mode === 'curve' && <label><Localized>{"Quantile "}</Localized><select value={q} onChange={(event) => setQ(Number(event.target.value))}><Localized>{[0.5, 0.85, 0.95].map((value) => <option key={value} value={value}><Localized>{"P"}</Localized><Localized>{value * 100}</Localized></option>)}</Localized></select></label>}</Localized>
    </div>
    <Localized>{parsed.error ? <div role="alert"><Localized>{parsed.error}</Localized></div> : <>
      <div style={{ fontSize: 13, color: '#a6a39c' }}><Localized>{parsed.samples.length}</Localized><Localized>{" released tickets · latest week "}</Localized><Localized>{latest?.week ?? '—'}</Localized><Localized>{" · N: commit→merge "}</Localized><Localized>{latest?.commitToMerge.length ?? 0}</Localized><Localized>{", merge→release "}</Localized><Localized>{latest?.mergeToRelease.length ?? 0}</Localized><Localized>{", total "}</Localized><Localized>{latest?.total.length ?? 0}</Localized></div>
      <Localized>{mode === 'curve' && <div style={{ fontSize: 12, color: '#a6a39c' }}><Localized>{"Forecast uses the latest consecutive weeks with measured values ("}</Localized><Localized>{data.teams[0].weekly.length}</Localized><Localized>{" available; at least 36 required). Missing values are skipped, not replaced by zero. "}</Localized><Localized>{teamId === 'all' ? 'All spaces combined; choose a space to model one team.' : ''}</Localized></div>}</Localized>
      <Localized>{mode === 'curve' ? <TeamForecast key={`${teamId}:${metric}:${q}`} data={data} /> : <TicketEstimate key={`${teamId}:${metric}`} tickets={parsed.tickets} metric={metric} />}</Localized>
    </>}</Localized>
  </div>;
}
