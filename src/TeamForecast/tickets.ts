import { TeamData } from './data';

export interface TTtmDbTicketRow {
  kaiten_card_id: string;
  space_id?: string | number | null;
  title?: string | null;
  first_commit_source?: string | null;
  first_commit_path_with_namespace?: string | null;
  first_commit_at: string | null;
  merged_to_main_at: string | null;
  released_at: string | null;
  technical_ttm_days: string | null;
}
export interface TTtmDbDashboardResponse {
  meta: unknown;
  tickets: TTtmDbTicketRow[];
  gitlab_projects: unknown[];
  release_tags: unknown[];
}
export type Metric = 'commitToMerge' | 'mergeToRelease' | 'total';
export interface TicketDurations {
  ticketId: string;
  week: string;
  commitToMerge: number | null;
  mergeToRelease: number | null;
  total: number | null;
}
export interface WeeklyBucket {
  week: string;
  commitToMerge: number[];
  mergeToRelease: number[];
  total: number[];
}
const DAY = 86400000;
export const METRIC_LABELS: Record<Metric, string> = {
  commitToMerge: 'First commit → merge', mergeToRelease: 'Merge → release', total: 'Technical T2M',
};
function timestamp(value: string | null): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}
function duration(start: number | null, end: number | null): number | null {
  if (start === null || end === null || end < start) return null;
  return (end - start) / DAY;
}
export function mondayUtc(timestampMs: number): string {
  const date = new Date(timestampMs);
  date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 6) % 7);
  return date.toISOString().slice(0, 10);
}
export function ticketDurations(ticket: TTtmDbTicketRow): TicketDurations | null {
  const released = timestamp(ticket.released_at);
  if (released === null) return null;
  const commit = timestamp(ticket.first_commit_at), merged = timestamp(ticket.merged_to_main_at);
  const raw = ticket.technical_ttm_days;
  const numeric = raw === null || raw.trim() === '' ? null : Number(raw);
  const total = numeric !== null && Number.isFinite(numeric) && numeric >= 0 ? numeric : duration(commit, released);
  return { ticketId: ticket.kaiten_card_id, week: mondayUtc(released),
    commitToMerge: duration(commit, merged), mergeToRelease: duration(merged, released), total };
}
export function quantile(values: number[], q: number): number | null {
  if (!Number.isFinite(q) || q < 0 || q > 1) throw new Error('Quantile must be between 0 and 1.');
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position), upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
export function buildBuckets(tickets: TTtmDbTicketRow[]): { buckets: WeeklyBucket[]; samples: TicketDurations[] } {
  const weeks = new Map<string, WeeklyBucket>();
  const samples: TicketDurations[] = [];
  tickets.forEach((ticket) => {
    const sample = ticketDurations(ticket);
    if (!sample) return;
    samples.push(sample);
    let bucket = weeks.get(sample.week);
    if (!bucket) { bucket = { week: sample.week, commitToMerge: [], mergeToRelease: [], total: [] }; weeks.set(sample.week, bucket); }
    for (const metric of ['commitToMerge', 'mergeToRelease', 'total'] as const) {
      const value = sample[metric];
      if (value !== null) bucket[metric].push(value);
    }
  });
  return { buckets: Array.from(weeks.values()).sort((a, b) => a.week.localeCompare(b.week)), samples };
}
export function weeklyPoints(buckets: WeeklyBucket[], metric: Metric, q: number) {
  return buckets.map((bucket) => ({ week: bucket.week, value: quantile(bucket[metric], q), n: bucket[metric].length }));
}
export function ticketForecastData(buckets: WeeklyBucket[], metric: Metric, q: number, name: string, demo: boolean): TeamData {
  const points = weeklyPoints(buckets, metric, q);
  // Keep only the latest uninterrupted weekly segment. Never invent zero values for missing data.
  const weekly: { week: string; medianDays: number; n: number }[] = [];
  for (const point of points) {
    if (point.value === null) { weekly.length = 0; continue; }
    if (weekly.length && Date.parse(point.week) - Date.parse(weekly[weekly.length - 1].week) !== 7 * DAY) weekly.length = 0;
    weekly.push({ week: point.week, medianDays: point.value, n: point.n });
  }
  return { demo, definition: `${METRIC_LABELS[metric]} · P${Math.round(q * 100)} by release week (Monday UTC) · days`, teams: [{ id: 'selected', name, weekly }] };
}
