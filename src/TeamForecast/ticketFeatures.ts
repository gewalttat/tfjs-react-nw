import { TTtmDbTicketRow } from './tickets';

export function hasTicketFeatures(ticket: TTtmDbTicketRow): boolean {
  return !!ticket.first_commit_at && Number.isFinite(Date.parse(ticket.first_commit_at));
}
export function featureSchema(training: TTtmDbTicketRow[]) {
  const dates = training.map((ticket) => Date.parse(ticket.first_commit_at!));
  const origin = Math.min(...dates);
  const timeScale = Math.max(86400000 * 7, Math.max(...dates) - origin);
  const spaces = Array.from(new Set(training.map((ticket) => String(ticket.space_id ?? 'unknown')))).sort();
  const counts = new Map<string, number>();
  training.forEach((ticket) => { const repo = ticket.first_commit_path_with_namespace; if (repo) counts.set(repo, (counts.get(repo) ?? 0) + 1); });
  const repos = Array.from(counts).sort((a, b) => b[1] - a[1]).slice(0, 16).map(([repo]) => repo);
  return { origin, timeScale, spaces, repos };
}
export function ticketFeatures(ticket: TTtmDbTicketRow, schema: ReturnType<typeof featureSchema>): number[] {
  const date = new Date(ticket.first_commit_at!);
  const weekday = date.getUTCDay() / 7 * Math.PI * 2, month = date.getUTCMonth() / 12 * Math.PI * 2;
  const space = String(ticket.space_id ?? 'unknown'), repo = ticket.first_commit_path_with_namespace;
  return [
    (date.getTime() - schema.origin) / schema.timeScale,
    Math.sin(weekday), Math.cos(weekday), Math.sin(month), Math.cos(month),
    Math.min(500, ticket.title?.length ?? 0) / 500,
    ticket.first_commit_source === 'gitlab' ? 1 : 0, ticket.first_commit_source === 'kaiten_comment' ? 1 : 0,
    ...schema.spaces.map((value) => space === value ? 1 : 0), schema.spaces.includes(space) ? 0 : 1,
    ...schema.repos.map((value) => repo === value ? 1 : 0), repo && schema.repos.includes(repo) ? 0 : 1,
  ];
}
