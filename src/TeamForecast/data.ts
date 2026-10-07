export interface Week { week: string; medianDays: number; n?: number }
export interface Team { id: string; name: string; weekly: Week[] }
export interface TeamData { definition: string; demo: boolean; teams: Team[] }
export const LOOKBACK = 6;
export const HOLDOUT = 8;
export function parseTeamData(value: unknown): TeamData {
  const data = value as TeamData;
  if (!data || typeof data.definition !== 'string' || typeof data.demo !== 'boolean' || !Array.isArray(data.teams) || !data.teams.length) throw new Error('Expected definition, demo, and non-empty teams in JSON.');
  const ids = new Set<string>();
  data.teams.forEach((team) => {
    if (!team || typeof team.id !== 'string' || typeof team.name !== 'string' || ids.has(team.id) || !Array.isArray(team.weekly) || team.weekly.length < 36) throw new Error('Each team needs a unique id, name, and at least 36 weekly observations.');
    ids.add(team.id);
    team.weekly.forEach((row, index) => {
      if (!row || typeof row.week !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(row.week) || !Number.isFinite(Date.parse(row.week)) || new Date(row.week).toISOString().slice(0, 10) !== row.week || typeof row.medianDays !== 'number' || !Number.isFinite(row.medianDays) || row.medianDays < 0) throw new Error('Each observation needs a valid YYYY-MM-DD week and non-negative medianDays.');
      if (index && Date.parse(row.week) - Date.parse(team.weekly[index - 1].week) !== 7 * 86400000) throw new Error('Weeks must be consecutive and sorted; fill missing weeks before forecasting.');
    });
  });
  return data;
}
export function normalization(values: number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const scale = Math.max(0.1, Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length));
  return { mean, scale };
}
export function windows(values: number[], mean: number, scale: number) {
  const inputs: number[][] = [], targets: number[][] = [];
  for (let index = LOOKBACK; index < values.length; index += 1) {
    inputs.push(values.slice(index - LOOKBACK, index).map((value) => (value - mean) / scale));
    targets.push([(values[index] - mean) / scale]);
  }
  return { inputs, targets };
}
export function mae(actual: number[], predicted: number[]) {
  return actual.reduce((sum, value, index) => sum + Math.abs(value - predicted[index]), 0) / actual.length;
}
export function futureWeek(last: string, offset: number) { return new Date(Date.parse(last) + offset * 7 * 86400000).toISOString().slice(0, 10); }
