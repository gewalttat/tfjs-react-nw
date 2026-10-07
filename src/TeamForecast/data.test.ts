import source from './teams.json';
import { HOLDOUT, LOOKBACK, futureWeek, mae, normalization, parseTeamData, windows } from './data';

it('loads JSON teams and rejects duplicate ids, invalid values, and missing weeks', () => {
  expect(parseTeamData(source).teams).toHaveLength(3);
  const duplicate = { ...source, teams: [source.teams[0], source.teams[0]] };
  expect(() => parseTeamData(duplicate)).toThrow();
  const invalid = JSON.parse(JSON.stringify(source));
  invalid.teams[0].weekly[0].medianDays = -1;
  expect(() => parseTeamData(invalid)).toThrow();
  const gap = JSON.parse(JSON.stringify(source));
  gap.teams[0].weekly.splice(5, 1);
  expect(() => parseTeamData(gap)).toThrow('consecutive');
});

it('builds lag windows and normalization solely from the history prefix', () => {
  const all = source.teams[0].weekly.map((row) => row.medianDays);
  const prefix = all.slice(0, -HOLDOUT);
  const { mean, scale } = normalization(prefix);
  const rows = windows(prefix, mean, scale);
  expect(rows.inputs).toHaveLength(prefix.length - LOOKBACK);
  expect(rows.inputs[0]).toEqual(prefix.slice(0, LOOKBACK).map((value) => (value - mean) / scale));
  expect(rows.targets[0][0]).toBeCloseTo((prefix[LOOKBACK] - mean) / scale);
  expect(rows.targets[rows.targets.length - 1][0]).toBeCloseTo((prefix[prefix.length - 1] - mean) / scale);
  expect(normalization(new Array(40).fill(10)).scale).toBeGreaterThan(0);
});

it('computes error in days and future weekly dates across years', () => {
  expect(mae([10, 12], [9, 15])).toBe(2);
  expect(futureWeek('2025-12-29', 1)).toBe('2026-01-05');
});
