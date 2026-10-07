import { buildBuckets, mondayUtc, quantile, ticketDurations, ticketForecastData, TTtmDbTicketRow } from './tickets';

const ticket: TTtmDbTicketRow = {
  kaiten_card_id: '1', first_commit_at: '2026-01-01T00:00:00Z',
  merged_to_main_at: '2026-01-03T12:00:00Z', released_at: '2026-01-05T00:00:00Z', technical_ttm_days: null,
};
it('extracts raw labels independently, honoring numeric total and zero durations', () => {
  expect(ticketDurations(ticket)).toEqual({ ticketId: '1', week: '2026-01-05', commitToMerge: 2.5, mergeToRelease: 1.5, total: 4 });
  expect(ticketDurations({ ...ticket, technical_ttm_days: '7.25' })?.total).toBe(7.25);
  expect(ticketDurations({ ...ticket, technical_ttm_days: '0' })?.total).toBe(0);
  expect(ticketDurations({ ...ticket, technical_ttm_days: '' })?.total).toBe(4);
  expect(ticketDurations({ ...ticket, first_commit_at: null })).toMatchObject({ commitToMerge: null, mergeToRelease: 1.5, total: null });
  expect(ticketDurations({ ...ticket, merged_to_main_at: null })).toMatchObject({ commitToMerge: null, mergeToRelease: null, total: 4 });
  expect(ticketDurations({ ...ticket, released_at: null })).toBeNull();
  expect(ticketDurations({ ...ticket, first_commit_at: '2026-01-10T00:00:00Z' })?.commitToMerge).toBeNull();
});
it('uses Monday UTC across timezone offsets and Sunday boundaries', () => {
  expect(mondayUtc(Date.parse('2026-01-05T01:00:00+03:00'))).toBe('2025-12-29');
  expect(mondayUtc(Date.parse('2026-01-05T00:00:00Z'))).toBe('2026-01-05');
});
it('keeps separate N values and raw samples before interpolating quantiles', () => {
  const { buckets, samples } = buildBuckets([ticket, { ...ticket, kaiten_card_id: '2', first_commit_at: null, technical_ttm_days: '10' }, { ...ticket, released_at: null }]);
  expect(samples).toHaveLength(2);
  expect(buckets[0].commitToMerge).toEqual([2.5]);
  expect(buckets[0].mergeToRelease).toEqual([1.5, 1.5]);
  expect(buckets[0].total).toEqual([4, 10]);
  expect(quantile(buckets[0].total, 0.85)).toBeCloseTo(9.1);
  expect(quantile([], 0.5)).toBeNull();
  expect(() => quantile([1], 2)).toThrow();
});
it('does not invent values for missing weeks or metrics', () => {
  const { buckets } = buildBuckets([ticket, { ...ticket, released_at: '2026-01-19T00:00:00Z' }]);
  const adapted = ticketForecastData(buckets, 'total', 0.5, 'Team', false);
  expect(adapted.teams[0].weekly).toHaveLength(1);
  expect(adapted.teams[0].weekly[0]).toMatchObject({ week: '2026-01-19', n: 1 });
});
