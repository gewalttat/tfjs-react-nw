import { featureSchema, hasTicketFeatures, ticketFeatures } from './ticketFeatures';
import { TTtmDbTicketRow } from './tickets';
const ticket: TTtmDbTicketRow = { kaiten_card_id: '1', space_id: 101, title: 'Task', first_commit_source: 'gitlab', first_commit_path_with_namespace: 'team/repo', first_commit_at: '2026-01-01T12:00:00Z', merged_to_main_at: null, released_at: null, technical_ttm_days: null };
it('features never change when future merge/release labels change', () => {
  const schema = featureSchema([ticket]);
  const inputs = ticketFeatures(ticket, schema);
  expect(inputs.every(Number.isFinite)).toBe(true);
  expect(ticketFeatures({ ...ticket, merged_to_main_at: '2026-01-05T00:00:00Z', released_at: '2026-01-20T00:00:00Z', technical_ttm_days: '100' }, schema)).toEqual(inputs);
});
it('handles unknown teams and repositories without changing feature dimensions', () => {
  const schema = featureSchema([ticket]);
  const inputs = ticketFeatures({ ...ticket, space_id: 999, first_commit_path_with_namespace: 'new/repo' }, schema);
  expect(inputs).toHaveLength(ticketFeatures(ticket, schema).length);
  expect(inputs.every(Number.isFinite)).toBe(true);
  expect(hasTicketFeatures({ ...ticket, first_commit_at: null })).toBe(false);
});
