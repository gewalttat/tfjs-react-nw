# T2M JSON integration

Pass the existing `/api/db` response into `<TicketForecast response={dashboard} />`.
This project does not contain the other application's API route or Kaiten synchronization code;
no direct request to Kaiten or `/api/db` is made by this component.
`dashboard.json` is a synthetic fixture with the same envelope.

Optional `getTeam(ticket)` returns `{ id, name }` to label spaces using your application's mapping.
Without it, grouping uses `space_id`; unknown spaces remain in the combined view.

Weekly mode extracts each duration independently, groups by release-week Monday UTC,
and calculates P50/P85/P95 using linear interpolation. It preserves each metric's N.
Missing metrics are not zero-filled. Forecasting requires 36 consecutive measured weeks;
only the latest continuous segment is used.

Ticket mode learns one selected duration per model. Features available at first commit:
space, first-commit repo/source, title length, and first-commit calendar/time features.
Merge/release timestamps and `technical_ttm_days` are labels only. Last 20% of release-ordered
examples are held out, with equal release timestamps kept together. Vocabulary and normalization
come only from the prefix. A historical-median baseline is scored before refitting on all labels.
The estimate is a complete duration, not remaining time. Current stored features are not historical
snapshots; validation cannot prove point-in-time accuracy if titles/repositories changed later.

No `raw`, task-size, type, or assignee fields are assumed to exist in your API response.

## Kaiten model references

Official card response: https://developers.kaiten.ru/cards/retrieve-card
It documents `size`, `size_unit`, `size_text`, `type_id`, `owner_id`, `responsible_id`,
`asap`, `created`, and `properties`. These need an explicit allowlisted mapping through
sync/database/your `/api/db` response before this browser predictor can use them.
Size may be absent; text sizes are not always numeric. Numeric zero workload is not proof
that a card has a real estimate. Property definitions must be mapped by company-specific IDs.

Card-list query: https://developers.kaiten.ru/cards/retrieve-card-list
`additional_card_fields` is documented for `description`; it is not a general promise that arbitrary
estimate fields can be fetched using that option. Check actual responses from your instance.

For honest historical validation, save the relevant features at the prediction time (e.g. first commit).
Do not feed terminal state, completion time, final time-spent totals, or accumulated blocked time into
an estimate made at work start.
