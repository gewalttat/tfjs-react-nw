import { decode, examples, idealLaunch, input, output, position } from './physics';

it('the minimum-speed teacher reaches elevated targets under different gravity', () => {
  for (const distance of [10, 60, 110]) for (const height of [0, 12, 35]) for (const gravity of [3, 9.81, 17]) {
    const target = { distance, height, gravity };
    const launch = idealLaunch(target);
    const arrival = distance / (launch.speed * Math.cos(launch.angle));
    expect(position(launch, gravity, arrival).y).toBeCloseTo(height, 8);
    const roundTrip = decode(output(launch));
    expect(roundTrip.angle).toBeCloseTo(launch.angle);
    expect(roundTrip.speed).toBeCloseTo(launch.speed);
    expect(input(target).every((value) => value >= -1 && value <= 1)).toBe(true);
  }
});

it('generates finite reproducible training examples and limits predictions', () => {
  expect(examples(10)).toEqual(examples(10));
  expect(examples(20).every((row) => [...row.inputs, ...row.targets].every(Number.isFinite))).toBe(true);
  expect(decode([100, -100]).speed).toBe(1);
});
