import { breed, createCar, fitness, network, onRoad, randomGenome, sensors, step, TRACK, WEIGHT_COUNT } from './simulation';

function seededRandom() {
  let seed = 12345;
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

it('detects road boundaries and initializes five bounded distance sensors', () => {
  expect(onRoad(TRACK.cx, TRACK.cy)).toBe(false);
  expect(onRoad(TRACK.cx + TRACK.rx + 1, TRACK.cy)).toBe(false);
  const car = createCar(new Array(WEIGHT_COUNT).fill(0));
  expect(onRoad(car.x, car.y)).toBe(true);
  expect(sensors(car)).toHaveLength(5);
  expect(sensors(car).every((value) => value > 0 && value <= 1)).toBe(true);
  expect(network(car.genome, [...sensors(car), car.speed / 4])).toEqual([0, 0]);
});

it('stops a driver that runs into the wall', () => {
  const car = createCar(new Array(WEIGHT_COUNT).fill(0));
  for (let tick = 0; tick < 1800; tick += 1) step(car);
  expect(car.alive).toBe(false);
  expect(car.ticks).toBeLessThan(1800);
  const x = car.x;
  step(car);
  expect(car.x).toBe(x);
});

it('preserves elite genomes and evolves farther-driving networks', () => {
  const random = seededRandom();
  let cars = Array.from({ length: 40 }, () => createCar(randomGenome(random)));
  let firstAverage = 0;
  let finalBest = 0;
  let finalAverage = 0;
  for (let generation = 0; generation < 12; generation += 1) {
    for (let tick = 0; tick < 1800 && cars.some((car) => car.alive); tick += 1) cars.forEach(step);
    const leader = [...cars].sort((a, b) => fitness(b) - fitness(a))[0];
    finalAverage = cars.reduce((sum, car) => sum + fitness(car), 0) / cars.length;
    if (generation === 0) firstAverage = finalAverage;
    finalBest = fitness(leader);
    const next = breed(cars, random);
    expect(next[0].genome).toEqual(leader.genome);
    expect(next.every((car) => car.alive && car.progress === 0)).toBe(true);
    cars = next;
  }
  expect(finalAverage).toBeGreaterThan(firstAverage * 2);
  expect(finalBest).toBeGreaterThan(1);
}, 20000);
