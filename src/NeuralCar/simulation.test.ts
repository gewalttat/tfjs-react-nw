import { breed, createCar, fitness, network, onRoad, randomGenome, sensors, step, stepRace, nextLiveHeat, trafficInputs, overlaps, INPUT_COUNT, POPULATION, MAX_LAPS, MAX_TICKS, TRACK, WEIGHT_COUNT } from './simulation';

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
  expect(network(car.genome, trafficInputs(car, []))).toEqual([0, 0]);
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
  let cars = Array.from({ length: POPULATION }, (_, index) => createCar(randomGenome(random), index % 3));
  let firstAverage = 0;
  let finalBest = 0;
  let finalAverage = 0;
  for (let generation = 0; generation < 12; generation += 1) {
    for (let tick = 0; tick < 1800 && cars.some((car) => car.alive); tick += 1) {
      for (let group = 0; group < cars.length; group += 3) stepRace(cars.slice(group, group + 3));
    }
    const leader = [...cars].sort((a, b) => fitness(b) - fitness(a))[0];
    finalAverage = cars.reduce((sum, car) => sum + fitness(car), 0) / cars.length;
    if (generation === 0) firstAverage = finalAverage;
    finalBest = fitness(leader);
    const next = breed(cars, random);
    expect(next.some((car) => car.genome.every((weight, index) => weight === leader.genome[index]))).toBe(true);
    expect(next.every((car) => car.alive && car.progress === 0)).toBe(true);
    cars = next;
  }
  expect(finalAverage).toBeGreaterThan(firstAverage * 2);
  expect(finalBest).toBeGreaterThan(1);
}, 20000);

it('distinguishes finishes from timeouts and leaves champion replay unlimited', () => {
  const timeout = createCar(new Array(WEIGHT_COUNT).fill(0));
  timeout.ticks = MAX_TICKS - 1;
  timeout.lastProgressTick = timeout.ticks;
  step(timeout);
  expect(timeout.stopReason).toBe('timeout');
  const champion = createCar(new Array(WEIGHT_COUNT).fill(0));
  champion.ticks = MAX_TICKS - 1;
  champion.lastProgressTick = champion.ticks;
  champion.progress = MAX_LAPS * 2 * Math.PI - 0.001;
  step(champion, true);
  expect(champion.alive).toBe(true);
  expect(champion.stopReason).toBeNull();
  const finisher = createCar(new Array(WEIGHT_COUNT).fill(0));
  finisher.progress = MAX_LAPS * 2 * Math.PI - 0.001;
  finisher.ticks = 4000;
  finisher.lastProgressTick = finisher.ticks;
  finisher.lapStart = 3600;
  step(finisher);
  expect(finisher.stopReason).toBe('finish');
  expect(finisher.bestLap).toBeCloseTo(401 / 60);
  expect(fitness({ ...finisher, ticks: 3000 })).toBeGreaterThan(fitness(finisher));
});

it('starts three cars apart and provides relative opponent positions and speed', () => {
  const cars = [0, 1, 2].map((slot) => createCar(new Array(WEIGHT_COUNT).fill(0), slot));
  expect(overlaps(cars[0], cars[1])).toBe(false);
  expect(overlaps(cars[1], cars[2])).toBe(false);
  const inputs = trafficInputs(cars[0], cars);
  expect(inputs).toHaveLength(INPUT_COUNT);
  expect(inputs.every(Number.isFinite)).toBe(true);
  expect(inputs[6]).toBeLessThan(0);
});

it('stops both cars on contact, penalizes collisions, and keeps wrecks as obstacles', () => {
  const a = createCar(new Array(WEIGHT_COUNT).fill(0));
  const b = createCar(new Array(WEIGHT_COUNT).fill(0));
  a.progress = b.progress = Math.PI * 2;
  stepRace([a, b]);
  expect(a.stopReason).toBe('collision');
  expect(b.stopReason).toBe('collision');
  expect(a.alive).toBe(false);
  expect(fitness(a)).toBeLessThan(fitness({ ...a, stopReason: 'timeout' }));
  const c = createCar(new Array(WEIGHT_COUNT).fill(0));
  c.x = a.x; c.y = a.y;
  stepRace([a, c]);
  expect(c.stopReason).toBe('collision');
});

it('can brake to a standstill and bases every decision on the same pre-move snapshot', () => {
  const genome = new Array(WEIGHT_COUNT).fill(0);
  genome[WEIGHT_COUNT - 1] = -100;
  const car = createCar(genome);
  for (let tick = 0; tick < 100; tick += 1) step(car);
  expect(car.speed).toBeLessThan(0.001);
  const cars = [0, 1, 2].map((slot) => createCar(randomGenome(seededRandom()), slot));
  const reverse = cars.map((entry) => ({ ...entry, genome: [...entry.genome] })).reverse();
  stepRace(cars); stepRace(reverse);
  expect(cars.map((entry) => [entry.x, entry.y])).toEqual(reverse.reverse().map((entry) => [entry.x, entry.y]));
});

it('cannot rotate while stopped and terminates drivers that make no forward progress', () => {
  const car = createCar(new Array(WEIGHT_COUNT).fill(0));
  car.speed = 0;
  const heading = car.heading;
  step(car, false, [1, -1]);
  expect(car.heading).toBe(heading);
  for (let tick = 0; tick < 300; tick += 1) step(car, false, [1, -1]);
  expect(car.stopReason).toBe('stalled');
  expect(car.alive).toBe(false);
});

it('switches away from a dead heat while other drivers train and stays when all finish', () => {
  const cars = Array.from({ length: 9 }, (_, i) => createCar(new Array(WEIGHT_COUNT).fill(0), i % 3));
  cars.slice(0, 3).forEach((car) => { car.alive = false; });
  expect(nextLiveHeat(cars, 0)).toBe(1);
  expect(nextLiveHeat(cars, 2)).toBe(2);
  cars.forEach((car) => { car.alive = false; });
  expect(nextLiveHeat(cars, 2)).toBe(2);
});
