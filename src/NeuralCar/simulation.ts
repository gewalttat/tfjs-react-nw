export const TRACK = { width: 800, height: 480, cx: 400, cy: 240, rx: 340, ry: 190, inner: 0.64 };
export const SENSOR_ANGLES = [-1.2, -0.6, 0, 0.6, 1.2];
export const SENSOR_RANGE = 130;
export const MAX_LAPS = 10;
export const MAX_TICKS = 6000;
export const POPULATION = 42;
export const INPUT_COUNT = 12;
export const HIDDEN_COUNT = 12;
export const WEIGHT_COUNT = (INPUT_COUNT + 1) * HIDDEN_COUNT + (HIDDEN_COUNT + 1) * 2;
export type Genome = number[];
export interface Car {
  genome: Genome; x: number; y: number; heading: number; speed: number;
  alive: boolean; progress: number; angle: number; ticks: number;
  stopReason: 'stalled' | 'collision' | 'crash' | 'wrong-way' | 'finish' | 'timeout' | null;
  lapStart: number; bestLap: number | null; peakProgress: number; lastProgressTick: number;
}

export function onRoad(x: number, y: number): boolean {
  const dx = (x - TRACK.cx) / TRACK.rx;
  const dy = (y - TRACK.cy) / TRACK.ry;
  const radiusSquared = dx * dx + dy * dy;
  return radiusSquared < 1 && radiusSquared > TRACK.inner * TRACK.inner;
}

export function createCar(genome: Genome, slot = 0): Car {
  const angle = -slot * 0.16;
  const x = TRACK.cx + TRACK.rx * 0.82 * Math.cos(angle);
  const y = TRACK.cy + TRACK.ry * 0.82 * Math.sin(angle);
  const heading = Math.atan2(TRACK.ry * Math.cos(angle), -TRACK.rx * Math.sin(angle));
  return { genome: [...genome], x, y, heading, speed: 2, alive: true, progress: 0, angle, ticks: 0, stopReason: null, lapStart: 0, bestLap: null, peakProgress: 0, lastProgressTick: 0 };
}

export function sensors(car: Car): number[] {
  return SENSOR_ANGLES.map((offset) => {
    const angle = car.heading + offset;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    for (let distance = 4; distance <= SENSOR_RANGE; distance += 4) {
      if (!onRoad(car.x + dx * distance, car.y + dy * distance)) return distance / SENSOR_RANGE;
    }
    return 1;
  });
}

export function trafficInputs(car: Car, opponents: Car[]): number[] {
  const nearby = opponents.filter((other) => other !== car && other.stopReason !== 'finish')
    .sort((a, b) => Math.hypot(a.x - car.x, a.y - car.y) - Math.hypot(b.x - car.x, b.y - car.y));
  const inputs: number[] = [];
  for (let index = 0; index < 2; index += 1) {
    const other = nearby[index];
    if (!other || Math.hypot(other.x - car.x, other.y - car.y) > SENSOR_RANGE) {
      inputs.push(1, 0, 0);
      continue;
    }
    const dx = other.x - car.x;
    const dy = other.y - car.y;
    const forward = (dx * Math.cos(car.heading) + dy * Math.sin(car.heading)) / SENSOR_RANGE;
    const lateral = (-dx * Math.sin(car.heading) + dy * Math.cos(car.heading)) / SENSOR_RANGE;
    const closingSpeed = ((other.alive ? other.speed : 0) * Math.cos(other.heading - car.heading) - car.speed) / 4;
    inputs.push(forward, lateral, closingSpeed);
  }
  return [...sensors(car), car.speed / 4, ...inputs];
}

export function network(genome: Genome, inputs: number[]): [number, number] {
  const hidden: number[] = [];
  for (let unit = 0; unit < HIDDEN_COUNT; unit += 1) {
    const offset = unit * (INPUT_COUNT + 1);
    let sum = genome[offset + INPUT_COUNT];
    for (let i = 0; i < INPUT_COUNT; i += 1) sum += (inputs[i] ?? 0) * genome[offset + i];
    hidden.push(Math.tanh(sum));
  }
  const outputs = [0, 1].map((unit) => {
    const offset = (INPUT_COUNT + 1) * HIDDEN_COUNT + unit * (HIDDEN_COUNT + 1);
    let sum = genome[offset + HIDDEN_COUNT];
    for (let i = 0; i < HIDDEN_COUNT; i += 1) sum += hidden[i] * genome[offset + i];
    return Math.tanh(sum);
  });
  return [outputs[0], outputs[1]];
}

export function overlaps(a: Car, b: Car): boolean {
  const dx = b.x - a.x, dy = b.y - a.y;
  for (const angle of [a.heading, a.heading + Math.PI / 2, b.heading, b.heading + Math.PI / 2]) {
    const distance = Math.abs(dx * Math.cos(angle) + dy * Math.sin(angle));
    const radius = (car: Car) => 7 * Math.abs(Math.cos(car.heading - angle)) + 4 * Math.abs(Math.sin(car.heading - angle));
    if (distance >= radius(a) + radius(b)) return false;
  }
  return true;
}

export function stepRace(cars: Car[], unlimited = false) {
  // Snapshot all decisions before any car moves, so ordering gives no advantage.
  const decisions = cars.map((car) => network(car.genome, trafficInputs(car, cars)));
  cars.forEach((car, index) => step(car, unlimited, decisions[index]));
  for (let i = 0; i < cars.length; i += 1) {
    for (let j = i + 1; j < cars.length; j += 1) {
      const a = cars[i], b = cars[j];
      if (a.stopReason === 'finish' || b.stopReason === 'finish' || (!a.alive && !b.alive)) continue;
      if (overlaps(a, b)) {
        for (const car of [a, b]) { car.alive = false; car.stopReason = 'collision'; }
      }
    }
  }
}

export function step(car: Car, unlimited = false, decision?: [number, number]) {
  if (!car.alive) return;
  const [steering, throttle] = decision ?? network(car.genome, trafficInputs(car, []));
  const targetSpeed = (throttle + 1) * 2;
  car.speed += (targetSpeed - car.speed) * 0.15;
  // Turning requires forward movement; a stopped car cannot rotate in place.
  car.heading += steering * car.speed * 0.028;
  car.x += Math.cos(car.heading) * car.speed;
  car.y += Math.sin(car.heading) * car.speed;
  car.ticks += 1;
  // Check the car's corners, rather than just its center.
  for (const forward of [-7, 7]) {
    for (const sideways of [-4, 4]) {
      const x = car.x + Math.cos(car.heading) * forward - Math.sin(car.heading) * sideways;
      const y = car.y + Math.sin(car.heading) * forward + Math.cos(car.heading) * sideways;
      if (!onRoad(x, y)) { car.alive = false; car.stopReason = 'crash'; }
    }
  }
  const angle = Math.atan2((car.y - TRACK.cy) / TRACK.ry, (car.x - TRACK.cx) / TRACK.rx);
  let delta = angle - car.angle;
  if (delta > Math.PI) delta -= 2 * Math.PI;
  if (delta < -Math.PI) delta += 2 * Math.PI;
  const previousLaps = Math.floor(car.progress / (2 * Math.PI));
  car.progress += delta;
  car.angle = angle;
  if (car.progress > car.peakProgress + 0.12) {
    car.peakProgress = car.progress;
    car.lastProgressTick = car.ticks;
  }
  if (car.alive && Math.floor(car.progress / (2 * Math.PI)) > previousLaps && car.progress >= 2 * Math.PI) {
    const lapSeconds = (car.ticks - car.lapStart) / 60;
    car.bestLap = Math.min(car.bestLap ?? Infinity, lapSeconds);
    car.lapStart = car.ticks;
  }
  if (car.alive) {
    if (car.progress < -0.3) car.stopReason = 'wrong-way';
    else if (car.ticks - car.lastProgressTick >= 300) car.stopReason = 'stalled';
    else if (!unlimited && car.progress >= MAX_LAPS * 2 * Math.PI) car.stopReason = 'finish';
    else if (!unlimited && car.ticks >= MAX_TICKS) car.stopReason = 'timeout';
    if (car.stopReason) car.alive = false;
  }
}

export function laps(car: Car): number {
  return Math.max(0, car.progress / (2 * Math.PI));
}

export function fitness(car: Car): number {
  const distance = Math.min(MAX_LAPS, laps(car));
  // Only finishers get the speed bonus: survival and distance come first.
  return distance * (car.stopReason === 'collision' ? 0.7 : 1) + (car.stopReason === 'finish' ? MAX_TICKS / Math.max(1, car.ticks) : 0);
}

export function randomGenome(random = Math.random): Genome {
  const genome = Array.from({ length: WEIGHT_COUNT }, () => (random() * 2 - 1) * 0.6);
  // Initially prioritize discovering driving; traffic weights can evolve later.
  for (let unit = 0; unit < HIDDEN_COUNT; unit += 1) {
    for (let input = 6; input < INPUT_COUNT; input += 1) genome[unit * (INPUT_COUNT + 1) + input] *= 0.15;
  }
  genome[WEIGHT_COUNT - 1] = 0.5;
  return genome;
}

export function breed(cars: Car[], random = Math.random): Car[] {
  const ranked = [...cars].sort((a, b) => fitness(b) - fitness(a));
  const next = ranked.map((_, index) => {
    if (index < 3) return createCar(ranked[index].genome, index % 3);
    if (index >= ranked.length - 4) return createCar(randomGenome(random), index % 3);
    const parent = ranked[Math.floor(random() * Math.min(8, ranked.length))].genome;
    const mutation = index % 3 === 0 ? 0.6 : 0.18;
    return createCar(parent.map((weight) => random() < 0.3 ? weight + (random() * 2 - 1) * mutation : weight), index % 3);
  });
  for (let i = next.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [next[i], next[j]] = [next[j], next[i]];
  }
  return next.map((car, index) => createCar(car.genome, index % 3));
}

export function nextLiveHeat(cars: Car[], current: number): number {
  const count = Math.ceil(cars.length / 3);
  for (let offset = 0; offset < count; offset += 1) {
    const heat = (current + offset) % count;
    if (cars.slice(heat * 3, heat * 3 + 3).some((car) => car.alive)) return heat;
  }
  return current;
}
