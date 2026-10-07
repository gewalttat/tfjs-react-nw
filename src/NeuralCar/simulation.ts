export const TRACK = { width: 800, height: 480, cx: 400, cy: 240, rx: 340, ry: 190, inner: 0.64 };
export const SENSOR_ANGLES = [-1.2, -0.6, 0, 0.6, 1.2];
export const SENSOR_RANGE = 130;
export const WEIGHT_COUNT = 74; // (6 inputs + bias) * 8 + (8 hidden + bias) * 2
export type Genome = number[];
export interface Car {
  genome: Genome; x: number; y: number; heading: number; speed: number;
  alive: boolean; progress: number; angle: number; ticks: number;
}

export function onRoad(x: number, y: number): boolean {
  const dx = (x - TRACK.cx) / TRACK.rx;
  const dy = (y - TRACK.cy) / TRACK.ry;
  const radiusSquared = dx * dx + dy * dy;
  return radiusSquared < 1 && radiusSquared > TRACK.inner * TRACK.inner;
}

export function createCar(genome: Genome): Car {
  const x = TRACK.cx + TRACK.rx * 0.82;
  return { genome: [...genome], x, y: TRACK.cy, heading: Math.PI / 2, speed: 2, alive: true, progress: 0, angle: 0, ticks: 0 };
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

export function network(genome: Genome, inputs: number[]): [number, number] {
  const hidden: number[] = [];
  for (let unit = 0; unit < 8; unit += 1) {
    let sum = genome[unit * 7 + 6];
    for (let i = 0; i < 6; i += 1) sum += inputs[i] * genome[unit * 7 + i];
    hidden.push(Math.tanh(sum));
  }
  const outputs = [0, 1].map((unit) => {
    let sum = genome[56 + unit * 9 + 8];
    for (let i = 0; i < 8; i += 1) sum += hidden[i] * genome[56 + unit * 9 + i];
    return Math.tanh(sum);
  });
  return [outputs[0], outputs[1]];
}

export function step(car: Car) {
  if (!car.alive) return;
  const [steering, throttle] = network(car.genome, [...sensors(car), car.speed / 4]);
  car.heading += steering * 0.085;
  const targetSpeed = 1.2 + (throttle + 1) * 1.4;
  car.speed += (targetSpeed - car.speed) * 0.15;
  car.x += Math.cos(car.heading) * car.speed;
  car.y += Math.sin(car.heading) * car.speed;
  car.ticks += 1;
  // Check the car's corners, rather than just its center.
  for (const forward of [-7, 7]) {
    for (const sideways of [-4, 4]) {
      const x = car.x + Math.cos(car.heading) * forward - Math.sin(car.heading) * sideways;
      const y = car.y + Math.sin(car.heading) * forward + Math.cos(car.heading) * sideways;
      if (!onRoad(x, y)) car.alive = false;
    }
  }
  const angle = Math.atan2((car.y - TRACK.cy) / TRACK.ry, (car.x - TRACK.cx) / TRACK.rx);
  let delta = angle - car.angle;
  if (delta > Math.PI) delta -= 2 * Math.PI;
  if (delta < -Math.PI) delta += 2 * Math.PI;
  car.progress += delta;
  car.angle = angle;
  if (car.progress < -0.3 || car.ticks >= 1800 || car.progress >= 6 * Math.PI) car.alive = false;
}

export function fitness(car: Car): number {
  return Math.max(0, car.progress / (2 * Math.PI));
}

export function randomGenome(random = Math.random): Genome {
  return Array.from({ length: WEIGHT_COUNT }, () => (random() * 2 - 1) * 1.2);
}

export function breed(cars: Car[], random = Math.random): Car[] {
  const ranked = [...cars].sort((a, b) => fitness(b) - fitness(a));
  return ranked.map((_, index) => {
    if (index < 3) return createCar(ranked[index].genome);
    if (index >= ranked.length - 4) return createCar(randomGenome(random));
    const parent = ranked[Math.floor(random() * Math.min(8, ranked.length))].genome;
    const mutation = index % 3 === 0 ? 0.6 : 0.18;
    return createCar(parent.map((weight) => random() < 0.3 ? weight + (random() * 2 - 1) * mutation : weight));
  });
}
