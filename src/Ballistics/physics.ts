export interface Target { distance: number; height: number; gravity: number }
export interface Launch { angle: number; speed: number }
export function idealLaunch(target: Target): Launch {
  const length = Math.hypot(target.distance, target.height);
  return { angle: Math.atan2(target.height + length, target.distance), speed: Math.sqrt(target.gravity * (target.height + length)) };
}
export function position(launch: Launch, gravity: number, time: number) {
  return { x: launch.speed * Math.cos(launch.angle) * time, y: launch.speed * Math.sin(launch.angle) * time - gravity * time * time / 2 };
}
export function input(target: Target) { return [(target.distance - 60) / 50, (target.height - 17.5) / 17.5, (target.gravity - 10) / 7]; }
export function output(launch: Launch) { return [launch.angle / (Math.PI / 2) * 2 - 1, launch.speed / 60 * 2 - 1]; }
export function decode(values: number[]): Launch { return { angle: Math.min(85, Math.max(5, (values[0] + 1) * 45)) * Math.PI / 180, speed: Math.min(60, Math.max(1, (values[1] + 1) * 30)) }; }
export function examples(count: number, seed = 42) {
  function random() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
  return Array.from({ length: count }, () => {
    const target = { distance: 10 + random() * 100, height: random() * 35, gravity: 3 + random() * 14 };
    return { inputs: input(target), targets: output(idealLaunch(target)) };
  });
}
