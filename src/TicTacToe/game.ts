export type Player = 1 | -1;
export type Board = number[];
const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

export function outcome(board: Board): number | null {
  for (const [a, b, c] of LINES) {
    if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) return board[a];
  }
  return board.every((cell) => cell !== 0) ? 0 : null;
}

export function turn(board: Board): Player {
  return board.filter((cell) => cell !== 0).length % 2 === 0 ? 1 : -1;
}

export function encode(board: Board, player: Player): number[] {
  return [player, -player, 0].flatMap((value) => board.map((cell) => cell === value ? 1 : 0));
}

const cache = new Map<string, number>();
function minimax(board: Board, player: Player): number {
  const result = outcome(board);
  if (result !== null) return result * player;
  const key = `${board.join(',')}:${player}`;
  const cached = cache.get(key);
  if (cached !== undefined) return cached;
  let best = -1;
  for (let i = 0; i < 9; i += 1) {
    if (board[i] !== 0) continue;
    const next = [...board];
    next[i] = player;
    best = Math.max(best, -minimax(next, -player as Player));
  }
  cache.set(key, best);
  return best;
}

export function moveValues(board: Board, player: Player): number[] {
  return board.map((cell, index) => {
    if (cell !== 0) return 0;
    const next = [...board];
    next[index] = player;
    return -minimax(next, -player as Player);
  });
}

export function bestMove(board: Board, values: number[]): number {
  let best = -1;
  for (let i = 0; i < 9; i += 1) {
    if (board[i] === 0 && (best === -1 || values[i] > values[best])) best = i;
  }
  return best;
}

export function trainingPositions(): { board: Board; inputs: number[]; targets: number[] }[] {
  const visited = new Set<string>();
  const positions: { board: Board; inputs: number[]; targets: number[] }[] = [];
  function visit(board: Board, player: Player) {
    const key = board.join(',');
    if (visited.has(key) || outcome(board) !== null) return;
    visited.add(key);
    positions.push({ board, inputs: encode(board, player), targets: moveValues(board, player) });
    board.forEach((cell, index) => {
      if (cell !== 0) return;
      const next = [...board];
      next[index] = player;
      visit(next, -player as Player);
    });
  }
  visit(new Array(9).fill(0), 1);
  return positions;
}
