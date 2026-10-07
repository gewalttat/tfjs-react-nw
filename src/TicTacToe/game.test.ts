import { bestMove, encode, moveValues, outcome, trainingPositions, turn } from './game';

it('detects wins, draws, and unfinished games', () => {
  expect(outcome([1, 1, 1, -1, -1, 0, 0, 0, 0])).toBe(1);
  expect(outcome([1, -1, 1, 1, -1, -1, -1, 1, 1])).toBe(0);
  expect(outcome(new Array(9).fill(0))).toBeNull();
});

it('encodes the board from the current player perspective', () => {
  const encoded = encode([1, -1, 0, 0, 0, 0, 0, 0, 0], -1);
  expect(encoded).toHaveLength(27);
  expect(encoded.slice(0, 3)).toEqual([0, 1, 0]);
  expect(encoded.slice(9, 12)).toEqual([1, 0, 0]);
  expect(encoded.slice(18, 21)).toEqual([0, 0, 1]);
});

it('teaches immediate wins and blocks forced losses, excluding occupied cells', () => {
  const win = [1, 1, 0, -1, -1, 0, 0, 0, 0];
  const values = moveValues(win, 1);
  expect(values[2]).toBe(1);
  expect(bestMove(win, values)).toBe(2);
  const block = [-1, -1, 0, 1, 0, 0, 0, 1, 0];
  const blockValues = moveValues(block, 1);
  expect(blockValues[2]).toBeGreaterThan(blockValues[4]);
  expect(bestMove([1, 0, 0, 0, 0, 0, 0, 0, 0], [100, 0.2, 0.8, 0, 0, 0, 0, 0, 0])).toBe(2);
});

it('generates unique reachable nonterminal positions and optimal play always draws', () => {
  const positions = trainingPositions();
  expect(positions.length).toBeGreaterThan(4000);
  expect(new Set(positions.map((row) => row.board.join(','))).size).toBe(positions.length);
  expect(positions.every((row) => outcome(row.board) === null && row.targets.length === 9)).toBe(true);
  const board = new Array(9).fill(0);
  while (outcome(board) === null) {
    const player = turn(board);
    board[bestMove(board, moveValues(board, player))] = player;
  }
  expect(outcome(board)).toBe(0);
});
