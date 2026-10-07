import { Localized } from '../i18n/Locale';
import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { TrainingCharts, TrainingEpoch } from '../Training/TrainingCharts';
import { bestMove, Board, encode, outcome, Player, trainingPositions, turn } from './game';

function predict(model: tf.Sequential, board: Board): number[] {
  return tf.tidy(() => Array.from((model.predict(tf.tensor2d([encode(board, turn(board))])) as tf.Tensor).dataSync()));
}

export function TicTacToe() {
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [board, setBoard] = useState<Board>(new Array(9).fill(0));
  const [human, setHuman] = useState<Player>(1);
  const [scores, setScores] = useState<number[]>([]);
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [training, setTraining] = useState(false);
  const [status, setStatus] = useState('Train a tiny neural network, then play against it.');
  const [agreement, setAgreement] = useState<number | null>(null);
  const modelRef = useRef<tf.Sequential | null>(null);
  const active = useRef(true);
  const running = useRef(false);
  const result = outcome(board);
  const player = turn(board);

  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
      modelRef.current?.dispose();
      modelRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!model || result !== null || training) { setScores([]); return; }
    const values = predict(model, board);
    setScores(values);
    if (player === human) return;
    const timer = window.setTimeout(() => {
      const move = bestMove(board, values);
      if (move < 0) return;
      setBoard((previous) => {
        const next = [...previous];
        next[move] = player;
        return next;
      });
    }, 350);
    return () => window.clearTimeout(timer);
  }, [board, model, human, player, result, training]);

  async function train() {
    if (running.current) return;
    running.current = true;
    setTraining(true);
    setHistory([]);
    setAgreement(null);
    setBoard(new Array(9).fill(0));
    setStatus('Generating minimax examples…');
    let nextModel: tf.Sequential | null = null;
    const tensors: tf.Tensor[] = [];
    const optimizer = tf.train.adam(0.003);
    try {
      await tf.nextFrame();
      if (!active.current) return;
      const positions = trainingPositions();
      tf.util.shuffle(positions);
      const split = Math.floor(positions.length * 0.9);
      const trainingSet = positions.slice(0, split);
      const validationSet = positions.slice(split);
      const tensor = (rows: number[][]) => {
        const value = tf.tensor2d(rows);
        tensors.push(value);
        return value;
      };
      const xs = tensor(trainingSet.map((row) => row.inputs));
      const ys = tensor(trainingSet.map((row) => row.targets));
      const validationXs = tensor(validationSet.map((row) => row.inputs));
      const validationYs = tensor(validationSet.map((row) => row.targets));
      const network = tf.sequential();
      nextModel = network;
      network.add(tf.layers.dense({ inputShape: [27], units: 64, activation: 'relu' }));
      network.add(tf.layers.dense({ units: 32, activation: 'relu' }));
      network.add(tf.layers.dense({ units: 9, activation: 'tanh' }));
      network.compile({ optimizer, loss: 'meanSquaredError' });
      const start = performance.now();
      let epoch = 0;
      let lastUpdate = 0;
      await network.fit(xs, ys, {
        epochs: 40, batchSize: 64, shuffle: true,
        validationData: [validationXs, validationYs],
        callbacks: {
          onEpochBegin: async (index) => { epoch = index; },
          onBatchEnd: async (batch, logs) => {
            if (!active.current) { network.stopTraining = true; return; }
            const elapsedMs = performance.now() - start;
            if (elapsedMs - lastUpdate < 50) return;
            lastUpdate = elapsedMs;
            setHistory((previous) => [...previous, {
              epoch: epoch + (batch + 1) / Math.ceil(trainingSet.length / 64), elapsedMs, loss: logs?.loss,
            }]);
            await tf.nextFrame();
          },
          onEpochEnd: async (index, logs) => {
            if (!active.current) { network.stopTraining = true; return; }
            setHistory((previous) => [...previous, {
              epoch: index + 1, elapsedMs: performance.now() - start, loss: logs?.loss, validationLoss: logs?.val_loss,
            }]);
            setStatus(`Training: epoch ${index + 1}/40 · ${positions.length} positions`);
          },
        },
      });
      if (!active.current) return;
      const predictions = network.predict(validationXs) as tf.Tensor;
      tensors.push(predictions);
      const values = await predictions.array() as number[][];
      const optimal = validationSet.filter((row, index) => {
        const move = bestMove(row.board, values[index]);
        return row.targets[move] === row.targets[bestMove(row.board, row.targets)];
      }).length;
      if (!active.current) return;
      setAgreement(optimal / validationSet.length);
      modelRef.current?.dispose();
      modelRef.current = network;
      setModel(network);
      nextModel = null;
      setStatus('Network ready. Pick a square!');
    } catch (error) {
      console.error('Tic-tac-toe training failed', error);
      if (active.current) setStatus('Training failed. Try again.');
    } finally {
      tensors.forEach((tensor) => tensor.dispose());
      nextModel?.dispose();
      optimizer.dispose();
      running.current = false;
      if (active.current) setTraining(false);
    }
  }

  const gameStatus = !model ? 'Train the network to start' : result === 0 ? 'Draw!' : result !== null
    ? result === human ? 'You win!' : 'Network wins!'
    : player === human ? 'Your turn' : 'Network is thinking…';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <h3 style={{ margin: 0 }}><Localized>{"Neural tic-tac-toe"}</Localized></h3>
      <div style={{ color: '#a6a39c', fontSize: 13 }}><Localized>{"27 inputs → 64 → 32 → 9 move values. Learns from minimax; plays using its own predictions."}</Localized></div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        <button onClick={train} disabled={training}><Localized>{training ? 'training…' : model ? 'retrain network' : 'train network'}</Localized></button>
        <button disabled={training || !model} onClick={() => setBoard(new Array(9).fill(0))}><Localized>{"new game"}</Localized></button>
        <label style={{ alignSelf: 'center' }}><Localized>{"Play as"}</Localized><Localized>{' '}</Localized>
          <select value={human} disabled={training} onChange={(event) => { setHuman(Number(event.target.value) as Player); setBoard(new Array(9).fill(0)); }}>
            <option value={1}><Localized>{"X · first"}</Localized></option><option value={-1}><Localized>{"O · second"}</Localized></option>
          </select>
        </label>
      </div>
      <div style={{ fontSize: 13, color: '#a6a39c' }} aria-live="polite"><Localized>{status}</Localized></div>
      <Localized>{agreement !== null && <div><Localized>{"Optimal moves on held-out positions: "}</Localized><Localized>{(agreement * 100).toFixed(1)}</Localized><Localized>{"%"}</Localized></div>}</Localized>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 24 }}>
        <div>
          <div style={{ fontWeight: 700, marginBottom: 12 }} aria-live="polite"><Localized>{gameStatus}</Localized></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, maxWidth: 330 }}>
            <Localized>{board.map((cell, index) => (
              <button key={index} aria-label={`Square ${index + 1}: ${cell === 1 ? 'X' : cell === -1 ? 'O' : 'empty'}`}
                disabled={!model || training || result !== null || player !== human || cell !== 0}
                onClick={() => setBoard((previous) => { const next = [...previous]; next[index] = human; return next; })}
                style={{ aspectRatio: '1', padding: 8, boxShadow: 'none', opacity: 1,
                  background: cell !== 0 || scores.length === 0 ? '#303030' : `hsl(42, 25%, ${18 + ((scores[index] + 1) / 2) * 25}%)`,
                  border: '1px solid #55524c', fontSize: 32 }}>
                <Localized>{cell === 1 ? 'X' : cell === -1 ? 'O' : <span style={{ fontSize: 13 }}><Localized>{scores.length ? scores[index].toFixed(2) : '·'}</Localized></span>}</Localized>
              </button>
            ))}</Localized>
          </div>
          <p style={{ fontSize: 12, color: '#a6a39c', maxWidth: 330 }}><Localized>{"Move heatmap for the player to move: darker ≈ loss (−1), middle ≈ draw (0), lighter gold ≈ win (+1). Values are estimates, not probabilities."}</Localized></p>
        </div>
        <TrainingCharts history={history} classification={false} lossLabel="Move-value MSE"
          validationNote="10% of positions are held out. Optimal-move agreement accepts every equally good minimax move." />
      </div>
    </div>
  );
}
