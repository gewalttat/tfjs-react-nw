import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import * as tf from '@tensorflow/tfjs';
import { useRegressionTraining } from './useRegressionTraining';

it('records batches and validation, produces predictions, and releases tensors between runs', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const initialTensors = tf.memory().numTensors;
  let recorded: ReturnType<typeof useRegressionTraining>;
  function Harness() {
    const state = useRegressionTraining();
    recorded = state;
    return <>
      <button disabled={state.training} onClick={() => state.train({
        inputs: [1, 2, 3, 4], targets: [2, 4, 6, 8],
        validationInputs: [1.5, 2.5], validationTargets: [3, 5],
        predictInputs: [2], epochs: 2, learningRate: 0.01,
      })}>Train</button>
      <div>{state.status}</div>
    </>;
  }
  render(<Harness />);
  for (let run = 0; run < 2; run += 1) {
    fireEvent.click(screen.getByText('Train'));
    await waitFor(() => expect(screen.getByText('Training complete')).toBeTruthy());
    expect(recorded!.history).toHaveLength(4);
    expect(recorded!.history.filter((row) => row.validationLoss !== undefined)).toHaveLength(2);
    expect(recorded!.history.every((row) => row.elapsedMs >= 0 && Number.isFinite(row.loss))).toBe(true);
    expect(recorded!.result).toHaveLength(1);
    expect(Number.isFinite(recorded!.result[0])).toBe(true);
    expect(tf.memory().numTensors).toBe(initialTensors);
  }
});
