import React from 'react';
import * as tf from '@tensorflow/tfjs';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useInteractiveTraining } from './useInteractiveTraining';

it.each([false, true])('learns interactive data, refreshes previews, and frees tensors (classification=%s)', async (classification) => {
  await tf.setBackend('cpu');
  await tf.ready();
  const baseline = tf.memory().numTensors;
  let state: ReturnType<typeof useInteractiveTraining>;
  function Harness() {
    state = useInteractiveTraining();
    return <><button disabled={state.training} onClick={() => state.train({
      inputs: [[-1], [-0.5], [0.5], [1]],
      targets: classification ? [[1, 0], [1, 0], [0, 1], [0, 1]] : [[-0.5], [-0.25], [0.25], [0.5]],
      preview: [[-0.5], [0.5]], neurons: 4, classification, epochs: 40,
    })}>Train</button><div>{state.status}</div></>;
  }
  render(<Harness />);
  fireEvent.click(screen.getByText('Train'));
  await waitFor(() => expect(screen.getByText('Training complete')).toBeTruthy(), { timeout: 5000 });
  expect(state!.history).toHaveLength(40);
  expect(state!.history[39].loss!).toBeLessThan(state!.history[0].loss!);
  expect(state!.prediction).toHaveLength(classification ? 4 : 2);
  expect(state!.prediction.every(Number.isFinite)).toBe(true);
  if (classification) {
    expect(state!.prediction[0]).toBeGreaterThan(state!.prediction[1]);
    expect(state!.prediction[3]).toBeGreaterThan(state!.prediction[2]);
  }
  expect(tf.memory().numTensors).toBe(baseline);
});

it('retains a trained model for new inputs and frees it on reset', async () => {
  await tf.setBackend('cpu');
  const baseline = tf.memory().numTensors;
  let state: ReturnType<typeof useInteractiveTraining>;
  function Harness() {
    state = useInteractiveTraining();
    return <><button onClick={() => state.train({ inputs: [[-1], [1]], targets: [[-0.5], [0.5]], preview: [[0]], neurons: 4, classification: false, epochs: 2, retainModel: true })}>Train retained</button>
      <button onClick={state.reset}>Reset retained</button><div>{state.status}</div></>;
  }
  render(<Harness />);
  fireEvent.click(screen.getByText('Train retained'));
  await waitFor(() => expect(screen.getByText('Training complete')).toBeTruthy());
  expect(state!.infer([[0.25]])?.every(Number.isFinite)).toBe(true);
  expect(tf.memory().numTensors).toBeGreaterThan(baseline);
  fireEvent.click(screen.getByText('Reset retained'));
  expect(state!.infer([[0.25]])).toBeNull();
  expect(tf.memory().numTensors).toBe(baseline);
});
