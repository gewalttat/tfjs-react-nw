import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { TrainingEpoch } from './TrainingCharts';

export function useRegressionTraining() {
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [result, setResult] = useState<number[]>([]);
  const [status, setStatus] = useState('Ready to train');
  const [training, setTraining] = useState(false);
  const running = useRef(false);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);

  async function train(config: {
    inputs: number[]; targets: number[];
    validationInputs: number[]; validationTargets: number[];
    predictInputs: number[]; epochs: number; learningRate: number;
  }) {
    if (running.current) return;
    running.current = true;
    setTraining(true);
    setHistory([]);
    setResult([]);
    setStatus('Training…');
    const model = tf.sequential();
    const optimizer = tf.train.sgd(config.learningRate);
    const tensors: tf.Tensor[] = [];
    const tensor = (values: number[]) => {
      const value = tf.tensor2d(values, [values.length, 1]);
      tensors.push(value);
      return value;
    };
    try {
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      model.compile({ loss: 'meanAbsoluteError', optimizer });
      const xs = tensor(config.inputs);
      const ys = tensor(config.targets);
      const validationXs = tensor(config.validationInputs);
      const validationYs = tensor(config.validationTargets);
      const start = performance.now();
      let epoch = 0;
      await model.fit(xs, ys, {
        epochs: config.epochs,
        validationData: [validationXs, validationYs],
        callbacks: {
          onEpochBegin: async (index) => { epoch = index; },
          onBatchEnd: async (batch, logs) => {
            if (!active.current) { model.stopTraining = true; return; }
            setHistory((previous) => [...previous, {
              epoch: epoch + (batch + 1) / Math.ceil(config.inputs.length / 32),
              elapsedMs: performance.now() - start,
              loss: logs?.loss,
            }]);
            await tf.nextFrame();
          },
          onEpochEnd: async (index, logs) => {
            if (!active.current) { model.stopTraining = true; return; }
            setHistory((previous) => [...previous, {
              epoch: index + 1,
              elapsedMs: performance.now() - start,
              loss: logs?.loss,
              validationLoss: logs?.val_loss,
            }]);
            setStatus(`Training: epoch ${index + 1}/${config.epochs}`);
          },
        },
      });
      if (!active.current) return;
      const prediction = model.predict(tensor(config.predictInputs)) as tf.Tensor;
      tensors.push(prediction);
      const values = Array.from(await prediction.data());
      if (active.current) {
        setResult(values);
        setStatus('Training complete');
      }
    } catch (error) {
      console.error('Regression training failed', error);
      if (active.current) setStatus('Training failed. Try again.');
    } finally {
      tensors.forEach((value) => value.dispose());
      model.dispose();
      optimizer.dispose();
      running.current = false;
      if (active.current) setTraining(false);
    }
  }
  return { history, result, status, training, train };
}
