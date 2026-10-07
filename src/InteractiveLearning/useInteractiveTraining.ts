import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { TrainingEpoch } from '../Training/TrainingCharts';

export function useInteractiveTraining() {
  const [history, setHistory] = useState<TrainingEpoch[]>([]);
  const [prediction, setPrediction] = useState<number[]>([]);
  const [training, setTraining] = useState(false);
  const [status, setStatus] = useState('Ready to train');
  const active = useRef(true);
  const running = useRef(false);
  const modelRef = useRef<tf.Sequential | null>(null);
  const stopping = useRef(false);
  const [modelVersion, setModelVersion] = useState(0);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; if (modelRef.current) { if (running.current) modelRef.current.stopTraining = true; else { modelRef.current.dispose(); modelRef.current = null; } } };
  }, []);

  function reset() {
    if (running.current) return;
    modelRef.current?.dispose(); modelRef.current = null;
    setHistory([]); setPrediction([]); setStatus('Data changed. Train to update the prediction.');
  }
  function stop() {
    stopping.current = true;
    if (modelRef.current) modelRef.current.stopTraining = true;
  }
  async function train(options: { inputs: number[][]; targets: number[][]; preview: number[][]; neurons: number; classification: boolean; epochs?: number; batchSize?: number; learningRate?: number; retainModel?: boolean }) {
    if (running.current) return;
    running.current = true; stopping.current = false;
    setTraining(true); setHistory([]); setPrediction([]); setStatus('Training…');
    modelRef.current?.dispose();
    const model = tf.sequential();
    let completed = false;
    modelRef.current = model;
    const optimizer = tf.train.adam(options.learningRate ?? 0.01);
    const tensors: tf.Tensor[] = [];
    const tensor = (values: number[][]) => { const result = tf.tensor2d(values); tensors.push(result); return result; };
    try {
      model.add(tf.layers.dense({ inputShape: [options.inputs[0].length], units: options.neurons, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: options.neurons, activation: 'tanh' }));
      model.add(tf.layers.dense({ units: options.targets[0].length, activation: options.classification ? 'softmax' : 'linear' }));
      model.compile({ optimizer, loss: options.classification ? 'categoricalCrossentropy' : 'meanSquaredError', metrics: options.classification ? ['accuracy'] : [] });
      const xs = tensor(options.inputs), ys = tensor(options.targets), preview = tensor(options.preview);
      const start = performance.now();
      const epochs = options.epochs ?? (options.classification ? 300 : 500);
      const updatePreview = async () => {
        const output = model.predict(preview) as tf.Tensor;
        try { const values = Array.from(await output.data()); if (active.current) setPrediction(values); }
        finally { output.dispose(); }
      };
      await model.fit(xs, ys, {
        epochs, batchSize: Math.min(options.batchSize ?? 32, options.inputs.length), shuffle: true,
        callbacks: {
          onBatchEnd: async () => { if (!active.current || stopping.current) model.stopTraining = true; },
          onEpochEnd: async (epoch, logs) => {
            if (!active.current) { model.stopTraining = true; return; }
            setHistory((previous) => [...previous, { epoch: epoch + 1, elapsedMs: performance.now() - start, loss: logs?.loss, accuracy: logs?.acc ?? logs?.accuracy }]);
            setStatus(`Epoch ${epoch + 1}/${epochs}`);
            if (epoch % 5 === 0) await updatePreview();
            await tf.nextFrame();
          },
        },
      });
      completed = true;
      if (active.current) { await updatePreview(); setStatus(stopping.current ? 'Stopped. Current prediction retained.' : 'Training complete'); }
    } catch (error) {
      console.error('Interactive training failed', error);
      if (active.current) setStatus('Training failed. Try again.');
    } finally {
      tensors.forEach((value) => value.dispose());
      if (!(completed && options.retainModel && active.current)) { model.dispose(); modelRef.current = null; }
      else setModelVersion((version) => version + 1);
      optimizer.dispose();
      running.current = false; if (active.current) setTraining(false);
    }
  }
  function infer(inputs: number[][]): number[] | null {
    if (!modelRef.current || running.current) return null;
    return tf.tidy(() => Array.from((modelRef.current!.predict(tf.tensor2d(inputs)) as tf.Tensor).dataSync()));
  }
  return { infer, modelVersion, history, prediction, training, status, train, stop, reset };
}
