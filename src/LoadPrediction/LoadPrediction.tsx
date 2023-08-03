import React, { useCallback, useEffect, useMemo, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { trainData } from './dataset';

const SMALL_FILE = 1;
const BIG_FILE = 100;
const HUGE_FILE = 10000;

export function LoadPrediction() {
  const [result, setResult] = useState<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]>();
  const [modelLosses, setModelLosses] = useState<number[]>([]);
  const [modelEpoch, setModelEpoch] = useState<number[]>([]);

  // a model for linear regression
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
  }));
  model.summary();
  const optimizer = tf.train.sgd(0.0005);
  model.compile({ loss: 'meanAbsoluteError', optimizer: optimizer });

  // static visualization
  // let k = 0;
  // let b = 0;
  // model.setWeights([tf.tensor2d([k], [1, 1]), tf.tensor1d([b])]);

  // generate some synthetic data for training
  const xs = tf.tensor2d(trainData.sizeMB, [20, 1]);
  const ys = tf.tensor2d(trainData.timeSec, [20, 1]);

  // creating surface
  const surface = useMemo(() => ({ name: 'show.fitCallbacks', tab: 'Training' }), []);

  // callback to train and set model results
  const watchTraining = useCallback(() => {
    model.fit(xs, ys, {
      epochs: 200, callbacks: tfvis.show.fitCallbacks(surface, ['loss'])
      //replace w user input later
    })
    .then((res) => {
      setModelLosses(() => res.history.loss as number[]);
      setModelEpoch(() => res.epoch);
    })
    .then(() => setResult(() => model.predict(tf.tensor2d([[SMALL_FILE], [BIG_FILE], [HUGE_FILE]], [3, 1]))))
  }, [model, surface, xs, ys]);

  function getEveryNth(arr: number[], nth: number) {
    const result = [];
  
    for (let index = 0; index < arr.length; index += nth) {
      result.push(arr[index]);
    }
  
    return result;
  }

  useEffect(() => {
    if (modelLosses.length && modelEpoch.length) {
      const steppedLosses = getEveryNth(modelLosses, 20);
      const steppedEpochs = getEveryNth(modelEpoch, 20);
      const xTickLabels = steppedEpochs.map((i) => String(i));
      const values = steppedLosses.map((i) => [i]);

      tfvis.render.heatmap({ name: 'heatmap', tab: 'Training'}, {
          values,
          xTickLabels,
          yTickLabels: ['loss']
        }, {
          width: 500,
          height: 300,
          xLabel: 'value',
          yLabel: 'epoch',
          colorMap: 'blues'
        })
    }
  }, [modelEpoch, modelLosses])


  return (
    // render predicate
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {/* tfjs ts issue */}
      {/*//@ts-ignore */}
      {result?.arraySync().map((i, index) => <span key={`${i}${index}`}>{`${Math.floor(Number(i))} sec`}</span>)}
      <button onClick={watchTraining} style={{ width: 150 }}>watch training</button>
    </div>
  )
}