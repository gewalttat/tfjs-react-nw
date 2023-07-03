import React, { useCallback, useMemo, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { trainData } from './dataset';

const SMALL_FILE = 1;
const BIG_FILE = 100;
const HUGE_FILE = 10000;

export function Visualization() {
  const [result, setResult] = useState<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]>();

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
    }).then(() => setResult(() => model.predict(tf.tensor2d([[SMALL_FILE], [BIG_FILE], [HUGE_FILE]], [3, 1]))));
  }, [model, surface, xs, ys])

  return (
    // render predicate
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {/*//@ts-ignore */}
      {result?.arraySync().map((i, index) => <span key={`${i}${index}`}>{i}</span>)}
      <button onClick={watchTraining} style={{ width: 150 }}>watch training</button>
    </div>
  )
}