import React, { useEffect, useState } from 'react';
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
  model.compile({loss: 'meanAbsoluteError', optimizer: optimizer});

  // Initialize to k=0, b=0 for pretty illustration purposes.  
  // You may want to remove this to see how the model looks with
  // random initialization
  let k = 0;
  let b = 0;
  model.setWeights([tf.tensor2d([k], [1, 1]), tf.tensor1d([b])]);

  // generate some synthetic data for training
  const xs = tf.tensor2d(trainData.sizeMB, [20, 1]);
  const ys = tf.tensor2d(trainData.timeSec, [20, 1]);

 const surface = { name: 'show.fitCallbacks', tab: 'Training' };

  // Train the model w a lot of epochs
  model.fit(xs, ys, {
    epochs: 200, callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc'])});

  useEffect(() => {
    // set up tensor predicate to component state
    if (!result) {
      setResult(() => model.predict(tf.tensor2d([[SMALL_FILE], [BIG_FILE], [HUGE_FILE]], [3, 1])));
    }
  }, [result, model]);

  return (
    // render predicate
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {/*//@ts-ignore */}
      {result?.arraySync().map((i, index) => <span key={`${i}${index}`}>{i}</span>)}
    </div>
  )
}