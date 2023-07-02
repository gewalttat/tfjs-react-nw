//@ts-nocheck
import React, { FC, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { trainData } from './dataset';

const SMALL_FILE = 1;
const BIG_FILE = 100;
const HUGE_FILE = 10000;

export const Visualization: FC = () => {
    const [result, setResult] = useState<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]>();

    // a model for linear regression
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    // compile w sgd alg
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    let k = 0;
    let b = 0;
    model.setWeights([tf.tensor2d([k], [1, 1]), tf.tensor1d([b])]);

    // generate some synthetic data for training
    const xs = tf.tensor2d(trainData.sizeMB, [20, 1]);
    const ys = tf.tensor2d(trainData.timeSec, [20, 1]);

    // Train the model w a lot of epochs
    model.fit(xs, ys, {epochs: 200});
    
    useEffect(() => {
    // set up tensor predicate to component state
      if (!result) setResult(() => model.predict(tf.tensor2d([[SMALL_FILE], [BIG_FILE], [HUGE_FILE]], [3, 1])))
    }, [result, model]);

  return (
    // render predicate
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {result?.arraySync().map((i, index) => <span key={`${i}${index}`}>{i}</span>)}
      </div>
  )
}