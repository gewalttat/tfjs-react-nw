import React from 'react';
import { trainData, testData } from './dataset';
import { TrainingCharts } from '../Training/TrainingCharts';
import { useRegressionTraining } from '../Training/useRegressionTraining';

const FILE_SIZES = [1, 100, 10000];

export function LoadPrediction() {
  const { history, result, status, training, train } = useRegressionTraining();
  const watchTraining = () => train({
    inputs: trainData.sizeMB, targets: trainData.timeSec,
    validationInputs: testData.sizeMB, validationTargets: testData.timeSec,
    predictInputs: FILE_SIZES, epochs: 200, learningRate: 0.0005,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ margin: 0 }}>Load time prediction</h3>
      <div style={{ fontSize: 13, color: '#a7b2c7' }}>{status}</div>
      {result.map((value, index) => <span key={index}>{`${FILE_SIZES[index]} MB → ${value.toFixed(3)} sec`}</span>)}
      <button onClick={watchTraining} disabled={training} style={{ width: 150 }}>{training ? 'training…' : 'watch training'}</button>
      <TrainingCharts history={history} classification={false} lossLabel="MAE loss (sec)"
        validationNote="Validation uses a separate test dataset. MAE is the average absolute error in seconds." />
    </div>
  );
}
