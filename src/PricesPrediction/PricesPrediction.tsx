import React from 'react';
import { trainData, testData } from './dataset';
import { TrainingCharts } from '../Training/TrainingCharts';
import { useRegressionTraining } from '../Training/useRegressionTraining';

const HOME_SIZES = [1200, 2500, 4200];
const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

export function PricesPrediction() {
  const { history, result, status, training, train } = useRegressionTraining();
  const watchTraining = () => train({
    inputs: trainData.sizeSqft, targets: trainData.priceUSD,
    validationInputs: testData.sizeSqft, validationTargets: testData.priceUSD,
    predictInputs: HOME_SIZES, epochs: 300, learningRate: 0.00005,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ margin: 0 }}>Real estate price prediction</h3>
      <div style={{ fontSize: 13, color: '#a7b2c7' }}>{status}</div>
      {result.map((value, index) => <span key={index}>{`${HOME_SIZES[index]} sqft → ${formatter.format(value)}`}</span>)}
      <button onClick={watchTraining} disabled={training} style={{ width: 180 }}>{training ? 'training…' : 'watch training'}</button>
      <TrainingCharts history={history} classification={false} lossLabel="MAE loss (USD)"
        validationNote="Validation uses a separate test dataset. MAE is the average absolute error in dollars." />
    </div>
  );
}
