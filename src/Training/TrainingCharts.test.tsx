import React from 'react';
import { render, screen } from '@testing-library/react';
import { TrainingCharts } from './TrainingCharts';

const history = [
  { epoch: 0.5, elapsedMs: 50, loss: 0.8, accuracy: 0.6 },
  { epoch: 1, elapsedMs: 100, loss: 0.4, validationLoss: 0.5, accuracy: 0.8, validationAccuracy: 0.75 },
];

it('shows live batch data, epoch validation, and heatmap values in milliseconds', () => {
  const { container } = render(<TrainingCharts history={history} />);
  expect(screen.getByText('Training heatmap')).toBeTruthy();
  expect(screen.getAllByText('Training time (ms)')).toHaveLength(2);
  expect(container.querySelector('title')?.textContent).toContain('50 ms');
  expect(container.querySelector('[title="Val acc, epoch 1.00, 100 ms: 0.7500"]')).toBeTruthy();
  expect(container.querySelector('[title="Val acc, epoch 0.50, 50 ms: not measured"]')).toBeTruthy();
});

it('shows regression MAE and a heatmap without a classification accuracy chart', () => {
  render(<TrainingCharts history={history} classification={false} lossLabel="MAE loss (USD)" />);
  expect(screen.getByText('MAE loss (USD) · lower is better')).toBeTruthy();
  expect(screen.queryByText('Accuracy · higher is better')).toBeNull();
  expect(screen.getAllByRole('img')).toHaveLength(1);
  expect(screen.getByText('Training heatmap')).toBeTruthy();
});
