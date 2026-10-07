import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

jest.mock('./DigitRecognition/DigitRecognition', () => ({ DigitRecognition: () => <input aria-label="drawing state" defaultValue="" /> }));
jest.mock('./LoadPrediction/LoadPrediction', () => ({ LoadPrediction: () => <div>Load experiment</div> }));

it('navigates, preserves an opened experiment, and switches shell language', () => {
  render(<App />);
  fireEvent.change(screen.getByLabelText('drawing state'), { target: { value: 'kept' } });
  fireEvent.click(screen.getByRole('button', { name: /Время загрузки/ }));
  expect(screen.getByText('Load experiment')).toBeTruthy();
  expect(screen.getByLabelText('drawing state').closest('section')?.hidden).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: /Рукописные цифры/ }));
  expect((screen.getByLabelText('drawing state') as HTMLInputElement).value).toBe('kept');
  fireEvent.click(screen.getByRole('button', { name: 'Switch interface language' }));
  expect(screen.getByRole('heading', { name: 'Handwritten digits' })).toBeTruthy();
});
