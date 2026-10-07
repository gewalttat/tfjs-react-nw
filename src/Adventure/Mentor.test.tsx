import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Mentor } from './Mentor';

it('reveals the explanation, advances the lesson, and resets for another model', () => {
  jest.useFakeTimers();
  const { container, rerender, unmount } = render(<Mentor key="image" id="image" english={false} />);
  expect(container.querySelector('.wizard-talking')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Показать всё' }));
  expect(container.querySelector('.wizard-talking')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Расскажи ещё →' }));
  expect(screen.getByRole('status').textContent).toContain('Координаты Фурье');
  act(() => { jest.advanceTimersByTime(5000); });
  expect(screen.getByRole('button', { name: 'Ещё раз ↺' })).toBeTruthy();
  rerender(<Mentor key="car-en" id="car" english />);
  expect(screen.getByRole('status').textContent).toContain('Three cars');
  expect(screen.getByText('1 / 2')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Show all' })).toBeTruthy();
  unmount();
  expect(jest.getTimerCount()).toBe(0);
  jest.useRealTimers();
});
