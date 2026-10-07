import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useAmbience } from './useAmbience';

function Harness({ model = 'digits' }: { model?: string }) {
  const audio = useAmbience(model);
  return <button disabled={audio.starting} onClick={audio.toggle}>{audio.unavailable ? 'Unavailable' : audio.enabled ? 'On' : 'Off'}</button>;
}

it('stays silent until requested and closes the engine on unmount', async () => {
  const node = () => ({ connect: jest.fn(), disconnect: jest.fn(), start: jest.fn(), stop: jest.fn(),
    frequency: { value: 0, setValueAtTime: jest.fn(), linearRampToValueAtTime: jest.fn() }, gain: { value: 0, setValueAtTime: jest.fn(), linearRampToValueAtTime: jest.fn(), exponentialRampToValueAtTime: jest.fn() } });
  jest.useFakeTimers();
  const close = jest.fn().mockResolvedValue(undefined);
  const oscillator = jest.fn(node);
  const constructor = jest.fn(() => ({ sampleRate: 100, currentTime: 0, state: 'running', destination: {},
    resume: jest.fn().mockResolvedValue(undefined), suspend: jest.fn().mockResolvedValue(undefined), close,
    createGain: node, createOscillator: oscillator, createBiquadFilter: node, createBufferSource: node,
    createBuffer: () => ({ getChannelData: () => new Float32Array(300) }),
  }));
  const descriptor = Object.getOwnPropertyDescriptor(window, 'AudioContext');
  Object.defineProperty(window, 'AudioContext', { configurable: true, value: constructor });
  try {
    const { rerender, unmount } = render(<Harness />);
    expect(constructor).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText('Off'));
    await screen.findByText('On');
    expect(constructor).toHaveBeenCalledTimes(1);
    const count = oscillator.mock.calls.length;
    rerender(<Harness model="car" />);
    await waitFor(() => expect(oscillator.mock.calls.length).toBe(count + 4));
    const beforeWildlife = oscillator.mock.calls.length;
    act(() => { jest.advanceTimersByTime(6000); });
    expect(oscillator.mock.calls.length).toBeGreaterThan(beforeWildlife);
    unmount();
    expect(jest.getTimerCount()).toBe(0);
    expect(close).toHaveBeenCalledTimes(1);
  } finally {
    jest.useRealTimers();
    if (descriptor) Object.defineProperty(window, 'AudioContext', descriptor);
    else delete (window as any).AudioContext;
  }
});
