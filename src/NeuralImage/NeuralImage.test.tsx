import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { NeuralImage } from './NeuralImage';

const mockTrain = jest.fn();
const mockReset = jest.fn();
jest.mock('../InteractiveLearning/useInteractiveTraining', () => ({
  useInteractiveTraining: () => ({ history: [], prediction: [], training: false, status: 'Ready', train: mockTrain, reset: mockReset, stop: jest.fn() }),
}));
jest.mock('../Training/TrainingCharts', () => ({ TrainingCharts: () => null }));
const drawImage = jest.fn();
const close = jest.fn();
const decode = jest.fn();
let imageData: Uint8ClampedArray;

beforeEach(() => {
  jest.clearAllMocks();
  imageData = new Uint8ClampedArray(32 * 32 * 4);
  for (let i = 0; i < imageData.length; i += 4) imageData.set([255, 128, 0, 255], i);
  Object.defineProperty(window, 'createImageBitmap', { configurable: true, value: decode });
  jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ({
    createLinearGradient: () => ({ addColorStop: jest.fn() }), fillRect: jest.fn(),
    beginPath: jest.fn(), arc: jest.fn(), fill: jest.fn(), moveTo: jest.fn(), lineTo: jest.fn(),
    getImageData: () => ({ data: imageData }),
    createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    putImageData: jest.fn(), clearRect: jest.fn(), drawImage,
  } as any));
});
afterEach(() => jest.restoreAllMocks());

it('crops a selected image and trains on its RGB pixels', async () => {
  decode.mockResolvedValue({ width: 80, height: 40, close });
  const { container } = render(<NeuralImage />);
  const file = new File(['image'], 'photo.png', { type: 'image/png' });
  fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } });
  await waitFor(() => expect(close).toHaveBeenCalledTimes(1));
  expect(decode).toHaveBeenCalledWith(file);
  expect(drawImage).toHaveBeenCalledWith(expect.anything(), 20, 0, 40, 40, 0, 0, 32, 32);
  fireEvent.click(screen.getByText('train image'));
  const options = mockTrain.mock.calls[0][0];
  expect(options.inputs).toHaveLength(1024);
  expect(options.targets).toHaveLength(1024);
  expect(options.targets[0]).toEqual([1, 128 / 127.5 - 1, -1]);
  expect(options.preview).toHaveLength(4096);
});

it('blocks training until decoding finishes and reports unreadable files', async () => {
  let reject!: (error: Error) => void;
  decode.mockReturnValue(new Promise((_, failure) => { reject = failure; }));
  const { container } = render(<NeuralImage />);
  fireEvent.change(container.querySelector('input[type="file"]')!, {
    target: { files: [new File(['broken'], 'broken.png', { type: 'image/png' })] },
  });
  expect((screen.getByText('train image') as HTMLButtonElement).disabled).toBe(true);
  reject(new Error('Decode failed'));
  await screen.findByText('Could not read this image. Try PNG or JPEG.');
  expect((screen.getByText('train image') as HTMLButtonElement).disabled).toBe(false);
  expect(mockTrain).not.toHaveBeenCalled();
});
