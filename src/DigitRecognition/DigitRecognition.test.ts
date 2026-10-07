import { decodeMnistLabels, preprocessDigitCanvas } from './DigitRecognition';

afterEach(() => jest.restoreAllMocks());

describe('decodeMnistLabels', () => {
  it('keeps ten label bytes per image instead of interpreting bytes as digit indices', () => {
    const labels = new Uint8Array([
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(Array.from(decodeMnistLabels(labels, 2))).toEqual(Array.from(labels.slice(0, 20)));
  });

  it('rejects a label file with fewer labels than images', () => {
    expect(() => decodeMnistLabels(new Uint8Array(19), 2)).toThrow();
  });
});

describe('preprocessDigitCanvas', () => {
  function sourceCanvas(drawn: boolean) {
    const data = new Uint8ClampedArray(280 * 280 * 4).fill(255);
    if (drawn) {
      const offset = (100 * 280 + 100) * 4;
      data[offset] = data[offset + 1] = data[offset + 2] = 0;
    }
    return {
      width: 280,
      height: 280,
      getContext: () => ({ getImageData: () => ({ data }) }),
    } as unknown as HTMLCanvasElement;
  }

  it('returns zero input for an empty white canvas', () => {
    expect(Array.from(preprocessDigitCanvas(sourceCanvas(false)))).toEqual(new Array(784).fill(0));
  });

  it('maps white background to zero and black strokes to one, with gray antialiasing', () => {
    const data = new Uint8ClampedArray(28 * 28 * 4).fill(255);
    data[0] = data[1] = data[2] = 0;
    data[4] = data[5] = data[6] = 128;
    const context = {
      fillStyle: '',
      fillRect: jest.fn(),
      drawImage: jest.fn(),
      getImageData: () => ({ data }),
    };
    jest.spyOn(document, 'createElement').mockReturnValue({
      width: 28,
      height: 28,
      getContext: () => context,
    } as unknown as HTMLCanvasElement);

    const pixels = preprocessDigitCanvas(sourceCanvas(true));
    expect(pixels).toHaveLength(784);
    expect(pixels[0]).toBe(1);
    expect(pixels[1]).toBeCloseTo(1 - 128 / 255);
    expect(pixels[2]).toBe(0);
    expect(context.drawImage).toHaveBeenCalled();
  });
});
