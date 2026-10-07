import React, { useEffect, useRef, useState } from 'react';
import { useInteractiveTraining } from '../InteractiveLearning/useInteractiveTraining';
import { TrainingCharts } from '../Training/TrainingCharts';

function coordinates(x: number, y: number, fourier: boolean): number[] {
  const values = [x * 2 - 1, y * 2 - 1];
  if (fourier) for (const frequency of [1, 2, 4, 8]) values.push(Math.sin(x * Math.PI * frequency), Math.cos(x * Math.PI * frequency), Math.sin(y * Math.PI * frequency), Math.cos(y * Math.PI * frequency));
  return values;
}
function preset(kind: string): number[] {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 32;
  const context = canvas.getContext('2d')!;
  if (kind === 'sunset') {
    const gradient = context.createLinearGradient(0, 0, 0, 32); gradient.addColorStop(0, '#312e81'); gradient.addColorStop(0.6, '#fb7185'); gradient.addColorStop(1, '#fbbf24');
    context.fillStyle = gradient; context.fillRect(0, 0, 32, 32);
    context.fillStyle = '#fde68a'; context.beginPath(); context.arc(22, 15, 5, 0, Math.PI * 2); context.fill();
    context.fillStyle = '#0f172a'; context.beginPath(); context.moveTo(0, 32); context.lineTo(0, 26); context.lineTo(10, 18); context.lineTo(22, 27); context.lineTo(32, 21); context.lineTo(32, 32); context.fill();
  } else {
    for (let y = 0; y < 8; y += 1) for (let x = 0; x < 8; x += 1) {
      context.fillStyle = (x + y) % 2 ? '#38bdf8' : '#fbbf24'; context.fillRect(x * 4, y * 4, 4, 4);
    }
  }
  return Array.from(context.getImageData(0, 0, 32, 32).data);
}
function paint(canvas: HTMLCanvasElement | null, pixels: number[], size: number, predicted = false) {
  const context = canvas?.getContext('2d'); if (!context || !pixels.length) return;
  const image = context.createImageData(size, size);
  for (let i = 0; i < size * size; i += 1) {
    for (let channel = 0; channel < 3; channel += 1) image.data[i * 4 + channel] = predicted ? (pixels[i * 3 + channel] + 1) * 127.5 : pixels[i * 4 + channel];
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

export function NeuralImage() {
  const [pixels, setPixels] = useState<number[]>([]);
  const [fourier, setFourier] = useState(true);
  const [neurons, setNeurons] = useState(32);
  const [error, setError] = useState('');
  const original = useRef<HTMLCanvasElement | null>(null), reconstruction = useRef<HTMLCanvasElement | null>(null);
  const uploadVersion = useRef(0);
  const { history, prediction, training, status, train, stop, reset } = useInteractiveTraining();
  useEffect(() => { setPixels(preset('sunset')); return () => { uploadVersion.current += 1; }; }, []);
  useEffect(() => { paint(original.current, pixels, 32); }, [pixels]);
  useEffect(() => {
    reconstruction.current?.getContext('2d')?.clearRect(0, 0, 64, 64);
    paint(reconstruction.current, prediction, 64, true);
  }, [prediction]);
  function change(next: number[]) { setPixels(next); reset(); setError(''); }
  async function upload(file?: File) {
    if (!file) return;
    const version = ++uploadVersion.current;
    try {
      const bitmap = await createImageBitmap(file);
      if (version !== uploadVersion.current) { bitmap.close(); return; }
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 32;
      const context = canvas.getContext('2d')!; context.fillStyle = '#ffffff'; context.fillRect(0, 0, 32, 32);
      const side = Math.min(bitmap.width, bitmap.height);
      context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, 32, 32); bitmap.close();
      change(Array.from(context.getImageData(0, 0, 32, 32).data));
    } catch { if (version === uploadVersion.current) setError('Could not read this image. Try PNG or JPEG.'); }
  }
  function start() {
    const inputs = Array.from({ length: 1024 }, (_, i) => coordinates(((i % 32) + 0.5) / 32, (Math.floor(i / 32) + 0.5) / 32, fourier));
    const targets = inputs.map((_, i) => [0, 1, 2].map((channel) => pixels[i * 4 + channel] / 127.5 - 1));
    const preview = Array.from({ length: 4096 }, (_, i) => coordinates(((i % 64) + 0.5) / 64, (Math.floor(i / 64) + 0.5) / 64, fourier));
    train({ inputs, targets, preview, neurons, classification: false, epochs: 300, batchSize: 128, learningRate: 0.005 });
  }
  return <div style={{ display: 'grid', gap: 16 }}>
    <h3 style={{ margin: 0 }}>Neural image reconstruction</h3>
    <div style={{ color: '#a7b2c7', fontSize: 13 }}>A tiny network learns coordinates → RGB from 1,024 pixels. Watch the image emerge during training.</div>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <button onClick={training ? stop : start} disabled={!pixels.length}>{training ? 'stop training' : 'train image'}</button>
      <button disabled={training} onClick={() => change(preset('sunset'))}>sunset</button><button disabled={training} onClick={() => change(preset('checker'))}>checkerboard</button>
      <label>Upload image <input type="file" accept="image/*" disabled={training} onChange={(event) => { upload(event.target.files?.[0]); event.target.value = ''; }} /></label>
      <label>Neurons <select disabled={training} value={neurons} onChange={(event) => { setNeurons(Number(event.target.value)); reset(); }}>{[16, 32, 64].map((value) => <option key={value}>{value}</option>)}</select></label>
      <label><input type="checkbox" checked={fourier} disabled={training} onChange={(event) => { setFourier(event.target.checked); reset(); }} /> Fourier coordinates</label>
    </div>
    <div style={{ fontSize: 13, color: '#a7b2c7' }} aria-live="polite">{error || status}</div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div><div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div>Original · 32×32<canvas ref={original} width={32} height={32} aria-label="Original image" style={{ display: 'block', width: 160, height: 160, imageRendering: 'pixelated', marginTop: 8 }} /></div>
        <div>Network · 64×64<canvas ref={reconstruction} width={64} height={64} aria-label="Reconstructed image" style={{ display: 'block', width: 256, height: 256, imageRendering: 'pixelated', background: '#1e293b', marginTop: 8 }} /></div>
      </div><p style={{ color: '#a7b2c7', fontSize: 12 }}>Uploads are center-cropped to a square. Fourier coordinates help learn fine detail. Larger output evaluates intermediate coordinates; it does not recover missing original detail. Updates every 5 epochs.</p></div>
      <TrainingCharts history={history} classification={false} lossLabel="RGB MSE" validationNote="Loss uses the 32×32 training image; no separate validation set." />
    </div>
  </div>;
}
