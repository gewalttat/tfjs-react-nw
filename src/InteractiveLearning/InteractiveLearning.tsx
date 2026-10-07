import { Localized } from '../i18n/Locale';
import React, { useEffect, useRef, useState } from 'react';
import { TrainingCharts } from '../Training/TrainingCharts';
import { useInteractiveTraining } from './useInteractiveTraining';

interface Point { x: number; y: number; label: number }
const COLORS = ['#38bdf8', '#fb7185', '#fbbf24'];
const RGB = [[56, 189, 248], [251, 113, 133], [251, 191, 36]];
const WIDTH = 480, HEIGHT = 320, MAP_WIDTH = 60, MAP_HEIGHT = 40;

function mapExample(spiral = false): Point[] {
  if (spiral) return Array.from({ length: 120 }, (_, i) => {
    const label = i % 3, t = Math.floor(i / 3) / 40;
    const angle = t * 4.5 + label * Math.PI * 2 / 3;
    return { x: 0.5 + Math.cos(angle) * (0.07 + t * 0.4), y: 0.5 + Math.sin(angle) * (0.07 + t * 0.4), label };
  });
  return Array.from({ length: 36 }, (_, i) => {
    const label = i % 3, angle = Math.floor(i / 3) * 2.4;
    const centers = [[0.25, 0.3], [0.75, 0.3], [0.5, 0.75]];
    return { x: centers[label][0] + Math.cos(angle) * 0.12, y: centers[label][1] + Math.sin(angle) * 0.12, label };
  });
}
function curveExample(kind: string): Point[] {
  return Array.from({ length: 81 }, (_, i) => {
    const x = i / 80;
    const y = kind === 'step' ? x < 0.5 ? 0.25 : 0.75 : kind === 'abs' ? 0.8 - Math.abs(x - 0.5) * 1.2 : 0.5 + Math.sin(x * Math.PI * 4) * 0.3;
    return { x, y, label: 0 };
  });
}

export function InteractiveLearning({ classification }: { classification: boolean }) {

  const [points, setPoints] = useState<Point[]>(() => classification ? mapExample() : curveExample('sine'));
  const [label, setLabel] = useState(0);
  const [neurons, setNeurons] = useState(16);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef(false);
  const previousPoint = useRef<Point | null>(null);
  const { history, prediction, training, status, train, stop, reset } = useInteractiveTraining();

  useEffect(() => {
    const context = canvasRef.current?.getContext('2d');
    if (!context) return;
    context.fillStyle = '#242424'; context.fillRect(0, 0, WIDTH, HEIGHT);
    if (classification && prediction.length) {
      const field = document.createElement('canvas'); field.width = MAP_WIDTH; field.height = MAP_HEIGHT;
      const fieldContext = field.getContext('2d');
      if (fieldContext) {
        const pixels = fieldContext.createImageData(MAP_WIDTH, MAP_HEIGHT);
        for (let i = 0; i < MAP_WIDTH * MAP_HEIGHT; i += 1) {
          for (let channel = 0; channel < 3; channel += 1) {
            pixels.data[i * 4 + channel] = RGB.reduce((sum, rgb, index) => sum + rgb[channel] * prediction[i * 3 + index], 0) * 0.55;
          }
          pixels.data[i * 4 + 3] = 255;
        }
        fieldContext.putImageData(pixels, 0, 0); context.drawImage(field, 0, 0, WIDTH, HEIGHT);
      }
    }
    context.strokeStyle = 'rgba(148,163,184,0.18)'; context.lineWidth = 1;
    for (let i = 1; i < 10; i += 1) {
      context.beginPath(); context.moveTo(i * WIDTH / 10, 0); context.lineTo(i * WIDTH / 10, HEIGHT); context.stroke();
      context.beginPath(); context.moveTo(0, i * HEIGHT / 10); context.lineTo(WIDTH, i * HEIGHT / 10); context.stroke();
    }
    if (!classification) {
      const ordered = [...points].sort((a, b) => a.x - b.x);
      context.strokeStyle = '#dedbd2'; context.lineWidth = 2; context.beginPath();
      ordered.forEach((point, i) => i ? context.lineTo(point.x * WIDTH, point.y * HEIGHT) : context.moveTo(point.x * WIDTH, point.y * HEIGHT)); context.stroke();
      if (prediction.length) {
        context.strokeStyle = '#c4ab72'; context.lineWidth = 1.2; context.beginPath();
        prediction.forEach((value, i) => {
          const x = i / (prediction.length - 1) * WIDTH, y = (value + 1) / 2 * HEIGHT;
          if (i) context.lineTo(x, y); else context.moveTo(x, y);
        }); context.stroke();
      }
    }
    points.forEach((point) => {
      context.beginPath(); context.arc(point.x * WIDTH, point.y * HEIGHT, classification ? 5 : 2.5, 0, Math.PI * 2);
      context.fillStyle = classification ? COLORS[point.label] : '#dedbd2'; context.fill();
      if (classification) { context.strokeStyle = '#eeece6'; context.lineWidth = 1.5; context.stroke(); }
    });
  }, [points, prediction, classification]);

  function getPoint(event: React.PointerEvent<HTMLCanvasElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)), y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)), label };
  }
  function addCurvePoint(point: Point) {
    const previous = previousPoint.current;
    previousPoint.current = point;
    setPoints((current) => {
      const byX = new Map(current.map((entry) => [Math.round(entry.x * 100), entry]));
      const first = Math.round((previous?.x ?? point.x) * 100), last = Math.round(point.x * 100);
      for (let bin = Math.min(first, last); bin <= Math.max(first, last); bin += 1) {
        const t = first === last ? 1 : (bin - first) / (last - first);
        byX.set(bin, { x: bin / 100, y: previous ? previous.y + (point.y - previous.y) * t : point.y, label: 0 });
      }
      return Array.from(byX.values());
    });
    reset();
  }
  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (training || event.button !== 0) return;
    event.preventDefault();
    const point = getPoint(event);
    if (classification) {
      setPoints((current) => event.shiftKey ? current.filter((entry) => Math.hypot((entry.x - point.x) * WIDTH, (entry.y - point.y) * HEIGHT) > 12) : [...current, point]);
      reset();
    } else {
      drawing.current = true; previousPoint.current = null;
      event.currentTarget.setPointerCapture(event.pointerId); addCurvePoint(point);
    }
  }
  function stopDrawing(event: React.PointerEvent<HTMLCanvasElement>) {
    drawing.current = false; previousPoint.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  function watchTraining() {
    const preview = classification ? Array.from({ length: MAP_WIDTH * MAP_HEIGHT }, (_, i) => [((i % MAP_WIDTH) + 0.5) / MAP_WIDTH * 2 - 1, (Math.floor(i / MAP_WIDTH) + 0.5) / MAP_HEIGHT * 2 - 1])
      : Array.from({ length: 161 }, (_, i) => [i / 160 * 2 - 1]);
    train({ inputs: points.map((point) => classification ? [point.x * 2 - 1, point.y * 2 - 1] : [point.x * 2 - 1]),
      targets: points.map((point) => classification ? [0, 1, 2].map((value) => value === point.label ? 1 : 0) : [point.y * 2 - 1]),
      preview, neurons, classification });
  }
  function changePoints(next: Point[]) { setPoints(next); reset(); }
  const canTrain = points.length >= 3 && (!classification || new Set(points.map((point) => point.label)).size >= 2);

  return <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
    <h3 style={{ margin: 0 }}><Localized>{classification ? 'Color classification map' : 'Draw a function'}</Localized></h3>
    <div style={{ color: '#a6a39c', fontSize: 13 }}><Localized>{classification ? 'Choose a class and click to add examples. Shift-click removes nearby points. The background shows learned class probabilities.' : 'Drag to draw y as a function of x. Each horizontal position has one target value. Light is your curve; gold is the network prediction.'}</Localized></div>
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
      <Localized>{classification && COLORS.map((color, index) => <button key={color} aria-pressed={label === index} onClick={() => setLabel(index)}
        style={{ background: color, outline: label === index ? '2px solid white' : 'none', outlineOffset: 2 }}><Localized>{"Class "}</Localized><Localized>{index + 1}</Localized></button>)}</Localized>
      <label><Localized>{"Neurons per layer"}</Localized><Localized>{' '}</Localized><select value={neurons} disabled={training} onChange={(event) => { setNeurons(Number(event.target.value)); reset(); }}>
        <Localized>{[4, 16, 64].map((value) => <option key={value} value={value}><Localized>{value}</Localized></option>)}</Localized>
      </select></label>
      <button onClick={training ? stop : watchTraining} disabled={!training && !canTrain}><Localized>{training ? 'stop training' : 'train network'}</Localized></button>
      <button disabled={training} onClick={() => changePoints([])}><Localized>{"clear"}</Localized></button>
      <Localized>{classification ? <><button disabled={training} onClick={() => changePoints(mapExample())}><Localized>{"clusters"}</Localized></button><button disabled={training} onClick={() => changePoints(mapExample(true))}><Localized>{"spiral"}</Localized></button></>
        : ['sine', 'step', 'abs'].map((kind) => <button key={kind} disabled={training} onClick={() => changePoints(curveExample(kind))}><Localized>{kind}</Localized></button>)}</Localized>
    </div>
    <div style={{ fontSize: 13, color: '#a6a39c' }} aria-live="polite"><Localized>{status}</Localized><Localized>{" · "}</Localized><Localized>{points.length}</Localized><Localized>{" examples"}</Localized><Localized>{!canTrain && ' · add at least 3 points'}</Localized></div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
      <div><canvas ref={canvasRef} width={WIDTH} height={HEIGHT} onPointerDown={pointerDown} onPointerMove={(event) => { if (drawing.current && !training) addCurvePoint(getPoint(event)); }}
        onPointerUp={stopDrawing} onPointerCancel={stopDrawing} aria-label={classification ? 'Interactive class points and probability map' : 'Draw a target curve and watch the fitted function'}
        style={{ width: '100%', border: '1px solid #55524c', borderRadius: 16, touchAction: 'none', cursor: training ? 'wait' : 'crosshair' }} />
        <p style={{ color: '#a6a39c', fontSize: 12 }}><Localized>{"Two hidden layers with "}</Localized><Localized>{neurons}</Localized><Localized>{" neurons each. Change the data or layer size and train again to compare. Predictions update every 5 epochs."}</Localized></p>
      </div>
      <TrainingCharts history={history} classification={classification} lossLabel={classification ? 'Cross-entropy' : 'Curve MSE'} validationNote="Metrics use your training examples. There is no separate validation set in this experiment." />
    </div>
  </div>;
}
