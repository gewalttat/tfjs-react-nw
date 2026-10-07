import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { TrainingCharts, TrainingEpoch } from '../Training/TrainingCharts';

const GRID_SIZE = 28;
const CANVAS_SIZE = 280;
const CLASS_COUNT = 10;
const MNIST_IMAGES_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
const MNIST_SAMPLE_COUNT = 12000;

function emptyImage(): Float32Array {
  return new Float32Array(GRID_SIZE * GRID_SIZE).fill(0);
}

export function decodeMnistLabels(labels: Uint8Array, sampleCount: number): Float32Array {
  if (labels.length < sampleCount * CLASS_COUNT) {
    throw new Error('MNIST labels do not match the image count.');
  }
  return Float32Array.from(labels.subarray(0, sampleCount * CLASS_COUNT));
}

export function preprocessDigitCanvas(canvas: HTMLCanvasElement): Float32Array {
  const sourceContext = canvas.getContext('2d');
  if (!sourceContext) {
    return emptyImage();
  }

  const sourceData = sourceContext.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width;
  let maxX = -1;
  let minY = canvas.height;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const offset = (y * canvas.width + x) * 4;
      const red = sourceData[offset];
      const green = sourceData[offset + 1];
      const blue = sourceData[offset + 2];
      const brightness = (red + green + blue) / 3;

      if (brightness < 245) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (minX === canvas.width) {
    return emptyImage();
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = Math.min((GRID_SIZE - 8) / cropWidth, (GRID_SIZE - 8) / cropHeight);
  const boxWidth = Math.max(1, cropWidth * scale);
  const boxHeight = Math.max(1, cropHeight * scale);

  const targetCanvas = document.createElement('canvas');
  targetCanvas.width = GRID_SIZE;
  targetCanvas.height = GRID_SIZE;
  const targetContext = targetCanvas.getContext('2d');

  if (!targetContext) {
    return emptyImage();
  }

  targetContext.fillStyle = '#ffffff';
  targetContext.fillRect(0, 0, GRID_SIZE, GRID_SIZE);

  const offsetX = (GRID_SIZE - boxWidth) / 2;
  const offsetY = (GRID_SIZE - boxHeight) / 2;
  targetContext.drawImage(
    canvas,
    minX,
    minY,
    cropWidth,
    cropHeight,
    offsetX,
    offsetY,
    boxWidth,
    boxHeight
  );

  const imageData = targetContext.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
  const normalized = new Float32Array(GRID_SIZE * GRID_SIZE);

  for (let i = 0; i < imageData.data.length; i += 4) {
    const brightness = imageData.data[i];
    normalized[i / 4] = 1 - brightness / 255;
  }

  return normalized;
}

async function loadMnistDataSet(limit: number) {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const mnistImage = new Image();
    mnistImage.crossOrigin = 'anonymous';
    mnistImage.onload = () => resolve(mnistImage);
    mnistImage.onerror = () => reject(new Error('Unable to load MNIST sprite.'));
    mnistImage.src = MNIST_IMAGES_URL;
  });

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas 2D context is unavailable.');
  }

  if (image.width !== GRID_SIZE * GRID_SIZE) {
    throw new Error('Unexpected MNIST sprite dimensions.');
  }
  const totalImages = Math.min(limit, image.height);
  const chunkSize = 5000;
  const dataset = new Float32Array(totalImages * GRID_SIZE * GRID_SIZE);

  for (let chunkIndex = 0; chunkIndex < Math.ceil(totalImages / chunkSize); chunkIndex += 1) {
    const chunkStart = chunkIndex * chunkSize;
    canvas.width = image.width;
    canvas.height = Math.min(chunkSize, totalImages - chunkStart);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, chunkStart, image.width, canvas.height, 0, 0, image.width, canvas.height);

    const chunkImageData = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const offsetStart = chunkStart * GRID_SIZE * GRID_SIZE;

    for (let index = 0; index < chunkImageData.length; index += 4) {
      const pixelIndex = offsetStart + index / 4;
      dataset[pixelIndex] = chunkImageData[index] / 255;
    }
  }

  const labelsResponse = await fetch(MNIST_LABELS_URL);
  if (!labelsResponse.ok) {
    throw new Error('Unable to load MNIST labels.');
  }

  const labelsBuffer = new Uint8Array(await labelsResponse.arrayBuffer());
  const sampleCount = totalImages;
  const images = dataset.slice(0, sampleCount * GRID_SIZE * GRID_SIZE);
  const labels = decodeMnistLabels(labelsBuffer, sampleCount);

  return { images, labels, sampleCount };
}

async function buildOfficialStyleModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [GRID_SIZE, GRID_SIZE, 1],
    filters: 8,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: CLASS_COUNT, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

export function DigitRecognition() {
  const modelRef = useRef<tf.Sequential | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingEpoch[]>([]);
  const [status, setStatus] = useState('Loading MNIST model...');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  const isReady = useMemo(() => model !== null && status === 'ready', [model, status]);

  useEffect(() => {
    let active = true;
    let ownedModel: tf.Sequential | null = null;

    async function trainModel() {
      let xs: tf.Tensor4D | undefined;
      let ys: tf.Tensor2D | undefined;
      try {
        setTrainingHistory([]);
        setStatus('Loading MNIST data...');
        const { images, labels, sampleCount } = await loadMnistDataSet(MNIST_SAMPLE_COUNT);
        if (!active) return;

        xs = tf.tensor4d(images, [sampleCount, GRID_SIZE, GRID_SIZE, 1], 'float32');
        ys = tf.tensor2d(labels, [sampleCount, CLASS_COUNT], 'float32');
        const newModel = await buildOfficialStyleModel();
        ownedModel = newModel;
        if (!active) return;

        setStatus('Training on real MNIST digits...');
        const trainingStart = performance.now();
        let currentEpoch = 0;
        let lastChartUpdate = 0;
        const batchesPerEpoch = Math.ceil(Math.floor(sampleCount * 0.9) / 64);
        await newModel.fit(xs, ys, {
          epochs: 6,
          batchSize: 64,
          validationSplit: 0.1,
          shuffle: true,
          callbacks: {
            onEpochBegin: async (epoch) => { currentEpoch = epoch; },
            onEpochEnd: async (epoch, logs) => {
              if (!active) {
                newModel.stopTraining = true;
                return;
              }
              setTrainingHistory((history) => [...history, {
                epoch: epoch + 1,
                elapsedMs: performance.now() - trainingStart,
                loss: logs?.loss,
                validationLoss: logs?.val_loss,
                accuracy: logs?.acc ?? logs?.accuracy,
                validationAccuracy: logs?.val_acc ?? logs?.val_accuracy,
              }]);
              const accuracy = logs?.val_acc ?? logs?.val_accuracy;
              setStatus(`Training: epoch ${epoch + 1}/6${accuracy === undefined ? '' : `, validation accuracy ${(accuracy * 100).toFixed(1)}%`}`);
            },
            onBatchEnd: async (batch, logs) => {
              if (!active) {
                newModel.stopTraining = true;
                return;
              }
              const elapsedMs = performance.now() - trainingStart;
              if (elapsedMs - lastChartUpdate >= 50) {
                lastChartUpdate = elapsedMs;
                setTrainingHistory((history) => [...history, {
                  epoch: currentEpoch + (batch + 1) / batchesPerEpoch,
                  elapsedMs,
                  loss: logs?.loss,
                  accuracy: logs?.acc ?? logs?.accuracy,
                }]);
                await tf.nextFrame();
              }
            },
          },
        });

        if (active) {
          modelRef.current = newModel;
          setModel(newModel);
          setStatus('ready');
        }
      } catch (error) {
        console.error('MNIST model training failed', error);
        ownedModel?.dispose();
        ownedModel = null;
        if (active) {
          setModel(null);
          setStatus('Training failed. Check the console and reload to retry.');
        }
      } finally {
        xs?.dispose();
        ys?.dispose();
        if (!active) {
          ownedModel?.dispose();
          ownedModel = null;
        }
      }
    }

    void trainModel();

    return () => {
      active = false;
      if (ownedModel) {
        ownedModel.stopTraining = true;
        // Training owns the tensors until fit completes.
        if (ownedModel === modelRef.current) {
          ownedModel.dispose();
          ownedModel = null;
          modelRef.current = null;
        }
      }
    };
  }, []);

  const getCanvasPoint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }

    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE;
    const y = ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE;

    return { x, y };
  };

  const isActivePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.pointerType === 'mouse') {
      return event.buttons === 1;
    }

    return event.pressure > 0 || event.pointerType === 'touch';
  };

  const startDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isActivePointer(event)) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    event.preventDefault();
    setPrediction(null);
    setConfidence(0);
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = 12;
    context.strokeStyle = '#111111';

    const { x, y } = getCanvasPoint(event);
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + 0.01, y + 0.01);
    context.stroke();
    canvas.setPointerCapture(event.pointerId);
  };

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isActivePointer(event) || !canvasRef.current?.hasPointerCapture(event.pointerId)) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    event.preventDefault();
    const { x, y } = getCanvasPoint(event);
    context.lineTo(x, y);
    context.stroke();
  };

  const stopDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(0);
  };

  const predictDigit = async () => {
    if (!model || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const normalizedPixels = preprocessDigitCanvas(canvas);
    if (!normalizedPixels.some((pixel) => pixel > 0)) {
      setPrediction(null);
      setConfidence(0);
      return;
    }
    const tensorInput = tf.tensor4d(normalizedPixels, [1, GRID_SIZE, GRID_SIZE, 1], 'float32');
    const predictionTensor = model.predict(tensorInput) as tf.Tensor;
    const probabilities = await predictionTensor.data();
    const bestIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
    const bestScore = probabilities[bestIndex];

    setPrediction(bestIndex);
    setConfidence(bestScore);

    tensorInput.dispose();
    predictionTensor.dispose();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }

    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = 12;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = '#111111';
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ margin: 0 }}>Digit recognition</h3>
      <div style={{ fontSize: 13, color: '#a7b2c7' }}>
        {status === 'ready' ? 'Real MNIST model ready' : status}
      </div>

      <TrainingCharts history={trainingHistory} />

      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onPointerDown={startDrawing}
        onPointerMove={draw}
        onPointerUp={stopDrawing}
        onPointerCancel={stopDrawing}
        onPointerLeave={stopDrawing}
        style={{
          width: 220,
          height: 220,
          borderRadius: 18,
          border: '1px solid rgba(148, 163, 184, 0.4)',
          background: '#ffffff',
          boxShadow: '0 16px 28px rgba(15, 23, 42, 0.2)',
          cursor: 'crosshair',
          touchAction: 'none',
          imageRendering: 'pixelated',
        }}
      />

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button onClick={predictDigit} disabled={!isReady} style={{ width: 120 }}>
          predict
        </button>
        <button onClick={clearCanvas} style={{ width: 100 }}>clear</button>
      </div>

      <div style={{ minHeight: 26, fontSize: 18, fontWeight: 700 }}>
        {prediction !== null ? `Predicted: ${prediction}` : 'Draw a digit'}
      </div>

      <div style={{ fontSize: 13, color: '#b6c3d8' }}>
        {confidence > 0 ? `Confidence: ${(confidence * 100).toFixed(1)}%` : 'Confidence: —'}
      </div>
    </div>
  );
}
