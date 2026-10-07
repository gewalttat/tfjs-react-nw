import React from 'react';
import './App.css';
import { LoadPrediction } from './LoadPrediction/LoadPrediction';
import { PricesPrediction } from './PricesPrediction/PricesPrediction';
import { NeuralCar } from './NeuralCar/NeuralCar';
import { TicTacToe } from './TicTacToe/TicTacToe';
import { DigitRecognition } from './DigitRecognition/DigitRecognition';

function App() {
  return (
    <div className="App">
      <div className="page-shell">
        <header className="hero-card">
          <span className="eyebrow">TensorFlow.js experiments</span>
          <h1>ML playground</h1>
          <p>
            A tiny browser-based lab for regression, handwritten digits, neural tic-tac-toe, and evolving race cars.
          </p>
        </header>

        <div className="demo-grid">
          <section className="section-card">
            <LoadPrediction />
          </section>

          <section className="section-card">
            <PricesPrediction />
          </section>

          <section className="section-card">
            <DigitRecognition />
          </section>
        </div>

        <section className="section-card">
          <TicTacToe />
        </section>

        <section className="section-card">
          <NeuralCar />
        </section>
      </div>
    </div>
  );
}

export default App;
