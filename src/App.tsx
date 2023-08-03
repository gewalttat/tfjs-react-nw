import React from 'react';
import './App.css';
import { LoadPrediction } from './LoadPrediction/LoadPrediction';
// import { PricesPrediction } from './PricesPrediction/PricesPrediction';

function App() {
  return (
    <div className="App">
      <LoadPrediction />
      {/* <PricesPrediction /> */}
    </div>
  );
}

export default App;
