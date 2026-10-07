import { Mentor } from './Adventure/Mentor';
import { LocaleContext } from './i18n/Locale';
import React, { useState } from 'react';
import './App.css';
import { LoadPrediction } from './LoadPrediction/LoadPrediction';
import { PricesPrediction } from './PricesPrediction/PricesPrediction';
import { TicketForecast } from './TeamForecast/TicketForecast';
import { NeuralImage } from './NeuralImage/NeuralImage';
import { Ballistics } from './Ballistics/Ballistics';
import { InteractiveLearning } from './InteractiveLearning/InteractiveLearning';
import { NeuralCar } from './NeuralCar/NeuralCar';
import { TicTacToe } from './TicTacToe/TicTacToe';
import { DigitRecognition } from './DigitRecognition/DigitRecognition';

const experiments = [
  { id: 'digits', title: 'Рукописные цифры', english: 'Handwritten digits', category: 'Распознавание', kind: 'Classification', architecture: '28 × 28 → CNN → 10', description: 'Нарисуйте цифру и наблюдайте, как свёрточная сеть учится её распознавать.', component: <DigitRecognition /> },
  { id: 'map', title: 'Карта классов', english: 'Classification map', category: 'Распознавание', kind: 'Classification', architecture: '2 → N → N → 3', description: 'Разметьте точки и исследуйте, как размер сети меняет границы классов.', component: <InteractiveLearning classification /> },
  { id: 'curve', title: 'Аппроксимация', english: 'Function fitting', category: 'Регрессия', kind: 'Regression', architecture: '1 → N → N → 1', description: 'Нарисуйте функцию. Сравните исходную кривую и её нейросетевое приближение.', component: <InteractiveLearning classification={false} /> },
  { id: 'load', title: 'Время загрузки', english: 'Load time', category: 'Регрессия', kind: 'Regression', architecture: '1 → 1', description: 'Прогноз времени загрузки по размеру файла с проверкой на отдельных примерах.', component: <LoadPrediction /> },
  { id: 'prices', title: 'Недвижимость', english: 'House prices', category: 'Регрессия', kind: 'Regression', architecture: '1 → 1', description: 'Линейная модель стоимости дома: площадь, прогноз и ошибка на проверочной выборке.', component: <PricesPrediction /> },
  { id: 'image', title: 'Нейросетевая картинка', english: 'Neural image', category: 'Реконструкция', kind: 'Reconstruction', architecture: 'XY + Fourier → N → N → RGB', description: 'Посмотрите, как изображение проявляется из координат и обученных весов.', component: <NeuralImage /> },
  { id: 'tictactoe', title: 'Крестики-нолики', english: 'Tic-tac-toe', category: 'Симуляции', kind: 'Game', architecture: '27 → 64 → 32 → 9', description: 'Обучите сеть на позициях minimax и сыграйте против её собственных предсказаний.', component: <TicTacToe /> },
  { id: 'car', title: 'Нейрогонки', english: 'Neural racing', category: 'Симуляции', kind: 'Evolution', architecture: '12 → 12 → 2', description: 'Три водителя на трассе. Отбор и мутации учат их ехать быстрее и избегать столкновений.', component: <NeuralCar /> },
  { id: 'ballistics', title: 'Баллистика', english: 'Ballistics', category: 'Симуляции', kind: 'Physics', architecture: '3 → 32 → 32 → 2', description: 'Задайте цель и гравитацию: сеть подберёт угол и скорость запуска.', component: <Ballistics /> },
  { id: 'team', title: 'T2M команды', english: 'Team T2M', category: 'Прогнозирование', kind: 'Forecast', architecture: 'History / ticket → MLP → days', description: 'Недельные квантили и оценки тикетов. Проверяйте прогнозы на скрытой истории.', component: <TicketForecast /> },
];

function App() {
  const [selected, setSelected] = useState('digits');
  const [visited, setVisited] = useState(['digits']);
  const [english, setEnglish] = useState(false);
  const experiment = experiments.find((item) => item.id === selected)!;
  function select(id: string) { setSelected(id); setVisited((current) => current.includes(id) ? current : [...current, id]); }
  return <LocaleContext.Provider value={english ? 'en' : 'ru'}><div className="App">
    <aside className="lab-sidebar">
      <button className="lab-brand" onClick={() => select('digits')}><span className="brand-mark">✦</span><span>ML <b>PLAYGROUND</b><small>TENSORFLOW.JS · EXPERIMENTS</small></span></button>
      <div className="sidebar-caption">{english ? 'EXPERIMENTS' : 'ЭКСПЕРИМЕНТЫ'} <span>{experiments.length.toString().padStart(2, '0')}</span></div>
      <nav aria-label={english ? 'Experiments' : 'Эксперименты'}>
        {experiments.map((item, index) => <button key={item.id} className={`nav-experiment ${selected === item.id ? 'is-active' : ''}`} aria-current={selected === item.id ? 'page' : undefined} onClick={() => select(item.id)}>
          <span className="nav-index">{(index + 1).toString().padStart(2, '0')}</span><span>{english ? item.english : item.title}</span><span className="nav-dot" />
        </button>)}
      </nav>
      <div className="sidebar-bottom"><span className="online-dot" /> {english ? 'Computes in your browser' : 'Вычисления в браузере'}<small>TensorFlow.js · local models</small></div>
    </aside>
    <main className="lab-main">
      <Mentor key={`${selected}-${english}`} id={selected} english={english} />
      <div className="lab-topbar"><span>ML PLAYGROUND <span className="breadcrumb-separator">/</span> {english ? experiment.english : experiment.title}</span><span className="local-badge">LOCAL RUNTIME</span></div>
      <div className="workspace">
        <header className="experiment-header">
          <div className="eyebrow">EXPERIMENT {(experiments.indexOf(experiment) + 1).toString().padStart(2, '0')} <span>/ {english ? experiment.kind : experiment.category}</span></div>
          <h1>{english ? experiment.english : experiment.title}</h1>
          <p>{english ? 'Explore the model, train it locally, and inspect its predictions and learning metrics.' : experiment.description}</p>
          <div className="experiment-facts"><span><small>{english ? 'MODEL' : 'МОДЕЛЬ'}</small>{experiment.architecture}</span><span><small>{english ? 'EXECUTION' : 'ВЫЧИСЛЕНИЯ'}</small>{english ? 'On device' : 'На устройстве'}</span><span><small>{english ? 'OBSERVABILITY' : 'НАБЛЮДЕНИЕ'}</small>{english ? 'Live charts + heatmap' : 'Графики + хитмап'}</span></div>
        </header>
        {experiments.filter((item) => visited.includes(item.id)).map((item) => <section key={item.id} className="section-card experiment-panel" hidden={item.id !== selected} aria-label={english ? item.english : item.title}>{item.component}</section>)}
        <footer className="lab-footer"><span>ML PLAYGROUND <span className="footer-divider">·</span> {english ? 'Small models. Visible learning.' : 'Маленькие модели. Наглядное обучение.'}</span><button className="language-toggle" onClick={() => setEnglish((value) => !value)} aria-label="Switch interface language">◎ {english ? 'RU · Русский' : 'EN · English'}</button></footer>
      </div>
    </main>
  </div></LocaleContext.Provider>;
}
export default App;
