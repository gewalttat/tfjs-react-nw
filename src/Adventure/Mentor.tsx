import React, { useEffect, useState } from 'react';
import { AdventureScene, Wizard } from './AdventureScene';

const lessons: Record<string, { ru: string[]; en: string[] }> = {
  digits: { ru: ['Добро пожаловать, ученик! Здесь сеть узнаёт рукописные цифры. Обучи её на примерах MNIST, затем нарисуй свою цифру: ответ и вероятности покажут, чему она научилась.', 'Свёртки ищут линии и изгибы. Загляни в хитмап: там видно, какие цифры сеть путает. Высокая уверенность ещё не гарантирует верный ответ — проверь разные почерки.'], en: ['Welcome, apprentice! This network recognizes handwritten digits. Train it on MNIST examples, then draw a digit: its answer and probabilities reveal what it learned.', 'Convolutions look for strokes and curves. Check the heatmap to see which digits get confused. Confidence is no guarantee: try different handwriting styles.'] },
  map: { ru: ['Расставь точки трёх цветов. Сеть получает только две координаты и учится определять класс. Цветной фон покажет её решение в каждом уголке карты.', 'Добавь точку на чужую территорию и обучи снова. Граница изменится! Сравни маленькую и большую сеть: сложность модели влияет на то, какие формы она может выучить.'], en: ['Place points of three colors. Given two coordinates, the network learns a class. The colored background shows its decision across the whole map.', 'Place a point in another class’s territory and retrain. Watch the boundary change! Compare small and large networks to see which shapes they can learn.'] },
  curve: { ru: ['Нарисуй кривую из точек. Сеть учится превращать координату X в значение Y — так работает аппроксимация. Сравни её линию с исходной.', 'Поменяй число нейронов и обучи заново. Мало нейронов — сложные изгибы могут потеряться. Смотри и на ошибку, и на форму между точками: одной цифры недостаточно.'], en: ['Draw a curve with points. The network learns to turn X into Y: this is function approximation. Compare its line with your original.', 'Change the neuron count and retrain. A small network may miss complex bends. Inspect both the error and the shape between points.'] },
  load: { ru: ['Сколько ждать загрузки? Эта маленькая модель связывает размер файла со временем. Обучи её и введи новый размер, чтобы получить оценку.', 'Это учебная зависимость: в реальности скорость сети и сервер тоже влияют на загрузку. Проверочная ошибка показывает качество на примерах, которых модель не видела.'], en: ['How long will a download take? This tiny model links file size to load time. Train it, then enter a new size for an estimate.', 'This is a simplified relationship: real downloads also depend on the network and server. Validation error measures performance on unseen examples.'] },
  prices: { ru: ['Здесь сеть оценивает стоимость дома по площади. Обучи линейную модель, затем поменяй площадь и посмотри на прогноз.', 'Линейная модель ищет наклон и смещение. Она полезна как простая отправная точка, но район и состояние дома ей неизвестны. Проверяй ошибку на отложенных примерах.'], en: ['Here the network estimates a house price from its area. Train the linear model, then change the area to inspect its prediction.', 'The model learns a slope and an offset. It is a simple baseline: neighborhood and condition are absent. Check its error on held-out examples.'] },
  image: { ru: ['Выбери закат, шахматную доску или свою картинку. Сеть получает координаты и учится выдавать цвет пикселя. Нажми обучение — изображение постепенно проявится!', 'Источник уменьшен до 32 × 32. Координаты Фурье помогают с деталями. Сравни результат с ними и без них: выход 64 × 64 вычисляет промежуточные цвета, а не возвращает утраченную детализацию.'], en: ['Choose a sunset, checkerboard, or your own image. The network learns coordinates → pixel color. Start training and watch the picture emerge!', 'The source is reduced to 32 × 32. Fourier features help with detail. Try with and without them: the 64 × 64 output evaluates intermediate colors, not lost original detail.'] },
  tictactoe: { ru: ['Обучи соперника и сыграй с ним! Учитель minimax подсказывает хорошие ходы на учебных позициях, а сеть пытается выучить эти решения.', 'Во время игры ход выбирает сама сеть. Она может ошибаться: маленькая модель приближает стратегию учителя. Изучи вероятности ходов и найди её слабое место.'], en: ['Train an opponent and play! A minimax teacher provides good moves for training positions; the network learns to imitate those decisions.', 'During play the network chooses its own moves. It can make mistakes: a small model approximates the teacher’s strategy. Inspect move probabilities and find its weak spot.'] },
  car: { ru: ['Три машины видят стены и соперников через датчики. Их сети выбирают поворот и тягу. Запусти эволюцию: удачные водители оставляют потомков с небольшими мутациями.', 'Это отбор, а не обучение по правильным ответам. След чемпиона покажет траекторию. Сравни поколения по дистанции, столкновениям и финишам, затем испытай лучших в гонке.'], en: ['Three cars sense walls and rivals. Their networks choose steering and throttle. Start evolution: successful drivers produce offspring with small mutations.', 'This uses selection rather than labeled answers. The champion trail shows its route. Compare generations by distance, collisions, and finishes, then race the best drivers.'] },
  ballistics: { ru: ['Поставь цель и выбери гравитацию. Сеть учится подбирать угол и скорость броска по примерам физических траекторий. После обучения попробуй попасть!', 'Сравни предсказанную траекторию с целью. Здесь особенно хорошо видно разницу между низкой ошибкой обучения и точным попаданием в новой ситуации.'], en: ['Set a target and gravity. The network learns launch angle and speed from physical trajectories. Train it, then try a shot!', 'Compare the predicted trajectory with the target. This makes the difference between low training error and accurate performance in a new situation easy to see.'] },
  team: { ru: ['Здесь прогнозируем время доставки задач по JSON-истории. Выбери команду и изучи недельные квантили или оценку отдельного тикета.', 'Прогноз проверяется на более поздних данных: будущее нельзя подмешивать в обучение. Сравни с простой базовой оценкой и учитывай число тикетов — редкая история даёт мало оснований для уверенности.'], en: ['Here we forecast delivery time from JSON history. Select a team and inspect weekly quantiles or an individual ticket estimate.', 'Evaluation uses later data: future information must not leak into training. Compare with a simple baseline and consider ticket counts; sparse history offers little certainty.'] },
};

export function Mentor({ id, english }: { id: string; english: boolean }) {
  const pages = lessons[id][english ? 'en' : 'ru'];
  const [page, setPage] = useState(0);
  const [visible, setVisible] = useState(0);
  const [replay, setReplay] = useState(0);
  const text = pages[page];
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) { setVisible(text.length); return; }
    setVisible(0);
    const timer = window.setInterval(() => setVisible((count) => {
      if (count >= text.length) { window.clearInterval(timer); return text.length; }
      return Math.min(count + 3, text.length);
    }), 25);
    return () => window.clearInterval(timer);
  }, [text, replay]);
  const talking = visible < text.length;
  return <header className="adventure-hero">
    <AdventureScene />
    <div className="mentor-stage"><Wizard talking={talking} />
      <div className="mentor-dialog pixel-frame">
        <div className="mentor-title"><span>{english ? 'ARCHIVIST · GUIDE TO SMALL NETWORKS' : 'АРХИВАРИУС · ХРАНИТЕЛЬ МАЛЫХ СЕТЕЙ'}</span><span>{page + 1} / {pages.length}</span></div>
        <p aria-hidden="true">{text.slice(0, visible)}<span className="dialog-cursor">▌</span></p>
        <span className="sr-only" role="status">{text}</span>
        <div className="mentor-actions">
          <span>{english ? 'No magic. Just weights and data.' : 'Никакой магии. Только веса и данные.'}</span>
          <button onClick={() => talking ? setVisible(text.length) : page < pages.length - 1 ? setPage(page + 1) : (setPage(0), setReplay((value) => value + 1))}>{talking ? (english ? 'Show all' : 'Показать всё') : page < pages.length - 1 ? (english ? 'Tell me more →' : 'Расскажи ещё →') : (english ? 'Read again ↺' : 'Ещё раз ↺')}</button>
        </div>
      </div>
    </div>
  </header>;
}
