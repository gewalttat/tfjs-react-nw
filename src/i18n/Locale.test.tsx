import React from 'react';
import { render, screen } from '@testing-library/react';
import { LocaleContext, Localized, translate } from './Locale';
it('translates controls and dynamic training status without changing numbers', () => {
  expect(translate('Training: epoch 3/6, validation accuracy 94.2%', 'ru')).toBe('Обучение: эпоха 3/6, точность проверки 94.2%');
  const { rerender } = render(<LocaleContext.Provider value="ru"><Localized>train network</Localized></LocaleContext.Provider>);
  expect(screen.getByText('Обучить сеть')).toBeTruthy();
  rerender(<LocaleContext.Provider value="en"><Localized>train network</Localized></LocaleContext.Provider>);
  expect(screen.getByText('train network')).toBeTruthy();
});
