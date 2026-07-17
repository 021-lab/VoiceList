'use strict';

export const seedState = {
  snapshot: {
    items: [
      { id: 'milk1', parentId: null, order: 10, status: 'Open', line1: 'Молоко 3.2%', line2: '2 пакета, магазин у дома', collapsed: false, tags: [] },
      { id: 'bread', parentId: null, order: 20, status: 'Open', line1: 'Хлеб ржаной', line2: '', collapsed: false, tags: [] },
      { id: 'borod', parentId: 'bread', order: 10, status: 'Open', line1: 'Бородинский', line2: '400 г', collapsed: false, tags: [] },
      { id: 'stoli', parentId: 'bread', order: 20, status: 'Open', line1: 'Столичный', line2: '500 г', collapsed: false, tags: [] },
      { id: 'apple', parentId: null, order: 30, status: 'Focus', line1: 'Яблоки', line2: 'Голден, ~1.5 кг', collapsed: false, tags: ['Купить'] },
      { id: 'goldn', parentId: 'apple', order: 10, status: 'Open', line1: 'Голден', line2: '500 г', collapsed: false, tags: [] },
      { id: 'grnsm', parentId: 'apple', order: 20, status: 'Open', line1: 'Гренни Смит', line2: '400 г', collapsed: false, tags: [] },
      { id: 'fudji', parentId: 'apple', order: 30, status: 'Pause', line1: 'Фуджи', line2: '300 г', collapsed: false, tags: [] },
      { id: 'pozzd', parentId: 'fudji', order: 10, status: 'Done', line1: 'Позззд', line2: '', collapsed: false, tags: ['Дом'] },
      { id: 'voovo', parentId: 'pozzd', order: 10, status: 'Open', line1: 'Воовоага', line2: '', collapsed: false, tags: [] },
      { id: 'first', parentId: 'voovo', order: 10, status: 'Focus', line1: 'Первый позад', line2: '', collapsed: false, tags: ['Купить'] },
      { id: 'cofee', parentId: null, order: 40, status: 'Open', line1: 'Кофе', line2: 'Арабика, зерно, 250 г', collapsed: false, tags: [] },
      { id: 'tooth', parentId: null, order: 50, status: 'Open', line1: 'Зубная пастааоаоа', line2: '', collapsed: false, tags: ['Дом', 'Важное'] },
      { id: 'shamp', parentId: null, order: 60, status: 'Archive', line1: 'Шампунь', line2: 'Для нормальных волос', collapsed: false, tags: [] }
    ]
  },
  actionLog: []
};
