'use strict';

export const seedState = {
  snapshot: {
    items: [
      { id: 'milk1', parentId: null, order: 10, status: 'Open', title: 'Молоко 3.2%', collapsed: false, tags: [] },
      { id: 'bread', parentId: null, order: 20, status: 'Open', title: 'Хлеб ржаной', collapsed: false, tags: [] },
      { id: 'borod', parentId: 'bread', order: 10, status: 'Open', title: 'Бородинский', collapsed: false, tags: [] },
      { id: 'stoli', parentId: 'bread', order: 20, status: 'Open', title: 'Столичный', collapsed: false, tags: [] },
      { id: 'apple', parentId: null, order: 30, status: 'Focus', title: 'Яблоки', collapsed: false, tags: ['Купить'] },
      { id: 'goldn', parentId: 'apple', order: 10, status: 'Open', title: 'Голден', collapsed: false, tags: [] },
      { id: 'grnsm', parentId: 'apple', order: 20, status: 'Open', title: 'Гренни Смит', collapsed: false, tags: [] },
      { id: 'fudji', parentId: 'apple', order: 30, status: 'Pause', title: 'Фуджи', collapsed: false, tags: [] },
      { id: 'pozzd', parentId: 'fudji', order: 10, status: 'Done', title: 'Позззд', collapsed: false, tags: ['Дом'] },
      { id: 'voovo', parentId: 'pozzd', order: 10, status: 'Open', title: 'Воовоага', collapsed: false, tags: [] },
      { id: 'first', parentId: 'voovo', order: 10, status: 'Focus', title: 'Первый позад', collapsed: false, tags: ['Купить'] },
      { id: 'cofee', parentId: null, order: 40, status: 'Open', title: 'Кофе', collapsed: false, tags: [] },
      { id: 'tooth', parentId: null, order: 50, status: 'Open', title: 'Зубная пастааоаоа', collapsed: false, tags: ['Дом', 'Важное'] },
      { id: 'shamp', parentId: null, order: 60, status: 'Archive', title: 'Шампунь', collapsed: false, tags: [] }
    ]
  },
  actionLog: []
};
