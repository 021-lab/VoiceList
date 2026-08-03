import * as R from './resolver.js';
const T = (id, title, status='open') => ({id, title, status});
const graph = [
  T('1','Ремонт'), T('2','Ванная','in_progress'), T('3','Установить смеситель'),
  T('4','Спальня'), T('5','Купить обои'), T('6','Свет','blocked'),
  T('7','Купить лампочку'), T('8','Покупки'), T('9','Купить смеситель'),
];
const cases = [
  ['купить смеситель', r => r.kind==='one' && r.id==='9' && r.why==='R1'],
  ['купи смеситель',   r => r.kind==='one' && r.id==='9' && r.why==='R2'],  // стем + полное покрытие
  ['смеситель',        r => r.kind==='vector' && r.why==='tie' && r.candidates.length===2],
  ['ванную',           r => r.kind==='one' && r.id==='2'],
  ['купить',           r => r.kind==='vector' && r.candidates.length===3],
  ['сместитель',       r => r.kind==='vector' && r.candidates.length===2],
  ['свиточка',         r => r.kind==='empty'],
  ['Купить Смеситель!',r => r.kind==='one' && r.id==='9' && r.why==='R1'],
];
let fail = 0;
for (const [q, check] of cases) {
  const res = R.resolveOverTasks(q, graph);
  const ok = check(res);
  console.log((ok?'PASS':'FAIL'), JSON.stringify(q), '->', JSON.stringify(res));
  if (!ok) fail++;
}
// дубли имён -> инвариант 1/2
const dup = R.resolveOverTasks('ремонт', [...graph, T('10','Ремонт')]);
console.log(dup.kind==='vector' && dup.why==='R1-dup' ? 'PASS' : 'FAIL', '"ремонт" (дубль) ->', JSON.stringify(dup));
if (!(dup.kind==='vector')) fail++;
// closed/superseded вне охвата
const closed = R.resolveOverTasks('покупки', graph.map(t=>t.id==='8'?{...t,status:'closed'}:t));
console.log(closed.kind==='empty' ? 'PASS' : 'FAIL', '"покупки" (closed) ->', JSON.stringify(closed));
if (closed.kind!=='empty') fail++;
process.exit(fail ? 1 : 0);
