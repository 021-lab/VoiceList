const DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;

function pad(value) {
  return String(value).padStart(2, '0');
}

export function isDeadline(value) {
  const match = DATE_PATTERN.exec(String(value || ''));
  if (!match) return false;
  const [, year, month, day] = match.map(Number);
  const date = new Date(year, month - 1, day);
  return date.getFullYear() === year && date.getMonth() === month - 1 && date.getDate() === day;
}

export function deadlineFromToday(days, now = new Date()) {
  const date = new Date(now.getFullYear(), now.getMonth(), now.getDate() + days);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function deadlineDaysFromToday(deadline, now = new Date()) {
  if (!isDeadline(deadline)) return null;
  const [, year, month, day] = DATE_PATTERN.exec(deadline).map(Number);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(year, month - 1, day);
  return Math.round((target - today) / 86_400_000);
}

export function compareByDeadline(left, right, now = new Date()) {
  const leftDays = deadlineDaysFromToday(left.deadline, now);
  const rightDays = deadlineDaysFromToday(right.deadline, now);
  if (leftDays === null && rightDays === null) return (left.order || 0) - (right.order || 0);
  if (leftDays === null) return 1;
  if (rightDays === null) return -1;
  return leftDays - rightDays || (left.order || 0) - (right.order || 0);
}

export function deadlineDaysLabel(deadline, now = new Date()) {
  const days = deadlineDaysFromToday(deadline, now);
  return days === null ? '' : String(days);
}
