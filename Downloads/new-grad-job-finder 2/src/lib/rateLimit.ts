// Minimal in-memory rate limiter. Sufficient for a single-user, single
// -process personal app - not meant to survive multiple server instances.
const hits: number[] = [];

export function isRateLimited(maxPerMinute = 20): boolean {
  const now = Date.now();
  const oneMinuteAgo = now - 60_000;
  while (hits.length > 0 && hits[0] < oneMinuteAgo) hits.shift();
  if (hits.length >= maxPerMinute) return true;
  hits.push(now);
  return false;
}
