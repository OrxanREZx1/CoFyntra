'use client';

import { useSyncExternalStore } from 'react';
import { formatTimestamp } from '@/lib/utils';

const MINUTE = 60_000;

// Re-render once a minute so "5m ago" stays fresh.
function subscribe(onChange: () => void) {
  const id = setInterval(onChange, MINUTE);
  return () => clearInterval(id);
}

// Rounded to the minute so the snapshot is stable between renders.
function getSnapshot() {
  return Math.floor(Date.now() / MINUTE) * MINUTE;
}

// No current time on the server / during prerender or hydration.
function getServerSnapshot() {
  return null;
}

/**
 * Renders a relative time ("5m ago"). The current time is only read on the
 * client, so the value is never frozen into a prerendered page.
 */
export function RelativeTime({ timestamp }: { timestamp: string }) {
  const now = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  return (
    <time dateTime={timestamp}>
      {now === null
        ? new Date(timestamp).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            timeZone: 'UTC',
          })
        : formatTimestamp(timestamp, now)}
    </time>
  );
}
