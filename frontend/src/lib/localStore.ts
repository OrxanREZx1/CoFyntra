/**
 * Tiny typed localStorage store for useSyncExternalStore.
 *
 * - Server / first hydration render: `getServerSnapshot()` returns `undefined`
 *   ("not known yet"), so server HTML never depends on browser storage and
 *   hydration never mismatches.
 * - Snapshots are cached by raw string, so React sees a stable reference.
 * - Updates propagate to this tab (custom event) and other tabs ('storage').
 * - Every storage access is guarded: blocked storage degrades to "empty".
 */
export interface LocalStore<T> {
  get(): T | null;
  set(value: T | null): void;
  subscribe(onChange: () => void): () => void;
  getSnapshot(): T | null;
  getServerSnapshot(): undefined;
}

const SAME_TAB_EVENT = 'confyntra:local-store';

export function createLocalStore<T>(key: string, parse: (raw: unknown) => T | null): LocalStore<T> {
  let cachedRaw: string | null | undefined;
  let cachedValue: T | null = null;

  const readRaw = (): string | null => {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const get = (): T | null => {
    const raw = readRaw();
    if (raw === cachedRaw) return cachedValue;
    cachedRaw = raw;
    try {
      cachedValue = raw === null ? null : parse(JSON.parse(raw));
    } catch {
      cachedValue = null;
    }
    return cachedValue;
  };

  return {
    get,
    set(value) {
      try {
        if (value === null) window.localStorage.removeItem(key);
        else window.localStorage.setItem(key, JSON.stringify(value));
      } catch {
        // Storage unavailable: nothing persists, which the UI tolerates.
      }
      window.dispatchEvent(new CustomEvent(SAME_TAB_EVENT, { detail: key }));
    },
    subscribe(onChange) {
      const onStorage = (e: StorageEvent) => e.key === key && onChange();
      const onLocal = (e: Event) => (e as CustomEvent<string>).detail === key && onChange();
      window.addEventListener('storage', onStorage);
      window.addEventListener(SAME_TAB_EVENT, onLocal);
      return () => {
        window.removeEventListener('storage', onStorage);
        window.removeEventListener(SAME_TAB_EVENT, onLocal);
      };
    },
    getSnapshot: get,
    getServerSnapshot: () => undefined,
  };
}
