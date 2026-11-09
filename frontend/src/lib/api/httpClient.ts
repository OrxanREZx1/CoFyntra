import axios from 'axios';
import { ApiError, normalizeError } from './errors';

/**
 * Centralized HTTP client for the Confyntra backend.
 *
 * - The base URL is public configuration, NOT a secret. Never put API keys or
 *   other credentials in NEXT_PUBLIC_* variables — they are shipped to the browser.
 * - `withCredentials` sends the session cookie, so the backend can derive the
 *   founder's identity from the authenticated session rather than trusting a
 *   client-supplied founder ID.
 * - No automatic retries here: retries are user-driven (chat) or handled by SWR,
 *   so requests are never retried twice by stacked mechanisms.
 * - Every rejection is normalized into an `ApiError`.
 */
export const httpClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL,
  timeout: 30_000,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
});

httpClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => Promise.reject(normalizeError(error)),
);

httpClient.interceptors.request.use((config) => {
  if (process.env.NEXT_PUBLIC_BACKEND_MODE === 'api' && !config.baseURL) {
    return Promise.reject(
      new ApiError('unavailable', {
        detail: 'Live API mode is enabled, but NEXT_PUBLIC_API_BASE_URL is not configured.',
      }),
    );
  }
  return config;
});
