import axios from 'axios';

/**
 * Normalized error categories shared by every service (HTTP or demo).
 * UI code switches on `kind` and never inspects raw Axios errors.
 */
export type ApiErrorKind =
  | 'network' // no connection / request never reached a server
  | 'unavailable' // server reachable but down or failing (5xx)
  | 'timeout'
  | 'rate_limited'
  | 'unauthorized' // missing/expired session or forbidden
  | 'invalid_response' // response did not match the expected shape
  | 'empty_response' // AI returned no content
  | 'conflict' // stale version / concurrent change (HTTP 409)
  | 'invalid_request' // the server rejected the action (HTTP 400/422)
  | 'not_found'
  | 'canceled'
  | 'unknown';

const USER_MESSAGES: Record<ApiErrorKind, string> = {
  network: "Couldn't reach the server. Check your connection and try again.",
  unavailable: 'The service is temporarily unavailable. Please try again shortly.',
  timeout: 'The request took too long. Please try again.',
  rate_limited: "You're sending messages too quickly. Wait a moment and retry.",
  unauthorized: 'Your session has expired. Please sign in again.',
  invalid_response: 'Received an unexpected response. Please try again.',
  empty_response: 'The assistant returned an empty reply. Please try again.',
  conflict: 'This was changed by someone else. Showing the latest version.',
  invalid_request: "That action isn't allowed right now.",
  not_found: 'This item no longer exists.',
  canceled: 'The request was canceled.',
  unknown: 'Something went wrong. Please try again.',
};

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  /** Seconds to wait before retrying, when the server says so (429). */
  readonly retryAfterSeconds?: number;
  /** Specific, user-safe explanation (e.g. which rule blocked an action). */
  readonly detail?: string;

  constructor(
    kind: ApiErrorKind,
    options: { message?: string; detail?: string; status?: number; retryAfterSeconds?: number; cause?: unknown } = {},
  ) {
    super(options.message ?? options.detail ?? USER_MESSAGES[kind], { cause: options.cause });
    this.name = 'ApiError';
    this.kind = kind;
    this.status = options.status;
    this.retryAfterSeconds = options.retryAfterSeconds;
    this.detail = options.detail;
  }

  /** Concise, user-facing description. Never includes message content. */
  get userMessage(): string {
    if (this.detail) return this.detail;
    if (this.kind === 'rate_limited' && this.retryAfterSeconds) {
      return `You're sending messages too quickly. Try again in ${this.retryAfterSeconds}s.`;
    }
    return USER_MESSAGES[this.kind];
  }
}

function parseRetryAfter(value: unknown): number | undefined {
  if (typeof value !== 'string' && typeof value !== 'number') return undefined;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds) : undefined;
}

/** A short, plain-text `message` from an error response body, if present. */
function serverDetail(error: { response?: { data?: unknown } }): string | undefined {
  const data = error.response?.data;
  if (data && typeof data === 'object' && 'message' in data) {
    const message = (data as { message: unknown }).message;
    if (typeof message === 'string' && message.length > 0 && message.length <= 300) return message;
  }
  return undefined;
}

/** Converts anything thrown by a request into an ApiError. */
export function normalizeError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;

  if (axios.isCancel(error)) return new ApiError('canceled', { cause: error });

  if (axios.isAxiosError(error)) {
    const status = error.response?.status;

    if (status === undefined) {
      if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
        return new ApiError('timeout', { cause: error });
      }
      return new ApiError('network', { cause: error });
    }
    if (status === 401 || status === 403) return new ApiError('unauthorized', { status, cause: error });
    if (status === 429) {
      return new ApiError('rate_limited', {
        status,
        retryAfterSeconds: parseRetryAfter(error.response?.headers?.['retry-after']),
        cause: error,
      });
    }
    if (status === 409) return new ApiError('conflict', { status, detail: serverDetail(error), cause: error });
    if (status === 400 || status === 422) {
      return new ApiError('invalid_request', { status, detail: serverDetail(error), cause: error });
    }
    if (status === 404) return new ApiError('not_found', { status, cause: error });
    if (status === 408 || status === 504) return new ApiError('timeout', { status, cause: error });
    if (status >= 500) return new ApiError('unavailable', { status, cause: error });
    return new ApiError('unknown', { status, cause: error });
  }

  return new ApiError('unknown', { cause: error });
}
