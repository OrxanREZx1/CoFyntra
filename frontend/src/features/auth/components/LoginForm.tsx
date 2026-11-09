'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useId, useRef, useState, useTransition } from 'react';
import { BACKEND_MODE } from '@/lib/config';
import { readDemoProfile, saveDemoProfile } from '@/features/onboarding/profileStore';
import { startDemoSession } from '../demoSession';
import { nextDemoRoute } from '../useDemoAccess';
import { isAdminDemoLogin } from '../lib/demoCredentials';
import { validateLogin, type FieldErrors, type LoginValues } from '../lib/validation';
import { DemoAuthNotice, Divider, ExistingSessionBanner, ExploreDemoButton, primaryButton, Spinner } from './authUi';
import { FormField } from './FormField';

const ADMIN_DEMO_PROFILE = {
  reason: 'have-cofounder',
  role: 'founder',
  projectName: 'FoundersPath',
  idea: 'AI tools that help startup co-founders connect their discoveries and make decisions together.',
} as const;

export function LoginForm() {
  const id = useId();
  const router = useRouter();
  const [values, setValues] = useState<LoginValues>({ email: '', password: '' });
  const [touched, setTouched] = useState<Partial<Record<keyof LoginValues, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);
  const [authUnavailable, setAuthUnavailable] = useState(false);
  const [pending, startTransition] = useTransition();
  const emailRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);

  const errors: FieldErrors<LoginValues> = validateLogin(values);
  const show = (k: keyof LoginValues) => (submitted || touched[k] ? errors[k] : undefined);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (errors.email) return emailRef.current?.focus();
    if (errors.password) return passwordRef.current?.focus();
    if (BACKEND_MODE === 'api') {
      setAuthUnavailable(true);
      return;
    }
    const isAdminDemo = isAdminDemoLogin(values.email, values.password);
    startTransition(() => {
      // This is a local demo shortcut; credentials are never stored or sent.
      if (isAdminDemo) saveDemoProfile(ADMIN_DEMO_PROFILE);
      startDemoSession(isAdminDemo ? 'Admin' : null);
      setValues({ email: '', password: '' });
      router.push(nextDemoRoute(readDemoProfile() !== null));
    });
  };

  return (
    <div className="space-y-5">
      <ExistingSessionBanner />
      <form onSubmit={onSubmit} noValidate className="space-y-4" aria-label="Log in">
        {authUnavailable && (
          <p role="alert" className="text-xs text-[var(--color-error)]">
            Live sign-in is unavailable until an authentication API contract is provided.
          </p>
        )}
        <FormField
          id={`${id}-email`}
          label="Email"
          type="email"
          inputMode="email"
          autoComplete="email"
          inputRef={emailRef}
          value={values.email}
          onChange={(e) => setValues((v) => ({ ...v, email: e.target.value }))}
          onBlur={() => setTouched((t) => ({ ...t, email: true }))}
          error={show('email')}
        />
        <div>
          <FormField
            id={`${id}-password`}
            label="Password"
            revealable
            autoComplete="current-password"
            inputRef={passwordRef}
            value={values.password}
            onChange={(e) => setValues((v) => ({ ...v, password: e.target.value }))}
            onBlur={() => setTouched((t) => ({ ...t, password: true }))}
            error={show('password')}
          />
          <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
            {BACKEND_MODE === 'demo' ? (
              <>
                Password recovery isn&apos;t available. Use <span className="font-medium">admin@gmail.com</span> /{' '}
                <span className="font-medium">admin</span> to open the demo chat.
              </>
            ) : (
              'Live sign-in is unavailable until an authentication API contract is provided.'
            )}
          </p>
        </div>
        <button type="submit" className={primaryButton} disabled={pending}>
          {pending && <Spinner />} {pending ? 'Entering demo…' : 'Log in'}
        </button>
      </form>
      <DemoAuthNotice />
      <Divider label="or" />
      <ExploreDemoButton />
      <p className="text-center text-sm text-[var(--color-text-muted)]">
        New to Confyntra?{' '}
        <Link href="/signup" className="font-medium text-[var(--color-text-primary)] underline underline-offset-4 hover:text-[var(--color-secondary)]">
          Create an account
        </Link>
      </p>
    </div>
  );
}
