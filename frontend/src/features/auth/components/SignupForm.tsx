'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useId, useState, useTransition } from 'react';
import { BACKEND_MODE } from '@/lib/config';
import { clearDemoProfile } from '@/features/onboarding/profileStore';
import { startDemoSession } from '../demoSession';
import { validateSignup, type FieldErrors, type SignupValues } from '../lib/validation';
import { DemoAuthNotice, Divider, ExistingSessionBanner, ExploreDemoButton, primaryButton, Spinner } from './authUi';
import { FormField } from './FormField';

const EMPTY: SignupValues = { name: '', email: '', password: '', confirmPassword: '' };
const ORDER: (keyof SignupValues)[] = ['name', 'email', 'password', 'confirmPassword'];

export function SignupForm() {
  const id = useId();
  const router = useRouter();
  const [values, setValues] = useState<SignupValues>(EMPTY);
  const [touched, setTouched] = useState<Partial<Record<keyof SignupValues, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);
  const [authUnavailable, setAuthUnavailable] = useState(false);
  const [pending, startTransition] = useTransition();

  const errors: FieldErrors<SignupValues> = validateSignup(values);
  const show = (k: keyof SignupValues) => (submitted || touched[k] ? errors[k] : undefined);
  const bind = (k: keyof SignupValues) => ({
    value: values[k],
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => setValues((v) => ({ ...v, [k]: e.target.value })),
    onBlur: () => setTouched((t) => ({ ...t, [k]: true })),
    error: show(k),
  });

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    const firstInvalid = ORDER.find((k) => errors[k]);
    if (firstInvalid) {
      document.getElementById(`${id}-${firstInvalid}`)?.focus();
      return;
    }
    if (BACKEND_MODE === 'api') {
      setAuthUnavailable(true);
      return;
    }
    startTransition(() => {
      // Demo entry: keep only the display name. Email and password are
      // discarded here and never stored or sent anywhere.
      startDemoSession(values.name);
      clearDemoProfile(); // a new signup starts onboarding fresh
      setValues((v) => ({ ...v, email: '', password: '', confirmPassword: '' }));
      router.push('/onboarding');
    });
  };

  return (
    <div className="space-y-5">
      <ExistingSessionBanner />
      <form onSubmit={onSubmit} noValidate className="space-y-4" aria-label="Create your account">
        {authUnavailable && (
          <p role="alert" className="text-xs text-[var(--color-error)]">
            Live account creation is unavailable until an authentication API contract is provided.
          </p>
        )}
        <FormField id={`${id}-name`} label="Full name" autoComplete="name" {...bind('name')} />
        <FormField id={`${id}-email`} label="Email" type="email" inputMode="email" autoComplete="email" {...bind('email')} />
        <FormField
          id={`${id}-password`}
          label="Password"
          revealable
          autoComplete="new-password"
          hint="At least 8 characters, with a letter and a number."
          {...bind('password')}
        />
        <FormField id={`${id}-confirmPassword`} label="Confirm password" revealable autoComplete="new-password" {...bind('confirmPassword')} />
        <button type="submit" className={primaryButton} disabled={pending}>
          {pending && <Spinner />} {pending ? 'Setting up your demo…' : 'Create account'}
        </button>
      </form>
      <DemoAuthNotice />
      <Divider label="or" />
      <ExploreDemoButton />
      <p className="text-center text-sm text-[var(--color-text-muted)]">
        Already have an account?{' '}
        <Link href="/login" className="font-medium text-[var(--color-text-primary)] underline underline-offset-4 hover:text-[var(--color-secondary)]">
          Log in
        </Link>
      </p>
    </div>
  );
}
