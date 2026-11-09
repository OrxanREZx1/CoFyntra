import type { Metadata } from 'next';
import { LoginForm } from '@/features/auth/components/LoginForm';

export const metadata: Metadata = { title: 'Log in' };

export default function LoginPage() {
  return (
    <>
      <h1 className="font-display text-3xl tracking-tight text-[var(--color-text-primary)]">Welcome back</h1>
      <p className="mt-2 mb-7 text-sm text-[var(--color-text-secondary)]">Log in to your Confyntra workspace.</p>
      <LoginForm />
    </>
  );
}
