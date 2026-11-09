import type { Metadata } from 'next';
import { SignupForm } from '@/features/auth/components/SignupForm';

export const metadata: Metadata = { title: 'Create your account' };

export default function SignupPage() {
  return (
    <>
      <h1 className="font-display text-3xl tracking-tight text-[var(--color-text-primary)]">Create your account</h1>
      <p className="mt-2 mb-7 text-sm text-[var(--color-text-secondary)]">
        Start a shared workspace for you and your co-founder.
      </p>
      <SignupForm />
    </>
  );
}
