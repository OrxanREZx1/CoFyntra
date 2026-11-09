'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useId, useRef, useState, useTransition } from 'react';
import { AlertCircle, ArrowLeft, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Spinner } from '@/features/auth/components/authUi';
import { saveDemoProfile } from '../profileStore';
import {
  EMPTY_DRAFT,
  FOUNDER_ROLES,
  ONBOARDING_REASONS,
  ONBOARDING_STEP_COUNT,
  validateStep,
  type OnboardingDraft,
} from '../lib/onboarding';
import { ChoiceStep } from './ChoiceStep';
import { OnboardingProgress } from './OnboardingProgress';
import { StartupStep } from './StartupStep';

const STEPS = [
  { title: 'What brings you to Confyntra?', description: 'Pick the one that fits best.' },
  { title: 'What best describes you?', description: 'This helps us frame your workspace.' },
  { title: 'What are you working on?', description: "Tell us a little about your startup idea or the problem you're trying to solve." },
];

export function OnboardingWizard({ displayName }: { displayName: string | null }) {
  const id = useId();
  const router = useRouter();
  const [step, setStep] = useState(0);
  // Single draft for all steps, so going Back never loses answers.
  const [draft, setDraft] = useState<OnboardingDraft>(EMPTY_DRAFT);
  const [showError, setShowError] = useState(false);
  const [finishing, startTransition] = useTransition();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const firstRender = useRef(true);

  // Move focus to the new question so screen readers announce it.
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    headingRef.current?.focus();
  }, [step]);

  const error = validateStep(step, draft);
  const isLast = step === ONBOARDING_STEP_COUNT - 1;
  const errorId = `${id}-error`;
  const headingId = `${id}-heading`;

  const update = (patch: Partial<OnboardingDraft>) => {
    setDraft((d) => ({ ...d, ...patch }));
    setShowError(false);
  };

  const next = (e: React.FormEvent) => {
    e.preventDefault();
    if (error) return setShowError(true);
    if (!isLast) {
      setShowError(false);
      return setStep((s) => s + 1);
    }
    startTransition(() => {
      saveDemoProfile(draft);
      router.push('/workspace');
    });
  };

  const firstName = displayName?.split(/\s+/)[0];
  const { title, description } = STEPS[step];

  return (
    <form onSubmit={next} noValidate className="w-full max-w-xl">
      <OnboardingProgress step={step} total={ONBOARDING_STEP_COUNT} />

      <div key={step} className="animate-step-in mt-8">
        {step === 0 && (
          <p className="mb-2 text-sm text-[var(--color-text-muted)]">
            Welcome{firstName ? `, ${firstName}` : ''}. Three quick questions and you&apos;re in.
          </p>
        )}
        <h1
          id={headingId}
          ref={headingRef}
          tabIndex={-1}
          className="font-display text-3xl sm:text-4xl tracking-tight text-[var(--color-text-primary)] outline-none"
        >
          {title}
        </h1>
        <p className="mt-2 mb-7 text-[15px] text-[var(--color-text-secondary)]">{description}</p>

        {step === 0 && (
          <ChoiceStep
            name="reason"
            headingId={headingId}
            options={ONBOARDING_REASONS}
            value={draft.reason}
            onChange={(reason) => update({ reason })}
            invalid={showError && Boolean(error)}
            errorId={errorId}
          />
        )}
        {step === 1 && (
          <ChoiceStep
            name="role"
            headingId={headingId}
            options={FOUNDER_ROLES}
            value={draft.role}
            onChange={(role) => update({ role })}
            invalid={showError && Boolean(error)}
            errorId={errorId}
          />
        )}
        {step === 2 && (
          <StartupStep
            idPrefix={id}
            projectName={draft.projectName}
            idea={draft.idea}
            onChange={update}
            invalid={showError && Boolean(error)}
            errorId={errorId}
          />
        )}
      </div>

      <div className="min-h-6 mt-4" aria-live="polite">
        {showError && error && (
          <p id={errorId} className="flex items-center gap-1.5 text-sm text-[var(--color-error)]">
            <AlertCircle className="w-4 h-4 flex-shrink-0" aria-hidden="true" /> {error}
          </p>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => {
            setShowError(false);
            setStep((s) => Math.max(0, s - 1));
          }}
          disabled={step === 0 || finishing}
          className="inline-flex items-center gap-1.5 h-11 px-4 rounded-xl text-sm font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)] disabled:opacity-0 disabled:pointer-events-none"
        >
          <ArrowLeft className="w-4 h-4" aria-hidden="true" /> Back
        </button>
        <button
          type="submit"
          disabled={finishing}
          className={cn(
            'inline-flex items-center gap-2 h-11 px-6 rounded-xl text-sm font-semibold transition-colors',
            'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)] disabled:opacity-60',
          )}
        >
          {finishing ? <Spinner /> : null}
          {isLast ? (finishing ? 'Opening your workspace…' : 'Enter your workspace') : 'Continue'}
          {!finishing && <ArrowRight className="w-4 h-4" aria-hidden="true" />}
        </button>
      </div>

      <p className="mt-10 text-xs text-[var(--color-text-muted)]">
        Demo: your answers are saved only in this browser to personalize the sample workspace. They aren&apos;t sent to a
        server or used to train or tailor any AI.
      </p>
    </form>
  );
}
