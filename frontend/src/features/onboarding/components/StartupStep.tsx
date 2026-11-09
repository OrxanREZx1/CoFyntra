'use client';

import { cn } from '@/lib/utils';
import { inputClass } from '@/features/auth/components/FormField';
import { ONBOARDING_LIMITS } from '../lib/onboarding';

const EXAMPLES = [
  'A tool that helps students find part-time work that fits their class schedule.',
  'Reusable packaging for campus cafés.',
  "Still exploring — we know small clinics struggle with appointment no-shows.",
];

interface StartupStepProps {
  idPrefix: string;
  projectName: string;
  idea: string;
  onChange: (patch: { projectName?: string; idea?: string }) => void;
  invalid: boolean;
  errorId?: string;
}

export function StartupStep({ idPrefix, projectName, idea, onChange, invalid, errorId }: StartupStepProps) {
  return (
    <div className="space-y-5">
      <div>
        <label htmlFor={`${idPrefix}-name`} className="block mb-1.5 text-sm font-medium text-[var(--color-text-primary)]">
          Project name <span className="font-normal text-[var(--color-text-muted)]">(optional)</span>
        </label>
        <input
          id={`${idPrefix}-name`}
          value={projectName}
          onChange={(e) => onChange({ projectName: e.target.value })}
          maxLength={ONBOARDING_LIMITS.projectName + 20}
          placeholder="e.g. FoundersPath"
          autoComplete="off"
          className={inputClass}
        />
      </div>
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <label htmlFor={`${idPrefix}-idea`} className="text-sm font-medium text-[var(--color-text-primary)]">
            Your idea or the problem you&apos;re solving
          </label>
          <span
            className={cn(
              'text-xs',
              idea.length > ONBOARDING_LIMITS.ideaMax ? 'text-[var(--color-error)]' : 'text-[var(--color-text-muted)]',
            )}
          >
            {idea.length}/{ONBOARDING_LIMITS.ideaMax}
          </span>
        </div>
        <textarea
          id={`${idPrefix}-idea`}
          value={idea}
          onChange={(e) => onChange({ idea: e.target.value })}
          rows={4}
          placeholder="e.g. A tool that helps students find part-time work that fits their class schedule."
          aria-invalid={invalid || undefined}
          aria-describedby={[`${idPrefix}-examples`, invalid && errorId].filter(Boolean).join(' ')}
          className={cn(inputClass, 'h-auto py-3 resize-y min-h-[112px] leading-relaxed')}
        />
        <div id={`${idPrefix}-examples`} className="mt-2.5">
          <p className="text-xs text-[var(--color-text-muted)] mb-1.5">A sentence or two is plenty. For example:</p>
          <ul className="space-y-1">
            {EXAMPLES.map((ex) => (
              <li key={ex} className="text-xs text-[var(--color-text-secondary)] pl-3 border-l-2 border-[var(--color-surface-accent)]">
                {ex}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
