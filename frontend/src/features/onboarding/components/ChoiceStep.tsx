'use client';

import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChoiceStepProps<T extends string> {
  name: string;
  headingId: string;
  options: readonly { id: T; label: string }[];
  value: T | null;
  onChange: (value: T) => void;
  invalid: boolean;
  errorId?: string;
}

/** Single-choice question using native radios (arrow keys work out of the box). */
export function ChoiceStep<T extends string>({ name, headingId, options, value, onChange, invalid, errorId }: ChoiceStepProps<T>) {
  return (
    <fieldset aria-labelledby={headingId} aria-invalid={invalid || undefined} aria-describedby={invalid ? errorId : undefined}>
      <div className="grid gap-2.5">
        {options.map((o) => {
          const checked = value === o.id;
          return (
            <label
              key={o.id}
              className={cn(
                'group flex items-center gap-3 px-4 py-3.5 rounded-xl border cursor-pointer transition-colors',
                'bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-active)]',
                'has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[var(--color-focus)]',
                checked ? 'border-[var(--color-secondary)] bg-[var(--color-secondary-muted)]' : 'border-[var(--color-border)]',
              )}
            >
              <input
                type="radio"
                name={name}
                value={o.id}
                checked={checked}
                onChange={() => onChange(o.id)}
                className="sr-only"
              />
              <span
                className={cn(
                  'w-5 h-5 rounded-full border flex items-center justify-center flex-shrink-0 transition-colors',
                  checked
                    ? 'bg-[var(--color-secondary)] border-[var(--color-secondary)] text-[var(--color-on-secondary)]'
                    : 'border-[var(--color-border-active)]',
                )}
                aria-hidden="true"
              >
                {checked && <Check className="w-3 h-3" />}
              </span>
              <span className="text-sm sm:text-[15px] text-[var(--color-text-primary)]">{o.label}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
