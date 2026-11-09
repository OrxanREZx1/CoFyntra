import { cn } from '@/lib/utils';

export function OnboardingProgress({ step, total }: { step: number; total: number }) {
  return (
    <div className="flex items-center gap-3">
      <div
        role="progressbar"
        aria-label="Onboarding progress"
        aria-valuemin={1}
        aria-valuemax={total}
        aria-valuenow={step + 1}
        aria-valuetext={`Question ${step + 1} of ${total}`}
        className="flex gap-1.5 flex-1"
      >
        {Array.from({ length: total }, (_, i) => (
          <span
            key={i}
            className={cn(
              'h-1.5 flex-1 rounded-full transition-colors duration-300',
              i <= step ? 'bg-[var(--color-secondary)]' : 'bg-[var(--color-border)]',
            )}
          />
        ))}
      </div>
      <span className="text-xs font-medium text-[var(--color-text-muted)] whitespace-nowrap" aria-hidden="true">
        {step + 1} of {total}
      </span>
    </div>
  );
}
