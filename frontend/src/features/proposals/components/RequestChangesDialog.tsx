'use client';

import { useId, useState } from 'react';
import { cn } from '@/lib/utils';
import { Dialog } from '@/components/ui/Dialog';
import type { DecisionProposal } from '@/types';
import { LIMITS } from '../lib/proposalMachine';

interface RequestChangesDialogProps {
  proposal: DecisionProposal;
  isSubmitting: boolean;
  onSubmit: (reason: string) => Promise<boolean>;
  onClose: () => void;
}

export function RequestChangesDialog({ proposal, isSubmitting, onSubmit, onClose }: RequestChangesDialogProps) {
  const id = useId();
  const [reason, setReason] = useState('');
  const [touched, setTouched] = useState(false);
  const trimmed = reason.trim();
  const error = !trimmed ? 'Briefly explain what should change.' : trimmed.length > LIMITS.reason ? 'Please keep it shorter.' : undefined;

  const submit = async () => {
    setTouched(true);
    if (error || isSubmitting) return;
    if (await onSubmit(trimmed)) onClose();
  };

  const btn = 'inline-flex items-center justify-center gap-1.5 h-9 px-3.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50';

  return (
    <Dialog
      title={`Request changes · version ${proposal.version}`}
      description={
        <>
          Approval of <span className="text-[var(--color-text-secondary)]">“{proposal.title}”</span> will be paused until
          a revised version is published.
        </>
      }
      onClose={onClose}
      footer={
        <>
          <button type="button" onClick={onClose} className={cn(btn, 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]')}>
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={isSubmitting}
            className={cn(btn, 'bg-[var(--color-error)] text-[var(--color-bg-base)] hover:opacity-90')}
          >
            {isSubmitting && (
              <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" aria-hidden="true" />
            )}
            Request changes
          </button>
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        noValidate
      >
        <div className="flex items-baseline justify-between mb-1">
          <label htmlFor={`${id}-reason`} className="text-xs font-medium text-[var(--color-text-primary)]">
            What should change?
          </label>
          <span className="text-[10px] text-[var(--color-text-muted)]">
            {reason.length}/{LIMITS.reason}
          </span>
        </div>
        <textarea
          id={`${id}-reason`}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          rows={4}
          autoFocus
          placeholder="e.g. Let's interview working professionals too before choosing a segment."
          aria-invalid={touched && Boolean(error)}
          aria-describedby={touched && error ? `${id}-err` : undefined}
          className="w-full rounded-lg px-3 py-2 text-base sm:text-sm outline-none resize-y bg-[var(--color-bg-base)] border border-[var(--color-border)] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-disabled)] focus-visible:border-[var(--color-primary)]"
        />
        {touched && error && (
          <p id={`${id}-err`} className="mt-1 text-[11px] text-[var(--color-error)]">
            {error}
          </p>
        )}
      </form>
    </Dialog>
  );
}
