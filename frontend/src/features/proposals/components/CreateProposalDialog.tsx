'use client';

import { useId, useState } from 'react';
import { AlertTriangle, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Dialog } from '@/components/ui/Dialog';
import { EpistemicBadge } from '@/components/ui/EpistemicBadge';
import { FOUNDERS } from '@/mocks';
import type { DecisionProposal, KnowledgeConflict, KnowledgeItem, ProposalContent } from '@/types';
import { LIMITS } from '../lib/proposalMachine';

export type ProposalDialogMode =
  | { kind: 'create' }
  | { kind: 'edit-draft'; proposal: DecisionProposal }
  | { kind: 'revise'; proposal: DecisionProposal };

interface CreateProposalDialogProps {
  mode: ProposalDialogMode;
  items: KnowledgeItem[];
  knowledgeConflicts: readonly KnowledgeConflict[];
  isSubmitting: boolean;
  /** Resolves true on success (the dialog then closes). */
  onSubmit: (content: ProposalContent, options: { submit: boolean; changeNote: string | null }) => Promise<boolean>;
  onClose: () => void;
}

const inputClass = cn(
  'w-full rounded-lg px-3 py-2 text-base sm:text-sm outline-none',
  'bg-[var(--color-bg-base)] border border-[var(--color-border)] text-[var(--color-text-primary)]',
  'placeholder:text-[var(--color-text-disabled)] focus-visible:border-[var(--color-primary)]',
);

function TextField(props: {
  id: string;
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
  max: number;
  placeholder?: string;
  multiline?: boolean;
  error?: string;
}) {
  const near = props.value.length > props.max * 0.8;
  const describedBy = [props.hint && `${props.id}-hint`, props.error && `${props.id}-err`].filter(Boolean).join(' ') || undefined;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label htmlFor={props.id} className="text-xs font-medium text-[var(--color-text-primary)]">
          {props.label}
        </label>
        {near && (
          <span className={cn('text-[10px]', props.value.length > props.max ? 'text-[var(--color-error)]' : 'text-[var(--color-text-muted)]')}>
            {props.value.length}/{props.max}
          </span>
        )}
      </div>
      {props.hint && (
        <p id={`${props.id}-hint`} className="mb-1.5 text-[11px] text-[var(--color-text-muted)]">
          {props.hint}
        </p>
      )}
      {props.multiline ? (
        <textarea
          id={props.id}
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          placeholder={props.placeholder}
          rows={3}
          aria-invalid={Boolean(props.error)}
          aria-describedby={describedBy}
          className={cn(inputClass, 'resize-y min-h-[72px]')}
        />
      ) : (
        <input
          id={props.id}
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          placeholder={props.placeholder}
          aria-invalid={Boolean(props.error)}
          aria-describedby={describedBy}
          className={inputClass}
        />
      )}
      {props.error && (
        <p id={`${props.id}-err`} className="mt-1 text-[11px] text-[var(--color-error)]">
          {props.error}
        </p>
      )}
    </div>
  );
}

function KnowledgePicker(props: {
  legend: string;
  hint: string;
  items: KnowledgeItem[];
  selected: string[];
  disabledIds: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <fieldset>
      <legend className="text-xs font-medium text-[var(--color-text-primary)]">{props.legend}</legend>
      <p className="mb-1.5 text-[11px] text-[var(--color-text-muted)]">{props.hint}</p>
      <div className="max-h-44 overflow-y-auto rounded-lg border border-[var(--color-border)] divide-y divide-[var(--color-border)]">
        {props.items.length === 0 && <p className="p-3 text-[11px] text-[var(--color-text-muted)]">No shared findings yet.</p>}
        {props.items.map((item) => {
          const checked = props.selected.includes(item.id);
          const disabled = !checked && props.disabledIds.includes(item.id);
          return (
            <label
              key={item.id}
              className={cn(
                'flex items-start gap-2 px-2.5 py-2 cursor-pointer hover:bg-[var(--color-bg-hover)]',
                checked && 'bg-[var(--color-primary-muted)]',
                disabled && 'opacity-40 cursor-not-allowed',
              )}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={() => props.onToggle(item.id)}
                className="mt-0.5 accent-[var(--color-primary)]"
              />
              <span className="min-w-0 flex-1">
                <span className="block text-xs text-[var(--color-text-primary)] leading-snug wrap-anywhere">
                  <span className="font-mono text-[10px] text-[var(--color-text-muted)]">{item.id}</span> {item.finding}
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-1.5">
                  <EpistemicBadge status={item.epistemicStatus} compact />
                  <span className="text-[10px] text-[var(--color-text-muted)]">{FOUNDERS[item.contributorId]?.name ?? item.contributorName}</span>
                  {disabled && <span className="text-[10px] text-[var(--color-text-muted)]">· selected in the other list</span>}
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function CreateProposalDialog({ mode, items, knowledgeConflicts, isSubmitting, onSubmit, onClose }: CreateProposalDialogProps) {
  const id = useId();
  const source = mode.kind === 'create' ? undefined : mode.proposal;
  const [title, setTitle] = useState(source?.title ?? '');
  const [direction, setDirection] = useState(source?.direction ?? '');
  const [nextAction, setNextAction] = useState(source?.nextAction ?? '');
  const [supporting, setSupporting] = useState<string[]>(source?.supportingKnowledgeIds ?? []);
  const [conflicting, setConflicting] = useState<string[]>(source?.conflictingKnowledgeIds ?? []);
  const [changeNote, setChangeNote] = useState('');
  const [showErrors, setShowErrors] = useState(false);

  const errors = {
    title: !title.trim() ? 'Give the decision a short title.' : title.trim().length > LIMITS.title ? 'Too long.' : undefined,
    direction: !direction.trim() ? 'Describe the direction you propose.' : direction.trim().length > LIMITS.direction ? 'Too long.' : undefined,
    nextAction: !nextAction.trim() ? 'Name the concrete next step.' : nextAction.trim().length > LIMITS.nextAction ? 'Too long.' : undefined,
    changeNote: changeNote.trim().length > LIMITS.changeNote ? 'Too long.' : undefined,
  };
  const valid = !Object.values(errors).some(Boolean);

  const toggle = (list: string[], set: (v: string[]) => void, itemId: string) =>
    set(list.includes(itemId) ? list.filter((x) => x !== itemId) : [...list, itemId]);

  // Suggestions come ONLY from supplied conflict relationships.
  const suggested = knowledgeConflicts
    .flatMap((c) => {
      const [a, b] = c.itemIds;
      if (supporting.includes(a)) return [b];
      if (supporting.includes(b)) return [a];
      return [];
    })
    .filter((x, i, arr) => arr.indexOf(x) === i && !conflicting.includes(x) && !supporting.includes(x));

  const submit = async (submitForApproval: boolean) => {
    setShowErrors(true);
    if (!valid || isSubmitting) return;
    const ok = await onSubmit(
      { title, direction, nextAction, supportingKnowledgeIds: supporting, conflictingKnowledgeIds: conflicting },
      { submit: submitForApproval, changeNote: changeNote.trim() || null },
    );
    if (ok) onClose();
  };

  const err = (k: keyof typeof errors) => (showErrors ? errors[k] : undefined);
  const btn = 'inline-flex items-center justify-center gap-1.5 h-9 px-3.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50';
  const spinner = <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" aria-hidden="true" />;

  const titleText =
    mode.kind === 'create' ? 'Propose a decision' : mode.kind === 'edit-draft' ? 'Edit draft' : `Revise proposal · version ${mode.proposal.version + 1}`;

  return (
    <Dialog
      title={titleText}
      description={
        mode.kind === 'revise'
          ? 'Publishing creates a new version.'
          : 'A decision becomes official only when both founders approve the same version.'
      }
      onClose={onClose}
      footer={
        <>
          <button type="button" onClick={onClose} className={cn(btn, 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]')}>
            Cancel
          </button>
          {mode.kind !== 'revise' && (
            <button
              type="button"
              onClick={() => submit(false)}
              disabled={isSubmitting}
              className={cn(btn, 'border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)]')}
            >
              Save draft
            </button>
          )}
          <button
            type="button"
            onClick={() => submit(true)}
            disabled={isSubmitting}
            className={cn(btn, 'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)]')}
          >
            {isSubmitting && spinner}
            {mode.kind === 'revise' ? `Publish version ${mode.proposal.version + 1}` : 'Propose for approval'}
          </button>
        </>
      }
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          void submit(true);
        }}
        noValidate
      >
        {mode.kind === 'revise' && (
          <div className="flex items-start gap-2 p-2.5 rounded-lg bg-[var(--color-warning-muted)] border border-[var(--color-warning)]/25">
            <AlertTriangle className="w-3.5 h-3.5 mt-px text-[var(--color-warning)] flex-shrink-0" aria-hidden="true" />
            <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)]">
              Previous approvals will be cleared because this proposal changed. Both founders must approve version{' '}
              {mode.proposal.version + 1} again.
            </p>
          </div>
        )}

        <TextField id={`${id}-title`} label="Title" value={title} onChange={setTitle} max={LIMITS.title} placeholder="e.g. Initial Customer Segment" error={err('title')} />
        <TextField
          id={`${id}-direction`}
          label="Proposed direction"
          hint="What should the startup commit to?"
          value={direction}
          onChange={setDirection}
          max={LIMITS.direction}
          placeholder="e.g. Focus on university student founders as our first customer segment."
          multiline
          error={err('direction')}
        />
        <TextField
          id={`${id}-next`}
          label="Next action"
          hint="The concrete step you'll take if this is approved."
          value={nextAction}
          onChange={setNextAction}
          max={LIMITS.nextAction}
          placeholder="e.g. Interview five university student founders."
          multiline
          error={err('nextAction')}
        />

        <KnowledgePicker
          legend="Supporting findings"
          hint="Shared findings that support this direction."
          items={items}
          selected={supporting}
          disabledIds={conflicting}
          onToggle={(x) => toggle(supporting, setSupporting, x)}
        />

        <div className="space-y-1.5">
          <KnowledgePicker
            legend="Conflicting findings"
            hint="Shared findings that point another way. Including them helps your co-founder review honestly."
            items={items}
            selected={conflicting}
            disabledIds={supporting}
            onToggle={(x) => toggle(conflicting, setConflicting, x)}
          />
          {suggested.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-[var(--color-text-muted)]">
              <span>Known conflicts with your supporting findings:</span>
              {suggested.map((x) => (
                <button
                  key={x}
                  type="button"
                  onClick={() => setConflicting((c) => [...c, x])}
                  className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md border border-[var(--color-warning)]/40 text-[var(--color-warning)] font-mono text-[10px] hover:bg-[var(--color-warning-muted)]"
                  aria-label={`Add ${x} as a conflicting finding`}
                >
                  <Plus className="w-2.5 h-2.5" aria-hidden="true" /> {x}
                </button>
              ))}
            </div>
          )}
        </div>

        {mode.kind === 'revise' && (
          <TextField
            id={`${id}-note`}
            label="What changed? (optional)"
            value={changeNote}
            onChange={setChangeNote}
            max={LIMITS.changeNote}
            placeholder="e.g. Added working professionals to the interviews."
            error={err('changeNote')}
          />
        )}
        <button type="submit" className="hidden" aria-hidden="true" tabIndex={-1} />
      </form>
    </Dialog>
  );
}
