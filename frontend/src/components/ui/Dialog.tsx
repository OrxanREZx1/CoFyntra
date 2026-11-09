'use client';

import { useEffect, useId, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DialogProps {
  title: string;
  description?: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
}

/**
 * Modal dialog built on the native <dialog> element (focus trapping, Esc to
 * close and inert background come from the browser). Mount it to open it and
 * unmount it to close it, so its contents start fresh each time.
 */
export function Dialog({ title, description, onClose, children, footer, className }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => dialog?.close();
  }, []);

  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      onCancel={(e) => {
        e.preventDefault(); // let React state drive closing
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose(); // backdrop click
      }}
      className={cn(
        'm-auto p-0 w-[calc(100%-1.5rem)] max-w-lg max-h-[min(92dvh,760px)] rounded-2xl',
        'bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] border border-[var(--color-border)] shadow-2xl',
        'backdrop:bg-[var(--color-overlay)] backdrop:backdrop-blur-[2px]',
        className,
      )}
    >
      <div className="flex flex-col max-h-[min(92dvh,760px)]">
        <header className="flex items-start gap-3 px-5 pt-4 pb-3 border-b border-[var(--color-border)] flex-shrink-0">
          <div className="flex-1 min-w-0">
            <h2 id={titleId} className="text-sm font-semibold">
              {title}
            </h2>
            {description && (
              <div id={descId} className="mt-0.5 text-xs text-[var(--color-text-muted)] leading-relaxed">
                {description}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="p-1 -mr-1 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
          >
            <X className="w-4 h-4" />
          </button>
        </header>
        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4">{children}</div>
        {footer && (
          <footer className="flex flex-wrap items-center justify-end gap-2 px-5 py-3 border-t border-[var(--color-border)] flex-shrink-0">
            {footer}
          </footer>
        )}
      </div>
    </dialog>
  );
}
