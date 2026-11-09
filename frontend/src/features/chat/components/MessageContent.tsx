import { Fragment, type ReactNode } from 'react';

/**
 * Safe, minimal formatter for chat text. Builds React elements only — never
 * HTML strings — so message content cannot inject markup or scripts.
 *
 * Supported: paragraphs (blank line), line breaks, "- " / "* " bullet lists,
 * "1. " numbered lists, **bold**, *italic*, and `inline code`.
 */

const INLINE_PATTERN = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\s\n][^*\n]*\*)/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  return text.split(INLINE_PATTERN).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key} className="font-semibold">{part.slice(2, -2)}</strong>;
    }
    if (part.length > 2 && part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={key} className="px-1 py-0.5 rounded bg-[var(--color-bg-hover)] font-mono text-[0.85em]">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.length > 2 && part.startsWith('*') && part.endsWith('*')) {
      return <em key={key}>{part.slice(1, -1)}</em>;
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

function renderLines(lines: string[], keyPrefix: string): ReactNode[] {
  return lines.map((line, i) => (
    <Fragment key={`${keyPrefix}-${i}`}>
      {i > 0 && <br />}
      {renderInline(line, `${keyPrefix}-${i}`)}
    </Fragment>
  ));
}

const BULLET = /^\s*[-*]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;

export function MessageContent({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/);

  return (
    <div className="space-y-2.5">
      {blocks.map((block, b) => {
        const lines = block.split('\n').filter((l) => l.trim() !== '');
        if (lines.length === 0) return null;
        const key = `b${b}`;

        if (lines.every((l) => BULLET.test(l))) {
          return (
            <ul key={key} className="list-disc pl-5 space-y-1">
              {lines.map((l, i) => (
                <li key={i}>{renderInline(l.replace(BULLET, ''), `${key}-${i}`)}</li>
              ))}
            </ul>
          );
        }
        if (lines.every((l) => NUMBERED.test(l))) {
          return (
            <ol key={key} className="list-decimal pl-5 space-y-1">
              {lines.map((l, i) => (
                <li key={i}>{renderInline(l.replace(NUMBERED, ''), `${key}-${i}`)}</li>
              ))}
            </ol>
          );
        }
        return <p key={key}>{renderLines(lines, key)}</p>;
      })}
    </div>
  );
}
