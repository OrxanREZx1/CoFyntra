'use client';

import type { DecisionProposal, FounderId } from '@/types';
import { decisionFor } from '../lib/proposalMachine';

export interface ProposalGroup {
  id: 'needs-you' | 'in-review' | 'drafts' | 'decided';
  title: string;
  proposals: DecisionProposal[];
}

const newestFirst = (a: DecisionProposal, b: DecisionProposal) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt);

/** Groups proposals by what the current founder needs to do. */
export function groupProposals(proposals: readonly DecisionProposal[], founderId: FounderId): ProposalGroup[] {
  const needsYou: DecisionProposal[] = [];
  const inReview: DecisionProposal[] = [];
  const drafts: DecisionProposal[] = [];
  const decided: DecisionProposal[] = [];

  for (const p of proposals) {
    if (p.status === 'COMMITTED') decided.push(p);
    else if (p.status === 'DRAFT') {
      // Unsubmitted drafts are only listed for their author.
      if (p.createdByFounderId === founderId) drafts.push(p);
    } else if (
      (p.status === 'PENDING_APPROVAL' && !decisionFor(p, founderId)) ||
      (p.status === 'CHANGES_REQUESTED' && p.createdByFounderId === founderId)
    ) {
      needsYou.push(p);
    } else inReview.push(p);
  }

  const groups: ProposalGroup[] = [
    { id: 'needs-you', title: 'Needs your decision', proposals: needsYou.sort(newestFirst) },
    { id: 'in-review', title: 'In review', proposals: inReview.sort(newestFirst) },
    { id: 'drafts', title: 'Your drafts', proposals: drafts.sort(newestFirst) },
    { id: 'decided', title: 'Decided', proposals: decided.sort(newestFirst) },
  ];
  return groups.filter((g) => g.proposals.length > 0);
}

interface ProposalListProps {
  groups: ProposalGroup[];
  renderProposal: (p: DecisionProposal) => React.ReactNode;
}

export function ProposalList({ groups, renderProposal }: ProposalListProps) {
  return (
    <div className="space-y-5">
      {groups.map((g) => (
        <section key={g.id} aria-labelledby={`proposals-${g.id}`} className="space-y-2.5">
          <h3
            id={`proposals-${g.id}`}
            className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]"
          >
            {g.title} · {g.proposals.length}
          </h3>
          {g.proposals.map(renderProposal)}
        </section>
      ))}
    </div>
  );
}
