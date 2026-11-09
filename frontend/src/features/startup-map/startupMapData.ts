import type { EpistemicStatus, KnowledgeItem, CommittedStartupState } from '@/types';

export interface StartupMapFinding {
  label: string;
  status: string;
}

export interface StartupMapData {
  problem: string | null;
  approvedDirection: string | null;
  candidateSegments: StartupMapFinding[];
  marketFindings: string[];
  assumptions: string[];
  evidence: string[];
  nextAction: { text: string; status: 'Approved' | 'Proposed' } | null;
}

const STATUS_LABELS: Record<EpistemicStatus, string> = {
  hypothesis: 'Hypothesis',
  'reported-evidence': 'Reported',
  'verified-evidence': 'Verified',
  'unverified-ai-claim': 'Unverified',
  inconclusive: 'Inconclusive',
};

export function deriveStartupMapData(
  items: KnowledgeItem[],
  committedState: CommittedStartupState | undefined,
): StartupMapData {
  const firstFinding = (category: KnowledgeItem['category']) =>
    items.find((item) => item.category === category)?.finding ?? null;

  const committedDecision = committedState?.decision;

  return {
    problem: firstFinding('Problem'),
    approvedDirection: committedDecision?.direction ?? null,
    candidateSegments: items
      .filter((item) => item.category === 'Target Customer')
      .map((item) => ({ label: item.finding, status: STATUS_LABELS[item.epistemicStatus] })),
    marketFindings: items.filter((item) => item.category === 'Market').map((item) => item.finding),
    assumptions: items.filter((item) => item.category === 'Assumption').map((item) => item.finding),
    evidence: items.filter((item) => item.category === 'Evidence').map((item) => item.finding),
    nextAction: committedDecision ? { text: committedDecision.nextAction, status: 'Approved' } : null,
  };
}
