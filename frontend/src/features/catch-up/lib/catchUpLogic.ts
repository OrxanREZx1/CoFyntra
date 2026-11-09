// Pure catch-up logic (type-only imports, no React, no mocks).
import type {
  CatchUpFinding,
  CatchUpProposalRef,
  CatchUpSummary,
  EpistemicStatus,
  FounderId,
  SharedKnowledgeSnapshot,
} from '@/types';

const STATUS_PHRASE: Record<EpistemicStatus, string> = {
  hypothesis: 'hypothesis',
  'reported-evidence': 'founder-reported evidence',
  'verified-evidence': 'verified evidence',
  'unverified-ai-claim': 'unverified AI claim',
  inconclusive: 'inconclusive finding',
};

/** Statuses presented as discoveries; the rest are assumptions to examine. */
const DISCOVERY_STATUSES: ReadonlySet<EpistemicStatus> = new Set([
  'reported-evidence',
  'verified-evidence',
  'inconclusive',
]);

function plural(n: number, one: string, many: string) {
  return `${n} ${n === 1 ? one : many}`;
}

/**
 * DEMO ONLY: deterministically assembles a catch-up from shared knowledge
 * records. It is template text, not AI output: findings are quoted verbatim
 * and every finding references real item IDs from the given snapshot.
 */
export function buildDemoCatchUp(args: {
  snapshot: SharedKnowledgeSnapshot;
  requestingFounderId: FounderId;
  coFounderId: FounderId;
  coFounderName: string;
  pendingProposals?: CatchUpProposalRef[];
}): CatchUpSummary {
  const { snapshot, requestingFounderId, coFounderId, coFounderName, pendingProposals } = args;
  const byId = new Map(snapshot.items.map((i) => [i.id, i]));

  const coItems = snapshot.items
    .filter((i) => i.contributorId === coFounderId)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
  const coIds = new Set(coItems.map((i) => i.id));

  const findings: CatchUpFinding[] = [];

  for (const item of coItems.filter((i) => DISCOVERY_STATUSES.has(i.epistemicStatus))) {
    findings.push({
      id: `discovery-${item.id}`,
      kind: 'discovery',
      text: `New ${STATUS_PHRASE[item.epistemicStatus]} from ${coFounderName} (${item.category.toLowerCase()}): “${item.finding}”`,
      knowledgeItemIds: [item.id],
    });
  }

  for (const item of coItems.filter((i) => !DISCOVERY_STATUSES.has(i.epistemicStatus))) {
    findings.push({
      id: `assumption-${item.id}`,
      kind: 'assumption',
      text: `${coFounderName}'s ${STATUS_PHRASE[item.epistemicStatus]} (${item.category.toLowerCase()}): “${item.finding}”`,
      knowledgeItemIds: [item.id],
    });
  }

  for (const conflict of snapshot.conflicts) {
    if (!conflict.itemIds.some((id) => coIds.has(id))) continue;
    const [a, b] = conflict.itemIds.map((id) => byId.get(id));
    if (!a || !b) continue;
    const owner = (id: string, founder: FounderId, name: string) =>
      founder === requestingFounderId ? `your finding ${id}` : `${name}'s finding ${id}`;
    const lead = owner(a.id, a.contributorId, a.contributorName);
    findings.push({
      id: `conflict-${conflict.id}`,
      kind: 'conflict',
      text: `${lead.charAt(0).toUpperCase()}${lead.slice(1)} and ${owner(b.id, b.contributorId, b.contributorName)} may point in different directions.`,
      knowledgeItemIds: [a.id, b.id],
    });
  }

  const discoveries = findings.filter((f) => f.kind === 'discovery').length;
  const assumptions = findings.filter((f) => f.kind === 'assumption').length;
  const conflicts = findings.filter((f) => f.kind === 'conflict').length;

  const summary =
    coItems.length === 0
      ? `${coFounderName} hasn't shared any findings yet.`
      : `${coFounderName} has shared ${plural(coItems.length, 'finding', 'findings')}: ` +
        `${plural(discoveries, 'discovery', 'discoveries')} and ${plural(assumptions, 'assumption', 'assumptions')} to examine` +
        (conflicts ? `, plus ${plural(conflicts, 'potential conflict', 'potential conflicts')} worth discussing.` : '.');

  const times = coItems.map((i) => i.createdAt).sort();
  return {
    projectId: snapshot.projectId,
    requestingFounderId,
    coFounderId,
    coFounderName,
    summary,
    findings,
    knowledgeItemIds: [...new Set(findings.flatMap((f) => f.knowledgeItemIds))],
    pendingProposals,
    coverage: { itemCount: coItems.length, from: times[0], to: times[times.length - 1] },
  };
}

export interface ReferenceValidation {
  summary: CatchUpSummary;
  /** Referenced IDs that don't exist in this project's shared knowledge. */
  unresolvedIds: string[];
}

/**
 * Ensures a catch-up (from any source) only references shared knowledge
 * items that exist in the selected project. Unknown references are removed;
 * findings left with no valid reference are dropped rather than shown
 * without provenance.
 */
export function validateCatchUpReferences(
  summary: CatchUpSummary,
  projectId: string,
  knownItemIds: ReadonlySet<string>,
): ReferenceValidation {
  if (summary.projectId !== projectId) {
    return {
      summary: { ...summary, projectId, summary: '', findings: [], knowledgeItemIds: [] },
      unresolvedIds: summary.knowledgeItemIds,
    };
  }
  const unresolved = new Set<string>();
  const findings = summary.findings
    .map((f) => {
      const valid = f.knowledgeItemIds.filter((id) => {
        const ok = knownItemIds.has(id);
        if (!ok) unresolved.add(id);
        return ok;
      });
      return { ...f, knowledgeItemIds: valid };
    })
    .filter((f) => f.knowledgeItemIds.length > 0);

  for (const id of summary.knowledgeItemIds) if (!knownItemIds.has(id)) unresolved.add(id);

  return {
    summary: {
      ...summary,
      findings,
      knowledgeItemIds: summary.knowledgeItemIds.filter((id) => knownItemIds.has(id)),
    },
    unresolvedIds: [...unresolved],
  };
}
