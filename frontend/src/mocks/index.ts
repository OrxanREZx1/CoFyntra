// ============================================================
// Confyntra — Mock Data
// Phase 1: Realistic demo data for the FoundersPath project.
// IMPORTANT: This data is NOT persisted. Replace with real API
// calls when the backend contract (docs/sync_contract.md) is finalized.
// ============================================================

import type {
  Founder,
  Project,
  ChatMessage,
  KnowledgeItem,
  KnowledgeConflict,
  DecisionProposal,
  CommittedStartupState,
  FounderId,
} from '@/types';

// ─── Founders ───────────────────────────────────────────────

export const FOUNDERS: Record<string, Founder> = {
  'founder-a': {
    id: 'founder-a',
    name: 'Alex Chen',
    avatar: 'AC',
    color: 'var(--color-founder-a)', // theme token
  },
  'founder-b': {
    id: 'founder-b',
    name: 'Jordan Park',
    avatar: 'JP',
    color: 'var(--color-founder-b)', // theme token
  },
};

export const FOUNDERS_LIST: Founder[] = Object.values(FOUNDERS);

// ─── Project ────────────────────────────────────────────────

export const MOCK_PROJECT: Project = {
  id: 'project-foundersppath-01',
  name: 'FoundersPath',
  tagline: 'AI tools for startup founders',
  createdAt: '2026-10-15T09:00:00Z',
};

// ─── Chat Messages — Founder A (Alex) ───────────────────────
// Private to Founder A. Must never appear in Founder B's view.

export const FOUNDER_A_MESSAGES: ChatMessage[] = [
  {
    id: 'msg-a-01',
    role: 'user',
    content:
      "I've been thinking about who our initial customers should be. We built FoundersPath to help founders, but I want to narrow down. Who do you think fits best?",
    createdAt: '2026-11-08T10:05:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
  {
    id: 'msg-a-02',
    role: 'assistant',
    content:
      "Great question. Based on the problem you've described — fragmented AI-assisted research between co-founders — the most likely early adopters are **first-time startup founders** who are actively in the discovery phase. University student founders often have more time to experiment with new tools, lower switching costs, and high motivation to validate their ideas quickly. They're also often remote-working with a co-founder, which makes the synchronization problem more acute.\n\nThat said, this is a hypothesis. Have you talked to any student founders recently?",
    createdAt: '2026-11-08T10:05:45Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
  {
    id: 'msg-a-03',
    role: 'user',
    content:
      "Yes! I spoke with 3 student teams at the university accelerator last week. All of them had exactly this problem — one founder doing deep AI research on customers while the other builds, then they sync up and realize they've been working on different assumptions.",
    createdAt: '2026-11-08T10:08:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
  {
    id: 'msg-a-04',
    role: 'assistant',
    content:
      "That's strong **reported evidence** — not yet verified, but it's direct customer observation, which is valuable. Three teams with the same pattern suggests this isn't coincidental.\n\nKey things to validate next:\n1. **Willingness to pay** — student budgets are tight. Would they pay, or expect a free tier?\n2. **Frequency of the problem** — how often do they sync? Daily? Weekly?\n3. **Current workaround** — what are they using now? Notion docs, Slack threads?\n\nI'd recommend running 3 more interviews with the specific question: *'What do you do when you and your co-founder realize you've been researching different directions?'*",
    createdAt: '2026-11-08T10:09:15Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
  {
    id: 'msg-a-05',
    role: 'user',
    content:
      "They all said they use Notion or just a shared doc, but it becomes a mess quickly. One team said they had a 2-week period where they both thought they were building for different customers. That's a real pain point.",
    createdAt: '2026-11-08T10:11:30Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
  {
    id: 'msg-a-06',
    role: 'assistant',
    content:
      "That 2-week divergence story is **gold**. That's exactly the kind of specific, emotionally resonant moment that makes for a compelling problem narrative. Document that verbatim — it could become your founding story.\n\n**My recommendation:** Position university student co-founder teams as your initial target segment. They have the problem acutely, they're accessible for iteration, and early success with them builds toward the broader professional founder market.",
    createdAt: '2026-11-08T10:12:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-a',
  },
];

// ─── Chat Messages — Founder B (Jordan) ─────────────────────
// Private to Founder B. Must never appear in Founder A's view.

export const FOUNDER_B_MESSAGES: ChatMessage[] = [
  {
    id: 'msg-b-01',
    role: 'user',
    content:
      "I'm trying to figure out if we're targeting the right customers. I keep thinking university students might not be the right fit — they don't have money and their startups don't last. What do you think?",
    createdAt: '2026-11-08T14:20:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
  {
    id: 'msg-b-02',
    role: 'assistant',
    content:
      "That's a valid concern. University students as a segment have advantages (accessible, eager to try tools, representative of the problem), but also real challenges: lower willingness to pay, high churn as they graduate or pivot, and startups that don't survive long enough to become advocates.\n\nAn alternative worth considering: **working professionals** (25–40) who are founding their second or third startup, or doing it alongside employment. They tend to have:\n- More stable financial situations (can pay)\n- More structured ways of working (more likely to pay for productivity tools)\n- Stronger sense of urgency (less tolerance for wasted time)\n\nHave you done any interviews with that group?",
    createdAt: '2026-11-08T14:21:30Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
  {
    id: 'msg-b-03',
    role: 'user',
    content:
      "Actually yes — I interviewed 5 people last week. All working professionals, all with side startup projects. 4 out of 5 said they had experienced exactly this — their co-founder was doing research in a completely different direction without realizing it.",
    createdAt: '2026-11-08T14:24:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
  {
    id: 'msg-b-04',
    role: 'assistant',
    content:
      "Four out of five is a compelling signal. For working professionals, the problem might be even more acute because they have **less time to sync**, which means divergence can compound over weeks or months before it surfaces.\n\nImportant to note: this is **reported evidence from interviews** — it confirms the problem exists in this segment, but you'd want to validate that they'd actually use and pay for a tool like FoundersPath.\n\nOne key question: did any of them mention how much they'd be willing to pay? Even ballpark figures help.",
    createdAt: '2026-11-08T14:25:45Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
  {
    id: 'msg-b-05',
    role: 'user',
    content:
      'Two of them mentioned they currently pay for Notion Teams and Linear — so definitely open to paying for productivity tools. One specifically said "$20–30/month per founder feels right for something that saves that kind of confusion."',
    createdAt: '2026-11-08T14:27:00Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
  {
    id: 'msg-b-06',
    role: 'assistant',
    content:
      "That price anchor is very useful — $20–30/month is in the range of professional SaaS tools they already buy. That's meaningful willingness-to-pay signal.\n\n**Summary so far:** Working professionals as a segment show strong problem-market fit with early willingness-to-pay signals. This may warrant prioritizing over student founders, especially if monetization is important early on. I'd recommend 5 more interviews to confirm the pattern and test whether they'd actually use a tool in their workflow.",
    createdAt: '2026-11-08T14:28:30Z',
    projectId: MOCK_PROJECT.id,
    founderId: 'founder-b',
  },
];

// ─── Shared Knowledge (L2) — DEMO ───────────────────────────
// Illustrative demo findings, NOT verified market research. Excerpts are
// the short, approved-to-share quotes a founder chose to surface — never
// full private transcripts. Source refs are opaque provenance identifiers.

export const MOCK_KNOWLEDGE_ITEMS: KnowledgeItem[] = [
  {
    id: 'K-101',
    projectId: MOCK_PROJECT.id,
    category: 'Target Customer',
    finding: 'University students are likely to be our first customers.',
    contributorId: 'founder-a',
    contributorName: 'Alex Chen',
    sourceRef: 'src-a-7f3c',
    supportingExcerpt: 'I believe student founders struggle most with knowing what to do next.',
    createdAt: '2026-11-08T10:06:00Z',
    epistemicStatus: 'hypothesis',
  },
  {
    id: 'K-102',
    projectId: MOCK_PROJECT.id,
    category: 'Evidence',
    finding: 'Working professionals reported difficulty deciding their next startup action.',
    contributorId: 'founder-b',
    contributorName: 'Jordan Park',
    sourceRef: 'src-b-2d91',
    supportingExcerpt:
      'I spoke with professionals who are building businesses, and several described that problem.',
    createdAt: '2026-11-08T14:24:00Z',
    epistemicStatus: 'reported-evidence',
  },
  {
    id: 'K-103',
    projectId: MOCK_PROJECT.id,
    category: 'Problem',
    finding: 'Co-founders can work from different customer assumptions for weeks without noticing.',
    contributorId: 'founder-a',
    contributorName: 'Alex Chen',
    sourceRef: 'src-a-91be',
    supportingExcerpt:
      'One team said they had a 2-week period where they both thought they were building for different customers.',
    createdAt: '2026-11-08T10:11:30Z',
    epistemicStatus: 'reported-evidence',
  },
  {
    id: 'K-104',
    projectId: MOCK_PROJECT.id,
    category: 'Market',
    finding: 'Some interviewed professionals already pay for productivity tools and named $20–30/month as a fair price.',
    contributorId: 'founder-b',
    contributorName: 'Jordan Park',
    sourceRef: 'src-b-44a0',
    supportingExcerpt: '$20–30/month per founder feels right for something that saves that kind of confusion.',
    createdAt: '2026-11-08T14:27:00Z',
    epistemicStatus: 'reported-evidence',
  },
  {
    id: 'K-105',
    projectId: MOCK_PROJECT.id,
    category: 'Assumption',
    finding: 'Student founders may adopt new tools faster because their switching costs are low.',
    contributorId: 'founder-a',
    contributorName: 'Alex Chen',
    sourceRef: 'src-a-c210',
    supportingExcerpt: 'University student founders often have more time to experiment with new tools.',
    createdAt: '2026-11-08T10:05:45Z',
    epistemicStatus: 'unverified-ai-claim',
  },
  {
    id: 'K-106',
    projectId: MOCK_PROJECT.id,
    category: 'Target Customer',
    finding: 'Working professionals may experience the problem more intensely than students.',
    contributorId: 'founder-b',
    contributorName: 'Jordan Park',
    sourceRef: 'src-b-8e57',
    supportingExcerpt: 'I keep thinking university students might not be the right fit.',
    createdAt: '2026-11-08T14:20:00Z',
    epistemicStatus: 'hypothesis',
  },
  {
    // Deliberately has no provenance: exercises the UI's neutral fallbacks.
    id: 'K-107',
    projectId: MOCK_PROJECT.id,
    category: 'Assumption',
    finding: 'Divergence may compound for longer among working professionals, who have less time to sync.',
    contributorId: 'founder-b',
    contributorName: 'Jordan Park',
    sourceRef: null,
    supportingExcerpt: null,
    createdAt: '2026-11-08T14:25:45Z',
    epistemicStatus: 'hypothesis',
  },
  {
    id: 'K-108',
    projectId: MOCK_PROJECT.id,
    category: 'Market',
    finding: 'It is unclear whether student founders would pay for the product.',
    contributorId: 'founder-a',
    contributorName: 'Alex Chen',
    sourceRef: 'src-a-5b6d',
    supportingExcerpt: 'Student budgets are tight — none of the teams said whether they would pay.',
    createdAt: '2026-11-08T10:12:00Z',
    epistemicStatus: 'inconclusive',
  },
];

/**
 * Predefined demo conflict relationships. In production these come from the
 * backend; the frontend never infers conflicts itself.
 */
export const MOCK_KNOWLEDGE_CONFLICTS: KnowledgeConflict[] = [
  {
    id: 'CF-1',
    projectId: MOCK_PROJECT.id,
    itemIds: ['K-101', 'K-106'],
    explanation: 'These findings point to different initial customer segments.',
  },
];

// ─── Decision Proposals (L3) — DEMO seed ────────────────────
// Starts with one proposal awaiting both founders and NO committed direction.

export const MOCK_DECISION_PROPOSALS: DecisionProposal[] = [
  {
    id: 'P-001',
    projectId: MOCK_PROJECT.id,
    createdByFounderId: 'founder-a',
    title: 'Initial Customer Segment',
    direction: 'Focus on university student founders as our first customer segment.',
    nextAction:
      'Interview five university student founders and compare their problem urgency against working professionals.',
    supportingKnowledgeIds: ['K-101'],
    conflictingKnowledgeIds: ['K-106', 'K-102'],
    version: 1,
    status: 'PENDING_APPROVAL',
    approvals: [],
    changeNote: null,
    history: [],
    committedRevision: null,
    createdAt: '2026-11-08T15:30:00Z',
    updatedAt: '2026-11-08T15:30:00Z',
  },
];

// ─── Committed Startup State (L3) — DEMO seed ───────────────

export const MOCK_COMMITTED_STATE: CommittedStartupState = {
  projectId: MOCK_PROJECT.id,
  revision: 0, // nothing committed yet
  decision: null,
};

// ─── Mock Message Maps (by founder) ─────────────────────────

export const CHAT_MESSAGES_BY_FOUNDER: Record<FounderId, ChatMessage[]> = {
  'founder-a': FOUNDER_A_MESSAGES,
  'founder-b': FOUNDER_B_MESSAGES,
};

