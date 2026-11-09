/**
 * DEMO ONLY — deterministic, scripted assistant replies.
 *
 * These are keyword-matched templates, not LLM output. They deliberately do
 * not claim to research anything, analyze data, or save shared knowledge.
 * Real inference is owned by the separate AI/backend track.
 */

interface DemoReplyRule {
  pattern: RegExp;
  reply: string;
}

const RULES: DemoReplyRule[] = [
  {
    pattern: /\b(interview(ed|s)?|spoke|talked|survey(ed)?|conversations?)\b/i,
    reply:
      'Interviews are a strong source of signal. To make them easier to compare later:\n\n' +
      '1. **How many people** did you speak with, and how did you find them?\n' +
      '2. Which problem did they describe **in their own words**?\n' +
      '3. What are they doing about it **today**?\n\n' +
      'Which of those answers surprised you most?',
  },
  {
    pattern: /\b(pay|pric(e|es|ing)|revenue|budget|monetiz\w*|subscription)\b/i,
    reply:
      "Pricing questions are easier once the buyer is clear. A few things to pin down:\n\n" +
      '- **Who pays** — the founder personally, or a company budget?\n' +
      '- What do they **already pay for** that solves a similar problem?\n' +
      '- Has anyone named a price unprompted?\n\n' +
      'Do you have any direct willingness-to-pay signal yet, or is this still an assumption?',
  },
  {
    pattern: /\b(students?|universit(y|ies)|college|professionals?|customers?|segments?|users?|audience|target)\b/i,
    reply:
      "Choosing an initial segment is a key decision. Before committing, it helps to compare candidates on:\n\n" +
      '- **Pain intensity** — how often and how badly do they hit the problem?\n' +
      '- **Reachability** — can you talk to ten of them this week?\n' +
      '- **Ability to pay** — now, or only later?\n\n' +
      'Which segment are you leaning toward, and what evidence is that based on so far?',
  },
  {
    pattern: /\b(compet\w*|alternatives?|notion|workarounds?|existing tools?)\b/i,
    reply:
      'Understanding current workarounds tells you what you need to beat. Worth clarifying:\n\n' +
      '- What tool or habit do people use **today**?\n' +
      '- Where exactly does it **break down**?\n' +
      '- What would make switching worth the effort?\n\n' +
      'Which workaround comes up most often in what you have heard?',
  },
  {
    pattern: /\b(problems?|pain( points?)?|struggl\w*|frustrat\w*|hard|difficult)\b/i,
    reply:
      'Let\'s sharpen the problem statement. Could you describe:\n\n' +
      '1. **Who** experiences it?\n' +
      '2. **When** does it show up — what triggers it?\n' +
      '3. **What it costs** them (time, money, missed opportunities)?\n\n' +
      'A recent, specific example would help most.',
  },
  {
    pattern: /\b(next steps?|what should (we|i) do|prioriti[sz]\w*|focus|plan)\b/i,
    reply:
      'To pick a next step, it helps to name the **riskiest assumption** first — the one that would hurt most if wrong.\n\n' +
      'What do you currently believe about your customer that you have the least evidence for?',
  },
];

const DEFAULT_REPLY =
  "Thanks — let's dig into that. To keep this grounded:\n\n" +
  '- Is this something you have **observed**, or a **hypothesis** for now?\n' +
  '- Who would it matter to most?\n\n' +
  'What would you need to see to feel confident about it?';

export function getDemoReply(userContent: string): string {
  return RULES.find((rule) => rule.pattern.test(userContent))?.reply ?? DEFAULT_REPLY;
}
