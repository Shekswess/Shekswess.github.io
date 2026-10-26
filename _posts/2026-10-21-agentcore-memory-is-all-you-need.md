---
layout: "post"
title: "(AgentCore) Memory is All You Need"
description: "What changes in cost, risk and quality when AI agents are designed to remember what truly matters?"
author: "shekswess"
date: "2026-10-21 00:00:00 +0000"
permalink: "/agentcore-memory-is-all-you-need.html"
categories: ["AI", "AWS"]
tags: ["AgentCore", "Memory", "Agentic AI", "Amazon Bedrock"]
image: "/assets/images/agentcore-memory-is-all-you-need/6ad8d9c4f33e3d00c21581e6_agentcore-memory-blog-post.webp"
source_title: "(AgentCore) Memory is All You Need"
source_url: "https://www.loka.com/blog/agentcore-memory-is-all-you-need"
source_authors: ["Andrej Brova", "Nina Cvetkovska", "Bojan Jakimovski", "Ivana Karalieva Petreska", "Nora Shannon Johnson"]
---

_**Original Source of the blog post: [(AgentCore) Memory is All You Need](https://www.loka.com/blog/agentcore-memory-is-all-you-need)**_

_**Written with my colleagues at Loka and AWS: Andrej Brova, Nina Cvetkovska, Ivana Karalieva Petreska, and Nora Shannon Johnson.**_

![AgentCore Memory: cost, risk, and quality when AI agents remember](/assets/images/agentcore-memory-is-all-you-need/6ad8d9c4f33e3d00c21581e6_agentcore-memory-blog-post.webp)

**Executive summary**

An AI agent that remembers what matters can cost less to run, work reliably in regulated environments, and become more useful over time. The difference comes down to how memory is designed; not how much information is stored. Memory shapes what the model sees each time it responds, influencing the cost, risk, and quality of every interaction.

Drawing on Loka's work with Amazon Bedrock AgentCore Memory, this paper explores four key aspects of designing memory for agents that deliver real value:

1. **Cost:** Memory is one of the biggest levers for controlling token costs. In one Loka engagement, redesigning memory reduced the context retrieved per turn by 70%. A more consistent prompt layout cut prompt-token costs by a further 80% from the second turn onward.
2. **Risk:** Decisions about what to retain, who can access it, and when to delete it start with the first record. AgentCore Memory provides the managed infrastructure, while your team defines the policies that govern how memory is used.
3. **Value:** Remembering a fact is only half the equation. Agents deliver value when they can apply what they remember to make better decisions and take more useful actions.
4. **Where to start:** Three practical steps to help you design memory effectively before your next agent pilot.

<figure><div><img alt="Retrieved context per turn fell from 1,518 tokens to 452 on a retrieval hit and 24 on a miss after the memory redesign." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b433dfd2fbd7fc195340_706d0aff.png" style="height: auto;"/></div></figure>

## 1. What it takes for agent pilots to reach production

Forecasts put 40% of enterprise applications shipping with task-specific agents by the end of 2026, up from under 5% a year earlier.<sup><a href="#source-1">1</a></sup> By a separate measure, only around 17% of organizations report having deployed agents at all, while more than 60% expect to within two years.<sup><a href="#source-2">2</a></sup>

<figure><div><img alt="Agent adoption: 17% of organizations have deployed agents, while more than 60% expect to deploy within two years." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa25813_5ad44eb2.png" style="height: auto;"/></div></figure>

Where agents are running, they run narrowly. Among enterprises above $1 billion in revenue, the share scaling agents reached 40% this year, up from 27% last year, while smaller organizations reported 22%, unchanged.<sup><a href="#source-3">3</a></sup> Whatever is driving that growth isn't reaching most of the market, and even across organizations of every size, the reach within a company stays shallow: in any single business function, no more than one organization in ten reported scaling agents, and most of those that did were live in only one or two functions.<sup><a href="#source-4">4</a></sup>

More than 40% of agentic projects are expected to be canceled by the end of 2027, for three stated reasons: escalating costs, unclear business value, and inadequate risk controls.<sup><a href="#source-5">5</a></sup> Model capability is not among them. Costs escalate because context grows, and memory is what decides what goes into it: the write path decides what is kept, while the retrieval path decides what is loaded on each call. Context size is an output of memory policy, not an independent knob. Risk controls fail because data retention decisions are made by engineers who do not know they are making them.

Context is what the model sees on a single call. Memory is what the system keeps between calls (the raw session record and the durable facts distilled from it) and selectively loads back into context when it is relevant. The industry's answer so far has been bigger context windows, but window size governs what a model can see, not what it can reliably use. Model accuracy degrades as input length increases, well before the advertised limit. Across eighteen frontier models tested, performance fell in every case.<sup><a href="#source-6">6</a></sup> The benchmark literature agrees: commercial assistants and long-context models show roughly a 30% accuracy drop retaining information across sustained interactions;<sup><a href="#source-7">7</a></sup> even at 10,000–100,000 tokens, well inside any current window, models struggle with recalling specific events correctly;<sup><a href="#source-8">8</a></sup> and in conversations averaging 300 turns across up to 35 sessions, models and retrieval both help, and both still lag human performance by a wide margin.<sup><a href="#source-9">9</a></sup> Cost moves the same direction: every token carried forward is a token paid for on every call, so a strategy of "keep more in context" makes the bill grow with the conversation.

Memory is the way out of that trade-off, but it is not a drop-in fix. Storing more is not the answer any more than holding more in context was: agents that nearly max out today's most-cited recall benchmarks still perform poorly the moment they have to act on what they recorded earlier in a real agentic loop.<sup><a href="#source-10">10</a></sup> Recall and use are different capabilities. Note that the 300-turn benchmark cited above, where models still lagged humans by a wide margin, is the same one now reported as close to saturated. That two-year gap is the argument: recall got solved, but use did not.

What works is selectivity on both sides of the store: deciding what is worth writing and deciding how much of it is worth loading on any given turn. That is the principle human memory runs on: consolidate what recurs, discard what doesn't, and reconstruct the rest from cues instead of storing everything. That argues for treating memory as a design problem rather than a capacity problem. Designed well, it is the largest cost lever available, holds up in regulated settings, and stores facts that actually change behavior.

**What this means for leaders:** The projects that reach production treat memory as a business decision, not a storage setting. It sets what each interaction costs, which data you keep and for how long, and whether the agent improves with use. Defining a memory strategy early is what allows a pilot to scale.

Amazon Bedrock AgentCore Memory provides that foundation as a managed service: short-term and long-term memory, isolation tied to verified identity, and configurable expiry. The sections that follow cover the design decisions that sit on top of it, where cost, risk, and value are actually decided.

## 2. Memory decides your token bill

Model tokens make up the large majority of a production agent's cost. A single complex turn can involve six to eight model calls against a context in the low hundreds of thousands of tokens. Memory determines whether that context carries five retrieved facts or an entire replayed history. It's where the biggest savings live when designed deliberately, and where spend grows quietly when it isn't, surfacing only on the invoice.

<figure><div><img alt="Illustrative comparison of growing cumulative costs without memory and a predictable per-turn budget with memory." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa25819_4b4d0cd4.png" style="height: auto;"/></div></figure>

### Where the money goes.

In the analytics deployment, a natural-language analytics agent running over a large, sensitive operational dataset, model inference accounted for roughly 95% of running cost. Memory operations, billed on records held and retrieval calls made, sat well below inference even at production volume. That split follows from the pricing model more than from this particular workload: every token is billed on every call and scales with the length of the conversation, while memory is billed on what is stored and what is retrieved, neither of which grows the same way. Any agent that carries its history forward in context lands in roughly the same place.

What moves the ratio is scale: inference dominates absolute spend at production volume, while at pilot volume the always-on costs dominate the per-request economics. The costs that deserved attention were the fixed ones that don't scale down: the managed vector store behind long-term memory carries an always-on charge, provisioned once per customer regardless of usage, which made it the largest fixed cost in the deployment. Query execution itself averaged around five seconds against total response times of roughly a minute. Nine-tenths of the wall clock, and the matching share of the bill, went to model reasoning rather than to task completion.<sup><a href="#source-12">12</a></sup>

### Why memory reduces cost.

Three mechanisms, in order of impact:

1. Retrieval replaces replay: without memory, continuity means carrying the whole conversation forward on every call, and a bounded retrieval step stands in for that, so cost stops growing with session length.
2. Bounded context enables bounded cost: a fixed retrieval budget makes cost per turn predictable, which matters more for pricing an agent than its average cost does.
3. Fewer turns per resolution: an agent that remembers a stated constraint doesn't ask again, and every avoided clarification removes a full turn, which is six to eight model calls rather than one.

### Where memory increases cost.

Memory isn't free, and the easiest cost to miss is its interaction with prompt caching. Providers cache the stable prefix of a prompt and bill it at a steep discount on later calls, cutting agentic cost substantially. A careless memory implementation, however, destroys that saving. If it injects freshly retrieved text into the cached prefix, that prefix is different on every turn, so it no longer matches the cache, and the full prompt is billed again. In that same analytics deployment, the memory layer was rebuilt for exactly this reason.

The first version pulled roughly 1,500 tokens of loosely matched context into every turn. Restructuring what was stored and applying a relevance threshold before results reached the model cut that to around 450 tokens on a match and 25 on a miss, reductions of roughly 70% and 98%, respectively. Holding the stable part of the prompt identical across turns reduced prompt-token cost by a further 80% from the second turn onward. Set against the pipeline it replaced, the memory-equipped agent answered the same benchmark set with median response time down about 15% and every prior correctness defect eliminated. Those figures were captured with the memory store deliberately emptied, so they are a floor rather than a ceiling.<sup><a href="#source-12">12</a></sup> One number moved the wrong way. Response-time variability rose, and the slowest requests got slower: retrieval adds a variable-latency step in front of the model, and on a miss you pay that latency for none of the token saving. The median improved; the tail did not.<sup><a href="#source-12">12</a></sup>

**What this means for leaders:** set against the pipeline it replaced, the memory-equipped agent answered the same questions faster, with every prior correctness defect eliminated, while loading less than a third of the retrieved memory per turn. That is the return on designing memory deliberately.

<figure><div><img alt="Retrieved-context token counts before and after the memory rebuild: 1,518 before, 452 on a hit, and 24 on a miss." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa2581c_08c204ba.png" style="height: auto;"/></div></figure>

## 3. Memory in regulated, data-sensitive industries

In regulated, data-sensitive industries, the decision that lowers cost often raises exposure. Long-term memory holding a personal interaction is a retention decision before it's an engineering one. A cache that lets one tenant's data surface in another's context is an isolation failure. A store that cannot forget on request will eventually appear in an audit finding. Most teams reach these questions after the memory layer is built, making them significantly more expensive to remediate.

**What this means for leaders:** For CIOs, CISOs, and compliance leads, memory is where AI governance becomes concrete. Only around one in five companies report mature agent governance today.<sup><a href="#source-11">11</a></sup> The organizations that settle retention, isolation, and deletion up front move through compliance review faster and reach production sooner.

### What must be decided before the first record is written.

Retention must be set by fact type rather than by store: identity and preferences may persist, session context should expire in hours, and anything derived from a transaction inherits that transaction's lifecycle. Isolation belongs below the application, where each tenant's namespace comes from their verified identity token rather than from a value the calling code passes in. Tenant separation handled in app code is one bug away from a cross-tenant leak. Deletion must target precisely the right records. Semantic stores match by similarity, and similarity applied to a deletion request will take out the target's neighbors along with it. Every derived record needs a link back to what produced it, or a user's "forget me" request leaves the derivation intact under a different name. And where compliance requires it, an agent whose behavior shifts under a silent model upgrade is a governance problem, controlled by pinning versions and testing against a stable baseline.

<figure><div><img alt="Memory pyramid separating ephemeral session state, user preferences stable across sessions, and durable derived facts." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa25820_996ad77d.png" style="height: auto;"/></div></figure>

### Platform support.

AgentCore Memory provides short-term and long-term memory as managed resources, with namespaces derived from token claims, policy-based access control evaluated at the infrastructure layer, and records that support configurable expiry. What it doesn't decide for you is retention policy, durability rules per fact type, or the retrieval logic above it, and that is where the regulated-environment work lives.

This split works in the customer's favor: AWS runs and scales the memory infrastructure, and your team keeps full control of the policies that govern it.

<figure><div><img alt="Agent reasoning loop retrieves and writes memory containing session state, preferences, and facts while answering a user question." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa25823_390e24e9.png" style="height: auto;"/></div></figure>

### In practice.

Both examples below are engagements as delivered, not recommended patterns. They are included because the two reached opposite conclusions about how much to remember.

A natural-language analytics agent answering questions over a large, sensitive operational dataset stores three things: a running conversation summary, plus (persisting across sessions) each user's stated preferences and a small set of facts about how they work. The two are held apart deliberately: a stated preference is given directly by the user, such as a preferred query language, while a working fact is inferred from behavior, such as always wanting the query before the summary. They earn different retention rules, because what a user stated can be shown back to them and corrected, while what was inferred cannot. Retention was set at 90 days, recorded explicitly as a placeholder rather than a policy, with a standing rule that personally identifiable data cannot enter memory until a retention policy is formally agreed with the customer's compliance function. Isolation runs below the application: each customer has a separate memory store, keyed by a verified identity token rather than by anything the calling code supplies. Compliance review produced three concrete changes:

1. Session credentials cut from thirty days to eight hours
2. Identifiable fields excluded from any generated summary
3. The storage layer is closed to public network access entirely, at a fixed always-on cost that does not scale down with usage.

Compliance review required those three. A separate, self-initiated change, made after a control built for exactly this purpose failed in practice, is the one worth passing on. A content-filter rule meant to catch identifiable data was removed, because in this domain the vocabulary of legitimate analysis reads the same as the vocabulary of the risk, and the filter was rejecting valid answers. Data protection moved to controlling who can reach the data; instead of screening answers for sensitive-looking words, the system now checks each requester's permissions when the query runs and returns only what that person is cleared to see. Content filtering and data protection are different controls, and using one to do the other's job produces a system nobody can use.<sup><a href="#source-12">12</a></sup>

A second agent reviews a completed piece of work against an external governing standard, in a domain where more than one standard can plausibly apply and only a human expert can say which one governs the case at hand. The agent surfaces the candidate standards; a human reviewer picks the one that applies; the agent completes the comparison. Memory here is deliberately minimal: case context and candidate standards for the duration of the pause, and nothing retained afterward. The flow is one request with one human checkpoint rather than a conversation, so persisting it would create a retention obligation for no functional gain. The reusable reference standard was pushed into a separate shared library holding only reference material, which is why it can be shared across customer sites while memory cannot. Isolation is per-site at the infrastructure layer, with a separate runtime, access role, and memory resource for each site, so a misrouted session cannot reach another site's data because the permission does not exist. Every governance question here was raised before anything was built: where model processing sits relative to the existing data-protection agreement, whether the system can explain how it reached a conclusion, and who owns the retention and encryption decisions.<sup><a href="#source-13">13</a></sup>

## 4. Getting memory right in production

Storing facts and using them are different problems. Most implementations solve the first and assume the second follows. That is how an agent ends up repeating a stored preference back to a user and then ignoring it in the same session.

**What this means for leaders:** For leaders, this is the difference between an agent that impresses in a demo, and one users still trust after a hundred sessions. Each pattern below is invisible on a dashboard, and each has a fix that costs far less at design time than in production.

On the way in, a working system resolves what the current turn is about, decides which memory applies, pulls more candidates than it needs, then ranks and trims to a fixed budget. It never runs a single search over everything and hopes the top match is relevant. On the way out, new facts are extracted, checked against what's already stored, given a durability rule appropriate to the type of facts they represent, and linked back to the interaction that produced them, so a later deletion request can find every trace.

Four failure patterns show up repeatedly once the retrieval itself is working correctly:

- **Exclusion failure.** A remembered "don't" is treated as a suggestion, not a rule, once enough other context points the other way. The fix is architectural: exclusions belong in code as filters applied before results reach the model, not as sentences asking the model to remember what to avoid.
- **Long-term compounding.** More stored records means noisier ranking and a rising bill, and it arrives without any warning signal. Nothing breaks; it just silently worsens. Durability must be a property of the fact type, decided at design time, rather than a risk untangled under pressure from a store nobody has audited.
- **Hidden cache cost.** Memory changes every turn because the query changes. Placed carelessly in the prompt, it quietly multiplies the bill even while every individual answer looks correct. A cache invalidates from the first changed token onward, so memory injected high in the prompt throws away everything below it, tool definitions and history included, while the system prompt above it still hits. The aggregate hit rate stays healthy while the billed token count climbs. Order the prompt by how often each part changes and put memory last.
- **Stale memory vs. live truth.** A remembered fact and a fresh system check look identical in the context, since both are just text, and a model will sometimes trust the more confident sounding one even when it's months old. Anything the user will act on, such as price, availability, status, or balance, must come from a live check made on the current turn, never from memory, no matter how confident the remembered version sounds.

## 5. Where to start

Each step below takes days, not quarters, and together they turn memory into a measurable advantage.

1. Map your fact types before writing the first record. Identity, preference, session, and transaction-derived facts each need a different retention rule, and deciding this at schema design costs an afternoon; retrofitting it means auditing a store nobody has looked at in a year.
2. Put the cost model in front of engineering before build, not after. The biggest line item is rarely the one people budget for, and the fixed, always-on costs are the ones that don't show up until the first invoice.
3. Pilot memory on one bounded workflow with a measured baseline. "With memory" versus "without" should be something you can show, not just claim.

**What this means for leaders:** Start with a bounded workflow, define retention rules before storing data, and agree on the cost model before engineering begins. These decisions make the pilot measurable and give the team a practical path to scale.

## 6. The path forward

Every step between an agent pilot and a production system that remembers has a known path. The infrastructure is managed, the failure patterns are documented, and the key design decisions fit into a schema review and a clear cost model.

Loka has walked that path with AWS across regulated industries, from healthcare and life sciences to financial services. As an AWS Premier Tier Partner and 2024 AWS Innovation Partner of the Year, Loka has delivered 500+ GenAI and agentic projects since 2023, with half now in or moving to production.

Planning an agent pilot, or scaling one that already works? Talk to your AWS account team or reach out to Loka to map your memory design against the three steps above.

**What this means for leaders:** Treat memory as part of the production plan from the start. A clear approach to cost, governance, and retrieval helps teams move from promising pilots to agents that are ready to scale.

### Appendix: Evidence

*Anonymized; measured internally.*

The baseline, without memory, was a fixed four-step pipeline: build the prompt, generate the query, run it, summarize. It held no conversation state and reused no prior work. We measured six architecture variants against a benchmark set, then ran an extended set of 54 requests including three multi-turn conversations. Relative to that baseline, the memory-equipped agent reduced median response time by roughly 15% and removed every correctness defect the baseline produced. Response-time variability rose: the slowest requests got slower.<sup><a href="#source-12">12</a></sup>

<figure><div><img alt="Typical and slowest response times: stateless baseline 56 and 101 seconds, with memory 48 and 140 seconds, and reuse layer empty 58 and 181 seconds." loading="lazy" src="/assets/images/agentcore-memory-is-all-you-need/6ad8b362ca3c4f20dfa2582a_b92c6976.png" style="height: auto;"/></div></figure>

Twelve conversational turns were then run with the reuse layer deliberately emptied, so any continuity had to come from memory alone. Follow-up questions that omitted their own subject, such as asking to filter results further or which entries recurred most often, resolved against the previous turn instead of making the user repeat themselves, and in the conversation where each question built on the last, response times fell at every turn, by more than half.. One result cuts the other way and is worth stating plainly: a question repeated word for word at the end of a session performed no better than the same question at the start. Remembering a conversation and reusing prior work are different capabilities.<sup><a href="#source-12">12</a></sup>

The second engagement, a human-in-the-loop assessment agent, has a defined baseline (state previously held by the calling application, now made native to the platform), but results aren't in yet. A quality baseline against a customer-supplied ground-truth set is a scoped deliverable there, not a completed measurement.<sup><a href="#source-13">13</a></sup>

### Sources

<ol role="list" start="1"><li id="source-1">Gartner, "Gartner Predicts 40% of Enterprise Apps Will Feature Task-Specific AI Agents by 2026," 26 August 2025. <a href="https://www.gartner.com/en/newsroom/press-releases/2025-08-26-gartner-predicts-40-percent-of-enterprise-apps-will-feature-task-specific-ai-agents-by-2026-up-from-less-than-5-percent-in-2025">Gartner Predicts 40% of Enterprise Apps Will Feature Task-Specific AI Agents by 2026, Up from Less Than 5% in 2025</a></li><li id="source-2">Gartner, Hype Cycle for Agentic AI, 2026. <a href="https://www.gartner.com/en/articles/hype-cycle-for-agentic-ai">2026 Hype Cycle for Agentic AI | Gartner</a></li><li id="source-3">McKinsey, State of AI Global Survey, 2026. <a href="https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai">The State of AI: Global Survey 2026 | McKinsey</a></li><li id="source-4">McKinsey, State of AI Global Survey, 2025. <a href="https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/november%202025/the-state-of-ai-2025-agents-innovation_cmyk-v1.pdf">The state of AI in 2025: Agents, innovation, and transformation</a></li><li id="source-5">Gartner, "Gartner Predicts Over 40% of Agentic AI Projects Will Be Canceled by End of 2027," 25 June 2025. <a href="https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027">Gartner Predicts Over 40% of Agentic AI Projects Will Be Canceled by End of 2027</a></li><li id="source-6">Hong, K.; Troynikov, A.; Huber, J. "Context Rot: How Increasing Input Tokens Impacts LLM Performance." Chroma, July 2025. <a href="https://www.trychroma.com/research/context-rot">Context Rot: How Increasing Input Tokens Impacts LLM Performance | Chroma</a></li><li id="source-7">Wu, D. et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." ICLR 2025, arXiv:2410.10813. <a href="https://arxiv.org/abs/2410.10813">[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory</a></li><li id="source-8">Huet, A.; Ben Houidi, Z.; Rossi, D. "Episodic Memories Generation and Evaluation Benchmark for Large Language Models." ICLR 2025, arXiv:2501.13121. <a href="https://arxiv.org/abs/2501.13121">[2501.13121] Episodic Memories Generation and Evaluation Benchmark for Large Language Models</a></li><li id="source-9">Maharana, A. et al. "Evaluating Very Long-Term Conversational Memory of LLM Agents." ACL 2024, arXiv:2402.17753. <a href="https://arxiv.org/abs/2402.17753">[2402.17753] Evaluating Very Long-Term Conversational Memory of LLM Agents</a></li><li id="source-10">He, Z. et al. "MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks." arXiv:2602.16313, 18 February 2026. <a href="https://arxiv.org/abs/2602.16313">[2602.16313] MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks</a></li><li id="source-11">"<a href="https://stateofopensource.ai/">The State of Open Source AI</a>," v1.1, September 2026, <a href="http://stateofopensource.ai">stateofopensource.ai<br/><br/></a><em>Internal engagements, anonymized, not independently verifiable and provided for transparency about where this data originates:<br/></em></li></ol>
<ol role="list" start="12"><li id="source-12">An internal Loka engagement: a natural-language analytics agent built against a large, sensitive operational dataset, including its memory-cost redesign and evaluation runs. The engagement was in active delivery, not yet fully in production, at the time of writing; figures are drawn from delivery and evaluation documentation.</li><li id="source-13">An internal Loka engagement: an assessment agent that compares a professional's work product against a reference standard, including a human-in-the-loop pause. Figures are drawn from the engagement's design and assessment documentation; deployment status was not independently confirmed at the time </li></ol>
