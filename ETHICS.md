# Ethics & Guardrails

This is the most important file in the repository. Read it before pointing this tool at real people.

## Why this public version uses fictional data

This repository demonstrates a technique: take public membership metadata (who speaks where, who maintains what, who sits on which board), and from that structure alone compute who brokers between communities, who is best positioned to move information, and how information would spread under different framings. No message content required — just affiliations.

That technique is genuinely useful for understanding communities. It is also dual-use. A ranked list of real, named individuals labeled "optimal seeds for information diffusion" is exactly the kind of artifact that is harmless in a research context but easy to misread — or to weaponize — once it is public and stripped of context.

So we made a deliberate choice. The analysis was originally built and explored on a small set of **real** public figures. That version is kept private. **This public repository ships with an entirely fictional dataset** — invented people, companies, conferences, podcasts, and projects — hand-authored to reproduce the *structure* (clusters joined by a few brokers) without naming anyone real. You get to see exactly how the method works, and learn from it, without a real person being labeled and ranked in a public repo they never consented to.

If you fork this and substitute real data, the guardrails below are written for you.

## What this project is

A research and teaching tool for understanding the **structure of influence** in a community. It maps associations between individuals and the entities (conferences, podcasts, governance bodies, OSS projects) where reputations are visible, then computes centrality measures and simulates how information flows through the resulting network.

The intended use is structural understanding — answering questions like "which communities are most tightly intertwined?", "who bridges otherwise-disconnected groups?", and "if information had to travel through this network, what shape would the spread take?"

## What this project is not

- **Not a targeting list.** The "Seed Optimization" section identifies structurally important nodes. Treating that output as a list of real people to manipulate would be a misuse. Real influence campaigns are visible, traceable, and subject to professional reputation costs that the model does not represent.

- **Not a model of trust or credibility.** The SIR diffusion model treats every Informed node as equally likely to share information. Real people exercise judgment about what to repeat, who to repeat it to, and whether a piece of information is worth their reputation. The model does not capture any of that.

- **Not a model of platform dynamics.** Information spread through social media, group chats, and in-person conversation has very different mechanics than a uniform contagion process. The simulation is a conceptual abstraction, not a forecast.

## If you supply real data

Every association should be derived from genuinely public information — conference programs, published board/committee membership, podcast feeds, OSS governance documents, public newsletters. No private data, no scraping of protected sources, no inference from non-public signals.

And then:

1. **Do not use seed-optimization output to plan covert outreach.** If you contact someone the model identifies as a high-impact seed, the contact should be honest about your goals and your relationship to them. Framing information as "exclusive" or "gossip" specifically to exploit network structure is the kind of misuse this guardrail is for.

2. **Do not present model output as forecasts.** "This campaign will reach 80% of the community in two weeks" because a simulation produced that number would be misleading. The simulation produces numbers under stylized parameters with no calibration to real-world outcomes.

3. **Do not publish a real dataset alongside influence rankings without context.** A ranked list of "top brokers" in isolation invites being read as a value judgment about who matters. The structural metrics measure one specific thing — position in a co-affiliation graph — and nothing else. (This is the whole reason the public version of this repo is fictional.)

4. **Respect the people in any real dataset.** Public participation in conferences and podcasts is a signal of professional interest, not consent to be the subject of network analysis. If anyone asks to be removed, remove them.

5. **Consider keeping a real-data version private.** The combination of named individuals + structural rankings + diffusion targeting can be misread or weaponized if surfaced publicly, even when it is harmless in a research context.

## Limitations the dashboard does not surface

- **Sample bias.** Any hand-curated dataset is small and selected by its author. It will over-represent some communities and under-represent others. Centrality rankings are conditional on the sampled network and would shift dramatically with different inclusions. (The fictional dataset here is explicitly engineered to have brokers — real networks are messier.)

- **Edge weight sensitivity.** Treating "keynote speaker" as weight 3 and "attendee" as weight 1 is a defensible heuristic but ultimately arbitrary. Different weight schemes produce different rankings.

- **Recency.** Any real dataset is a snapshot. People change roles, leave companies, and join new communities. Re-curate before drawing current conclusions.

- **Missing edge types.** Personal friendships, mentorship, prior-collaborator history, geographic proximity, alma-mater ties — none of these are in the bipartite graph. Real influence networks include all of them.

## If something here makes you uncomfortable

That is a signal worth listening to. Stop, talk to someone you trust, and reconsider whether the use case you have in mind is one this tool should support.
