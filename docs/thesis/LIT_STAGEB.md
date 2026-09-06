# What writers may say — Stage B noise (lit agent → CoS, 6 Sep 2026)

Vault notes: `Literature_Review/Papers/Reed_1979.md` (abstract only);
`Papers/Sethi_2005.md` (CUDARE WP 13 Oct 2004 in hand);
`Concepts/StageB_Multiplicative_Growth_Noise.md`.
Keep-list **not** expanded. Gap 01/02 not demonstrated. Alex signs before freeze.

1. Reed (1979, *JEEM* 6(4):350–363) is a constant-escapement *feedback* policy on a stochastic stock–recruitment model — **publisher abstract only; PDF not in hand**. Do not quote the body.
2. The same abstract: in most cases the optimal stochastic escapement is no less than the deterministic one.
3. Do not cite Reed as the source of \((R_0,K,T,\texttt{rate\_tenths})=(8,40,36,9)\). Those integers are a method fact.
4. Sethi, Costello, Fisher, Hanemann & Karp (2005, *JEEM* 50(2); verified against CUDARE WP 13 Oct 2004) distinguish **growth**, **measurement**, and **implementation** uncertainty.
5. This thesis uses **growth** uncertainty only: true integer stock in the prompt; harvest as written (scarcity rule unchanged); \(\xi\) multiplies the growth increment.
6. Sethi WP §5.4: with only high growth uncertainty, constant escapement remains qualitatively optimal (Reed’s cell). Measurement error is what changes the policy shape. Do not import Sethi’s 42%/56% profit/extinction numbers or \(K=100\).
7. Sethi’s growth shock is multiplicative on recruitment. Footnote 3 attributes that *entry* to Reed (1979) — Sethi’s claim about Reed, not a Reed-PDF sentence.
8. Live \(\varepsilon\) is **not** Reed’s \(zG(s)\) and **not** Sethi’s uniform \(z^g\). It is \(\xi\in\{0.7,1.0,1.3\}\) equally likely, mean one, on the **integer logistic growth increment** at \((8,40,36,0.9)\), then round-half-even (`xi_tenths` 7/10/13).
9. \(\xi=1.0\) is bit-identical to the deterministic increment. Absorbing zero: \(\xi\) is never applied at post-harvest \(R=0\).
10. Not additive \(\pm 1\) on stock. Not \(\mathcal{N}(0,5)\) on \(S\sim 80\). `stochastic_cpr_env.py` is historical.
11. Piche et al. (arXiv:2511.19405, keep-list) use CRN so return variance in a group comes from actions, not the environment. Stage B paired \(\xi\) tables are that device for **measurement**, not a new algorithm.
12. Writers may call the law a discrete, mean-one, three-point analogue of multiplicative *growth* uncertainty. They may not say it *is* Reed’s optimal-escapement policy, Sethi’s three-shock manager, or a lognormal recruitment draw.
13. Cite Sethi from the working paper in hand (WP pages). Cite Reed from the publisher abstract only.

LIVE_FACTS still wins on numbers. DP rows in LIVE_FACTS are the comparison target for GPU arms.