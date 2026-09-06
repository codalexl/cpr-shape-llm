# Limitations

These constraints limit what the executed contrasts can be said to show.

**The constant-strategy matrix is the wrong efficiency benchmark.** Mutual take-1 returns 36; one (0,0) then mutual 3 returns 105. Take-3 is the sustainable harvest at high stock. Results language that treats take-3 as smash and take-1 as cooperation is calibrated to the reduction, not the DP. Test B last-epoch return 72.4 is the constant hawk–dove cell, not the best response 107.

**Leave-2 is a statistic for a near-constant policy class.** Opening 2 dies only if the continuer stays on 2. Opening 0 and opening 1 versus a frozen 2 are different basins (\(8\to 11\) vs \(8\to 9\)). Last-epoch \(\pi(a\mid R)\) at \(R=9\) is ~90% take-2; nobody learned 0-then-3.

**The shaping null is close to built-in until someone conditions on stock.** A rational shaper against a learner who only avoids death by opening 1 then taking 2 has about one unit to gain (72 versus 71). Slow-LR naive–naive already reproduces who-leaves-2 not swapping.

**Whitened Test A is a three-seed family, not a law.** Seed 0 last-epoch leave-2 2/15, return 11.3, last-three-epoch survival 5/45. Seeds 1–2 left opening 2 (87% and 82% leave-2 last3; survival 97/225 and 80/225). The landmine reading is seed-dependent.

**Three seeds, last-epoch counts out of 15.** 13/15 has a wide interval. Whole-run survival averages over learning and is supplementary. Matched long naive and shaper-e50 are one seed each.

**The critic is weakly trained** (`vf_coef=0.01`, \(\gamma=1\), \(\lambda=0.97\), new value head). Opening advantages are close to return minus a poor baseline; later 2s smear. Centre versus whiten is entangled with that.

**Stage B GPU packets are not in yet.** In-repo DP says (1,1) and hawk–dove live with probability 1 under ξ; (2,2) dies; open-1-then-2s survives 80%. Until the four arms exist, do not write a noise finding. Retired ±1 Test A is not a substitute.
