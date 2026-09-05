import numpy as np
from itertools import product
R0,G,CEIL,H,NA = 20,2,20,30,4

def trans(R,a1,a2):
    if a1+a2 <= R:
        r1,r2,Rh = a1,a2,R-(a1+a2)
    else:
        cap=R//2; r1,r2,Rh = min(a1,cap),min(a2,cap),0
    Rn = min(CEIL,Rh+G) if Rh>0 else 0
    return r1,r2,Rn

# ---------- (a) exact best response to a CONSTANT opponent ----------
print("=== BEST RESPONSE (dynamic programming) vs a constant opponent ===")
print("opp | BR value | constant-match value | BR opening moves (R=20,t=1..8)")
for opp in range(NA):
    V=np.zeros((H+2,CEIL+1)); PI=np.zeros((H+2,CEIL+1),dtype=int)
    for t in range(H,0,-1):
        for R in range(CEIL+1):
            best,arg=-1,0
            for a in range(NA):
                r1,_,Rn=trans(R,a,opp)
                v=r1+V[t+1][Rn]
                if v>best: best,arg=v,a
            V[t][R],PI[t][R]=best,arg
    # roll the BR policy forward to see the actual play
    R,seq=R0,[]
    for t in range(1,9):
        a=PI[t][R]; seq.append(a); _,_,R=trans(R,a,opp)
    const={0:0,1:30,2:18,3:14}[opp]
    print(f" {opp}  |   {V[1][R0]:>5.0f}  |        {const:>3}          | {seq}")

# ---------- (b) Markov Perfect Equilibrium by backward induction ----------
print("\n=== MARKOV PERFECT EQUILIBRIUM (backward induction, symmetric) ===")
V1=np.zeros((H+2,CEIL+1)); V2=np.zeros((H+2,CEIL+1))
POL=np.zeros((H+2,CEIL+1),dtype=int); MULTI=0; NOPURE=0
for t in range(H,0,-1):
    for R in range(CEIL+1):
        Q1=np.zeros((NA,NA)); Q2=np.zeros((NA,NA))
        for a1,a2 in product(range(NA),repeat=2):
            r1,r2,Rn=trans(R,a1,a2)
            Q1[a1,a2]=r1+V1[t+1][Rn]; Q2[a1,a2]=r2+V2[t+1][Rn]
        ne=[(a1,a2) for a1,a2 in product(range(NA),repeat=2)
            if Q1[a1,a2]>=Q1[:,a2].max()-1e-9 and Q2[a1,a2]>=Q2[a1,:].max()-1e-9]
        if not ne: NOPURE+=1; a1s=a2s=0
        else:
            sym=[e for e in ne if e[0]==e[1]]
            pool=sym if sym else ne
            if len(pool)>1: MULTI+=1
            a1s,a2s=max(pool,key=lambda e:Q1[e]+Q2[e])   # payoff-dominant selection
        V1[t][R],V2[t][R]=Q1[a1s,a2s],Q2[a1s,a2s]; POL[t][R]=a1s
print(f"states with NO pure stage-NE : {NOPURE} / {(CEIL+1)*H}")
print(f"states with MULTIPLE stage-NE: {MULTI} / {(CEIL+1)*H}")
print(f"MPE value at (R=20, t=1)     : {V1[1][R0]:.0f} each")
R,seq=R0,[]
for t in range(1,16):
    a=POL[t][R]; seq.append(a); _,_,R=trans(R,a,a)
print(f"MPE play from R=20, first 15 steps: {seq}")
print(f"pool level after those steps: {R}")
print("\nMPE policy a(R) at t=1 :", [POL[1][r] for r in range(CEIL+1)])
print("MPE policy a(R) at t=25:", [POL[25][r] for r in range(CEIL+1)])
print("MPE policy a(R) at t=30:", [POL[30][r] for r in range(CEIL+1)])
