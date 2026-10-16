#!/usr/bin/env python3
"""MDEC v3.0 frozen research protocol implementation.

Implements the four occupation-specific labor markets, five experimental
conditions, finite committed package menus, qualification tiers, applicant and
employer utilities, employer commitment, conditional compatibility, disclosure
metrics, and paired-replication analysis described in MDEC v3.0.

The defaults are the frozen protocol. Use --smoke-test for a small validation
run, --pilot for the 20-replication calibration run, and --main for the main
experiment (200 replications initially, extended when MCSE > .005).
"""
from __future__ import annotations
import argparse, builtins, json, math, os, statistics
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Module-level parallel worker — must be defined at module scope so it is
# picklable by the 'spawn' multiprocessing context used on Windows.
# ─────────────────────────────────────────────────────────────────────────────
def _run_replication_worker(seed, scale, menu_scale,
                             quality_params_raw, quality_mixtures_raw,
                             rep, strategic_prevalence, markets_values):
    """Reconstruct a simulator in the worker process and run one replication.

    Parameters are plain Python scalars/dicts (fully picklable).
    MDECSimulator and Market are available because the full module is
    re-imported by each spawned worker process.
    """
    markets = None if markets_values is None else [Market(v) for v in markets_values]
    sim = MDECSimulator(seed, scale=scale, menu_scale=menu_scale)
    sim.quality_params   = {Market(k): v for k, v in quality_params_raw.items()}
    sim.quality_mixtures = {Market(k): v for k, v in quality_mixtures_raw.items()}
    return sim.run_replication(rep, strategic_prevalence, markets)

VERSION = "3.15-vacancy-repair"
MONTE_CARLO_STANDARD_ERROR_TARGET = 0.005
BASE_SEED = 20261011
PILOT_REPS = 2
MAIN_REPS = 20
MAX_ROUNDS = 5
Q_THRESHOLD = 70.0
P_THRESHOLD = 88.0
SINGLE_ACCEPT = .50
MULTI_ACCEPT = SINGLE_ACCEPT  # retained for backward compatibility; one reservation threshold is used
NONINFERIORITY_MARGIN = .02
IRS_MILEAGE_RATE = .67  # frozen synthetic constant; replace in empirical run metadata
IRS_MILEAGE_YEAR = 2024
DOLLAR_YEAR = 2026
OT_MULTIPLIER = 1.5
RN_MAX_ANNUAL_HOURS = 2600
RN_REGULAR_HOURS = 2080
STRATEGIC_PREVALENCE_LEVELS = (0.0, 0.10, 0.25)
RN_CALLBACK_PROBABILITY = 0.45
RN_CALLBACK_GAMMA_SHAPE = 2.0
RN_CALLBACK_GAMMA_SCALE = 1.0

class Market(str, Enum):
    STUDENT="student"; RN="rn"; PROFESSOR="professor"; CEO="ceo"
class Condition(str, Enum):
    TRAD="traditional"; TRAD_LOCK="traditional_lock"; MDEC="mdec_unlocked"; MDEC_LOCK="mdec_locked"; POSTED="posted_package"
class Tier(str, Enum):
    NOT_QUALIFIED="not_qualified"; QUALIFIED="qualified"; PREFERRED="preferred"
class Compat(str, Enum):
    INCOMPATIBLE="incompatible"; CONDITIONAL="conditional"; COMPATIBLE="compatible"

MARKET_SIZES = {
    Market.STUDENT:(1000,1000,5,5), Market.RN:(800,1000,4,5),
    Market.PROFESSOR:(300,150,8,4), Market.CEO:(120,100,10,3)
}
ARCHETYPES = {
 Market.STUDENT:{
  "income":{"hours":.50,"speed":.30,"wage":.15,"schedule":.05},
  "balanced":{"hours":.40,"speed":.25,"schedule":.20,"wage":.15},
  "low_commitment":{"schedule":.35,"hours":.30,"speed":.20,"wage":.15}},
 Market.RN:{
  "compensation":{"net_comp":.40,"effective_hourly":.30,"oncall":.20,"daily_travel":.10},
  "travel_burden":{"daily_travel":.35,"mileage":.30,"net_comp":.25,"oncall":.10},
  "lifestyle":{"daily_travel":.35,"mileage":.25,"effective_hourly":.25,"oncall":.15}},
 Market.PROFESSOR:{
  "compensation":{"salary":.60,"research":.25,"workload":.15},
  "research":{"research":.45,"salary":.35,"workload":.20},
  "workload":{"workload":.45,"salary":.35,"research":.20}},
 Market.CEO:{
  "wealth":{"total":.60,"equity":.20,"guaranteed":.10,"protection":.10},
  "equity":{"equity":.45,"total":.30,"authority":.15,"protection":.10},
  "risk":{"protection":.35,"guaranteed":.30,"total":.25,"authority":.10}}
}
TARGET_TIERS={Market.STUDENT:(.15,.50,.35),Market.RN:(.20,.55,.25),Market.PROFESSOR:(.25,.55,.20),Market.CEO:(.30,.55,.15)}

@dataclass
class Applicant:
    id:int; market:Market; archetype:str; weights:Dict[str,float]; latent_quality:float
    qualifications:Dict[str,float]; mandatory:Dict[str,bool]; thresholds:Dict[str,float]
    active:bool=True; hired:bool=False; accepted_vacancy:Optional[int]=None

@dataclass
class Package:
    values:Dict[str,float]

@dataclass
class Vacancy:
    id:int; market:Market; subtype:str; mandatory:Dict[str,bool]; qual_weights:Dict[str,float]
    q_menu:List[Package]; p_menu:List[Package]; q_default:int; p_default:int
    budget:float; active:bool=True; filled:bool=False; hired_applicant:Optional[int]=None
    closure_reason:Optional[str]=None; considered:set=field(default_factory=set); hold:list=field(default_factory=list)
    doctoral_program:bool=False; revision_count:int=0; commitment_violations:int=0; generation_attempts:int=0; generation_rejections:int=0
    dominance_violations:int=0; budget_violations:int=0; composition_violations:int=0

@dataclass
class EvalResult:
    applicant_id:int; vacancy_id:int; tier:Tier; score:float; compat:Compat
    package_index:Optional[int]; applicant_utility:float; employer_utility:float
    hard_pass:bool; disclosed_exact:int=0; disclosed_categorical:int=0
    ground_truth_viable:bool=False; human_escalation:bool=False; observed_compat:Optional[Compat]=None

@dataclass
class Offer:
    applicant_id: int
    vacancy_id: int
    package: Package
    utility: float
    employer_utility: float
    tier: Tier
    ground_truth_valid: bool

def clip(x,a=0.,b=1.): return float(min(b,max(a,x)))
def u_inc(x,h,a,p):
    if x<h:return 0.
    if x<a:return .5*(x-h)/max(1e-12,a-h)
    if x<p:return .5+.5*(x-a)/max(1e-12,p-a)
    return 1.
def u_dec(x,p,a,m):
    if x<=p:return 1.
    if x<=a:return 1-.5*(x-p)/max(1e-12,a-p)
    if x<=m:return .5-.5*(x-a)/max(1e-12,m-a)
    return 0.
def perturb_weights(base,rng):
    vals={k:max(0.,v+rng.normal(0,.03)) for k,v in base.items()}; s=sum(vals.values())
    return {k:v/s for k,v in vals.items()}
def truncnorm(rng,mean,sd,low,high,n=1):
    out=[]
    while len(out)<n:
        x=rng.normal(mean,sd,n*2)
        out.extend(x[(x>=low)&(x<=high)].tolist())
    return np.array(out[:n])
def lhs(rng,n,d):
    out=np.empty((n,d))
    for j in range(d): out[:,j]=(rng.permutation(n)+rng.random(n))/n
    return out

def stable_seed(*parts):
    import hashlib
    raw='|'.join(map(str,parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little') % (2**32-1)

class MDECSimulator:
    def __init__(self, seed=BASE_SEED, scale=1.0, menu_scale=1.0):
        self.seed=seed; self.scale=scale; self.menu_scale=menu_scale
        self.rng=np.random.default_rng(seed); self.current_replication=0
        self.quality_params={Market.STUDENT:[3.8,1.8],Market.RN:[3.2,2.1],Market.PROFESSOR:[2.8,2.4],Market.CEO:[2.4,2.8]}
        self.quality_mixtures={}
        self.metadata={"version":VERSION,"seed":seed,"dollar_year":DOLLAR_YEAR,"irs_rate":IRS_MILEAGE_RATE,"irs_year":IRS_MILEAGE_YEAR,"menu_scale":menu_scale,"privacy_scope":"output_only_compatibility_and_tier"}

    def auto_calibrate_quality(self, draws=20000):
        """Calibrate continuous latent-quality mixtures conditional on mandatory eligibility.

        The frozen 15/50/35 etc. tier targets are applied among applicants who
        satisfy mandatory credentials. Overall not-qualified shares additionally
        include mandatory failures and are reported separately.
        """
        components=[{"lo":0.05,"hi":0.69,"a":5.0,"b":2.5},{"lo":0.71,"hi":0.89,"a":2.2,"b":2.2},{"lo":0.91,"hi":0.995,"a":2.5,"b":2.0}]
        weight_sets={Market.STUDENT:[np.array([.25,.40,.25,.10])],Market.RN:[np.array([.25,.30,.25,.20])],Market.PROFESSOR:[np.array([.50,.15,.20,.15]),np.array([.35,.35,.15,.15]),np.array([.20,.50,.15,.15])],Market.CEO:[np.array([.30,.20,.20,.30])]}
        subtype_probs={Market.STUDENT:[1.],Market.RN:[1.],Market.PROFESSOR:[.35,.40,.25],Market.CEO:[1.]};mandatory_probs={Market.STUDENT:[.98,.90],Market.RN:[.98,.80,.85],Market.PROFESSOR:[.98,.90],Market.CEO:[.95,.80]}
        rng=np.random.default_rng(stable_seed(self.seed,"mixture_quality_calibration"));report={};all_pass=True
        for m,target in TARGET_TIERS.items():
            weights=np.array(target,dtype=float);weights=weights/weights.sum();self.quality_mixtures[m]={"weights":weights.tolist(),"components":components,"noise_sd":.015}
            comp=rng.choice(3,size=draws,p=weights);z=np.empty(draws)
            for c,spec in enumerate(components):
                ix=np.where(comp==c)[0];z[ix]=spec["lo"]+(spec["hi"]-spec["lo"])*rng.beta(spec["a"],spec["b"],len(ix))
            q=np.clip(z[:,None]+rng.normal(0,.015,(draws,4)),0,1);types=rng.choice(len(weight_sets[m]),size=draws,p=subtype_probs[m]);score=np.array([100*np.dot(q[i],weight_sets[m][types[i]]) for i in range(draws)])
            conditional=(np.mean(score<Q_THRESHOLD),np.mean((score>=Q_THRESHOLD)&(score<P_THRESHOLD)),np.mean(score>=P_THRESHOLD));mand=float(np.prod(mandatory_probs[m]));overall=(1-mand+mand*conditional[0],mand*conditional[1],mand*conditional[2]);passed=all(abs(x-y)<=.03 for x,y in builtins.zip(conditional,target));all_pass &= passed
            report[m.value]={"mixture_weights":weights.tolist(),"components":components,"predicted_conditional":tuple(map(float,conditional)),"predicted_overall":tuple(map(float,overall)),"target_conditional":target,"target_scope":"mandatory_eligible_applicants","within_tolerance":bool(passed)}
        self.metadata["quality_mixtures"]={m.value:v for m,v in self.quality_mixtures.items()};self.metadata["quality_calibration_report"]=report;self.metadata["quality_calibration_passed"]=bool(all_pass);return report

    def load_calibration(self,path):
        data=json.loads(Path(path).read_text()); qp=data.get("quality_params",{}); qm=data.get("quality_mixtures",{})
        for m in Market:
            if m.value in qp:self.quality_params[m]=list(map(float,qp[m.value]))
            if m.value in qm:self.quality_mixtures[m]=qm[m.value]
        self.metadata["quality_params"]={m.value:v for m,v in self.quality_params.items()};self.metadata["quality_mixtures"]={m.value:v for m,v in self.quality_mixtures.items()}
        self.metadata["quality_calibration_passed"]=bool(data.get("quality_calibration_passed",True))

    def sizes(self,m):
        a,v,c,cap=MARKET_SIZES[m]
        return max(2,int(a*self.scale)),max(1,int(v*self.scale)),c,cap

    def applicants(self,m):
        n,_,_,_=self.sizes(m); names=list(ARCHETYPES[m]); apps=[]
        for i in range(n):
            arch=names[i%3]; w=perturb_weights(ARCHETYPES[m][arch],self.rng)
            if m in self.quality_mixtures:
                mix=self.quality_mixtures[m];c=int(self.rng.choice(3,p=mix["weights"]));spec=mix["components"][c];z=spec["lo"]+(spec["hi"]-spec["lo"])*self.rng.beta(spec["a"],spec["b"]);noise_sd=mix.get("noise_sd",.015)
            else:
                alpha,beta=self.quality_params[m];z=self.rng.beta(alpha,beta);noise_sd=.12
            quals={f"q{k}":clip(z+self.rng.normal(0,noise_sd)) for k in range(4)}
            mandatory=self._mandatory(m)
            thresholds=self._thresholds(m,arch)
            apps.append(Applicant(i,m,arch,w,z,quals,mandatory,thresholds))
        self.rng.shuffle(apps)
        for i,a in enumerate(apps):a.id=i
        return apps

    def _mandatory(self,m):
        probs={Market.STUDENT:[.98,.90],Market.RN:[.98,.80,.85],Market.PROFESSOR:[.98,.90],Market.CEO:[.95,.80]}[m]
        return {f"m{i}":bool(self.rng.random()<p) for i,p in enumerate(probs)}

    def _thresholds(self,m,arch):
        if m==Market.STUDENT:
            return {"min_hours":10.,"adequate_hours":15.,"preferred_hours":20.,"min_wage":14.25,"adequate_wage":15.,"preferred_wage":15.25}
        if m==Market.RN:
            return {"min_net":85000.,"adequate_net":95000.,"preferred_net":110000.,"min_eff":40.,"adequate_eff":47.,"preferred_eff":55.,"max_drive":5.,"adequate_day":10.,"preferred_day":7.,"max_mileage":72000.,"adequate_mileage":60000.,"preferred_mileage":48000.,"preferred_oncall":2.,"adequate_oncall":4.,"max_oncall":8.}
        if m==Market.PROFESSOR:
            base=float(truncnorm(self.rng,186000,20000,140000,260000)[0]); mult={"compensation":1.05,"research":1.,"workload":.97}[arch]
            hard=base*mult
            return {"min_salary":hard,"adequate_salary":hard*1.05,"preferred_salary":hard+self.rng.uniform(25000,50000),"max_load":{"compensation":4,"research":3,"workload":3}[arch]}
        # CEO threshold relative to applicant market expectation, later compared to actual package
        market=self._ceo_market_value(); z=clip(self.rng.normal(.80,.12),.55,1.10)
        return {"min_total":market*z,"min_equity":.35 if arch=="equity" else 0.,"min_guaranteed":.20 if arch=="risk" else 0.,"min_protection":2 if arch=="risk" else 0,"min_authority":{"wealth":1,"equity":2,"risk":1}[arch]}

    def vacancies(self,m):
        _,n,_,_=self.sizes(m); return [self._vacancy(m,i) for i in range(n)]

    def _vacancy(self,m,i):
        subtype={Market.STUDENT:"fast_food",Market.RN:"home_infusion",Market.PROFESSOR:self.rng.choice(["4/4","3/3","2/2"],p=[.35,.40,.25]),Market.CEO:"r3000"}[m]
        mandatory={f"m{k}":True for k in range({Market.STUDENT:2,Market.RN:3,Market.PROFESSOR:2,Market.CEO:2}[m])}
        doctoral = bool((subtype=="2/2" and self.rng.random()<.80) or (subtype=="3/3" and self.rng.random()<.20)) if m==Market.PROFESSOR else False
        qw=self._qual_weights(m,subtype); budget, qmenu, proposed_pmenu, stats=self._menus(m,subtype,doctoral)
        pmenu=self._enforce_preferred_dominance(qmenu,proposed_pmenu,m)
        qd=self._default_index(qmenu, preferred=False); pd=self._default_index(pmenu, preferred=True)
        vac=Vacancy(i,m,subtype,mandatory,qw,qmenu,pmenu,qd,pd,budget,doctoral_program=doctoral)
        # Final budget-safe repair: any promoted preferred point that violates
        # the locked vacancy budget is replaced by its paired qualified point.
        # Repair q_menu first (should be within budget from generator, but guard)
        _safe_q = None
        for j, pkg in enumerate(list(vac.q_menu)):
            if self._package_budget_ok(vac, pkg):
                _safe_q = pkg
            elif _safe_q is not None:
                vac.q_menu[j] = Package(dict(_safe_q.values))
        # Repair p_menu: use matching q_menu package as fallback
        for j, pkg in enumerate(list(vac.p_menu)):
            if not self._package_budget_ok(vac, pkg):
                fallback = vac.q_menu[j] if j < len(vac.q_menu) else (vac.q_menu[0] if vac.q_menu else pkg)
                vac.p_menu[j] = Package(dict(fallback.values))
        vac.generation_attempts=stats.get("attempts",0); vac.generation_rejections=stats.get("rejections",0)
        vac.dominance_violations=self._count_dominance_violations(vac.q_menu,vac.p_menu,m)
        vac.budget_violations=self._count_budget_violations(vac)
        vac.composition_violations=self._count_composition_violations(vac)
        return vac

    def _qual_weights(self,m,subtype):
        if m==Market.STUDENT:return {"q0":.25,"q1":.40,"q2":.25,"q3":.10}
        if m==Market.RN:return {"q0":.25,"q1":.30,"q2":.25,"q3":.20}
        if m==Market.PROFESSOR:
            return {"4/4":{"q0":.50,"q1":.15,"q2":.20,"q3":.15},"3/3":{"q0":.35,"q1":.35,"q2":.15,"q3":.15},"2/2":{"q0":.20,"q1":.50,"q2":.15,"q3":.15}}[subtype]
        return {"q0":.30,"q1":.20,"q2":.20,"q3":.30}

    def _menus(self,m,subtype,doctoral=False):
        n=int((100 if m==Market.CEO else 50)*self.menu_scale); n=max(10,n)
        if m==Market.STUDENT:return (*self._student_menus(n), {"attempts":n*4,"rejections":0})
        if m==Market.RN:return self._rn_menus(n)
        if m==Market.PROFESSOR:return (*self._prof_menus(n,subtype,doctoral), {"attempts":n*5,"rejections":0})
        return (*self._ceo_menus(n), {"attempts":n*2,"rejections":0})

    def _student_menus(self,n):
        wp=15.; h0=self.rng.triangular(5,15,25); hl=max(5,h0-5); hu=min(25,h0+5); z=self.rng.uniform(.95,1.10); budget=wp*h0*z
        pts=lhs(self.rng,n*4,3); feasible=[]
        for a,b,c in pts:
            wage=13.5+3*a; hours=hl+(hu-hl)*b; schedule=c
            if wage*hours<=budget+1e-9: feasible.append(Package({"wage":wage,"hours":hours,"schedule":schedule,"delay":float(self.rng.integers(1,15))}))
            if len(feasible)>=n*2:break
        while len(feasible)<n*2:
            safe_hours=min(hu,max(hl,min(h0,budget/wp)))
            feasible.append(Package({"wage":wp,"hours":safe_hours,"schedule":.7,"delay":7.}))
        feasible.sort(key=lambda p:(p.values["wage"]-13.5)/3+(p.values["hours"]-hl)/max(1,hu-hl)+p.values["schedule"])
        return budget,feasible[:n],feasible[-n:]

    def _rn_menus(self,n):
        """Generate diverse, internally feasible home-infusion packages.

        Feasibility is imposed by construction rather than by generating mostly
        infeasible visit schedules and filling the menu with identical fallbacks.
        The generator still records residual rejections for diagnostics.
        """
        target = n * 2
        menus = []
        attempts = 0
        rejected = 0
        max_attempts = target * 40

        while len(menus) < target and attempts < max_attempts:
            attempts += 1

            # First draw a feasible annual workload and allocate it across paid tasks.
            active = float(self.rng.triangular(1850, 2200, 2550))
            callback_h = float(self.rng.triangular(20, 55, 110))
            documentation_h = float(self.rng.triangular(120, 180, 260))
            travel_h = float(self.rng.triangular(650, 900, 1150))
            treatment_h = active - callback_h - documentation_h - travel_h
            if treatment_h <= 450:
                rejected += 1
                continue

            visits = int(self.rng.integers(220, 341))
            average_treatment = treatment_h / visits
            average_documentation = documentation_h / visits
            if not (2.0 <= average_treatment <= 4.25 and .25 <= average_documentation <= 1.2):
                rejected += 1
                continue

            speed = float(self.rng.triangular(42, 50, 58))
            target_miles = float(travel_h * speed)
            if not (42000 <= target_miles <= 72000):
                rejected += 1
                continue

            # Daily burden is generated directly within the hard bounds.
            max_oneway = float(self.rng.triangular(1.5, 3.25, 5.0))
            max_day = float(self.rng.triangular(7.0, 9.5, 12.75))
            oncall = float(self.rng.triangular(1.0, 4.0, 8.0))
            opportunities = int(round(oncall * 12))

            rc = float(self.rng.uniform(40, 50))
            rt = rc * float(self.rng.uniform(.90, 1.05))
            rd = rc * float(self.rng.uniform(.95, 1.05))
            cat_hours = np.array([treatment_h, travel_h, documentation_h, callback_h])
            cat_rates = np.array([rc, rt, rd, rc])
            regular = min(active, RN_REGULAR_HOURS)
            overtime = max(0.0, active - RN_REGULAR_HOURS)
            blended = float(np.dot(cat_hours, cat_rates) / active)
            cash = (regular * blended + overtime * blended * OT_MULTIPLIER
                    + opportunities * self.rng.uniform(75, 150)
                    + self.rng.uniform(0, 7000))

            actual_cost = target_miles * float(self.rng.triangular(.55, .70, .90))
            compliant = bool(self.rng.random() < .90)
            rate = IRS_MILEAGE_RATE if compliant else IRS_MILEAGE_RATE - .10
            reimbursement = target_miles * rate
            net = cash - max(0.0, actual_cost - reimbursement)
            effective = net / active

            paid_travel = bool(self.rng.random() < .95)
            no_overnight = bool(self.rng.random() < .95)
            vals = {
                "net_comp": net,
                "effective_hourly": effective,
                "annual_mileage": target_miles,
                "daily_duration": max_day,
                "oncall": oncall,
                "callback_hours": callback_h,
                "paid_patient": 1.0,
                "paid_travel": float(paid_travel),
                "mileage_compliant": float(compliant),
                "no_overnight": float(no_overnight),
                "max_oneway": max_oneway,
                "annual_hours": active,
                "fallback": 0.0,
            }
            if (active > RN_MAX_ANNUAL_HOURS or max_day > 13 or max_oneway > 5.5):
                rejected += 1
                continue
            menus.append(Package(vals))

        # Diverse feasible fallback family. Fallbacks are varied rather than clones.
        fallback_profiles = [
            (90000, 44, 68000, 11.5, 6.5, 2450),
            (95000, 47, 60000, 10.0, 4.5, 2300),
            (100000, 49, 54000, 9.0, 3.5, 2180),
            (106000, 51, 48000, 8.0, 2.5, 2080),
        ]
        fallback_count = 0
        while len(menus) < target:
            net, eff, miles, day, oncall, hours = fallback_profiles[fallback_count % len(fallback_profiles)]
            jitter = self.rng.uniform(.98, 1.02)
            menus.append(Package({
                "net_comp": net * jitter,
                "effective_hourly": eff * jitter,
                "annual_mileage": miles * self.rng.uniform(.97, 1.03),
                "daily_duration": day * self.rng.uniform(.97, 1.03),
                "oncall": oncall * self.rng.uniform(.95, 1.05),
                "callback_hours": 10 * oncall * self.rng.uniform(.9, 1.1),
                "paid_patient": 1.0,
                "paid_travel": 1.0,
                "mileage_compliant": 1.0,
                "no_overnight": 1.0,
                "max_oneway": min(5.0, 1.0 + miles / 18000),
                "annual_hours": hours * self.rng.uniform(.98, 1.02),
                "fallback": 1.0,
            }))
            fallback_count += 1

        # Higher score is more applicant-favorable; on-call is a burden.
        menus.sort(key=lambda p: (
            p.values["net_comp"]
            + 2000 * p.values["effective_hourly"]
            - p.values["annual_mileage"]
            - 3000 * p.values["daily_duration"]
            - 2500 * p.values["oncall"]
        ))
        return 125000., menus[:n], menus[-n:], {
            "attempts": attempts,
            "rejections": rejected,
            "fallbacks": fallback_count,
        }
    def _prof_menus(self,n,subtype,doctoral):
        cfg={"4/4":(180000,22000,140000,240000,0,5000,0,5000,1.04,8e3,4),"3/3":(215000,32000,160000,300000,5000,20000,0,15000,1.10,12e3,3),"2/2":(258000,48000,180000,400000,15000,50000,10000,30000,1.18,18e3,2)}[subtype]
        mean,sd,lo,hi,rlo,rhi,slo,shi,mult,course_cost,load=cfg; smid=(lo+hi)/2; budget=mult*smid
        pts=lhs(self.rng,n*5,4); menus=[]
        for x in pts:
            salary=float(truncnorm(self.rng,mean,sd,lo,hi)[0]); research=rlo+(rhi-rlo)*x[1]; summer=slo+(shi-slo)*x[2]; release=int(x[3]*2.999)
            if subtype=="2/2" and research<min(.10*budget,rhi):continue
            if salary+research+summer+release*course_cost<=budget:
                burdens={"4/4":(1,.3,.75,0),"3/3":(.7,.6,.55,.2 if doctoral else 0),"2/2":(.45,1,.5,.8 if doctoral else 0)}[subtype]
                b=[clip(v+self.rng.uniform(-.1,.1)) for v in burdens]; workload=1-clip(.45*b[0]+.25*b[1]+.15*b[2]+.15*b[3])
                research_u=clip(.45*((research-rlo)/max(1,rhi-rlo))+.30*((summer-slo)/max(1,shi-slo))+.25*float(doctoral))
                menus.append(Package({"salary":salary,"research":research_u,"workload":workload,"load":load,"research_dollars":research,"summer":summer,"course_release":release}))
            if len(menus)>=n*2:break
        fallback_research=max(rlo,min(rhi,.10*budget)) if subtype=="2/2" else rlo
        fallback_salary=min(mean,budget-fallback_research-slo)
        while len(menus)<n*2: menus.append(Package({"salary":fallback_salary,"research":.5,"workload":.5,"load":load,"research_dollars":fallback_research,"summer":slo,"course_release":0}))
        menus.sort(key=lambda p:p.values["salary"]+40000*p.values["research"]+40000*p.values["workload"])
        return budget,menus[:n],menus[-n:]

    def _ceo_market_value(self):
        p=self.rng.random()
        if p<.25:return 3e6+(p/.25)*1e6
        if p<.5:return 4e6+((p-.25)/.25)*3e6
        if p<.75:return 7e6+((p-.5)/.25)*3e6
        return 10e6+10e6*((p-.75)/.25)**2

    def _ceo_menus(self,n):
        v=self._ceo_market_value(); qlo=max(3e6,.85*v); qhi=min(20e6,v); plo=max(3e6,.95*v); phi=min(20e6,1.2*v)
        def gen(lo,hi):
            arr=[]
            for _ in range(n):
                total=self.rng.uniform(lo,hi); g,b,e=self.rng.dirichlet([4,4,12]); prot=int(self.rng.choice(5,p=[.05,.15,.35,.30,.15])); auth=int(self.rng.integers(0,5))
                arr.append(Package({"total":total,"guaranteed":g,"bonus":b,"equity":e,"protection":prot,"authority":auth}))
            return arr
        return 20e6,gen(qlo,qhi),gen(plo,phi)

    def _directions(self,m):
        """Return fresh, immutable string-key tuples for dominance/adaptation."""
        if m == Market.STUDENT:
            return ("wage", "hours", "schedule"), ("delay",)
        if m == Market.RN:
            return ("net_comp", "effective_hourly"), (
                "annual_mileage", "daily_duration", "oncall",
                "max_oneway", "annual_hours"
            )
        if m == Market.PROFESSOR:
            return (
                "salary", "research", "workload", "research_dollars",
                "summer", "course_release"
            ), ("load",)
        if m == Market.CEO:
            return ("total", "protection", "authority"), ()
        raise ValueError(f"Unsupported market: {m!r}")

    @staticmethod
    def _package_from_menu_item(item, label, index):
        """Return one Package and reject/repair accidental tuple wrappers."""
        if isinstance(item, tuple):
            packages=[part for part in item if isinstance(part,Package)]
            if len(packages)!=1:
                raise TypeError(f"{label}[{index}] malformed tuple: {item!r}")
            item=packages[0]
        if not isinstance(item,Package):
            raise TypeError(f"{label}[{index}] expected Package, got {type(item).__name__}")
        if not isinstance(item.values,dict):
            raise TypeError(f"{label}[{index}].values expected dict, got {type(item.values).__name__}")
        return item

    def _enforce_preferred_dominance(self,qmenu,pmenu,m):
        if not pmenu:
            raise ValueError(f"Empty preferred menu for {m.value}")
        benefit_keys,burden_keys=self._directions(m)
        out=[]
        for i,q_item in enumerate(qmenu):
            q=self._package_from_menu_item(q_item,"qmenu",i)
            source=self._package_from_menu_item(pmenu[i%len(pmenu)],"pmenu",i%len(pmenu))
            q_values=dict(q.values)
            p_values=dict(source.values)
            for key in benefit_keys:
                if key in q_values and key in p_values:
                    p_values[key]=max(p_values[key],q_values[key])
            for key in burden_keys:
                if key in q_values and key in p_values:
                    p_values[key]=min(p_values[key],q_values[key])
            p=Package(p_values)
            if m==Market.CEO:
                shares=np.array([p.values["guaranteed"],p.values["bonus"],p.values["equity"]]);shares=np.maximum(shares,0);shares=shares/shares.sum();p.values["guaranteed"],p.values["bonus"],p.values["equity"]=shares.tolist();p.values["total"]=clip(p.values["total"],3e6,20e6)
            feasible=True
            if m==Market.STUDENT:
                promoted_cost=p.values["wage"]*p.values["hours"]
                qualified_cost=q_values["wage"]*q_values["hours"]
                feasible=promoted_cost<=qualified_cost*1.05+1e-6
            elif m==Market.PROFESSOR:
                cost=p.values["salary"]+p.values.get("research_dollars",0)+p.values.get("summer",0)+p.values.get("course_release",0)*18000
                qcost=q_values["salary"]+q_values.get("research_dollars",0)+q_values.get("summer",0)+q_values.get("course_release",0)*18000
                feasible=cost<=max(qcost*1.20,qcost+50000)
            if not feasible:
                p=Package(q_values)
            out.append(p)
        return out

    def _count_dominance_violations(self,qmenu,pmenu,m):
        benefit_keys,burden_keys=self._directions(m)
        n=0
        pair_count=min(len(qmenu),len(pmenu))
        for i in range(pair_count):
            q=self._package_from_menu_item(qmenu[i],"qmenu",i)
            p=self._package_from_menu_item(pmenu[i],"pmenu",i)
            q_values=q.values; p_values=p.values
            ok=(
                all(p_values.get(key,-np.inf)>=q_values.get(key,-np.inf)-1e-9 for key in benefit_keys if key in q_values)
                and all(p_values.get(key,np.inf)<=q_values.get(key,np.inf)+1e-9 for key in burden_keys if key in q_values)
            )
            n+=0 if ok else 1
        return n

    def _package_budget_ok(self,v,p):
        x=p.values
        if v.market==Market.STUDENT:return x["wage"]*x["hours"]<=v.budget+1e-6
        if v.market==Market.PROFESSOR:
            cost=x["salary"]+x.get("research_dollars",0)+x.get("summer",0)+x.get("course_release",0)*{"4/4":8000,"3/3":12000,"2/2":18000}[v.subtype]
            return cost<=v.budget+1e-6
        if v.market==Market.CEO:return 3e6-1e-6<=x["total"]<=20e6+1e-6
        return x.get("annual_hours",0)<=RN_MAX_ANNUAL_HOURS+1e-6 and x.get("net_comp",0)<=v.budget+1e-6

    def _count_budget_violations(self,v):
        """Count budget violations in p_menu only.
        q_menu violations are repaired before matching and tracked separately.
        """
        return sum(not self._package_budget_ok(v,p) for p in v.p_menu)
    def _count_composition_violations(self,v):
        if v.market!=Market.CEO:return 0
        return sum(abs(p.values["guaranteed"]+p.values["bonus"]+p.values["equity"]-1)>1e-10 for p in v.q_menu+v.p_menu)

    def _default_index(self, menu, preferred=False):
        if not menu: return 0
        # menus are ordered from less to more applicant-favorable by generators
        target=.65 if preferred else .35
        return min(len(menu)-1,max(0,int(round(target*(len(menu)-1)))))

    def _package_vector(self, p, market):
        keys={Market.STUDENT:["wage","hours","schedule"],Market.RN:["net_comp","effective_hourly","annual_mileage","daily_duration","oncall"],Market.PROFESSOR:["salary","research","workload"],Market.CEO:["total","guaranteed","equity","protection","authority"]}[market]
        return np.array([float(p.values[k]) for k in keys],dtype=float)

    def _nearest_feasible(self, menu, default_idx, feasible, market):
        arr=np.array([self._package_vector(p,market) for p in menu]); lo=arr.min(0); span=np.where(arr.max(0)-lo<1e-12,1,arr.max(0)-lo)
        base=(arr[default_idx]-lo)/span
        scored=[]
        for idx,au,eu in feasible:
            d=np.abs((arr[idx]-lo)/span-base).mean(); scored.append((d,-(au+eu),-au,idx,au,eu))
        _,_,_,idx,au,eu=min(scored); return idx,au,eu

    def adapt_vacancy(self,v,condition,rng,strategic=False):
        if condition in (Condition.TRAD_LOCK,Condition.MDEC_LOCK,Condition.POSTED):
            v.commitment_violations += 1; return
        benefits,burdens=self._directions(v.market); step=.10 if strategic else .05
        # Snapshot limits ONCE before iterating menus to avoid stale references.
        allmenus_snap = []
        for item in (v.q_menu + v.p_menu):
            if isinstance(item, Package) and isinstance(item.values, dict):
                allmenus_snap.append(item)
        all_keys_snap = set()
        for _p in allmenus_snap:
            all_keys_snap.update(_p.values.keys())
        limits_snap = {
            k: (
                builtins.min(_p.values[k] for _p in allmenus_snap if k in _p.values),
                builtins.max(_p.values[k] for _p in allmenus_snap if k in _p.values)
            )
            for k in all_keys_snap
        }
        for menu in (v.q_menu, v.p_menu):
            for p in menu:
                if not isinstance(p, Package) or not isinstance(p.values, dict):
                    continue
                old=dict(p.values)
                for k in benefits:
                    if k in p.values:
                        lo,hi=limits_snap.get(k,(p.values[k],p.values[k])); p.values[k]=p.values[k]-step*(p.values[k]-lo) if strategic else p.values[k]+step*(hi-p.values[k])
                for k in burdens:
                    if k in p.values:
                        lo,hi=limits_snap.get(k,(p.values[k],p.values[k])); p.values[k]=p.values[k]+step*(hi-p.values[k]) if strategic else p.values[k]-step*(p.values[k]-lo)
                if v.market==Market.CEO:
                    sh=np.maximum([p.values["guaranteed"],p.values["bonus"],p.values["equity"]],0);sh=sh/sh.sum();_g,_b,_e=float(sh[0]),float(sh[1]),float(sh[2]);p.values["guaranteed"],p.values["bonus"],p.values["equity"]=_g,_b,_e;p.values["total"]=float(clip(p.values["total"],3e6,20e6))
                if not self._package_budget_ok(v,p): p.values=old
        # Re-establish weak preferred-menu dominance after adaptation and
        # preserve all budget/composition constraints.
        # Repair q_menu first so _enforce_preferred_dominance uses clean values.
        for j, pkg in enumerate(list(v.q_menu)):
            if not self._package_budget_ok(v, pkg):
                # Roll back to a safe fallback: use the package from the other end
                # of the menu (least-favorable for applicant = most budget-safe).
                v.q_menu[j] = Package(dict(v.q_menu[0].values))
        v.p_menu=self._enforce_preferred_dominance(v.q_menu,v.p_menu,v.market)
        for j,pkg in enumerate(list(v.p_menu)):
            if not self._package_budget_ok(v,pkg): v.p_menu[j]=Package(dict(v.q_menu[j].values))
        v.revision_count += 1

    def observed_applicant(self,a,condition,rng):
        import copy
        obs=copy.deepcopy(a)
        if condition not in (Condition.TRAD,Condition.TRAD_LOCK): return obs
        # omission, not distortion: omitted hard constraints become permissive; omitted weights are equalized
        for k in list(obs.thresholds):
            if rng.random()>.85:
                if k.startswith("min_"): obs.thresholds[k]=0
                elif k.startswith("max_"): obs.thresholds[k]=float("inf")
        kept={k:v for k,v in obs.weights.items() if rng.random()<.60}
        if not kept: kept={next(iter(obs.weights)):1.0}
        strengths={k:(v if rng.random()<.40 else 1.0) for k,v in kept.items()}; sm=sum(strengths.values());
        if sm<=0: strengths={next(iter(kept)):1.0}; sm=1.0
        obs.weights={k:v/sm for k,v in strengths.items()}
        return obs

    def posted_self_screen(self,a,v,round_no):
        # applicant sees summary defaults and visible ranges, then chooses whether to enter evaluation
        candidates=[v.q_menu[v.q_default],v.p_menu[v.p_default]]
        return any(self.hard_pass(a,p) and self.applicant_utility(a,p,round_no)>=.50 for p in candidates)

    def classify(self,a,v):
        # applicant mandatory values must satisfy employer requirements
        if not all(a.mandatory.get(k,False) for k in v.mandatory):return Tier.NOT_QUALIFIED,0.
        score=100*sum(v.qual_weights[k]*a.qualifications[k] for k in v.qual_weights)
        return (Tier.PREFERRED if score>=P_THRESHOLD else Tier.QUALIFIED if score>=Q_THRESHOLD else Tier.NOT_QUALIFIED),score

    def hard_pass(self,a,p):
        x=p.values; t=a.thresholds
        if a.market==Market.STUDENT:return x["hours"]>=t["min_hours"] and x["wage"]>=t["min_wage"]
        if a.market==Market.RN:return bool(x.get("paid_patient",0) and x["paid_travel"] and x["mileage_compliant"] and x["no_overnight"] and x["max_oneway"]<=t["max_drive"] and x["daily_duration"]<=13 and x["annual_hours"]<=RN_MAX_ANNUAL_HOURS)
        if a.market==Market.PROFESSOR:return x["salary"]>=t["min_salary"] and x["load"]<=t["max_load"]
        return x["total"]>=t["min_total"] and x["equity"]>=t["min_equity"] and x["guaranteed"]>=t["min_guaranteed"] and x["protection"]>=t["min_protection"] and x["authority"]>=t["min_authority"]

    def applicant_utility(self,a,p,round_no=1):
        x=p.values; w=a.weights; t=a.thresholds
        if a.market==Market.STUDENT:
            u={"hours":u_inc(x["hours"],t["min_hours"],t["adequate_hours"],t["preferred_hours"]),"wage":u_inc(x["wage"],t["min_wage"],t["adequate_wage"],t["preferred_wage"]),"speed":u_dec(x["delay"]+7*(round_no-1),1,14,42),"schedule":clip(x["schedule"])}
        elif a.market==Market.RN:
            u={"net_comp":u_inc(x["net_comp"],t["min_net"],t["adequate_net"],t["preferred_net"]),"effective_hourly":u_inc(x["effective_hourly"],t["min_eff"],t["adequate_eff"],t["preferred_eff"]),"daily_travel":u_dec(x["daily_duration"],t["preferred_day"],t["adequate_day"],13),"mileage":u_dec(x["annual_mileage"],t["preferred_mileage"],t["adequate_mileage"],t["max_mileage"]),"oncall":u_dec(x["oncall"],t["preferred_oncall"],t["adequate_oncall"],t["max_oncall"])}
        elif a.market==Market.PROFESSOR:
            u={"salary":u_inc(x["salary"],t["min_salary"],t["adequate_salary"],t["preferred_salary"]),"research":clip(x["research"]),"workload":clip(x["workload"])}
        else:
            u={"total":u_inc(x["total"],t["min_total"],t["min_total"]*1.1,t["min_total"]*1.3),"equity":clip((x["equity"]-.2)/.6),"guaranteed":clip((x["guaranteed"]-.1)/.4),"protection":x["protection"]/4,"authority":x["authority"]/4}
        return clip(sum(w[k]*u[k] for k in w))

    def employer_utility(self,a,v,p,score,tier):
        x=p.values; within=clip((score-(P_THRESHOLD if tier==Tier.PREFERRED else Q_THRESHOLD))/((100-P_THRESHOLD) if tier==Tier.PREFERRED else (P_THRESHOLD-Q_THRESHOLD)))
        if a.market==Market.STUDENT: pkg=clip(.4*min(1,x["hours"]/20)+.25*a.qualifications["q1"]+.25*a.qualifications["q0"]+.10*(1-clip((x["wage"]-13.5)/3)))
        elif a.market==Market.RN: pkg=clip(.30*a.qualifications["q0"]+.25*a.qualifications["q1"]+.20*(1.-builtins.min(1.,x["oncall"]/8.))+.25*(1-clip((x["net_comp"]-85000)/50000)))
        elif a.market==Market.PROFESSOR:
            wt={"4/4":(.50,.15,.20,.15),"3/3":(.35,.35,.15,.15),"2/2":(.20,.50,.15,.15)}[v.subtype]
            pkg=clip(wt[0]*a.qualifications["q0"]+wt[1]*a.qualifications["q1"]+wt[2]*a.qualifications["q2"]+wt[3]*(1-clip((x["salary"]-140000)/260000)))
        else: pkg=clip(.30*a.qualifications["q0"]+.20*a.qualifications["q1"]+.30*a.qualifications["q3"]+.20*(1-clip((x["total"]-3e6)/17e6)))
        return clip(.70*pkg+.30*within)

    def evaluate(self,a,v,condition,round_no,rng=None):
        rng = rng or self.rng
        ground_a=a
        observed_a=self.observed_applicant(a,condition,rng)
        tier,score=self.classify(observed_a,v)
        if tier==Tier.NOT_QUALIFIED:return EvalResult(a.id,v.id,tier,score,Compat.INCOMPATIBLE,None,0,0,False)
        menu=v.p_menu if tier==Tier.PREFERRED else v.q_menu; default=v.p_default if tier==Tier.PREFERRED else v.q_default
        feasible=[]
        for idx,p in enumerate(menu):
            hp=self.hard_pass(observed_a,p); au=self.applicant_utility(observed_a,p,round_no); eu=self.employer_utility(observed_a,v,p,score,tier)
            if hp and au>=.5 and eu>=.5:feasible.append((idx,au,eu))
        exact,cat=self.disclosure_schema_counts(a,condition)
        if not feasible:return EvalResult(a.id,v.id,tier,score,Compat.INCOMPATIBLE,None,0,0,False,exact,cat)
        d=[x for x in feasible if x[0]==default]
        if d: idx,au,eu=d[0]; comp=Compat.COMPATIBLE
        else:
            # finite-menu nearest-to-default in normalized index space
            idx,au,eu=self._nearest_feasible(menu,default,feasible,a.market); comp=Compat.CONDITIONAL
        return EvalResult(a.id,v.id,tier,score,comp,idx,au,eu,True,exact,cat)

    def disclosure_schema_counts(self,a,condition):
        exact_fields={Market.STUDENT:2,Market.RN:2,Market.PROFESSOR:2,Market.CEO:3}
        categorical_fields={Market.STUDENT:3,Market.RN:6,Market.PROFESSOR:3,Market.CEO:2}
        if condition in (Condition.TRAD,Condition.TRAD_LOCK):return exact_fields[a.market],categorical_fields[a.market]
        if condition in (Condition.MDEC,Condition.MDEC_LOCK):return 0,3
        return 0,5

    def privacy_thresholds(self,a):
        t=a.thresholds
        if a.market==Market.STUDENT: return t["min_hours"], t["preferred_wage"], "discrete", "continuous"
        if a.market==Market.RN: return t["max_drive"], t["min_net"], "continuous", "continuous"
        if a.market==Market.PROFESSOR: return t["min_salary"], t["max_load"], "continuous", "discrete"
        return t["min_total"], t["min_equity"], "continuous", "continuous"

    def market_ground_truth(self,apps,vacs,rep,market):
        viable_apps=set(); viable_vacs=set(); viable_pairs=set(); pref_vacs=set(); qual_vacs=set()
        # Exact finite-menu ground truth, chunked by vacancy. Used once per market/replication.
        for v in vacs:
            for a in apps:
                tier,score=self.classify(a,v)
                if tier==Tier.NOT_QUALIFIED: continue
                menu=v.p_menu if tier==Tier.PREFERRED else v.q_menu
                ok=False
                for p in menu:
                    if self.hard_pass(a,p) and self.applicant_utility(a,p,1)>=.5 and self.employer_utility(a,v,p,score,tier)>=.5:
                        ok=True;break
                if ok:
                    viable_pairs.add((a.id,v.id));viable_apps.add(a.id);viable_vacs.add(v.id)
                    if tier==Tier.PREFERRED:pref_vacs.add(v.id)
                    else:qual_vacs.add(v.id)
        return viable_pairs,viable_apps,viable_vacs,pref_vacs,qual_vacs

    def run_market_condition(self,m,condition,apps0,vacs0,rep=0,groundtruth=None,strategic_prevalence=0.0):
        import copy
        apps=copy.deepcopy(apps0); vacs=copy.deepcopy(vacs0)
        rng_sampling=np.random.default_rng(stable_seed(self.seed,rep,m.value,"shared_sampling"))
        rng_comm=np.random.default_rng(stable_seed(self.seed,rep,m.value,condition.value,"communication"))
        rng_human=np.random.default_rng(stable_seed(self.seed,rep,m.value,condition.value,"human"))
        rng_adapt=np.random.default_rng(stable_seed(self.seed,rep,m.value,condition.value,"adapt"))
        initial_evals=conditional=compat_escalations=choice_escalations=offers_rejected=0
        applicant_rejections=offers_not_selected=offers_below_reservation=0
        disclosed_exact=disclosed_cat=0; dyad_rows=[]; offer_rows=[]; app_rows=[]; vac_rows=[]
        evaluated_viable=dyad_false_rejects=0
        for rnd in range(1,MAX_ROUNDS+1):
            active_apps=[a for a in apps if a.active]; open_v=[v for v in vacs if v.active]
            if not active_apps or not open_v: break
            caps={a.id:0 for a in active_apps}; rng_sampling.shuffle(open_v); evals=[]
            for v in open_v:
                _,_,per,cap=self.sizes(m)
                candidates=[a for a in active_apps if a.id not in v.considered and caps[a.id]<cap]
                rng_sampling.shuffle(candidates); accepted_candidates=[]
                for a in candidates:
                    if condition==Condition.POSTED and not self.posted_self_screen(a,v,rnd): continue
                    accepted_candidates.append(a)
                    if len(accepted_candidates)>=per: break
                for a in accepted_candidates:
                    v.considered.add(a.id); caps[a.id]+=1
                    res=self.evaluate(a,v,condition,rnd,rng_comm); initial_evals+=1
                    disclosed_exact+=res.disclosed_exact; disclosed_cat+=res.disclosed_categorical
                    # ground truth evaluated with complete applicant data
                    gt=self.evaluate(a,v,Condition.MDEC_LOCK,rnd,np.random.default_rng(stable_seed(self.seed,rep,m.value,v.id,a.id,"gt")))
                    gt_viable=gt.tier!=Tier.NOT_QUALIFIED and gt.compat!=Compat.INCOMPATIBLE
                    if gt_viable: evaluated_viable+=1
                    if gt_viable and res.compat==Compat.INCOMPATIBLE: dyad_false_rejects+=1
                    if res.compat==Compat.CONDITIONAL:
                        conditional+=1; compat_escalations+=1
                        # Human resolves feasible soft case with .70; unresolved becomes incompatible
                        if rng_human.random()>.70: res.compat=Compat.INCOMPATIBLE; res.package_index=None
                    evals.append((a,v,res))
                    th1,th2,kind1,kind2=self.privacy_thresholds(a)
                    dyad_rows.append({"replication":rep,"market":m.value,"condition":condition.value,"round":rnd,"applicant_id":a.id,"vacancy_id":v.id,"tier":res.tier.value,"score":res.score,"compatibility":res.compat.value,"applicant_utility":res.applicant_utility,"employer_utility":res.employer_utility,"ground_truth_viable":gt_viable,"exact_disclosures":res.disclosed_exact,"categorical_disclosures":res.disclosed_categorical,"threshold_primary":th1,"threshold_secondary":th2,"threshold_primary_kind":kind1,"threshold_secondary_kind":kind2})
            round_offers=[]
            for v in open_v:
                choices=[z for z in evals if z[1].id==v.id and z[2].compat!=Compat.INCOMPATIBLE]
                if m==Market.PROFESSOR:
                    if rnd==1:
                        v.hold=sorted([(z[2].score,z[0].id,z[2]) for z in choices if z[2].tier==Tier.QUALIFIED],reverse=True)[:3]
                        choices=[z for z in choices if z[2].tier==Tier.PREFERRED]
                    elif rnd==2 and not any(z[2].tier==Tier.PREFERRED for z in choices):
                        held=[]
                        for _,aid,res in v.hold:
                            if apps[aid].active:
                                held.append((apps[aid],v,res))
                        choices += held
                if m==Market.CEO: choices=[z for z in choices if z[2].tier==Tier.PREFERRED]
                if not choices: continue
                choices.sort(key=lambda z:(z[2].tier==Tier.PREFERRED,z[2].employer_utility,z[2].score),reverse=True)
                a,_,res=choices[0]; menu=v.p_menu if res.tier==Tier.PREFERRED else v.q_menu; pkg=menu[res.package_index]
                true_u=self.applicant_utility(a,pkg,rnd)
                round_offers.append(Offer(a.id,v.id,pkg,true_u,res.employer_utility,res.tier,bool(self.hard_pass(a,pkg) and true_u>=.50 and res.employer_utility>=.50)))
                offer_rows.append({"replication":rep,"market":m.value,"condition":condition.value,"round":rnd,"applicant_id":a.id,"vacancy_id":v.id,"utility":true_u,"employer_utility":res.employer_utility,"tier":res.tier.value,"accepted":False,"disposition":"pending","ground_truth_valid":bool(self.hard_pass(a,pkg) and true_u>=.50 and res.employer_utility>=.50),"package_values":json.dumps(pkg.values,sort_keys=True)})
            byapp={}
            for o in round_offers: byapp.setdefault(o.applicant_id,[]).append(o)
            accepted=[]
            for aid,os_ in byapp.items():
                os_.sort(key=lambda o:o.utility,reverse=True); best=os_[0]
                if len(os_)>1 and abs(os_[0].utility-os_[1].utility)<=.05:
                    choice_escalations+=1

                # Receiving multiple offers changes which acceptable offer is selected,
                # not the applicant's reservation utility.
                acceptable = [o for o in os_ if o.ground_truth_valid and o.utility >= SINGLE_ACCEPT]
                if acceptable:
                    acceptable.sort(key=lambda o:o.utility, reverse=True)
                    accepted.append(acceptable[0])
                    offers_not_selected += len(os_) - 1
                else:
                    applicant_rejections += 1
                    offers_below_reservation += len(os_)
                    offers_rejected += len(os_)
            # Assign offer-level dispositions after applicant decisions.
            accepted_keys = {(o.applicant_id, o.vacancy_id) for o in accepted}
            for row in offer_rows:
                if row["round"] != rnd or row["disposition"] != "pending":
                    continue
                key = (row["applicant_id"], row["vacancy_id"])
                if key in accepted_keys:
                    continue
                row["disposition"] = ("below_reservation" if row["utility"] < SINGLE_ACCEPT
                                      or not row["ground_truth_valid"] else "not_selected")
            filled=set()
            for o in accepted:
                if o.vacancy_id in filled or not apps[o.applicant_id].active or not vacs[o.vacancy_id].active: continue
                apps[o.applicant_id].active=False; apps[o.applicant_id].hired=True; apps[o.applicant_id].accepted_vacancy=o.vacancy_id
                vacs[o.vacancy_id].active=False; vacs[o.vacancy_id].filled=True; vacs[o.vacancy_id].hired_applicant=o.applicant_id; filled.add(o.vacancy_id)
                for row in reversed(offer_rows):
                    if row["round"]==rnd and row["applicant_id"]==o.applicant_id and row["vacancy_id"]==o.vacancy_id:
                        row["accepted"]=True; row["disposition"]="accepted"; break
            # unlocked adaptation after unsuccessful round
            if condition in (Condition.TRAD,Condition.MDEC):
                for v in vacs:
                    if v.active:
                        self.adapt_vacancy(v,condition,rng_adapt,strategic=bool(rng_adapt.random()<strategic_prevalence))
                        # Full repair of both menus after adaptation
                        for _j,_pkg in enumerate(list(v.q_menu)):
                            if not self._package_budget_ok(v,_pkg): v.q_menu[_j]=Package(dict(v.q_menu[0].values))
                        v.p_menu=self._enforce_preferred_dominance(v.q_menu,v.p_menu,v.market)
                        for _j,_pkg in enumerate(list(v.p_menu)):
                            if not self._package_budget_ok(v,_pkg): v.p_menu[_j]=Package(dict(v.q_menu[_j].values))
                        v.budget_violations=self._count_budget_violations(v);v.composition_violations=self._count_composition_violations(v);v.dominance_violations=self._count_dominance_violations(v.q_menu,v.p_menu,v.market)
            if m==Market.CEO and rnd==3:
                for v in vacs:
                    if v.active:
                        had_pref=any(d["vacancy_id"]==v.id and d["tier"]==Tier.PREFERRED.value and d["ground_truth_viable"] for d in dyad_rows)
                        had_offer=any(o["vacancy_id"]==v.id for o in offer_rows)
                        v.closure_reason="preferred_rejected" if had_offer else "no_compatible_preferred" if not had_pref else "round_limit"
                        v.active=False
                break
            # observable exhaustion: no unevaluated dyads can be sampled
            if all(len([a for a in apps if a.active and a.id not in v.considered])==0 for v in vacs if v.active): break
        hires=sum(a.hired for a in apps); fill=hires/len(vacs); place=hires/len(apps)
        for a in apps: app_rows.append({"replication":rep,"market":m.value,"condition":condition.value,"applicant_id":a.id,"archetype":a.archetype,"hired":a.hired,"vacancy_id":a.accepted_vacancy})
        for v in vacs: vac_rows.append({"replication":rep,"market":m.value,"condition":condition.value,"vacancy_id":v.id,"subtype":v.subtype,"filled":v.filled,"closure_reason":v.closure_reason,"revisions":v.revision_count,"commitment_violations":v.commitment_violations,"generation_attempts":v.generation_attempts,"generation_rejections":v.generation_rejections,"dominance_violations":v.dominance_violations,"budget_violations":v.budget_violations,"composition_violations":v.composition_violations})
        viable_pairs,viable_apps,viable_vacs,pref_vacs,qual_vacs = groundtruth or (set(),set(),set(),set(),set())
        unplaced_viable=sum((not apps[i].hired) for i in viable_apps); unfilled_viable=sum((not vacs[i].filled) for i in viable_vacs)
        result={"market":m.value,"condition":condition.value,"hires":hires,"fill_rate":fill,"placement_rate":place,"initial_evaluations":initial_evals,"frr_a":unplaced_viable/max(1,len(viable_apps)),"frr_v":unfilled_viable/max(1,len(viable_vacs)),"pfvr_q":len(qual_vacs|pref_vacs)/max(1,len(vacs)),"pfvr_p":len(pref_vacs)/max(1,len(vacs)),"mdec_efficiency":hires/max(1,initial_evals),"conditional_rate":conditional/max(1,initial_evals),"compatibility_escalations":compat_escalations,"choice_escalations":choice_escalations,"offers_rejected":offers_rejected,"applicant_rejections":applicant_rejections,"offers_not_selected":offers_not_selected,"offers_below_reservation":offers_below_reservation,"exact_disclosures":disclosed_exact,"categorical_disclosures":disclosed_cat,"frr_d":dyad_false_rejects/max(1,evaluated_viable)}
        return result,dyad_rows,offer_rows,app_rows,vac_rows

    def run_replication(self,rep,strategic_prevalence=0.0,markets=None):
        rows=[]; dyads=[]; offers=[]; appout=[]; vacout=[]
        # replication-specific population stream
        self.rng=np.random.default_rng(stable_seed(self.seed,rep,"population"))
        selected_markets = list(Market) if markets is None else list(markets)
        for m in selected_markets:
            apps=self.applicants(m); vacs=self.vacancies(m)
            groundtruth=self.market_ground_truth(apps,vacs,rep,m)
            for c in Condition:
                res,d,o,a,v=self.run_market_condition(m,c,apps,vacs,rep,groundtruth,strategic_prevalence)
                rows.append({"replication":rep,**res}); dyads.extend(d); offers.extend(o); appout.extend(a); vacout.extend(v)
        return rows,dyads,offers,appout,vacout

    def run(self,reps,strategic_prevalence=0.0,markets=None,n_workers=None):
        """Run replications, optionally in parallel across CPU cores.

        n_workers=None  → use all available CPU cores
        n_workers=1     → single-process mode (safe for debugging)
        n_workers=k     → use exactly k worker processes
        """
        import multiprocessing as _mp
        import builtins as _b

        if n_workers is None:
            n_workers = _mp.cpu_count()

        rep_ids = list(range(1, reps + 1))
        rows=[]; dyads=[]; offers=[]; apps=[]; vacs=[]

        if n_workers <= 1 or reps <= 1:
            # Single-process path — original behaviour
            for r in rep_ids:
                a,b,c,d,e = self.run_replication(r, strategic_prevalence, markets)
                rows.extend(a); dyads.extend(b); offers.extend(c)
                apps.extend(d); vacs.extend(e)
        else:
            # Parallel path — serialize only picklable scalars/dicts
            qp  = {m.value: v for m, v in self.quality_params.items()}
            qmx = {m.value: v for m, v in self.quality_mixtures.items()}
            mv  = None if markets is None else [m.value for m in markets]
            _args = [
                (self.seed, self.scale, self.menu_scale, qp, qmx,
                 r, strategic_prevalence, mv)
                for r in rep_ids
            ]
            # 'spawn' works on Windows, Linux, and macOS
            ctx = _mp.get_context("spawn")
            n_proc = _b.min(n_workers, reps)
            with ctx.Pool(processes=n_proc) as pool:
                results = pool.starmap(_run_replication_worker, _args)
            for a,b,c,d,e in results:
                rows.extend(a); dyads.extend(b); offers.extend(c)
                apps.extend(d); vacs.extend(e)

        return (pd.DataFrame(rows), pd.DataFrame(dyads),
                pd.DataFrame(offers), pd.DataFrame(apps),
                pd.DataFrame(vacs))
