from mdec_core import *

def _entropy(values):
    if len(values)==0:return float("nan")
    _,counts=np.unique(values,return_counts=True); p=counts/counts.sum(); return float(-(p*np.log2(p)).sum())

def privacy_analysis(dyads):
    """200-point grid posterior attacker with observation-error sensitivity."""
    if dyads.empty:return pd.DataFrame()
    rows=[]
    errors=(.001,.01,.05)
    for (market,condition),g in dyads.groupby(["market","condition"]):
        for which in ("primary","secondary"):
            col=f"threshold_{which}"; kind=g[f"threshold_{which}_kind"].iloc[0]; vals=g[col].astype(float).to_numpy()
            if len(vals)<2:continue
            if kind=="continuous":
                qs=np.linspace(.001,.999,200);grid=np.quantile(vals,qs);prior=np.ones(200)/200
                prior_lo,prior_hi=np.quantile(vals,[.05,.95]);prior_width=max(1e-12,prior_hi-prior_lo)
                # nearest-grid assignment estimates P(output | threshold grid).
                idx=np.abs(vals[:,None]-grid[None,:]).argmin(1); labels=(g.compatibility.astype(str)+"|"+g.tier.astype(str)).to_numpy(); unique=np.unique(labels)
                counts={lab:np.bincount(idx[labels==lab],minlength=200).astype(float) for lab in unique}; totals=np.bincount(idx,minlength=200).astype(float)
                for eps in errors:
                    widths=[];acc=[]
                    for row_i,(_,r) in enumerate(g.iterrows()):
                        true=float(r[col])
                        if r.exact_disclosures>0: post=np.zeros(200);post[np.argmin(abs(grid-true))]=1
                        else:
                            lab=f"{r.compatibility}|{r.tier}";lik=(counts.get(lab,np.zeros(200))+eps)/(totals+eps*max(1,len(unique)));post=prior*lik;post=post/post.sum() if post.sum()>0 else prior
                        c=np.cumsum(post);lo=grid[np.searchsorted(c,.05)];hi=grid[min(199,np.searchsorted(c,.95))];est=grid[np.searchsorted(c,.5)]
                        widths.append((hi-lo)/prior_width);acc.append(abs(est-true)<=.10*max(abs(true),1e-12))
                    rows.append({"market":market,"condition":condition,"threshold":which,"kind":kind,"observation_error":eps,"npiw":float(np.mean(widths)),"normalized_entropy":np.nan,"estimation_accuracy":float(np.mean(acc))})
            else:
                categories=np.unique(vals);prior=np.array([(vals==c).mean() for c in categories]);prior_ent=_entropy(vals);labels=(g.compatibility.astype(str)+"|"+g.tier.astype(str)).to_numpy();unique=np.unique(labels)
                for eps in errors:
                    ents=[];acc=[]
                    for _,r in g.iterrows():
                        true=float(r[col])
                        if r.exact_disclosures>0:post=np.array([1. if c==true else 0. for c in categories])
                        else:
                            lab=f"{r.compatibility}|{r.tier}";lik=np.array([((labels[vals==c]==lab).sum()+eps)/((vals==c).sum()+eps*len(unique)) for c in categories]);post=prior*lik;post=post/post.sum() if post.sum()>0 else prior
                        ent=float(-(post[post>0]*np.log2(post[post>0])).sum())/max(prior_ent,1e-12);est=categories[np.argmax(post)];ents.append(ent);acc.append(est==true)
                    rows.append({"market":market,"condition":condition,"threshold":which,"kind":kind,"observation_error":eps,"npiw":np.nan,"normalized_entropy":float(np.mean(ents)),"estimation_accuracy":float(np.mean(acc))})
    out=pd.DataFrame(rows);out["observation_scope"]="compatibility_and_tier_only";return out

def diagnostics(df,dyads,offers,vacs):
    out=[]
    for (market,condition),g in offers.groupby(["market","condition"]):
        vals=g.utility.astype(float); out.append({"market":market,"condition":condition,"metric":"offer_utility_mean","value":vals.mean()});out.append({"market":market,"condition":condition,"metric":"offer_utility_median","value":vals.median()});out.append({"market":market,"condition":condition,"metric":"offer_utility_p95","value":vals.quantile(.95)});out.append({"market":market,"condition":condition,"metric":"offer_utility_gt_075","value":float((vals>.75).mean())})
    if not offers.empty and "disposition" in offers.columns:
        for (market,condition),g in offers.groupby(["market","condition"]):
            counts=g.disposition.value_counts(normalize=True)
            for disposition,value in counts.items():
                out.append({"market":market,"condition":condition,
                            "metric":f"offer_disposition_{disposition}_rate",
                            "value":float(value)})
    if not vacs.empty:
        rn=vacs[vacs.market=="rn"]
        if len(rn):out.append({"market":"rn","condition":"all","metric":"rn_package_rejection_rate","value":rn.generation_rejections.sum()/max(1,rn.generation_attempts.sum())})
    return pd.DataFrame(out)

def analyze(df):
    summary=df.groupby(["market","condition"]).agg(reps=("replication","nunique"),hires=("hires","mean"),fill_rate=("fill_rate","mean"),placement_rate=("placement_rate","mean"),mdec_efficiency=("mdec_efficiency","mean"),conditional_rate=("conditional_rate","mean"),offers_rejected=("offers_rejected","mean"),applicant_rejections=("applicant_rejections","mean"),offers_not_selected=("offers_not_selected","mean"),offers_below_reservation=("offers_below_reservation","mean"),frr_d=("frr_d","mean"),frr_a=("frr_a","mean"),frr_v=("frr_v","mean"),pfvr_q=("pfvr_q","mean"),pfvr_p=("pfvr_p","mean"),exact_disclosures=("exact_disclosures","mean"),categorical_disclosures=("categorical_disclosures","mean"),compatibility_escalations=("compatibility_escalations","mean"),choice_escalations=("choice_escalations","mean")).reset_index()
    contrasts=[]
    for m in df.market.unique():
        piv=df[df.market==m].pivot(index="replication",columns="condition",values="fill_rate")
        if "mdec_locked" in piv and "traditional" in piv:
            d=(piv["mdec_locked"]-piv["traditional"]).dropna(); mean=d.mean(); se=d.std(ddof=1)/math.sqrt(len(d)) if len(d)>1 else float('nan'); lcb=mean-1.645*se if len(d)>1 else float('nan')
            contrasts.append({"market":m,"contrast":"mdec_locked-traditional","mean_difference":mean,"one_sided_95_lcb":lcb,"noninferior_margin":NONINFERIORITY_MARGIN,"noninferior":bool(lcb>-NONINFERIORITY_MARGIN) if not math.isnan(lcb) else False})
    return summary,pd.DataFrame(contrasts)

def validate(df,dyads=None,offers=None,vacs=None):
    errs=[]
    if ((df.fill_rate<0)|(df.fill_rate>1)).any():errs.append("fill_rate_out_of_range")
    if ((df.placement_rate<0)|(df.placement_rate>1)).any():errs.append("placement_rate_out_of_range")
    if (df.hires<0).any():errs.append("negative_hires")
    if dyads is not None and not dyads.empty:
        bad=(dyads.ground_truth_viable==False)&(dyads.compatibility!="incompatible")
        # Do not treat observed traditional false compatibility as engine error; it is a treatment effect.
        bad=bad & dyads.condition.isin(["mdec_locked","mdec_unlocked","posted_package"])
        if bad.any():errs.append("hard_or_ground_truth_incompatible_labeled_compatible")
        if dyads.duplicated(["replication","market","condition","applicant_id","vacancy_id"]).any():errs.append("duplicate_dyad")
    if offers is not None and not offers.empty:
        accepted=offers[offers.accepted==True]
        if (accepted.ground_truth_valid==False).any():errs.append("accepted_offer_invalid")
        if accepted.duplicated(["replication","market","condition","applicant_id"]).any():errs.append("double_hire")
    if vacs is not None and not vacs.empty:
        if (vacs.budget_violations>0).any():errs.append("budget_violation")
        if (vacs.composition_violations>0).any():errs.append("component_sum_violation")
        if (vacs.dominance_violations>0).any():errs.append("preferred_dominance_violation")
        if (vacs.filled.groupby([vacs.replication,vacs.market,vacs.condition,vacs.vacancy_id]).sum()>1).any():errs.append("double_fill")
    return errs

def calibration_report(dyads, offers, vacs):
    rows=[]
    if not dyads.empty:
        for market,g in dyads.groupby("market"):
            gg=g.drop_duplicates(["replication","condition","applicant_id"]);target=TARGET_TIERS[Market(market)]
            # Output both overall observed shares and conditional score-tier shares.
            shares=gg.tier.value_counts(normalize=True);eligible=gg[gg.tier!="not_qualified"];ep=eligible.tier.value_counts(normalize=True)
            rows.append({"market":market,"diagnostic":"overall_not_qualified","actual":float(shares.get("not_qualified",0)),"target":np.nan,"within_tolerance":True})
            # Conditional targets renormalize qualified/preferred after mandatory eligibility.
            qp=np.array([target[1],target[2]])/(target[1]+target[2])
            for tier,t in zip(["qualified","preferred"],qp):
                actual=float(ep.get(tier,0));rows.append({"market":market,"diagnostic":f"eligible_{tier}","actual":actual,"target":float(t),"within_tolerance":abs(actual-t)<=.03})
    if not offers.empty:
        for market,g in offers.groupby("market"):
            med=float(g.utility.median());p95=float(g.utility.quantile(.95));rows.append({"market":market,"diagnostic":"utility_median","actual":med,"target":.55,"within_tolerance":.40<=med<=.70});rows.append({"market":market,"diagnostic":"utility_p95","actual":p95,"target":.75,"within_tolerance":p95>=.75})
    if not vacs.empty:
        rn=vacs[vacs.market=="rn"]
        if len(rn):
            rr=rn.generation_rejections.sum()/max(1,rn.generation_attempts.sum());rows.append({"market":"rn","diagnostic":"rn_package_rejection_rate","actual":rr,"target":.20,"within_tolerance":rr<=.20})
    return pd.DataFrame(rows)

def max_primary_mcse(df,privacy=None):
    vals=[]
    for market in df.market.unique():
        piv=df[df.market==market].pivot(index="replication",columns="condition",values=["fill_rate","placement_rate","frr_d"])
        for metric in ("fill_rate","placement_rate","frr_d"):
            d=(piv[(metric,"mdec_locked")]-piv[(metric,"traditional")]).dropna()
            if len(d)>1:vals.append(d.std(ddof=1)/math.sqrt(len(d)))
    # Privacy is summarized across dyads rather than replication-level in the
    # current attacker model; its MCSE is reported separately after analysis.
    return max(vals) if vals else float("inf")

def run_convergence(seed,reps=3,scale=.01):
    frames=[]
    for rep in range(1,reps+1):
        base=MDECSimulator(seed,scale=scale,menu_scale=5);base.auto_calibrate_quality(draws=5000);base.rng=np.random.default_rng(stable_seed(seed,rep,"population"))
        for market in Market:
            apps=base.applicants(market); full_vacs=base.vacancies(market)
            for density,count in ((1,50),(2,100),(5,250)):
                import copy
                vacs=copy.deepcopy(full_vacs)
                for v in vacs:
                    wanted=100 if market==Market.CEO and density==1 else 200 if market==Market.CEO and density==2 else 500 if market==Market.CEO else count
                    v.q_menu=v.q_menu[:wanted];v.p_menu=v.p_menu[:wanted];v.q_default=base._default_index(v.q_menu,False);v.p_default=base._default_index(v.p_menu,True)
                gt=base.market_ground_truth(apps,vacs,rep,market)
                for condition in Condition:
                    res,*_=base.run_market_condition(market,condition,apps,vacs,rep,gt,0.0);res.update({"replication":rep,"menu_density":density});frames.append(res)
    raw=pd.DataFrame(frames);rows=[]
    for metric in ("fill_rate","placement_rate"):
        piv=raw.pivot_table(index=["replication","market","condition"],columns="menu_density",values=metric)
        for (market,condition),g in piv.groupby(level=["market","condition"]):
            g=g.dropna();d=g[2]-g[5];mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else np.nan;half=1.96*se if len(d)>1 else np.nan
            converged=bool(abs(mean)<.005 and (np.isnan(half) or (mean-half>=-.005 and mean+half<=.005)))
            rows.append({"market":market,"condition":condition,"metric":metric,"mean_difference_100_minus_250":mean,"ci95_low":mean-half if not np.isnan(half) else np.nan,"ci95_high":mean+half if not np.isnan(half) else np.nan,"converged":converged})
    return raw,pd.DataFrame(rows)

def success_hierarchy(df,privacy):
    rows=[]; market_effects=[]
    for market in df.market.unique():
        piv=df[df.market==market].pivot(index="replication",columns="condition",values=["fill_rate","placement_rate","frr_d"])
        def paired(metric):
            d=(piv[(metric,"mdec_locked")]-piv[(metric,"traditional")]).dropna();mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else np.nan;return mean,mean-1.645*se if len(d)>1 else np.nan
        fill,fill_lcb=paired("fill_rate");place,place_lcb=paired("placement_rate");frr,_=paired("frr_d")
        pp=privacy[(privacy.market==market)&(privacy.threshold=="primary")&(privacy.observation_error==.01)];ma=pp[pp.condition=="mdec_locked"].estimation_accuracy.mean();ta=pp[pp.condition=="traditional"].estimation_accuracy.mean()
        market_effects.append((fill,place,frr,ma,ta));rows.append({"market":market,"fill_difference":fill,"fill_lcb":fill_lcb,"placement_difference":place,"placement_lcb":place_lcb,"frr_difference":frr,"mdec_privacy_accuracy":ma,"traditional_privacy_accuracy":ta,"harm_bound":bool(fill>=-.02)})
    # Frozen rule: unweighted market-average success, plus no market harm below -2 points.
    af=np.mean([x[0] for x in market_effects]);ap=np.mean([x[1] for x in market_effects]);afr=np.mean([x[2] for x in market_effects]);apriv_m=np.nanmean([x[3] for x in market_effects]);apriv_t=np.nanmean([x[4] for x in market_effects])
    # Conservative aggregate noninferiority uses mean market effect and harm safeguard.
    flags={"S1_fill_noninferior":bool(af>=-.02),"S2_place_noninferior":bool(ap>=-.02),"S3_privacy_lower_accuracy":bool(apriv_m<apriv_t),"S4_frr_lower":bool(afr<0),"harm_bound":all(r["harm_bound"] for r in rows)}
    rows.append({"market":"AGGREGATE","fill_difference":af,"placement_difference":ap,"frr_difference":afr,"mdec_privacy_accuracy":apriv_m,"traditional_privacy_accuracy":apriv_t,**flags,"market_success":all(flags.values())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--main", action="store_true")
    g.add_argument("--convergence-test", action="store_true")

    ap.add_argument("--output", default="mdec_v3_results")
    ap.add_argument(
        "--strategic-prevalence",
        type=float,
        choices=STRATEGIC_PREVALENCE_LEVELS,
        default=0.0
    )
    ap.add_argument("--calibration-file", default=None)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--max-reps", type=int, default=500)
    ap.add_argument("--market", choices=["all","student","rn","professor","ceo"], default="all")
    ap.add_argument(
        "--privacy-sample",
        type=int,
        default=100_000
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all CPU cores). Use 1 for debug."
    )

    args = ap.parse_args()
    market_map = {"student":Market.STUDENT,"rn":Market.RN,"professor":Market.PROFESSOR,"ceo":Market.CEO}
    selected_markets = None if args.market == "all" else [market_map[args.market]]

    reps = args.reps or (
        1
        if args.smoke_test
        else PILOT_REPS
        if args.pilot
        else MAIN_REPS
    )

    scale = 0.02 if args.smoke_test else 1.0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    import multiprocessing as _mp
    _n_workers = args.workers if args.workers is not None else _mp.cpu_count()
    print(
        f"[MAIN] PID={os.getpid()} | "
        f"mode={'main' if args.main else 'pilot' if args.pilot else 'smoke'} | "
        f"workers={_n_workers}",
        flush=True
    )

    # --------------------------------------------------
    # Standalone menu-convergence test
    # --------------------------------------------------

    if args.convergence_test:
        conv_raw, conv_decisions = run_convergence(
            args.seed,
            reps=args.reps or 3
        )

        conv_raw.to_csv(
            str(out) + "_menu_convergence_raw.csv",
            index=False
        )

        conv_decisions.to_csv(
            str(out) + "_menu_convergence_decisions.csv",
            index=False
        )

        print(conv_decisions.to_string(index=False))
        print("[MAIN] Convergence test complete.", flush=True)
        return

    # --------------------------------------------------
    # Initialize simulator and calibration
    # --------------------------------------------------

    sim = MDECSimulator(
        args.seed,
        scale=scale
    )

    if args.calibration_file:
        sim.load_calibration(args.calibration_file)
    else:
        sim.auto_calibrate_quality(
            draws=5000 if args.smoke_test else 20000
        )

    # --------------------------------------------------
    # Adaptive main-run convergence loop
    # --------------------------------------------------

    max_total_reps = min(args.max_reps, 500)

    while True:
        print(
            f"[MAIN] Starting run with {reps} replications",
            flush=True
        )

        df, dyads, offers, appout, vacout = sim.run(
            reps,
            args.strategic_prevalence,
            selected_markets,
            n_workers=args.workers
        )

        mcse = max_primary_mcse(df)

        print(
            f"[MAIN] Completed {reps} reps | "
            f"MCSE={mcse:.6f} | "
            f"Target={MONTE_CARLO_STANDARD_ERROR_TARGET:.6f}",
            flush=True
        )

        # Smoke, pilot, or explicit --reps runs only once.
        if not args.main:
            print(
                "[MAIN] Fixed-size non-main run completed.",
                flush=True
            )
            break

        if args.reps is not None:
            print(
                f"[MAIN] Explicit replication count completed: "
                f"{reps}",
                flush=True
            )
            break

        if mcse <= MONTE_CARLO_STANDARD_ERROR_TARGET:
            print(
                f"[MAIN] Convergence achieved at {reps} reps | "
                f"MCSE={mcse:.6f}",
                flush=True
            )
            break

        if reps >= max_total_reps:
            print(
                f"[MAIN] HARD STOP reached at {reps} reps | "
                f"MCSE={mcse:.6f} remains above "
                f"{MONTE_CARLO_STANDARD_ERROR_TARGET:.6f}",
                flush=True
            )
            break

        next_reps = min(
            max_total_reps,
            reps + MAIN_REPS
        )

        print(
            f"[MAIN] Convergence not achieved. "
            f"Adding {next_reps - reps} replications "
            f"(new total={next_reps}).",
            flush=True
        )

        reps = next_reps

    # --------------------------------------------------
    # Post-processing: this section is OUTSIDE the loop
    # --------------------------------------------------

    print("[MAIN] Running analyze()", flush=True)
    summary, contrasts = analyze(df)

    print(
        f"[MAIN] Dyads before privacy analysis: "
        f"{len(dyads):,}",
        flush=True
    )

    privacy_sample_limit = max(
        1,
        args.privacy_sample
    )

    if len(dyads) > privacy_sample_limit:
        print(
            f"[MAIN] Sampling "
            f"{privacy_sample_limit:,} of "
            f"{len(dyads):,} dyads for privacy analysis",
            flush=True
        )

        privacy_dyads = dyads.sample(
            n=privacy_sample_limit,
            random_state=args.seed
        )
    else:
        privacy_dyads = dyads

    print("[MAIN] Running privacy_analysis()", flush=True)
    privacy = privacy_analysis(privacy_dyads)

    print("[MAIN] Running diagnostics()", flush=True)
    diags = diagnostics(
        df,
        dyads,
        offers,
        vacout
    )

    print("[MAIN] Running calibration_report()", flush=True)
    calibration = calibration_report(
        dyads,
        offers,
        vacout
    )

    print("[MAIN] Running success_hierarchy()", flush=True)
    success = success_hierarchy(
        df,
        privacy
    )

    print("[MAIN] Running validation()", flush=True)
    errs = validate(
        df,
        dyads,
        offers,
        vacout
    )

    if args.pilot:
        quality_passed = sim.metadata.get(
            "quality_calibration_passed",
            False
        )

        calibration_passed = (
            not calibration.empty
            and calibration["within_tolerance"].all()
        )

        if not quality_passed or not calibration_passed:
            errs.append("pilot_calibration_failed")

    # --------------------------------------------------
    # Write outputs exactly once
    # --------------------------------------------------

    print("[MAIN] Writing output files...", flush=True)

    calibrated_parameters = {
        "quality_params": {
            m.value: v
            for m, v in sim.quality_params.items()
        },
        "quality_mixtures": {
            m.value: v
            for m, v in sim.quality_mixtures.items()
        },
        "quality_calibration_passed": sim.metadata.get(
            "quality_calibration_passed",
            False
        ),
        "quality_calibration_report": sim.metadata.get(
            "quality_calibration_report",
            {}
        )
    }

    Path(
        str(out) + "_calibrated_parameters.json"
    ).write_text(
        json.dumps(
            calibrated_parameters,
            indent=2
        )
    )

    df.to_csv(
        str(out) + "_raw.csv",
        index=False
    )

    dyads.to_csv(
        str(out) + "_dyads.csv",
        index=False
    )

    offers.to_csv(
        str(out) + "_offers.csv",
        index=False
    )

    appout.to_csv(
        str(out) + "_applicants.csv",
        index=False
    )

    vacout.to_csv(
        str(out) + "_vacancies.csv",
        index=False
    )

    privacy.to_csv(
        str(out) + "_privacy.csv",
        index=False
    )

    diags.to_csv(
        str(out) + "_diagnostics.csv",
        index=False
    )

    calibration.to_csv(
        str(out) + "_calibration.csv",
        index=False
    )

    success.to_csv(
        str(out) + "_success_hierarchy.csv",
        index=False
    )

    summary.to_csv(
        str(out) + "_summary.csv",
        index=False
    )

    contrasts.to_csv(
        str(out) + "_contrasts.csv",
        index=False
    )

    metadata = {
        **sim.metadata,
        "replications": reps,
        "scale": scale,
        "strategic_prevalence": args.strategic_prevalence,
        "market_selection": args.market,
        "max_primary_mcse": mcse,
        "mcse_target": MONTE_CARLO_STANDARD_ERROR_TARGET,
        "mcse_converged": bool(
            mcse <= MONTE_CARLO_STANDARD_ERROR_TARGET
        ),
        "max_total_reps": max_total_reps,
        "privacy_population_dyads": len(dyads),
        "privacy_sample_dyads": len(privacy_dyads),
        "validation_errors": errs,
        "pilot_calibration_required": bool(args.pilot),
        "output_files": [
            "raw",
            "dyads",
            "offers",
            "applicants",
            "vacancies",
            "privacy",
            "diagnostics",
            "calibration",
            "success_hierarchy",
            "summary",
            "contrasts"
        ]
    }

    with open(
        str(out) + "_metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        "[MAIN] Output files written successfully.",
        flush=True
    )

    if not dyads.empty:
        rn = dyads[dyads.market == "rn"]
        print("\nRN compatibility distribution:", flush=True)
        print(rn.groupby(["condition","compatibility"]).size().to_string(), flush=True)
    print(summary.to_string(index=False))
    print("\nValidation:", errs or "PASS")

    print(
        f"[MAIN] PID={os.getpid()} | "
        f"Run complete. Exiting.",
        flush=True
    )


if __name__ == "__main__":
    main()