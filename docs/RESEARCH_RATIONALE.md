# Research rationale: answers to the six assessment questions

This document accompanies the code in this repository (`run_pipeline.py`, `src/lexamb/`). References
are restricted to peer-reviewed venues where possible; arXiv-only items are marked **[preprint]**.
Verify every citation against the published version before submission.

---

## 1. Ultimate objective and rationale

**Objective.** The study measures, layer by layer, how much information about four lexical constructs
is linearly recoverable from the representation of an isolated word in decoder LLMs:

1. corpus frequency (`log_subtlex`);
2. number of related senses (`wn_n_all`);
3. number of unrelated meanings (published Rodd `ambiguity`; computed `gcide_etymologies`);
4. semantic distance between senses (`sim_top2`, `min_sim`, `mean_sim`).

It asks whether these profiles differ between matched base and instruction-tuned models. It also
asks whether a pre-registered model-derived processing proxy reproduces the *directional
dissociation* found in humans by Rodd, Gaskell & Marslen-Wilson (2002), JML 46: many related senses
help word recognition, while multiple unrelated meanings hurt it.

**Rationale.** In humans, "ambiguity" is not one variable. Senses and meanings have opposite
behavioural effects (Rodd et al., 2002; review: Eddington & Tokowicz, 2015, Psychon. Bull. Rev.).
Attractor-network models explain this through settling dynamics: Rodd, Gaskell & Marslen-Wilson
(2004), Cognitive Science; Armstrong & Plaut (2016), Lang. Cogn. Neurosci.

Transformer LLMs refine a residual-stream representation across depth, a process often described as
iterative inference (Jastrzębski et al., ICLR 2018; Lad et al., 2024 **[preprint]**). That gives a
natural "processing time" axis on which to test the same dissociation. The code turns this into a
measurable, falsifiable protocol. It keeps representation probes, output probabilities and the
behavioural-style proxy strictly separate, so that decodability is never mistaken for processing.

---

## 2. Research gaps addressed (with the closest prior work)

| # | Gap | Closest prior work and why it does not close the gap |
|---|---|---|
| G1 | **Sense count and frequency are confounded.** More frequent words have more senses (Zipf's meaning–frequency law; Piantadosi, Tily & Gibson, 2012, Cognition). A "polysemy probe" can succeed by reading frequency, and frequency is known to be strongly encoded in LM representations (Gong et al., NeurIPS 2018; Kobayashi et al., Findings ACL 2023). | Garí Soler & Apidianaki (TACL 2021) controlled frequency for BERT-type encoders. Beekhuizen, Armstrong & Stevenson (Cognitive Science, 2021) showed that *static* vectors encode sense number and relatedness. Neither studies layer-wise decoder LLMs, and neither estimates the information *beyond* covariates with a conditional estimator. |
| G2 | **Meanings (homonymy) versus senses (polysemy) in decoder LLM layers.** | Ma, Xie, Wang & Wang (Findings ACL 2025) study layer-wise homonym representations in BERT, GPT-2, Llama 3 and Qwen 2.5. They work with *contextualised* uses on synthetic sentences and report no single best layer. Haber & Poesio (Computational Linguistics, 2024) review polysemy in contextual LMs. Trott & Bergen (ACL 2021, RAW-C) and Nair et al. (2020) study relatedness in context. None uses the published Rodd factorial items (meanings × senses crossed in one 2×2 design), and none tests the opposite-direction dissociation. |
| G3 | **No processing-dynamics analogue of the Rodd dissociation in LLMs.** Prior LLM work reports decodability or similarity, not a directional, settling-type effect with a pre-registered orientation. | The attractor models (Rodd et al., 2004; Armstrong & Plaut, 2016) make depth/time predictions that, to our knowledge, have not been tested on transformer depth. Recent preprints examine how ambiguous-word representations diverge and reconverge across layers in *context* (e.g. arXiv 2608.01816, 2026 **[preprint]**). They do not test the Rodd factorial dissociation. |
| G4 | **Base vs instruct differences at the lexical level are untested with matched pairs and paired inference.** | Wu et al. (NAACL 2024) and Lin et al. (ICLR 2024, URIAL) analyse how instruction tuning shifts behaviour, attention and token distributions. They do not test word-level frequency or ambiguity information with paired, word-resampled confidence intervals across depth. |
| G5 | **Probe-methodology weaknesses common in this literature:** no control or baseline (Hewitt & Liang, EMNLP 2019); no correction across layers; leakage from items shared across folds; anisotropic/rogue dimensions dominating similarity (Ethayarajh, EMNLP 2019; Timkey & van Schijndel, EMNLP 2021); best-layer cherry-picking. | Addressed by design (section 4). |
| G6 | **Provenance mixing.** Published, human-designed items and dictionary-derived items differ systematically: the derived pool is longer, rarer and more orthographically isolated. | The two pools are never pooled. Stage 2 is labelled generalisation, not replication. |

**Honest correction.** Layer-wise analysis of ambiguous words in LLMs is *not* an untouched area
(Ma et al., 2025; the 2026 preprints above). The defensible novelty is the *combination* in section 5,
not "the first layer-wise ambiguity study".

---

## 3. Hypotheses and research questions

Research questions RQ1–RQ6 follow the specification. The testable hypotheses are below, with their
direction fixed in `configs/experiment.yaml` before running.

- **H1 (RQ1).** Frequency is decodable from the earliest layers, including the embeddings. The
  conditional increment beyond sense count and form controls is > 0 (max-T-corrected permutation
  test).
- **H2 (RQ2).** Sense count carries information *beyond* frequency, length and concreteness:
  ΔR² > 0 at ≥ 1 layer, max-T corrected. Its decodability centre of mass lies deeper than that of
  frequency. The second part is exploratory: centre-of-mass CIs are reported, not a formal test.
- **H3 (RQ3).** The published ambiguity factor is decodable (AUROC > 0.5) beyond sense count and
  controls in the published pool. In the derived pool, the GCIDE proxy carries information beyond
  sense count.
- **H4 (RQ4).** Semantic distance (`sim_top2` etc.) is decodable beyond sense count, i.e. distance and
  count are separable constructs in the representation.
- **H5 (RQ5).** Within each matched pair, decodability curves differ between base and instruct models,
  with the largest differences in late layers. This is two-sided: no direction is assumed. Any
  difference is attributed to "variant under its native input format", not to tuning alone.
- **H6 (RQ6, primary directional test).** On representational settling cost (RSC; higher = slower):
  β(senses: many − few) < 0 and β(ambiguity: ambiguous − unambiguous) > 0, both in one model, in
  Appendix B and in Appendix C separately. Appendix B's human ambiguity effect was weak, so a null
  there is anticipated and is not a failure.
- **H6b (exploratory time course).** Following Armstrong & Plaut (2016), the senses (facilitation)
  effect on the layer-wise distance-to-final emerges at shallower depth than the ambiguity
  (inhibition) effect. Predicted: effect-weighted depth difference > 0.
- **H7 (exploratory geometry).** In the 2×2 design, ambiguity is coded abstractly with respect to
  senses: its cross-condition generalisation exceeds the permutation null (CCGP; Bernardi et al.,
  Cell 2020).

---

## 4. How the hypotheses are tested (scientific controls)

| Requirement | Implementation (file) |
|---|---|
| No label leakage | Prompts are exactly `Word: {w}` or the native chat template with that content. Labels are joined only after extraction, and a unit test asserts the exact prompt text (`prompt_formatting.py`, `tests/`). |
| Correct representation site | Character-offset alignment to the target subtokens, with decoded-string verification. Last subtoken is primary, mean is sensitivity. The L+1 hidden-state convention and the final-norm status are checked at runtime (`extract_hidden_states.py`). |
| Valid estimator in d ≫ n | Ridge in dual form. α is chosen by *exact* leave-one-out error (Golub, Heath & Wahba, 1979), verified against the primal solution to 1e-14 (`stats_utils.py`, test). |
| Unique information (G1) | Covariates are *unpenalised* and partialled out inside each training fold by the Frisch–Waugh–Lovell theorem. ΔR² over the covariate-only baseline is a conditional probe (Hewitt et al., EMNLP 2021). A unit test shows that a layer encoding *only* frequency yields ΔR² ≈ 0 for sense count. |
| No item leakage | Repeated StratifiedGroupKFold by word, deduplicated within each analysis set. All scaling, residualisation, α selection and Platt calibration happen inside training folds. Folds are identical across models, which enables paired tests. |
| Null and multiplicity | Freedman–Lane permutation (Freedman & Lane, 1983), with α re-selected inside every permutation. The same permutations are used at every layer, giving per-layer p, BH-FDR across layers (Benjamini & Hochberg, 1995) and Westfall–Young max-T across layers. |
| Uncertainty | Word-level bootstrap CIs. Cell-stratified bootstrap for the factorial effects. HC3 standard errors (MacKinnon & White, 1985). |
| Anisotropy (G5) | The settling proxy uses per-layer, per-dimension standardisation before cosine (Timkey & van Schijndel, 2021), which also makes it invariant to the final RMSNorm gain. |
| Rodd dissociation (G3) | One effect-coded factorial model per appendix, with the interaction and controls. Operational pass rule: *supported*, *partial*, *directionally consistent* or *not recovered*. Run on the primary proxy plus secondary proxies (logit-lens KL; Belrose et al., 2023 **[preprint]**; surprisal; entropy). |
| Base vs instruct (G4) | Same words and same bootstrap draws for both members of a pair. Per-layer paired difference with BH, a centre-of-mass shift CI, paired Rodd-coefficient differences, and linear CKA (Kornblith et al., ICML 2019) as exploratory. |
| Robustness | Pooling (last/mean); control sets (none, primary, expanded, +log #subtokens); matched vs full pool; GCIDE zero excluded / recoded / indicator; noun senses; repeated CV seeds. |

---

## 5. Novelty and type of contribution

**Type of contribution.** This is a *methodological and empirical* contribution, not a new model
architecture and not a mere incremental benchmark. The novel elements, which to our knowledge have
not been combined before, are:

1. **The Rodd 2×2 factorial applied to transformer depth.** A pre-registered *representational settling
   cost* (RSC), defined as the area under 1 − cos(z_l, z_L) over relative depth, is used as a
   settling-time analogue. It links attractor-network accounts of the senses/meanings dissociation to
   the residual stream, and is tested with an explicit sign rule.
2. **A layer-wise time-course test of semantic-settling dynamics** (H6b). It turns Armstrong & Plaut's
   (2016) timing prediction into an effect-weighted-depth contrast with bootstrap CIs.
3. **Cross-condition generalisation (CCGP) on a psycholinguistic factorial design.** This imports an
   abstraction metric from systems neuroscience (Bernardi et al., 2020) to ask whether "unrelated
   meanings" is coded independently of "number of senses".
4. **Conditional, permutation-calibrated layer-wise probing** with unpenalised covariates (FWL in kernel
   form). It separates frequency, sense count, meanings proxy and semantic distance from each other,
   with max-T FWER control across layers. This addresses G1 directly rather than by item matching
   alone.
5. **Matched base/instruct pairs with paired word-level inference** on lexical constructs, without
   inventing missing counterparts.

**What it improves over prior practice:** confound-aware estimates (unique information rather than
raw decodability), calibrated nulls, multiplicity control across layers, no best-layer-only
reporting, provenance separation, and an explicit distinction between decodability and processing.

---

## 6. Are all gaps addressed, and can success be "100% assured"?

**Correction to the premise.** No honest study can assure in advance that "everything is 100%
achieved" or that the expected results will occur. That claim would itself be unscientific. Reviewers
at top venues would reject a paper that framed results as guaranteed. What the code *can* guarantee
is that the result is interpretable whichever way it comes out:

| Outcome | What it means | Still publishable? |
|---|---|---|
| H6 supported in B and/or C | LLM depth dynamics show a human-like senses/meanings dissociation under this proxy. | Yes: the central positive finding. |
| Only senses facilitation, ambiguity null | Matches the weak human B ambiguity effect, and is "partial" under the rule. | Yes, with the nuance reported. |
| Same-direction effects | The dissociation is *not* recovered: evidence that settling depth in LLMs indexes something else (e.g. frequency). | Yes, as an informative negative, provided power is reported. |
| H2 ΔR² ≈ 0 | Apparent polysemy decoding is explained by frequency and form, which corrects a common inference. | Yes. |

**Gap coverage.**

- **G1, G5, G6** are addressed by construction.
- **G2, G3 and G4** are addressed by the analyses, but the *conclusions* depend on data that do not yet
  exist. The full extraction has not been run.

**Residual limits that the code cannot remove:**

- Probe decodability does not show that the model uses the feature (Belinkov, Comput. Linguist. 2022;
  Elazar et al., TACL 2021). Causal interventions would be the next step.
- Appendix B (N = 128) and C (N = 92) are small. Report CIs, and treat C as a stimulus-set
  sensitivity analysis on text input.
- GCIDE etymologies and WordNet candidate labels are computed or rule-based, not human norms.
- Base vs instruct comparisons differ in input format (chat wrapper).

**How to judge the results:** use pre-registered sign rules, max-T and BH-corrected permutation
p-values, bootstrap CIs, and consistency across 6 pairs and 15 models rather than a single model.
Agreement across models, appendices and sensitivity variants is what makes a finding a reliable
backbone for a paper. A single significant layer is not.

---

### Key references (verify before citing)

- Armstrong & Plaut (2016). *Lang. Cogn. Neurosci.* 31(7), 940–966.
- Beekhuizen, Armstrong & Stevenson (2021). Probing lexical ambiguity: word vectors encode number and relatedness of senses. *Cognitive Science* 45(5).
- Belinkov (2022). Probing classifiers. *Computational Linguistics* 48(1).
- Bernardi et al. (2020). The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell* 183(4).
- Eddington & Tokowicz (2015). *Psychonomic Bulletin & Review* 22.
- Elazar, Ravfogel, Jacovi & Goldberg (2021). Amnesic probing. *TACL* 9.
- Garí Soler & Apidianaki (2021). Let's play mono-poly. *TACL* 9.
- Haber & Poesio (2024). Polysemy — evidence from linguistics, behavioral science, and contextualized language models. *Computational Linguistics* 50(1).
- Hewitt & Liang (2019). *EMNLP*.
- Hewitt, Ethayarajh, Liang & Manning (2021). Conditional probing. *EMNLP*.
- Kornblith et al. (2019). *ICML*.
- Ma, Xie, Wang & Wang (2025). Exploring layer-wise representations of English and Chinese homonymy in pre-trained language models. *Findings of ACL*.
- Rodd, Gaskell & Marslen-Wilson (2002). *J. Memory and Language* 46.
- Rodd, Gaskell & Marslen-Wilson (2004). *Cognitive Science* 28.
- Tenney, Das & Pavlick (2019). *ACL*.
- Timkey & van Schijndel (2021). *EMNLP*.
- Wu et al. (2024). From language modeling to instruction following. *NAACL*.
- Statistics: Benjamini & Hochberg (1995); Freedman & Lane (1983); Golub, Heath & Wahba (1979); MacKinnon & White (1985).
