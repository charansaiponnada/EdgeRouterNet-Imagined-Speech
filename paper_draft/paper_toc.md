# Table of Contents: EdgeRouterNet Paper Refinement Plan

This ToC outlines the section-by-section refinement strategy to ensure the paper meets the 6-page minimum and the technical standards for IEEE DELCON 2026.

## 1. Front Matter & Abstract
- **Title**: Refine for impact and SEO (SOTA visibility).
- **Abstract**: 5-sentence formula polish (What, Why, How, Evidence, Remarkable Result).
- **Keywords**: Ensure compliance with IEEE and BCI taxonomy.

## 2. Section I: Introduction
- **I-A. BCI Motivation**: Deepen clinical context (ALS, Locked-in syndrome).
- **I-B. The Problem**: Elaborate on EEG low SNR and non-stationarity.
- **I-C. Current Limitations**: Critique handcrafted features vs. generic CNNs.
- **I-D. Proposed Innovations**: Explicitly list the 4 key contributions.

## 3. Section II: Related Work
- **II-A. Cognitive Foundations**: Link imagined speech to the "dorsal stream".
- **II-B. BCI Architectures**: Compare EEGNet, DeepConvNet, and Transformers.
- **II-C. Metric Learning**: Discuss the evolution from Softmax to Angular Margin.

## 4. Section III: Methodology
- **III-A. Data & Preprocessing**: (Table I) Subject demographics & ICA artifacts.
- **III-B. EdgeRouterNet Core**: (Table II & Eq 1-2) CSA and Multi-scale blocks.
- **III-C. Attention Mechanisms**: (Eq 3) ECA and Temporal Attention Pooling.
- **III-D. ArcFace Loss**: (Eq 4-5) Geodesic distance & Multi-task formulation.
- **III-E. Ensemble Strategy**: (Alg 1) Adaptive Inference & Confidence Filtering.

## 5. Section IV: Experiments & Results
- **IV-A. Setup**: Cross-validation details (trial-safe) and hyperparameters.
- **IV-B. SOTA Comparison**: (Table III & Fig 2) Comparison against baselines.
- **IV-C. Per-Class Analysis**: (Table IV) Precision/Recall/F1 breakdown.
- **IV-D. Ablation Studies**: Dissecting the impact of CSA, ArcFace, and MS-Blocks.

## 6. Section V: Discussion & Impact
- **V-A. Phonetic Routing**: Technical analysis of phonetic priors.
- **V-B. Information Bottleneck**: How ArcFace filters EEG noise.
- **V-C. Socio-Economic Impact**: Caregiving costs and vocational inclusion.
- **V-D. Ethics & Privacy**: Neural privacy and biometric risks.

## 7. Section VI: Conclusion & Future Work
- **VI-A. Summary**: Recapping SOTA achievements.
- **VI-B. Future Directions**: EEG Foundation Models and real-time closed-loop.

## 8. Back Matter
- **References**: Final sync of `refs.bib` with 40+ citations.
- **Appendix (Optional)**: Check if any hyperparameter tables need moving to fit length.
