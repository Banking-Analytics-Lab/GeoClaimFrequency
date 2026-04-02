# GeoClaimFrequency 🛰️🗺️📊

This repository contains the formulation and results of the research project (2025–2026):

*"Revealing Geography-Driven Signals in Zone-Level Claim Frequency Models:  
An Empirical Study using Environmental and Visual Predictors"*

This work was conducted by Ph.D. candidate **Sherly Alfonso-Sánchez**, under the supervision of **Dr. Kristina G. Stankova** and **Dr. Cristián Bravo Roman** (2026).

---

## Abstract

Geographic context is often considered relevant for assessing motor insurance risk; however, public actuarial datasets typically provide limited location identifiers, restricting how this information can be incorporated into claim-frequency models.

This study investigates how geographic information from alternative data sources can be integrated into actuarial models for Motor Third Party Liability (MTPL) claim prediction under such constraints.

Using the BeMTPL97 dataset ([CAS Dataset Manual](https://cas.uqam.ca/pub/web/CASdatasets-manual.pdf)), we adopt a zone-level modeling framework and evaluate predictive performance on postcodes not observed during training.

Geographic information is introduced through two channels:
- Environmental indicators from OpenStreetMap (2014) and CORINE Land Cover 2000  
- Orthoimagery from 1995 released by the Belgian National Geographic Institute (for academic use)

We evaluate the predictive contribution of:
- Geographic coordinates  
- Environmental features
- Othoimagery
- Image embeddings  

across three baseline models:
- Generalized Linear Models (GLMs)  
- Regularized GLMs  
- Gradient-boosted trees
- Raw imagery is modeled using convolutional neural networks.

### Key Findings

- Augmenting actuarial variables with geographic information improves predictive accuracy  
- The best performance on average through the experiments is obtained by combining coordinates with environmental features at a 5 km scale  
- Smaller spatial scales also improve baseline models  
- Image embeddings do not improve performance when structured environmental features are available  
- However, when such features are absent, pretrained vision-transformer embeddings improve accuracy and stability, especially for regularized GLMs.  

Overall, the predictive value of geographic information depends less on model complexity than on how geography is represented, demonstrating that meaningful spatial signals can be extracted even with limited location data.

---

## Repository Structure

The scripts are organized in the following order:

1. **Preprocessing and Aggregation**  
2. **Alternative Geographic Data**  
3. **Data Splits by Experiment**  
4. **Zone-Level Frequency Models**
