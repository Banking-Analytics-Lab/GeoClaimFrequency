# GeoClaimFrequency 🛰️🗺️📊

This repository contains the formulation and results of the research project conducted in 2025-2026: *"Revealing Geography-Driven Signals in Zone-Level Claim Frequency Models:
An Empirical Study using Environmental and Visual Predictors"*; this work was done by Ph.D. candidate Ms. Sherly Alfonso-Sánchez, under the supervision of Dr. Kristina G. Stankova and Dr. Cristián Bravo Roman (2026). 

## Abstract 

Geographic context is often considered relevant for assessing motor insurance risk, yet public actuarial datasets typ-
ically provide limited location identifiers, constraining how this information can be incorporated in claim-frequency
models. This study examines how geographic information from alternative data sources can be incorporated into
actuarial models for Motor Third Party Liability (MTPL) claim prediction under such constraints.
Using the BeMTPL97 dataset, we adopt a zone-level modeling framework and evaluate predictive performance on
postcodes not observed during training. Geographic information is introduced through two channels: environmental
indicators from OpenStreetMap (2014) and CORINE Land Cover 2000, and orthoimagery from 1995 released by
the Belgian National Geographic Institute for academic use. We evaluate the predictive contribution of coordinates,
environmental features, and image embeddings across three baseline models: generalized linear models (GLMs),
regularized GLMs, and gradient-boosted trees, while raw imagery is modeled using convolutional neural networks.
Our results show that augmenting actuarial variables with constructed geographic information improves predictive
accuracy. Across experiments, both linear and tree-based models benefit most from combining coordinates with
environmental features extracted at 5 km scale, while smaller neighborhoods also improve baseline specifications.
Generally, image embeddings do not improve performance when structured environmental features are available;
however, when such features are absent, pretrained vision-transformer embeddings enhance accuracy and stability for
regularized GLMs. Our results show the predictive value of geographic information in zone-level MTPL frequency
models depends less on model complexity than on how geography is represented, and illustrate that geographic context
can be incorporated despite limited individual-level spatial information.
