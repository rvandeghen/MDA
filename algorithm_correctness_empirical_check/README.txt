Introduction
------------

This library is intended to demonstrate the correctness of the unsupervised
domain adaptation algorithm given in the paper entitled "Mixture Domain
Adaptation to Improve Semantic Segmentation in Real-World Surveillance" by
S�bastien Pi�rard, Anthony Cioppa, Ana�s Halin, Renaud Vandeghen,
Maxime Zanella, Beno�t Macq, Sa�d Mahmoudi, and Marc Van Droogenbroeck,
published at the "IEEE/CVF Winter Conference on Applications of Computer
Vision" (WACV), Hawa�, January 3-7 2023.

This library provides a reference matlab implemention of the algorithm.
Please note that the goal of this implementation is to provide an easy to
read code; this implementation is NOT optimized for processing speed.
Both the off-line and the on-the-fly steps are implemented. The source
models and the domain discriminator model are "exact up to a target shift"
(see the definition given in Section 3.1 of the paper). To ensure this,
these models are derived theoretically in this implementation, without 
using any learning set and any machine learning technique.

This code tests the algorithm on 10,000 random cases with a number of
evidences between 2 and 10, a number of hypotheses between 2 and 10,
a number of source domains also between 2 and 10, randomly chosen
probability measures in the source domains, and randomly chosen mixture
weights.

Compatibility
-------------

This code has been tested with MATLAB R2016b. Minor adaptations could be
required with other versions of MATLAB or with OCTAVE.

Usage
-----

In order to validate experimentally the correctness of the algorithm given
in Section 3.4 of the paper, run the command "check_our_algorithm ()".
It should take a few seconds to execute and return the string
'The algorithm seems to behave as an exact model.'.

It is not a formal proof of the correctness of the algorithm (such a proof
is provided in Section 3.3 of the paper). It just means that the posteriors
obtained by the algorithm for the target domain correspond (up to a small
tolerance, necessary for numerical reasons) to the true posteriors in the
target domain, in the 10,000 randomly chosen cases.

Implementation details
----------------------

All distributions are represented by three-dimentional matrices. The first
dimension stands for the the evidences, the second one for the hypotheses,
and the last one for the source domains. The elements of these matrices are
the probbailities for the corresponding evidence, hypothesis and source
domain.

