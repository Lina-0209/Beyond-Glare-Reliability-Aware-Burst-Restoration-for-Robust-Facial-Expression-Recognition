# Third-party boundaries and release checklist

The MIT license in this repository applies to project-owned code only. It does not
relicense dependencies, model weights, source photographs, expression labels, or
third-party source code that a user obtains separately. This file is an inventory,
not a legal certification that every possible use is permitted.

- **NAFNet**: the compact NAF-style blocks and project single-frame baseline are
  implemented in this tree. Acknowledge the NAFNet architecture when describing
  the method; do not describe it as a newly invented block. If any code originated
  verbatim from upstream, preserve the actual upstream notice before publication.
- **EmotiEffLib**: installed separately. The enet_b2_7 checkpoint and its training
  data are not covered by this repository's MIT license. Obtain the model from
  EmotiEffLib, review its terms and keep relevant attribution.
- **POSTER V2**: adapter code is included, but no upstream model implementation,
  checkpoint, or pretraining weights are vendored. Install upstream dependencies
  in a separate compatible environment. The AffectNet-7 output crosswalk is the
  project's recorded mapping; it is not a claim of new official label verification.
- **MediaPipe / OpenCV**: packages and face-detection assets have independent
  terms. `face_landmarker.task` is deliberately not redistributed here.
- **LPIPS / PyTorch / torchvision / ONNX Runtime** and other packages: retain their
  own licenses and any downloaded pretrained-backbone restrictions.
- **CelebAMaskHQ / FFHQ / RAF-DB / FER2013**: no source images or licensed labelled
  datasets are distributed. Dataset access, derivative generation, permitted uses
  and redistribution must be checked independently. Naming a synthetic dataset
  LensGlare-4 does not grant rights to its source images or derivatives.

Before public release, the owner should confirm authorship/provenance of the
project-owned files, all required architecture acknowledgments, and applicable
institutional or group publication requirements. Adding MIT is not a substitute
for that review. Do not include credentials, copyrighted reports, student imagery,
person-level records, or group infrastructure details in the public repository.
