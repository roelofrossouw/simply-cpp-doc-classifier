# Specimen documents

Made-up identity documents for the tests and demos: a smart ID card (also turned sideways), a
passport data page with a machine readable zone whose check digits agree, and a green ID book page.
The people, numbers and dates are invented, and each is marked "SPECIMEN - NOT A VALID DOCUMENT".

The JPEGs were rendered from the SVGs beside them (in Arial or Helvetica) and are kept as they
are, so the tests read the same pixels on every machine. sc-image's `sc::svg2png` draws text in
OpenCV's Hershey fonts, so it can't render them again like for like.

TODO: re-render them once sc-image draws text in real fonts.

Real documents never go in the repository: put them in `tests/resource/private/` (git-ignored),
where the tests that need them - real faces, real photographs - look for them, and skip without.
