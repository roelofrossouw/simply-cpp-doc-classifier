# simply-cpp-doc-classifier

`sc-doc-classifier` finds identity documents in photographs and scans, tells what they are, and cuts
them out straight and upright. It knows four kinds:

| `sc::doc::kind` | | Cut out at |
|---|---|---|
| `sa_driving_licence` | the South African driving licence card | 1012 × 638 (ID-1 at 300 dpi) |
| `sa_id_card` | the South African smart ID card | 1012 × 638 |
| `passport` | any country's passport data page | 1476 × 1039 (ID-3 at 300 dpi) |
| `sa_temporary_driving_licence` | the South African temporary driving licence (form TDL) | 2296 × 1590 (200 dpi) |

**Documentation:** the [simply-cpp wiki](https://github.com/roelofrossouw/simply-cpp/wiki) has the [sc-doc-classifier reference](https://github.com/roelofrossouw/simply-cpp/wiki/Reference-sc-doc-classifier).

```cpp
#include <sc_doc_classifier.h>

const sc::image photo{"scan.jpg"};
for (const auto &found: sc::doc::find_documents(photo, sc::doc::method::text)) {
    found.picture.save(sc::doc::short_name(found.type) + ".png");   // straight and upright
    found.confidence;      // how sure, from the printed words read in their places
    found.where.corners;   // where it lies in the photo, its top left first
}
```

The licence and the ID card are both ID-1 cards (85.6 × 54 mm, a bank card's size), often photographed
side by side, so a card's outline doesn't say which it is. Their printed words do, each in its own place
on the card: *DRIVING LICENCE*, *CARTA DE CONDUCAO*, *Lic. No.* on the one, *NATIONAL IDENTITY CARD*,
*Surname*, *Names* on the other. A passport page is known by its two machine readable lines, which ICAO
9303 puts in the same place on every passport, and their check digits. The temporary licence is a green
A4 form with labels in English and Afrikaans all over it. Each kind needs one of its own words read in
place: a passport has *Surname* and *REPUBLIC OF SOUTH AFRICA* too, but not *NATIONAL IDENTITY CARD*.

## Three ways to find one

| `sc::doc::method` | Finds the document by | Then |
|---|---|---|
| `contours` | closed outlines of edges with four corners, in a card's proportions; the sides refitted past the rounded corners | reads each for the kinds' words |
| `lines` | straight edges (line segments) paired into four sides at nearly right angles, so a broken outline still counts | reads each for the kinds' words |
| `text` | its printed labels, read from the whole image: where they are gives where it is (RANSAC, so misreads drop out) | snaps its edges to the nearest outline |

`text` is the one to use: on 132 sample photos and scans it found 57 of 60 ID cards, 42 of 43 passports
and all 6 licences, with no false finds; it also places a document whose edges run off the image. The
outline methods find fewer (41 to 44 ID cards) and take longer, but show what edges alone can do.

An outline is scored by how much of each side runs along an edge that crosses it the same way all
along (the card lighter, or darker, than what it's on), so a patterned background, which has edges
pointing every which way, doesn't make cards of its own. A document's words count only **where they
belong** on it: half a card, or the box printed inside one, isn't one. Of outlines that overlap (a
card, the plastic sleeve it's in, the boxes printed on it) each is tried in turn.

`find_cards()` gives the card-shaped outlines alone, `straightened()` cuts one out, `identify()` reads
a cut-out, and `find_driving_licences()`, `find_id_cards()`, `find_passports()` and
`find_temporary_licences()` look for one kind.

## The demos

Each shows one method on the images (or folders of images) given, writing each image with what was
found drawn on it and each document cut out:

```bash
sc-doc-classifier-text photos/ -o /tmp/found
sc-doc-classifier-lines photos/ --cards           # every card-shaped outline too
sc-doc-classifier-contours photos/ --find passport
```

## Building

```bash
cmake -B cmake-build-local -S .
cmake --build cmake-build-local
ctest --test-dir cmake-build-local
```

It needs sc-core, sc-image and sc-ai (OCR, with simply-cpp-models), OpenCV (4 or 5) and a C++20
compiler.

## License

Apache License 2.0, see [LICENSE](LICENSE).
