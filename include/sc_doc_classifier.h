#pragma once
// Finds documents in photographs and scans, tells what they are, and cuts them out straight.
//
// For now four kinds: the South African driving licence and smart ID card, both ID-1 cards
// (85.6 x 54 mm, the size of a bank card), any country's passport data page (ID-3, 125 x 88 mm), and
// the South African temporary driving licence (a green A4 form).
// The cards are often photographed side by side, and being the same size their outlines don't say
// which is which: their printed words do, each in its own place. A passport page is known by its
// machine readable lines and their check digits.
//
//     const sc::image photo{"scan.jpg"};
//     for (const auto &found: sc::doc::find_documents(photo, sc::doc::method::text))
//         found.picture.save(short_name(found.type) + ".png");   // straight and upright, size_of(type)
//
// Three ways to find a card, to compare (each demo shows one):
//
//   contours  the outlines of edges: a card stands out from what it lies on as a closed outline
//             with four corners, in a bank card's proportions.
//   lines     straight edges: a card's four sides as two pairs of lines, kept when its outline is
//             broken (a finger, glare, a card beside it of the same colour).
//   text      the documents' own printed words (DRIVING LICENCE, ID No., Lic. No.; NATIONAL IDENTITY
//             CARD, Surname, Names, ...; a passport's machine readable lines): where they are read
//             gives where the document is, even when its edges can't be seen.
//
// A card found by its outline is then read (OCR) to say what it is, and which way up.

#include <opencv.h>
#include <percent.h>
#include <point.h>
#include <polygon.h>
#include <size.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace sc::doc {
    // An ID-1 card: 85.6 x 54 mm
    inline constexpr double card_aspect = 85.6 / 54.0;

    // A straightened card's size, in pixels: ID-1 at 300 dpi
    inline const size_i card_size{1012, 638};

    enum class method { contours, lines, text };

    // A passport's data page: ID-3, 125 x 88 mm, at 300 dpi
    inline const size_i passport_page_size{1476, 1039};

    // A temporary driving licence (form TDL) to the outer edge of its green border, about 292 x 202
    // mm, at 200 dpi
    inline const size_i temporary_licence_size{2296, 1590};

    // sa_*: South Africa's own; passport: any country's data page (its machine readable lines are
    // the same everywhere)
    enum class kind { unknown, sa_driving_licence, sa_id_card, passport, sa_temporary_driving_licence };

    [[nodiscard]] std::string to_string(method how);
    [[nodiscard]] std::string to_string(kind what);
    // A word or two for file names: "licence", "id-card", "passport", "temporary-licence"
    [[nodiscard]] std::string short_name(kind what);

    // A kind's size straightened: card_size, passport_page_size or temporary_licence_size
    [[nodiscard]] size_i size_of(kind what);

    // Where a card-shaped thing lies in an image: its four corners in the image's pixels, clockwise
    // from the one that was its top left (as far as the outline can tell: a card upside down has
    // its corners the other way round until identify() turns it), and how card-like it looked.
    struct card {
        std::array<point, 4> corners;
        double fit = 0; // 0 to 1: how well the outline matches a card's shape and edges

        [[nodiscard]] polygon outline() const;
    };

    // Cards found by their outlines, best first, none overlapping. The image may be a photograph or a
    // scan, in colour or grey, any size (it is looked at about 1000 pixels across).
    [[nodiscard]] std::vector<card> find_cards(const image &picture, method how);

    // The card cut out of the image, straight: its corners mapped to a rectangle of size (in a card's
    // proportions, landscape).
    [[nodiscard]] image straightened(const image &picture, const card &where, size_i size = card_size);

    // What a straightened card is, read from its printed words.
    struct identification {
        kind type = kind::unknown;
        percent confidence{0};
        int rotation = 0;               // how far it was turned to be upright (0 or 180 degrees)
        std::vector<std::string> words; // the printed words that told it, as read
    };

    // Reads a straightened card or page (both ways up, any size: its words are placed as shares of
    // it) for each kind's printed words, each counted only where it belongs: half a card, or a box
    // printed inside one, isn't one. A passport is known by its machine readable lines and their
    // check digits.
    [[nodiscard]] identification identify(const image &document_picture);

    // A card found in an image.
    struct found_document {
        kind type = kind::unknown;
        card where;          // corners in the image, the card's top left first once upright
        image picture;       // straight and upright, size_of(type)
        percent confidence{0};
        method how = method::contours;
        std::vector<std::string> words;
    };

    // Cards of the kinds wanted (all when kinds is empty) in an image, most certain first: cards found
    // by how, kept when identify() says what they are (method::text finds and identifies in one go).
    [[nodiscard]] std::vector<found_document> find_documents(const image &picture, method how = method::contours,
                                                             const std::vector<kind> &kinds = {});

    // find_documents() for one kind
    [[nodiscard]] std::vector<found_document> find_driving_licences(const image &picture, method how = method::contours);
    [[nodiscard]] std::vector<found_document> find_id_cards(const image &picture, method how = method::contours);
    [[nodiscard]] std::vector<found_document> find_passports(const image &picture, method how = method::contours);
    [[nodiscard]] std::vector<found_document> find_temporary_licences(const image &picture, method how = method::contours);
}
