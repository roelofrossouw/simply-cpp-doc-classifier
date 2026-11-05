// Cards drawn on a background, tilted and in perspective, found by their outlines and their straight
// edges with the corners where they were drawn, and cut out at a card's size. No OCR here: what a card
// is needs real printing, which the demos show on photographs.
#include <sc_doc_classifier.h>

#include "sc_test.h"

#include <algorithm>
#include <cmath>

namespace {
    // A pale card on a darker, slightly textured background: corners clockwise from the top left
    sc::image drawn(const std::array<sc::point, 4> &corners, const sc::size_i size = {1200, 900}) {
        sc::image picture{size, sc::color{0.25, 0.3, 0.35}};
        for (int x = 0; x < size.width(); x += 40)
            picture.draw(sc::line{sc::point{x, 0}, sc::point{x + 300, size.height()}}, sc::appearance{sc::stroke{sc::color{0.3, 0.35, 0.4}, 2}});
        picture.draw(sc::polygon{std::vector<sc::point>(corners.begin(), corners.end())},
                     sc::appearance{sc::stroke{}, sc::fill{sc::color{0.93, 0.9, 0.85}}});
        return picture;
    }

    // The largest distance from a corner found to the corner drawn, whichever corner it starts at
    double corner_error(const sc::doc::card &found, const std::array<sc::point, 4> &drawn) {
        double best = 1e9;
        for (int shift = 0; shift < 4; ++shift) {
            double worst = 0;
            for (int i = 0; i < 4; ++i) {
                const auto a = found.corners[(i + shift) % 4];
                const auto b = drawn[i];
                worst = std::max(worst, std::hypot(a.x() - b.x(), a.y() - b.y()));
            }
            best = std::min(best, worst);
        }
        return best;
    }

    // A card of width pixels centred at (cx, cy), turned by degrees
    std::array<sc::point, 4> turned(const double cx, const double cy, const double width, const double degrees) {
        const double height = width / sc::doc::card_aspect;
        const double a = degrees * std::acos(-1) / 180;
        std::array<sc::point, 4> corners;
        const double xs[] = {-width / 2, width / 2, width / 2, -width / 2};
        const double ys[] = {-height / 2, -height / 2, height / 2, height / 2};
        for (int i = 0; i < 4; ++i)
            corners[i] = sc::point{cx + xs[i] * std::cos(a) - ys[i] * std::sin(a), cy + xs[i] * std::sin(a) + ys[i] * std::cos(a)};
        return corners;
    }
}

int main() {
    using sc::doc::method;

    SECTION("A card lying straight, and turned, found both ways with its corners");
    for (const double degrees: {0.0, 7.0, -18.0, 35.0}) {
        const auto corners = turned(600, 450, 700, degrees);
        const auto picture = drawn(corners);
        for (const auto how: {method::contours, method::lines}) {
            const auto cards = sc::doc::find_cards(picture, how);
            CHECK_EQ(cards.size(), 1u);
            if (cards.empty()) continue;
            CHECK_MSG(corner_error(cards[0], corners) < 6, sc::doc::to_string(how) + " at " + std::to_string(degrees) + " degrees");
            CHECK(cards[0].fit > 0.7);
        }
    }

    SECTION("In perspective: a photo taken at an angle");
    {
        const std::array<sc::point, 4> corners{sc::point{250, 200}, sc::point{980, 240}, sc::point{1010, 700}, sc::point{200, 690}};
        const auto picture = drawn(corners);
        for (const auto how: {method::contours, method::lines}) {
            const auto cards = sc::doc::find_cards(picture, how);
            CHECK_EQ(cards.size(), 1u);
            if (!cards.empty()) CHECK_MSG(corner_error(cards[0], corners) < 6, sc::doc::to_string(how));
        }
    }

    SECTION("Two cards side by side are two");
    {
        sc::image picture{sc::size_i{1600, 900}, sc::color{0.2, 0.25, 0.2}};
        const auto left = turned(420, 450, 620, 4);
        const auto right = turned(1180, 460, 620, -6);
        for (const auto &corners: {left, right})
            picture.draw(sc::polygon{std::vector<sc::point>(corners.begin(), corners.end())},
                         sc::appearance{sc::stroke{}, sc::fill{sc::color{0.95, 0.93, 0.9}}});
        for (const auto how: {method::contours, method::lines}) {
            const auto cards = sc::doc::find_cards(picture, how);
            CHECK_EQ(cards.size(), 2u);
            if (cards.size() == 2) {
                const double error = std::min(std::max(corner_error(cards[0], left), corner_error(cards[1], right)),
                                              std::max(corner_error(cards[0], right), corner_error(cards[1], left)));
                CHECK_MSG(error < 6, sc::doc::to_string(how));
            }
        }
    }

    SECTION("Not cards: a square, a long strip, nothing");
    {
        const std::array<sc::point, 4> square{sc::point{300, 200}, sc::point{800, 200}, sc::point{800, 700}, sc::point{300, 700}};
        const std::array<sc::point, 4> strip{sc::point{100, 400}, sc::point{1100, 400}, sc::point{1100, 550}, sc::point{100, 550}};
        for (const auto how: {method::contours, method::lines}) {
            CHECK(sc::doc::find_cards(drawn(square), how).empty());
            CHECK(sc::doc::find_cards(drawn(strip), how).empty());
            CHECK(sc::doc::find_cards(sc::image{sc::size_i{800, 600}, sc::color{0.5, 0.5, 0.5}}, how).empty());
        }
    }

    SECTION("Cut out straight at a card's size");
    {
        const auto corners = turned(600, 450, 700, 12);
        const auto picture = drawn(corners);
        const auto cards = sc::doc::find_cards(picture, method::contours);
        CHECK_EQ(cards.size(), 1u);
        if (!cards.empty()) {
            const auto flat = sc::doc::straightened(picture, cards[0]);
            CHECK_EQ(flat.size().width(), sc::doc::card_size.width());
            CHECK_EQ(flat.size().height(), sc::doc::card_size.height());
            CHECK_EQ(sc::doc::straightened(picture, cards[0], {400, 252}).size().width(), 400);
        }
        CHECK_THROWS_AS((void) sc::doc::find_cards(sc::image{}, method::contours), std::invalid_argument);
    }

    TEST_SUMMARY();
}
