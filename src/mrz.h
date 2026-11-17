#pragma once
// ICAO 9303 machine readable zones: the check digit both readers use (sc_doc_classifier.cpp to tell
// a passport page by its lines, identity.cpp to trust the fields read from them).

#include <cstddef>
#include <string_view>

namespace sc::doc_classifier::mrz {
    // A character's value in a check digit sum: digits 0 to 9, letters 10 to 35, anything else
    // (the < filler) 0.
    inline int value(const char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
        return 0;
    }

    // The check digit of field: the values weighted 7, 3, 1 in turn, summed, modulo 10.
    inline int check_digit(const std::string_view field) {
        constexpr int weights[] = {7, 3, 1};
        int total = 0;
        for (std::size_t i = 0; i < field.size(); ++i) total += value(field[i]) * weights[i % 3];
        return total % 10;
    }

    // Whether check is field's check digit.
    inline bool check_matches(const std::string_view field, const char check) {
        return check >= '0' && check <= '9' && check_digit(field) == check - '0';
    }
}
