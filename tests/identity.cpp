#include <sc/doc-classifier/identity.h>
#include <sc_test.h>
#include <sc/core/base64.h>
#include <sc/image/opencv.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

namespace {
    constexpr auto VALID = "8001015009087";
    constexpr auto VALID_FEMALE = "9912310001083";
    constexpr auto BAD_CHECK_DIGIT = "8001015009086";
    constexpr auto BAD_MONTH = "8013015009082";
    constexpr auto BAD_CITIZENSHIP = "8001015009285";
}

int main() {
    SECTION("Identity number validation");
    {
        CHECK(sc::doc::identity::valid_id_number(VALID));
        CHECK(sc::doc::identity::valid_id_number(VALID_FEMALE));
        CHECK(!sc::doc::identity::valid_id_number(BAD_CHECK_DIGIT));
        CHECK(!sc::doc::identity::valid_id_number(BAD_MONTH));
        CHECK(!sc::doc::identity::valid_id_number(BAD_CITIZENSHIP));
        CHECK(!sc::doc::identity::valid_id_number(""));
        CHECK(!sc::doc::identity::valid_id_number("800101500908"));
        CHECK(!sc::doc::identity::valid_id_number("80010150090855"));
        CHECK(!sc::doc::identity::valid_id_number("80010150090X5"));
    }

    SECTION("Fields derived from an identity number");
    {
        CHECK_EQ(sc::doc::identity::id_date_of_birth(VALID), string{"1980-01-01"});
        CHECK_EQ(sc::doc::identity::id_date_of_birth(VALID_FEMALE), string{"1999-12-31"});
        CHECK_EQ(sc::doc::identity::id_sex(VALID), string{"M"});
        CHECK_EQ(sc::doc::identity::id_sex(VALID_FEMALE), string{"F"});
        CHECK_EQ(sc::doc::identity::id_date_of_birth(BAD_CHECK_DIGIT), string{});
        CHECK_EQ(sc::doc::identity::id_sex(BAD_CHECK_DIGIT), string{});
    }

    SECTION("Empty result");
    {
        const sc::doc::identity nothing;
        CHECK(!nothing.has_id_number());
        CHECK(nothing.document_type() == sc::doc::identity::document::unknown);
        CHECK_EQ(nothing.document_type_name(), string{"unknown"});
        CHECK_EQ(static_cast<double>(nothing.confidence()), 0.0);
        CHECK(!nothing.citizen().has_value());
        CHECK_EQ(nothing.to_json()["id_number"].get<string>(), string{});
    }

    SECTION("Reading a passport");
    {
        // A specimen with made-up data (resource/samples).
        const sc::doc::identity passport{"resource/samples/passport.jpg"};
        CHECK_EQ(passport.id_number(), string{VALID});
        CHECK_EQ(passport.surname(), string{"SAMPLESON"});
        CHECK_EQ(passport.names(), string{"ALEX JORDAN"});
        CHECK_EQ(passport.passport_number(), string{"A12345678"});
        CHECK(passport.has_id_number());
        CHECK(passport.document_type() == sc::doc::identity::document::passport);
        CHECK(sc::doc::identity::valid_id_number(passport.id_number()));
        CHECK(passport.checks().mrz);
        CHECK(passport.checks().luhn);
        CHECK_BETWEEN(static_cast<double>(passport.confidence()), 90.0, 100.0);
        CHECK(!passport.passport_number().empty());
        CHECK(!passport.date_of_expiry().empty());
        CHECK_EQ(passport.date_of_birth(), sc::doc::identity::id_date_of_birth(passport.id_number()));
        CHECK_EQ(passport.sex(), sc::doc::identity::id_sex(passport.id_number()));
    }

    SECTION("An identity read elsewhere, from JSON");
    {
        // As a service sends it: to_json() with the lines read and each face's picture as base64 PNG
        sc::doc::identity card{std::filesystem::path{"resource/samples/card.jpg"}, true};
        auto sent = card.to_json();
        sent["text"] = card.all_text();
        for (size_t i = 0; i < card.faces().size(); ++i)
            if (const auto *picture = card.faces()[i].face_image())
                sent["faces"][i]["image"] = sc::base64::encode(picture->encoded("png"));
        auto read = sc::doc::identity::from_json(sent);
        CHECK_EQ(read.id_number(), card.id_number());
        CHECK(read.document_type() == card.document_type());
        CHECK_EQ(read.names(), card.names());
        CHECK_EQ(read.surname(), card.surname());
        CHECK_EQ(static_cast<double>(read.confidence()), static_cast<double>(card.confidence()));
        CHECK_EQ(read.rotation(), card.rotation());
        CHECK_EQ(read.all_text(), card.all_text());
        CHECK_EQ(read.faces().size(), card.faces().size());
        for (size_t i = 0; i < read.faces().size() && i < card.faces().size(); ++i) {
            CHECK(read.faces()[i].has_image());
            CHECK_BETWEEN(static_cast<double>(read.faces()[i] ^ card.faces()[i]), 99.9, 100.1);
        }
        CHECK_EQ(read.to_json().dump(), card.to_json().dump());
        CHECK_THROWS_AS(sc::doc::identity::from_json(nlohmann::ordered_json{{"version", 2}}), invalid_argument);
    }

    TEST_SUMMARY();
}
