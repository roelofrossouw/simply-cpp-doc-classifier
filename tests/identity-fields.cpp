#include <sc/doc-classifier/identity.h>
#include <sc/ai/onnx.h>
#include <sc_test.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    // Specimens with made-up data (resource/samples, rendered from the SVGs beside them).
    SECTION("Reading a smart ID card, photographed sideways");
    {
        // Sideways, so it covers turning it upright too (the upright card.jpg adds nothing but time).
        const sc::doc::identity card{"resource/samples/card-sideways.jpg"};
        CHECK(card.rotation() == 90 || card.rotation() == 270);
        CHECK(card.document_type() == sc::doc::identity::document::id_card);
        CHECK_EQ(card.id_number(), std::string{"8001015009087"});
        CHECK_EQ(card.surname(), std::string{"SAMPLESON"});
        CHECK_EQ(card.names(), std::string{"ALEX JORDAN"});
        CHECK_EQ(card.date_of_birth(), std::string{"1980-01-01"});
        CHECK_EQ(card.sex(), std::string{"M"});
        CHECK(card.checks().luhn && card.checks().date_of_birth && card.checks().sex);
    }

    SECTION("Reading an ID book, surname starting with VAN");
    {
        // VAN is also the Afrikaans label for surname: VAN DER MERWE was read as DER MERWE.
        const sc::doc::identity book{"resource/samples/book.jpg"};
        CHECK(book.document_type() == sc::doc::identity::document::id_book);
        CHECK_EQ(book.id_number(), std::string{"9912310001083"});
        CHECK_EQ(book.surname(), std::string{"VAN DER MERWE"});
        CHECK_EQ(book.names(), std::string{"ANNA-MARIE"});
        CHECK_EQ(book.sex(), std::string{"F"});
    }

    // Real documents are kept out of the repository, in resource/private/documents (git-ignored).
    // Where they are present, a partial card from them is read too - checked for shape only. A
    // real photo takes over 20 seconds on a CPU, so only where models are accelerated (a GPU,
    // or CoreML), unless SC_AI_SLOW_TESTS=1 asks for it.
    SECTION("Reading a partial smart ID card (private documents)");
    const char *slow_tests = std::getenv("SC_AI_SLOW_TESTS");
    const bool slow_ok = sc::onnx::accelerated() || (slow_tests && std::string{slow_tests} == "1");
    if (const std::filesystem::path card_path{"resource/private/documents/ID-10.jpg"}; !slow_ok) {
        std::cout << "   (models on the CPU: skipped; SC_AI_SLOW_TESTS=1 runs it)" << std::endl;
    } else if (std::filesystem::exists(card_path)) {
        const sc::doc::identity card{card_path};
        CHECK(!card.has_id_number());
        CHECK(card.document_type() == sc::doc::identity::document::id_card);
        CHECK(!card.names().empty());
    } else {
        std::cout << "   (no private documents: skipped)" << std::endl;
    }
    TEST_SUMMARY();
}
