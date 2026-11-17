#include <sc/doc-classifier/identity.h>
#include "mrz.h"
#include <sc/ai/facedetector.h>
#include <sc/image/image.h>
#include <sc/ai/ocr.h>
#include <sc/ai/onnx.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std;

namespace sc::doc_classifier {
    namespace impl {
        // A recognised line of text. Words are kept both spaced, for reading
        // labelled fields, and packed, because the identity number on a green
        // ID book is printed in widely separated groups that the recogniser
        // reports as separate words.
        struct text_line {
            string text; // words joined with single spaces
            string packed; // words joined with nothing
            string alnum; // packed, uppercased, punctuation removed
            double confidence{}; // mean word confidence, 0-100
            rect box;
        };

        static string upper(string value) {
            ranges::transform(value, value.begin(),
                              [](const unsigned char c) { return static_cast<char>(toupper(c)); });
            return value;
        }

        static string keep_alnum(const string &value) {
            string result;
            result.reserve(value.size());
            for (const unsigned char c: value) if (isalnum(c)) result += static_cast<char>(toupper(c));
            return result;
        }

        static string keep_digits(const string &value) {
            string result;
            result.reserve(value.size());
            for (const unsigned char c: value) if (isdigit(c)) result += static_cast<char>(c);
            return result;
        }

        static string trim(const string &value) {
            const auto begin = value.find_first_not_of(" \t\r\n:./,-");
            if (begin == string::npos) return {};
            const auto end = value.find_last_not_of(" \t\r\n:./,");
            return value.substr(begin, end - begin + 1);
        }

        static string clean_name(const string &value) {
            string result;
            for (const unsigned char c: upper(trim(value))) {
                if (isalpha(c) || c == ' ' || c == '-' || c == '\'') result += static_cast<char>(c);
            }
            return trim(result);
        }

        static size_t word_count(const string &value) {
            size_t count{};
            bool in_word = false;
            for (const unsigned char c: value) {
                if (isalpha(c)) {
                    if (!in_word) ++count;
                    in_word = true;
                } else {
                    in_word = false;
                }
            }
            return count;
        }

        // The space between two boxes side by side, or 0 when they overlap that way.
        static double horizontal_gap(const rect &a, const rect &b) {
            return max({0.0, a.left() - b.right(), b.left() - a.right()});
        }

        static double vertical_gap(const rect &a, const rect &b) {
            return max({0.0, a.top() - b.bottom(), b.top() - a.bottom()});
        }

        static constexpr array<string_view, 12> MONTHS = {
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
        };

        static int current_two_digit_year() {
            const auto now = chrono::system_clock::now();
            const auto days = chrono::floor<chrono::days>(now);
            const chrono::year_month_day today{days};
            return static_cast<int>(today.year()) % 100;
        }

        // Resolves a two-digit year: anything after this year belongs to the
        // previous century, so 82 is 1982 while 08 is 2008.
        static int resolve_year(const int two_digit_year) {
            return two_digit_year > current_two_digit_year() ? 1900 + two_digit_year : 2000 + two_digit_year;
        }

        static bool plausible_month_day(const int month, const int day) {
            static constexpr array<int, 12> LENGTHS = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            return month >= 1 && month <= 12 && day >= 1 && day <= LENGTHS[month - 1];
        }

        static string iso_date(const int year, const int month, const int day) {
            ostringstream date;
            date << setfill('0') << setw(4) << year << '-'
                 << setw(2) << month << '-'
                 << setw(2) << day;
            return date.str();
        }

        // Reads a printed date in any of the forms the three documents use:
        // 1984-03-24, 1984/03/24, 24 MAR 1984 or 240384. Returns an empty
        // string when nothing parses.
        static string parse_date(const string &value) {
            const string packed = keep_alnum(value);

            // yyyymmdd
            if (packed.size() >= 8 && all_of(packed.begin(), packed.begin() + 8, ::isdigit)) {
                const int year = stoi(packed.substr(0, 4));
                const int month = stoi(packed.substr(4, 2));
                const int day = stoi(packed.substr(6, 2));
                if (year >= 1900 && year <= 2100 && plausible_month_day(month, day))
                    return iso_date(year, month, day);
            }
            // ddMMMyyyy
            if (packed.size() >= 9) {
                for (size_t i = 0; i + 9 <= packed.size(); ++i) {
                    string window{packed.substr(i, 9)};
                    if (!isdigit(static_cast<unsigned char>(window[0]))) continue;
                    if (!isdigit(static_cast<unsigned char>(window[1]))) continue;
                    // OCR often reads the B in FEB as 8; repair only the
                    // month token, after the surrounding date shape matched.
                    for (size_t month_at = 2; month_at < 5; ++month_at)
                        if (window[month_at] == '8') window[month_at] = 'B';
                    const auto month = ranges::find(MONTHS, string_view{window}.substr(2, 3));
                    if (month == MONTHS.end()) continue;
                    if (!all_of(window.begin() + 5, window.end(), ::isdigit)) continue;
                    const int day = stoi(window.substr(0, 2));
                    const int month_number = static_cast<int>(ranges::distance(MONTHS.begin(), month)) + 1;
                    const int year = stoi(window.substr(5, 4));
                    if (year >= 1900 && year <= 2100 && plausible_month_day(month_number, day))
                        return iso_date(year, month_number, day);
                }
            }
            return {};
        }

        // The renderings of a date that could appear on these documents, packed
        // the same way recognised lines are so they can be searched for.
        static vector<string> date_renderings(const int year, const int month, const int day) {
            char iso[9], named[10], named_padded[10];
            snprintf(iso, sizeof(iso), "%04d%02d%02d", year, month, day);
            snprintf(named, sizeof(named), "%d%s%04d", day, string{MONTHS[month - 1]}.c_str(), year);
            snprintf(named_padded, sizeof(named_padded), "%02d%s%04d", day, string{MONTHS[month - 1]}.c_str(), year);
            return {iso, named, named_padded};
        }

        static bool luhn_valid(const string &digits) {
            int total = 0;
            for (size_t i = 0; i < digits.size(); ++i) {
                int value = digits[digits.size() - 1 - i] - '0';
                if (i % 2 == 1) {
                    value *= 2;
                    if (value > 9) value -= 9;
                }
                total += value;
            }
            return total % 10 == 0;
        }

        static bool mrz_check_matches(const string_view field, const char check) {
            return mrz::check_matches(field, check);
        }

        // Repairs the letters a recogniser most often substitutes for digits.
        // Safe to apply blindly because every field it touches is protected by
        // a check digit that will reject a wrong guess.
        static string as_digits(string field) {
            for (char &c: field) {
                switch (c) {
                    case 'O':
                    case 'Q':
                    case 'D': c = '0';
                        break;
                    case 'I':
                    case 'L': c = '1';
                        break;
                    case 'Z': c = '2';
                        break;
                    case 'G': c = '6';
                        break;
                    case 'S': c = '5';
                        break;
                    case 'B': c = '8';
                        break;
                    default: break;
                }
            }
            return field;
        }

        // The second line of a TD3 machine readable zone, which is where a
        // South African passport carries the holder's identity number.
        struct mrz_line2 {
            static constexpr size_t length = 44;

            string passport_number;
            string nationality;
            string date_of_birth; // ISO
            string sex;
            string date_of_expiry; // ISO
            string personal_number; // the 13-digit identity number
            bool composite_valid{};

            // Parses a 44-character window, returning false unless the birth
            // date, expiry and personal number check digits all agree.
            bool parse(const string_view raw) {
                if (raw.size() != length) return false;

                // The fields repaired here are numeric on a South African
                // passport, including the check digits themselves, so the
                // letters a recogniser most often substitutes for digits are
                // put back before anything is tested. A wrong repair just fails
                // its check digit and the window is discarded. The document
                // number, nationality and sex are left alone: they are
                // legitimately alphabetic.
                string window{raw};
                const auto repair = [&window](const size_t at, const size_t count) {
                    window.replace(at, count, as_digits(window.substr(at, count)));
                };
                repair(9, 1); // document number check digit
                repair(13, 7); // date of birth and its check digit
                repair(21, 7); // date of expiry and its check digit
                repair(28, 13); // the identity number, less its filler
                repair(42, 2); // personal number and composite check digits

                const string birth = window.substr(13, 6);
                const string expiry = window.substr(21, 6);
                const string personal = window.substr(28, 14);

                if (!mrz_check_matches(birth, window[19])) return false;
                if (!mrz_check_matches(expiry, window[27])) return false;
                if (!mrz_check_matches(personal, window[42])) return false;

                const int birth_month = stoi(birth.substr(2, 2));
                const int birth_day = stoi(birth.substr(4, 2));
                if (!plausible_month_day(birth_month, birth_day)) return false;
                date_of_birth = iso_date(resolve_year(stoi(birth.substr(0, 2))), birth_month, birth_day);

                const int expiry_month = stoi(expiry.substr(2, 2));
                const int expiry_day = stoi(expiry.substr(4, 2));
                if (!plausible_month_day(expiry_month, expiry_day)) return false;
                // A document that is current expires this century.
                date_of_expiry = iso_date(2000 + stoi(expiry.substr(0, 2)), expiry_month, expiry_day);

                passport_number = window.substr(0, 9);
                erase(passport_number, '<');
                nationality = window.substr(10, 3);
                sex = window[20] == 'M' || window[20] == 'F' ? string{window[20]} : string{};
                personal_number = personal;
                erase(personal_number, '<');

                // One digit covering the document number, both dates and the
                // personal number together, each with its own check digit.
                const string composite = window.substr(0, 10) + window.substr(13, 7) + window.substr(21, 22);
                composite_valid = mrz_check_matches(composite, window[43]);
                return true;
            }
        };

        // The first TD3 MRZ line contains the holder's name. The second
        // document-type character is deliberately not validated: OCR commonly
        // misreads its filler '<', while the issuing country and name separator
        // remain stable.
        struct mrz_line1 {
            static constexpr size_t length = 44;
            static constexpr size_t minimum_length = 20; // the rest is taken as lost fillers

            string surname;
            string names;

            bool parse(const string_view raw) {
                if (raw.size() != length || raw[0] != 'P') return false;
                if (!ranges::all_of(raw.substr(2, 3), [](const unsigned char c) { return isalpha(c); }))
                    return false;

                const string_view name_field = raw.substr(5);
                const size_t separator = name_field.find("<<");
                if (separator == string_view::npos) return false;
                const auto decode_name = [](string value) {
                    // A name field holds only letters and fillers: a digit in it is a misread
                    // letter that looks like it (NAID00, J0RDAN, 5IPHO).
                    for (char &c: value) {
                        switch (c) {
                            case '0': c = 'O'; break;
                            case '1': c = 'I'; break;
                            case '2': c = 'Z'; break;
                            case '5': c = 'S'; break;
                            case '6': c = 'G'; break;
                            case '8': c = 'B'; break;
                            default: break;
                        }
                    }
                    ranges::replace(value, '<', ' ');
                    return clean_name(value);
                };
                surname = decode_name(string{name_field.substr(0, separator)});
                names = decode_name(string{name_field.substr(separator + 2)});
                return !surname.empty() && !names.empty();
            }
        };

        // Text as an MRZ holds it: uppercase letters, digits and fillers (<).
        static string mrz_characters(const string &value) {
            string result;
            result.reserve(value.size());
            for (const unsigned char c: upper(value)) {
                if (isalnum(c) || c == '<') result += static_cast<char>(c);
            }
            return result;
        }

        struct mrz_result {
            mrz_line2 line2;
            size_t line_index;
        };

        // Finds a usable MRZ second line anywhere in the recognised text.
        static optional<mrz_result> find_mrz(const vector<text_line> &lines) {
            optional<mrz_result> best;
            for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
                const auto &line = lines[line_index];
                const string window = mrz_characters(line.packed);
                if (window.size() < mrz_line2::length) continue;
                for (size_t start = 0; start + mrz_line2::length <= window.size(); ++start) {
                    mrz_line2 candidate;
                    if (!candidate.parse(string_view{window}.substr(start, mrz_line2::length))) continue;
                    if (!identity::valid_id_number(candidate.personal_number)) continue;
                    mrz_result result{std::move(candidate), line_index};
                    if (result.line2.composite_valid) return result; // fully verified, stop looking
                    if (!best) best = std::move(result);
                }
            }
            return best;
        }

        static optional<mrz_line1> find_mrz_line1(const vector<text_line> &lines, const size_t line2_index) {
            const auto &line2 = lines[line2_index];
            optional<mrz_line1> best;
            double best_score = numeric_limits<double>::infinity();
            for (const auto &line: lines) {
                // The line, or the line with the rest of its row joined on (a line can be detected
                // in pieces); short of the full length only for lost fillers, which are put back.
                string row = line.packed;
                const double row_tolerance = line.box.height() / 2;
                vector<const text_line *> rest;
                for (const auto &other: lines) {
                    if (&other == &line || other.box.left() < line.box.right() - row_tolerance) continue;
                    if (std::abs(other.box.center().y() - line.box.center().y()) > row_tolerance) continue;
                    rest.push_back(&other);
                }
                ranges::sort(rest, {}, [](const text_line *l) { return l->box.left(); });
                for (const auto *other: rest) row += other->packed;

                string window = mrz_characters(row);
                if (window.size() < mrz_line1::minimum_length) continue;
                if (window.size() < mrz_line1::length) window.append(mrz_line1::length - window.size(), '<');

                const double below = line2.box.top() - line.box.bottom();
                const double beside = horizontal_gap(line.box, line2.box);
                const double max_gap = max(line.box.height(), line2.box.height()) * 3;
                if (below < -line2.box.height() || below > max_gap || beside > max_gap) continue;

                for (size_t start = 0; start + mrz_line1::length <= window.size(); ++start) {
                    mrz_line1 candidate;
                    if (!candidate.parse(string_view{window}.substr(start, mrz_line1::length))) continue;
                    const double score = max(0.0, below) + beside;
                    if (score >= best_score) continue;
                    best_score = score;
                    best = std::move(candidate);
                }
            }
            return best;
        }

        // Printed captions and headings, found anywhere in a line's letters.
        static constexpr auto CAPTIONS = to_array<string_view>({
            "SURNAME", "VANSURNAME", "GIVENNAMES", "FORENAMES", "NAMES", "VOORNAME", "PRENOMS", "NOM",
            "NATIONALIT", "COUNTRYOFBIRTH", "PLACEOFBIRTH", "GEBOORTEDISTRIK", "STATUS", "DATEISSUED",
            "DATEOFISSUE", "DATUMUITGEREIK", "DATEOFBIRTH", "GEBOORTEDATUM", "DATEOFEXPIRY", "DATEEXPIRES",
            "PASSPORTNO", "IDENTITYNUMBER", "IDENTITYNO", "IDNO", "SEX", "GESLAG",
            "SOUTHAFRICA", "REPUBLIC", "SABURGER", "SACITIZEN", "BINNELANDSESAKE", "ANDSESAKE", "CITIZEN",
            "IDENTITEITSDOKUMENT", "ADRESVERANDERING", "GEREGISTREERDE"
        });

        // Words printed on the documents themselves, as they read in English, Afrikaans and the
        // passport's French and Spanish: never a name. Matched loosely (they misread too - PASPORT,
        // PASSP0RT) anywhere in the letters, as bilingual captions run together (PASPOORTPASSPORT).
        // Words that are also names or parts of names (VAN, DU) are left out.
        static constexpr array<string_view, 24> DOCUMENT_WORDS = {
            "PASSPORT", "PASPOORT", "PASSEPORT", "PASAPORTE", "REPUBLIC", "REPUBLIEK", "AFRICA", "AFRIKA",
            "IDENTITY", "IDENTITEIT", "NATIONAL", "NATIONALITY", "NASIONALITEIT", "CITIZEN", "SURNAME",
            "FORENAMES", "SIGNATURE", "HANDTEKENING", "DEPARTMENT", "AFFAIRS", "BINNELANDSE", "DOCUMENT",
            "DOKUMENT", "AUTHORITY"
        };

        static size_t edit_distance(const string_view a, const string_view b) {
            vector<size_t> row(b.size() + 1);
            for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
            for (size_t i = 1; i <= a.size(); ++i) {
                size_t diagonal = row[0];
                row[0] = i;
                for (size_t j = 1; j <= b.size(); ++j) {
                    const size_t above = row[j];
                    row[j] = min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0 : 1)});
                    diagonal = above;
                }
            }
            return row[b.size()];
        }

        // Whether value holds one of DOCUMENT_WORDS, to within two letters for long words and one
        // for shorter ones. Checked once per text: the name searches ask about pairs of lines.
        static bool has_document_word(const string &value) {
            static thread_local unordered_map<string, bool> known;
            if (const auto found = known.find(value); found != known.end()) return found->second;
            string letters;
            for (const unsigned char c: value) if (isalpha(c)) letters += static_cast<char>(toupper(c));
            bool result = false;
            for (const auto word: DOCUMENT_WORDS) {
                const size_t allowed = word.size() >= 8 ? 2 : 1;
                for (size_t length = word.size() - 1; length <= word.size() + 1 && !result; ++length)
                    for (size_t at = 0; at + length <= letters.size() && !result; ++at)
                        result = edit_distance(string_view{letters}.substr(at, length), word) <= allowed;
                if (result) break;
            }
            if (known.size() > 10000) known.clear();
            known.emplace(value, result);
            return result;
        }

        // The other language's half of a bilingual label ("Surname / Van", "Sex / Geslag"): after a
        // label, text that is only one of these is more label, not its value. (VAN is no caption on
        // its own: it starts surnames like VAN DER MERWE.)
        static bool is_label_translation(const string &alnum) {
            static constexpr array<string_view, 14> words = {
                "VAN", "SURNAME", "VOORNAME", "VOORNAAM", "NAMES", "GIVENNAMES", "NASIONALITEIT", "NATIONALITY",
                "GESLAG", "SEX", "GEBOORTEDATUM", "DATEOFBIRTH", "GEBOORTEPLEK", "PLACEOFBIRTH"
            };
            return ranges::find(words, string_view{upper(alnum)}) != words.end();
        }

        static bool is_caption(const string &value) {
            return ranges::any_of(CAPTIONS, [&value](const string_view caption) {
                return value.find(caption) != string::npos;
            }) || has_document_word(value);
        }

        // Where text goes on after its first alnum_count letters and digits (npos when it has
        // fewer), so a value can be taken from after a label matched with its punctuation stripped.
        static size_t after_alnum(const string &text, const size_t alnum_count) {
            size_t seen = 0;
            for (size_t i = 0; i < text.size(); ++i) {
                if (!isalnum(static_cast<unsigned char>(text[i]))) continue;
                if (++seen == alnum_count) return i + 1;
            }
            return string::npos;
        }

        // Returns the value belonging to a label. Labels are matched on a
        // punctuation-free uppercase form because the small low-contrast
        // captions on these documents recognise poorly.
        static string value_for_label(const vector<text_line> &lines,
                                      const initializer_list<string_view> keys) {
            const text_line *best_value = nullptr;
            double best_score = numeric_limits<double>::infinity();

            for (size_t i = 0; i < lines.size(); ++i) {
                for (const auto key: keys) {
                    const auto at = lines[i].alnum.find(key);
                    if (at == string::npos) continue;
                    // VAN is a label (Afrikaans for surname) only on its own or as VAN/SURNAME:
                    // VAN DER MERWE is a surname, not the label VAN with DER MERWE after it.
                    if (key == "VAN") {
                        if (at != 0) continue;
                        const auto after = lines[i].text.find_first_not_of(" ", lines[i].text.find_first_not_of(" ") + 3);
                        if (after != string::npos && lines[i].text[after] != '/' && lines[i].text[after] != ':') continue;
                    }

                    const auto &text = lines[i].text;
                    const size_t label_end = after_alnum(text, at + key.size());
                    const string tail = label_end == string::npos ? string{} : trim(text.substr(label_end));
                    const bool separated = label_end != string::npos &&
                                           (label_end == text.size() || !isalnum(static_cast<unsigned char>(text[label_end])));
                    if (separated && !tail.empty() && !is_caption(keep_alnum(tail)) && !is_label_translation(keep_alnum(tail)))
                        return tail;

                    const auto &label = lines[i].box;
                    const double label_center_y = label.center().y();
                    const double same_row_tolerance = std::max(4.0, label.height() * 0.5);
                    const double max_horizontal_gap = std::max(label.height() * 4.0, label.width() * 1.5);
                    const double max_vertical_gap = std::max(label.height() * 6.0, label.width() * 1.5);

                    for (size_t j = 0; j < lines.size(); ++j) {
                        if (j == i || lines[j].text.empty() || is_caption(lines[j].alnum)) continue;

                        const auto &candidate = lines[j];
                        const auto &box = candidate.box;
                        const double beside = horizontal_gap(label, box);
                        if (beside > max_horizontal_gap) continue;

                        const double center_y_gap = std::abs(box.center().y() - label_center_y);
                        const bool same_row = box.left() >= label.left() &&
                                              center_y_gap <= same_row_tolerance;
                        double score{};
                        if (same_row) {
                            score = beside + center_y_gap * 2.0;
                        } else {
                            if (box.top() < label.top()) continue;
                            const double below = std::max(0.0, box.top() - label.bottom());
                            if (below > max_vertical_gap) continue;
                            score = below + beside * 0.5;
                        }
                        if (score < best_score) {
                            best_score = score;
                            best_value = &candidate;
                        }
                    }
                }
            }
            return best_value ? trim(best_value->text) : string{};
        }

        static bool any_line_contains(const vector<text_line> &lines, const string_view needle) {
            return ranges::any_of(lines, [needle](const text_line &line) {
                return line.alnum.find(needle) != string::npos;
            });
        }

        static ocr &recognizer() {
            static thread_local ocr instance;
            static thread_local const bool configured = (instance.set_auto_rotate(false), true); // rotations are chosen here
            (void) configured;
            return instance;
        }

        static vector<text_line> make_text_lines(const vector<ocr::line> &recognized_lines) {
            vector<text_line> lines;
            for (const auto &recognized: recognized_lines) {
                text_line line;
                line.text = trim(recognized.text);
                line.packed.reserve(line.text.size());
                for (const unsigned char c: line.text)
                    if (!isspace(c)) line.packed += static_cast<char>(c);
                line.alnum = keep_alnum(line.packed);
                line.confidence = static_cast<double>(recognized.confidence);
                line.box = recognized.box;
                if (!line.text.empty()) lines.push_back(std::move(line));
            }
            return lines;
        }

        // Lines read with less confidence are left out, except when looking for the MRZ.
        static constexpr double MinimumLineConfidence = 85;

        static vector<text_line> recognise(const image &input) {
            auto &reader = recognizer();
            reader.detect(input, 0); // every line: populate() leaves out the doubtful ones but for the MRZ
            return make_text_lines(reader.lines());
        }

        struct number_hit {
            string number;
            double confidence{};
        };

        static optional<number_hit> valid_number_in(const string &value, const double confidence) {
            const string digits = keep_digits(value);
            if (digits.size() < 13) return nullopt;
            for (size_t start = 0; start + 13 <= digits.size(); ++start) {
                const string candidate = digits.substr(start, 13);
                if (identity::valid_id_number(candidate)) return number_hit{candidate, confidence};
            }
            return nullopt;
        }

        // A number joined to its printed caption has a stronger provenance
        // than a coincidental thirteen-digit sequence elsewhere on the page.
        static optional<number_hit> find_labelled_id_number(const vector<text_line> &lines) {
            const string labelled = value_for_label(lines, {"IDENTITYNUMBER", "IDENTITYNO", "IDNO"});
            if (labelled.empty()) return nullopt;
            return valid_number_in(labelled, 0.0);
        }

        // Looks for a valid identity number in the digits of each line. The
        // number is printed in separated groups on a green ID book, so the
        // digits of a whole line are joined before the window slides over them.
        static optional<number_hit> find_id_number(const vector<text_line> &lines) {
            for (const auto &line: lines) {
                if (const auto number = valid_number_in(line.packed, line.confidence)) return number;
            }
            return nullopt;
        }

        class identity_reader {
        public:
            static identity read(const filesystem::path &image_path, const bool face_detection) {
                const image input{image_path.string()};
                return read(input, face_detection);
            }

            static identity read(const image &input, const bool face_detection) {
                if (input.empty()) throw invalid_argument{"Cannot read identity from an empty image"};
                identity best;
                best.face_detection_performed_ = face_detection;

                if (face_detection) {
                    static thread_local facedetector detector;
                    best.faces_ = detector.detect(input, true); // with each face's image (aligned, 112 x 112)
                }

                // Read it the way up the orientation model says. Only when that misses fields, try
                // it turned a quarter either way too: an upside-down page is caught already, as
                // sc::ocr reads lines the way up most of the page's lines are.
                const int upright = ocr::upright_rotation(input);
                best.rotation_ = upright;
                populate(best, recognise(turned(input, upright)));
                if (!needs_rotation_retry(best)) return best;

                for (const int rotation: {(upright + 90) % 360, (upright + 270) % 360}) {
                    identity candidate;
                    candidate.rotation_ = rotation;
                    candidate.faces_ = best.faces_;
                    candidate.face_detection_performed_ = face_detection;
                    populate(candidate, recognise(turned(input, rotation)));
                    if (more_verified(candidate, best)) best = std::move(candidate);
                }
                return best;
            }

        private:
            static image turned(const image &input, const int rotation) {
                image result{input};
                if (rotation) result.rotate(rotation);
                return result;
            }

            static bool needs_rotation_retry(const identity &result) {
                return !result.has_id_number() || result.names_.empty() || result.surname_.empty();
            }

            // Whether a reading at another rotation beats the orientation model's: only with an ID
            // number it lacks or more checks agreeing. Filling more fields doesn't count - sideways
            // lines can be read too, into the wrong fields.
            static bool more_verified(const identity &candidate, const identity &best) {
                if (candidate.has_id_number() != best.has_id_number()) return candidate.has_id_number();
                return candidate.checks_.score() > best.checks_.score();
            }

            static void populate(identity &result, const vector<text_line> &all_lines) {
                // The MRZ is looked for among every line read: check digits validate it, and
                // runs of fillers often read below the confidence the other fields need.
                vector<text_line> lines;
                for (const auto &line: all_lines)
                    if (line.confidence >= MinimumLineConfidence) lines.push_back(line);
                result.raw_text_.clear();
                result.raw_text_.reserve(lines.size());
                for (auto const &tl: lines) result.raw_text_.push_back(tl.text);
                const auto mrz = find_mrz(all_lines);
                const auto labelled = find_labelled_id_number(lines);
                const auto printed = labelled ? labelled : find_id_number(lines);

                result.document_ = classify(lines, mrz.has_value());

                if (mrz) {
                    result.id_number_ = mrz->line2.personal_number;
                    result.checks_.mrz = true;
                    result.passport_number_ = mrz->line2.passport_number;
                    result.date_of_expiry_ = mrz->line2.date_of_expiry;
                    result.sex_ = mrz->line2.sex;
                    result.nationality_ = mrz->line2.nationality;
                    result.ocr_confidence_ = mrz_line_confidence(all_lines);
                } else if (printed) {
                    result.id_number_ = printed->number;
                    result.ocr_confidence_ = printed->confidence;
                }

                if (!result.id_number_.empty()) {
                    result.checks_.luhn = true; // nothing reaches here without it
                    result.date_of_birth_ = identity::id_date_of_birth(result.id_number_);
                }
                if (mrz) {
                    if (const auto names = find_mrz_line1(all_lines, mrz->line_index)) {
                        result.surname_ = names->surname;
                        result.names_ = names->names;
                    }
                }
                read_fields(result, lines);
                if (!result.id_number_.empty()) cross_check(result, lines, mrz.has_value());
            }

            static double mrz_line_confidence(const vector<text_line> &lines) {
                for (const auto &line: lines) {
                    if (line.alnum.size() >= 40 && line.packed.find('<') != string::npos) return line.confidence;
                }
                return lines.empty() ? 0.0 : lines.front().confidence;
            }

            // Counts the markers each document carries rather than taking the
            // first match, because a photograph often loses some of them: a
            // card's small green captions and a worn book's Afrikaans labels
            // both recognise unreliably.
            static identity::document classify(const vector<text_line> &lines, const bool has_mrz) {
                if (has_mrz) return identity::document::passport; // decisive on its own

                const auto count = [&lines](const initializer_list<string_view> markers) {
                    return ranges::count_if(markers, [&lines](const string_view marker) {
                        return any_line_contains(lines, marker);
                    });
                };

                const auto passport = count({
                    "PASSEPORT", "PASSPORTNO", "SUDAFRICAIN", "AFRIQUEDUSUD",
                    "DATEOFEXPIRY"
                });
                const auto book = count({
                    "IDNO", "SABURGER", "SACITIZEN", "VANSURNAME", "VOORNAME",
                    "GEBOORTEDATUM", "GEBOORTEDISTRIK", "DATUMUITGEREIK",
                    "BINNELANDSESAKE", "IDENTITEITSDOKUMENT"
                });
                const auto card = count({
                    "IDENTITYCARD", "NATIONALIDENTITY", "IDENTITYNUMBER",
                    "COUNTRYOFBIRTH", "STATUS", "SOUTHAFRICA"
                });

                const auto best = max({passport, book, card});
                if (best == 0) return identity::document::unknown;
                if (best == passport) return identity::document::passport;
                if (best == book) return identity::document::id_book;
                return identity::document::id_card;
            }

            static void read_fields(identity &result, const vector<text_line> &lines) {
                const auto set_if_empty = [](string &field, const string &value) {
                    if (field.empty() && !value.empty()) field = value;
                };
                const auto is_name_value = [](const string &value) {
                    const string packed = keep_alnum(value);
                    return packed.size() >= 3 && packed != "IDN" && packed != "IDNO" && !is_caption(packed);
                };

                const string labelled_surname = clean_name(value_for_label(lines, {"SURNAME", "VAN"}));
                const string labelled_names = clean_name(value_for_label(lines, {
                                                                             "GIVENNAMES", "FORENAMES", "NAMES",
                                                                             "VOORNAME", "PRENOMS"
                                                                         }));
                if (is_name_value(labelled_surname)) set_if_empty(result.surname_, labelled_surname);
                if (is_name_value(labelled_names)) set_if_empty(result.names_, labelled_names);
                set_if_empty(result.nationality_, clean_name(value_for_label(lines, {"NATIONALIT"})));
                const auto country_of_birth = clean_name(value_for_label(lines, {
                                                                             "COUNTRYOFBIRTH", "PLACEOFBIRTH",
                                                                             "GEBOORTEDISTRIK"
                                                                         }));
                if (country_of_birth != "CITIZEN") set_if_empty(result.country_of_birth_, country_of_birth);
                if (result.country_of_birth_.empty()) {
                    const auto south_africa = ranges::find_if(lines, [](const text_line &line) {
                        return line.alnum == "SOUTHAFRICA";
                    });
                    if (south_africa != lines.end()) {
                        result.country_of_birth_ = "SOUTH AFRICA";
                        set_if_empty(result.nationality_, result.country_of_birth_);
                    }
                }
                set_if_empty(result.status_, clean_name(value_for_label(lines, {"STATUS"})));
                set_if_empty(result.date_of_issue_,
                             parse_date(value_for_label(lines, {"DATEISSUED", "DATEOFISSUE", "DATUMUITGEREIK"})));
                set_if_empty(result.date_of_expiry_,
                             parse_date(value_for_label(lines, {"DATEOFEXPIRY", "DATEEXPIRES"})));
                set_if_empty(result.date_of_birth_,
                             parse_date(value_for_label(lines, {"DATEOFBIRTH", "GEBOORTEDATUM", "DOB"})));
                if (result.document_ == identity::document::id_card &&
                    (result.surname_.empty() || result.names_.empty())) {
                    const auto [surname, names] = nearby_unlabelled_names(lines);
                    set_if_empty(result.surname_, surname);
                    set_if_empty(result.names_, names);
                }
                if (result.document_ == identity::document::id_card &&
                    (result.surname_.empty() || result.names_.empty())) {
                    const auto [surname, names] = structured_card_names(lines);
                    set_if_empty(result.surname_, surname);
                    set_if_empty(result.names_, names);
                }
                if (result.surname_.empty() && !result.names_.empty() &&
                    result.document_ == identity::document::id_book)
                    result.surname_ = nearby_unlabelled_name(lines, result.names_);
                if (result.surname_.empty() || result.names_.empty()) {
                    const auto [surname, names] = generic_names(lines);
                    set_if_empty(result.surname_, surname);
                    set_if_empty(result.names_, names);
                }

                if (result.passport_number_.empty()) {
                    const string passport = keep_alnum(value_for_label(lines, {"PASSPORTNO"}));
                    if (passport.size() >= 8 && passport.size() <= 9) result.passport_number_ = passport;
                }

                // Last resort: only use an unlabelled date when no validated
                // ID number or nearby birth-date caption has supplied one.
                if (result.date_of_birth_.empty()) {
                    for (const auto &line: lines) {
                        if (const string date = parse_date(line.text); !date.empty()) {
                            result.date_of_birth_ = date;
                            break;
                        }
                    }
                }
            }

            static pair<string, string> nearby_unlabelled_names(const vector<text_line> &lines) {
                const text_line *best_surname = nullptr;
                const text_line *best_names = nullptr;
                double best_gap = numeric_limits<double>::infinity();
                for (const auto &surname: lines) {
                    const string surname_text = clean_name(surname.text);
                    if (surname_text.empty() || word_count(surname_text) != 1 ||
                        is_caption(surname.alnum) || surname_text.size() < 3)
                        continue;
                    for (const auto &names: lines) {
                        const string names_text = clean_name(names.text);
                        if (&surname == &names || names_text.empty() || word_count(names_text) < 2 ||
                            is_caption(names.alnum))
                            continue;

                        const double below = names.box.top() - surname.box.bottom();
                        const double beside = horizontal_gap(surname.box, names.box);
                        const double text_height = max(
                            min(surname.box.width(), surname.box.height()),
                            min(names.box.width(), names.box.height()));
                        if (below < -text_height || below > text_height * 4 || beside > text_height * 2)
                            continue;

                        const double gap = hypot(max(0.0, below), beside);
                        if (gap >= best_gap) continue;
                        best_gap = gap;
                        best_surname = &surname;
                        best_names = &names;
                    }
                }
                if (!best_surname || !best_names) return {};
                return {clean_name(best_surname->text), clean_name(best_names->text)};
            }

            // Smart cards print given names before the surname in a compact
            // row or column. This is deliberately after captions, so an
            // arbitrary adjacent pair cannot replace a labelled field.
            static pair<string, string> structured_card_names(const vector<text_line> &lines) {
                const text_line *best_given = nullptr;
                const text_line *best_surname = nullptr;
                double best_score = numeric_limits<double>::infinity();
                for (const auto &given: lines) {
                    const string given_text = clean_name(given.text);
                    if (given_text.size() < 3 || is_caption(given.alnum)) continue;
                    for (const auto &surname: lines) {
                        const string surname_text = clean_name(surname.text);
                        if (&given == &surname || surname_text.size() < 3 || is_caption(surname.alnum))
                            continue;

                        const double height = max(1.0, max(given.box.height(), surname.box.height()));
                        const double center_y_gap = std::abs(given.box.center().y() - surname.box.center().y());
                        const double beside = horizontal_gap(given.box, surname.box);
                        const bool same_row = surname.box.left() >= given.box.left() &&
                                              center_y_gap <= height;
                        const bool next_row = surname.box.top() >= given.box.top() &&
                                              beside <= height * 2;
                        if (!same_row && !next_row) continue;

                        const double score = same_row
                                                 ? surname.box.left() - given.box.left() + center_y_gap
                                                 : surname.box.top() - given.box.top() + beside;
                        if (score >= best_score) continue;
                        best_score = score;
                        best_given = &given;
                        best_surname = &surname;
                    }
                }
                if (!best_given || !best_surname) return {};
                return {clean_name(best_given->text), clean_name(best_surname->text)};
            }

            static pair<string, string> generic_names(const vector<text_line> &lines) {
                vector<string> candidates;
                for (const auto &line: lines) {
                    const string name = clean_name(line.text);
                    if (name.size() < 4 || is_caption(line.alnum) || word_count(name) > 3)
                        continue;
                    candidates.push_back(name);
                }
                if (candidates.size() < 2) return {};
                const auto names = ranges::find_if(candidates, [](const string &value) {
                    return word_count(value) > 1;
                });
                if (names == candidates.end()) return {candidates[1], candidates[0]};
                const auto surname = ranges::find_if(candidates, [&names](const string &value) {
                    return &value != &*names && value.find(' ') == string::npos;
                });
                return {surname == candidates.end() ? string{} : *surname, *names};
            }

            static string nearby_unlabelled_name(const vector<text_line> &lines, const string &names) {
                const text_line *names_line = nullptr;
                for (const auto &line: lines) {
                    if (clean_name(line.text) == names) {
                        names_line = &line;
                        break;
                    }
                }
                if (!names_line) return {};

                const text_line *best = nullptr;
                double best_distance = numeric_limits<double>::infinity();
                for (const auto &candidate: lines) {
                    if (&candidate == names_line || candidate.alnum.size() < 3 ||
                        is_caption(candidate.alnum) ||
                        !ranges::all_of(candidate.text, [](const unsigned char c) {
                            return isalpha(c) || isspace(c) || c == '-' || c == '\'';
                        }))
                        continue;

                    const auto &a = names_line->box;
                    const auto &b = candidate.box;
                    const double distance = hypot(horizontal_gap(a, b), vertical_gap(a, b));
                    const double text_height = max(min(a.width(), a.height()), min(b.width(), b.height()));
                    if (distance > text_height * 3.0 || distance >= best_distance) continue;

                    best = &candidate;
                    best_distance = distance;
                }
                return best ? clean_name(best->text) : string{};
            }

            // Confirms the number against parts of the document it was not read
            // from: the printed birth date, and the printed sex.
            static void cross_check(identity &result, const vector<text_line> &lines, const bool from_mrz) {
                const string &number = result.id_number_;
                const int month = stoi(number.substr(2, 2));
                const int day = stoi(number.substr(4, 2));

                // Look for the birth date the number implies, rather than
                // parsing an unknown date, so the two centuries and all three
                // printed layouts are covered.
                for (const int year: {
                         resolve_year(stoi(number.substr(0, 2))),
                         1900 + stoi(number.substr(0, 2)),
                         2000 + stoi(number.substr(0, 2))
                     }) {
                    for (const auto &rendering: date_renderings(year, month, day)) {
                        if (any_line_contains(lines, rendering)) {
                            result.checks_.date_of_birth = true;
                            result.date_of_birth_ = iso_date(year, month, day);
                            break;
                        }
                    }
                    if (result.checks_.date_of_birth) break;
                }
                // An MRZ carries its own birth date under a check digit, which
                // is an independent confirmation in itself.
                if (from_mrz && !result.date_of_birth_.empty()) result.checks_.date_of_birth = true;

                const string expected = identity::id_sex(number);
                if (!result.sex_.empty()) {
                    result.checks_.sex = result.sex_ == expected;
                } else {
                    const string printed = printed_sex(lines);
                    if (!printed.empty()) {
                        result.sex_ = printed;
                        result.checks_.sex = printed == expected;
                    }
                }
                if (result.sex_.empty()) result.sex_ = expected;
            }

            // The sex is printed as a lone M or F after, beside or under its label: the rest of the
            // label's line, or the M or F line nearest the label to its right or below it. (Reading
            // order puts the next label on the same row, not the value under it, next.)
            static string printed_sex(const vector<text_line> &lines) {
                const auto is_sex = [](const string &value) { return value == "M" || value == "F"; };
                string nearest;
                double nearest_distance = numeric_limits<double>::infinity();
                for (const auto &label: lines) {
                    auto at = label.alnum.find("SEX");
                    size_t width = 3;
                    if (at == string::npos) {
                        at = label.alnum.find("GESLAG");
                        width = 6;
                    }
                    if (at == string::npos) continue;
                    if (const auto rest = label.alnum.substr(at + width); is_sex(rest)) return rest;

                    const auto &box = label.box;
                    const double reach = max(box.height() * 6.0, box.width() * 1.5);
                    for (const auto &candidate: lines) {
                        if (!is_sex(candidate.alnum)) continue;
                        const auto &value = candidate.box;
                        if (value.bottom() < box.top() || value.right() < box.left()) continue; // above or left of it
                        const double right = max(0.0, value.left() - box.right());
                        const double below = max(0.0, value.top() - box.bottom());
                        const double distance = hypot(right, below);
                        if (distance > reach || distance >= nearest_distance) continue;
                        nearest = candidate.alnum;
                        nearest_distance = distance;
                    }
                }
                return nearest;
            }
        };
    } // namespace impl

    int identity::verification::score() const noexcept {
        if (!luhn) return 0;
        return min(100, 60 + (date_of_birth ? 25 : 0) + (sex ? 10 : 0) + (mrz ? 35 : 0));
    }

    int identity::verification::passed() const noexcept {
        return (luhn ? 1 : 0) + (date_of_birth ? 1 : 0) + (sex ? 1 : 0) + (mrz ? 1 : 0);
    }

    nlohmann::ordered_json identity::verification::to_json() const {
        return nlohmann::ordered_json{
            {"luhn", luhn}, {"date_of_birth", date_of_birth}, {"sex", sex}, {"mrz", mrz}
        };
    }

    identity::identity(const filesystem::path &image_path, const bool face_detection) {
        read(image_path, face_detection);
    }

    identity::identity(const image &input, const bool face_detection) {
        read(input, face_detection);
    }

    bool identity::read(const filesystem::path &image_path, const bool face_detection) {
        *this = impl::identity_reader::read(image_path, face_detection);
        return has_id_number();
    }

    bool identity::read(const image &input, const bool face_detection) {
        *this = impl::identity_reader::read(input, face_detection);
        return has_id_number();
    }

    identity::document identity::document_type() const noexcept { return document_; }

    string identity::document_type_name() const {
        switch (document_) {
            case document::id_card: return "id_card";
            case document::id_book: return "id_book";
            case document::passport: return "passport";
            default: return "unknown";
        }
    }

    bool identity::has_id_number() const noexcept { return !id_number_.empty(); }
    const string &identity::id_number() const noexcept { return id_number_; }
    percent identity::confidence() const { return {static_cast<double>(checks_.score()), 0}; }
    const identity::verification &identity::checks() const noexcept { return checks_; }
    percent identity::ocr_confidence() const { return {ocr_confidence_, 1}; }
    const vector<face> &identity::faces() const noexcept { return faces_; }
    int identity::rotation() const noexcept { return rotation_; }
    const string &identity::surname() const noexcept { return surname_; }
    const string &identity::names() const noexcept { return names_; }
    const string &identity::date_of_birth() const noexcept { return date_of_birth_; }
    const string &identity::sex() const noexcept { return sex_; }
    const string &identity::nationality() const noexcept { return nationality_; }
    const string &identity::country_of_birth() const noexcept { return country_of_birth_; }
    const string &identity::status() const noexcept { return status_; }
    const string &identity::passport_number() const noexcept { return passport_number_; }
    const string &identity::date_of_issue() const noexcept { return date_of_issue_; }
    const string &identity::date_of_expiry() const noexcept { return date_of_expiry_; }

    optional<bool> identity::citizen() const {
        if (!has_id_number()) return nullopt;
        return id_number_[10] == '0';
    }

    string identity::all_text() {
        string result;
        for (const auto &text: raw_text_) {
            result += text + "\n";
        }
        return result;
    }

    bool identity::valid_id_number(const string &number) {
        if (number.size() != 13) return false;
        if (!ranges::all_of(number, [](const unsigned char c) { return isdigit(c); })) return false;
        if (!impl::plausible_month_day(stoi(number.substr(2, 2)), stoi(number.substr(4, 2)))) return false;
        if (number[10] != '0' && number[10] != '1') return false; // citizen or permanent resident
        return impl::luhn_valid(number);
    }

    string identity::id_date_of_birth(const string &number) {
        if (!valid_id_number(number)) return {};
        return impl::iso_date(impl::resolve_year(stoi(number.substr(0, 2))),
                              stoi(number.substr(2, 2)), stoi(number.substr(4, 2)));
    }

    string identity::id_sex(const string &number) {
        if (!valid_id_number(number)) return {};
        return stoi(number.substr(6, 4)) >= 5000 ? "M" : "F";
    }

    nlohmann::ordered_json identity::to_json() const {
        nlohmann::ordered_json result{
            {"version", 1},
            {"document_type", document_type_name()},
            {"id_number", id_number_},
            {"confidence", static_cast<double>(confidence())},
            {"verification", checks_.to_json()}
        };
        const auto add = [&result](const char *key, const string &value) {
            if (!value.empty()) result[key] = value;
        };
        add("surname", surname_);
        add("names", names_);
        add("date_of_birth", date_of_birth_);
        add("sex", sex_);
        if (const auto is_citizen = citizen()) result["citizen"] = *is_citizen;
        add("nationality", nationality_);
        add("country_of_birth", country_of_birth_);
        add("status", status_);
        add("passport_number", passport_number_);
        add("date_of_issue", date_of_issue_);
        add("date_of_expiry", date_of_expiry_);
        if (has_id_number()) result["ocr_confidence"] = static_cast<double>(ocr_confidence());
        if (face_detection_performed_) {
            result["faces"] = nlohmann::ordered_json::array();
            for (const auto &detected_face: faces_) result["faces"].push_back(detected_face.to_json());
        }
        result["rotation"] = rotation_;
        return result;
    }

    identity identity::from_json(const nlohmann::ordered_json &json) {
        if (!json.is_object() || json.value("version", 0) != 1)
            throw invalid_argument{"Unsupported identity JSON format"};
        identity result;
        const auto text = [&json](const char *key) { return json.contains(key) && json[key].is_string() ? json[key].get<string>() : string{}; };
        const auto type = text("document_type");
        result.document_ = type == "id_card" ? document::id_card : type == "id_book" ? document::id_book
                           : type == "passport" ? document::passport : document::unknown;
        result.id_number_ = text("id_number");
        if (json.contains("verification") && json["verification"].is_object()) {
            const auto &checks = json["verification"];
            result.checks_.luhn = checks.value("luhn", false);
            result.checks_.date_of_birth = checks.value("date_of_birth", false);
            result.checks_.sex = checks.value("sex", false);
            result.checks_.mrz = checks.value("mrz", false);
        }
        result.ocr_confidence_ = json.value("ocr_confidence", 0.0);
        result.rotation_ = json.value("rotation", 0);
        result.surname_ = text("surname");
        result.names_ = text("names");
        result.date_of_birth_ = text("date_of_birth");
        result.sex_ = text("sex");
        result.nationality_ = text("nationality");
        result.country_of_birth_ = text("country_of_birth");
        result.status_ = text("status");
        result.passport_number_ = text("passport_number");
        result.date_of_issue_ = text("date_of_issue");
        result.date_of_expiry_ = text("date_of_expiry");
        if (json.contains("faces") && json["faces"].is_array()) {
            result.face_detection_performed_ = true;
            for (const auto &each: json["faces"]) result.faces_.push_back(face::from_json(each));
        }
        // all_text() gives each line with a newline after it
        for (const auto lines = text("text"); const auto line: views::split(lines, '\n'))
            if (!ranges::empty(line)) result.raw_text_.emplace_back(ranges::begin(line), ranges::end(line));
        return result;
    }

    identity::operator nlohmann::ordered_json() const { return to_json(); }
    identity::operator string() const { return to_json().dump(); }

    std::ostream &operator<<(std::ostream &lhs, const identity &rhs) {
        return lhs << static_cast<string>(rhs);
    }
} // namespace sc::doc_classifier
