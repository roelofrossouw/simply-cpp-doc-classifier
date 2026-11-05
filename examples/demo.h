#pragma once
// What the three demos share: each looks for cards (driving licences, smart ID cards) in the images given, one way
// (sc::doc::method), and shows what it found, so the ways can be compared on the same images.
// Writes only to the output folder (default: the temporary directory's sc-doc-classifier).

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <params.h>
#include <sc.h>

#include <sc_doc_classifier.h>

namespace demo {
    inline std::vector<std::filesystem::path> images_in(const std::vector<std::string> &given) {
        std::vector<std::filesystem::path> images;
        auto is_image = [](const std::filesystem::path &p) {
            std::string ext = p.extension().string();
            for (auto &c: ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
        };
        for (const auto &g: given) {
            if (std::filesystem::is_directory(g)) {
                std::vector<std::filesystem::path> inside;
                for (const auto &entry: std::filesystem::directory_iterator(g))
                    if (entry.is_regular_file() && is_image(entry.path())) inside.push_back(entry.path());
                std::sort(inside.begin(), inside.end());
                images.insert(images.end(), inside.begin(), inside.end());
            } else images.emplace_back(g);
        }
        return images;
    }

    // Licences green, ID cards blue, passports purple, temporary licences orange
    inline std::string colour_of(const sc::doc::kind what) {
        switch (what) {
            case sc::doc::kind::sa_id_card: return "deepskyblue";
            case sc::doc::kind::passport: return "magenta";
            case sc::doc::kind::sa_temporary_driving_licence: return "orange";
            default: return "lime";
        }
    }

    inline int run(int argc, char **argv, const sc::doc::method how, const std::string &name, const std::string &description) {
        const sc::params args{argc, argv, {
            {"name", name},
            {"description", description},
            {"options", {
                {"out", {{"short", "o"}, {"type", "string"}, {"value", "DIR"},
                         {"default", (std::filesystem::temp_directory_path() / "sc-doc-classifier").string()},
                         {"help", "where to write each image with what was found drawn on it, and each card found cut out"}}},
                {"find", {{"type", "string"}, {"choices", {"all", "licence", "id-card", "passport", "temporary-licence"}}, {"default", "all"},
                          {"help", "what to look for: driving licences, smart ID cards, passports, temporary driving licences, or all"}}},
                {"cards", {{"type", "boolean"}, {"help", "also cut out every card-shaped outline, whatever it is"}}}}},
            {"extra", {
                {"images", {{"required", false}, {"help", "photographs or scans, or folders of them"}}}}}}};

        try {
            sc::console::title("simply-cpp doc-classifier: cards found by " + sc::doc::to_string(how));
            const auto images = images_in(args.as<std::vector<std::string> >("images", {}));
            if (images.empty()) {
                sc::console::note("Pass photographs or scans (or folders of them) to look in: " + name + " <image>...");
                return 0;
            }
            const std::filesystem::path out = args.as<std::string>("out");
            std::filesystem::create_directories(out);
            const bool cards = args.as<bool>("cards");
            const std::string find = args.as<std::string>("find");
            std::vector<sc::doc::kind> kinds;
            for (const auto what: {sc::doc::kind::sa_driving_licence, sc::doc::kind::sa_id_card, sc::doc::kind::passport,
                                   sc::doc::kind::sa_temporary_driving_licence})
                if (find == sc::doc::short_name(what)) kinds = {what};

            int with_card = 0;
            sc::timer all;
            for (const auto &path: images) {
                sc::console::heading(path.filename().string());
                try {
                    sc::image picture{path.string()};
                    sc::timer took;
                    const auto found = sc::doc::find_documents(picture, how, kinds);
                    const std::string time = std::string(took);
                    if (found.empty()) sc::console::note("nothing found, in " + time);
                    else ++with_card;

                    if (cards && how != sc::doc::method::text) {
                        int n = 0;
                        for (const auto &c: sc::doc::find_cards(picture, how)) {
                            picture.draw(c.outline(), sc::appearance{sc::stroke{sc::color{"yellow"}, 3}});
                            sc::doc::straightened(picture, c).save((out / (path.stem().string() + "-card-" + std::to_string(++n) + ".jpg")).string());
                        }
                    }
                    int n = 0;
                    for (const auto &doc: found) {
                        const std::string file = path.stem().string() + "-" + sc::doc::short_name(doc.type) + "-" + std::to_string(++n) + ".jpg";
                        doc.picture.save((out / file).string());
                        sc::console::note(sc::doc::to_string(doc.type) + ", " + std::string(doc.confidence) + " sure, outline fit " +
                                          std::to_string(doc.where.fit).substr(0, 4) + ", in " + time + " -> " + file);
                        std::string words;
                        for (const auto &w: doc.words) words += (words.empty() ? "" : " | ") + w;
                        sc::console::note("read: " + words);
                        picture.draw(doc.where.outline(), sc::appearance{sc::stroke{sc::color{colour_of(doc.type)}, 4}});
                        picture.draw(sc::circle{doc.where.corners[0], 10}, sc::appearance{sc::stroke{sc::color{"red"}, 4}});
                    }
                    picture.save((out / (path.stem().string() + "-found.jpg")).string(), 85);
                } catch (const std::exception &error) {
                    sc::console::note(std::string("couldn't look: ") + error.what());
                }
            }
            sc::console::heading("Summary");
            sc::console::note(std::to_string(with_card) + " of " + std::to_string(images.size()) + " images with a card found, in " +
                              std::string(all) + "; pictures in " + out.string());
        } catch (const std::exception &error) {
            std::cerr << name << ": " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
}
