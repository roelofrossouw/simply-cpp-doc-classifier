#include "sc_doc_classifier.h"

#include <ocr.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp> // contours, lines and transforms moved out of imgproc and calib3d
#else
#include <opencv2/calib3d.hpp>  // estimateAffinePartial2D
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace sc::doc {
    namespace {
        // Cards are looked for in a copy about this many pixels on its long side
        constexpr double working_size = 1000;

        // How far a card's outline may stray from a bank card's proportions (perspective, wear)
        constexpr double min_aspect = 1.35;
        constexpr double max_aspect = 1.85;

        // A card fills at least this part of the image
        constexpr double min_area = 0.02;

        using quad = std::array<cv::Point2f, 4>;

        // sc::image keeps OpenCV out of its header, so pixels cross over as an uncompressed file.
        // TODO: a way to share the matrix with sc-image instead (it has it in impl::internals).
        cv::Mat to_mat(const image &picture) {
            if (picture.empty()) throw std::invalid_argument("sc::doc: the image is empty");
            const std::string bytes = picture.encoded("bmp");
            const cv::Mat data(1, static_cast<int>(bytes.size()), CV_8U, const_cast<char *>(bytes.data()));
            cv::Mat result = cv::imdecode(data, cv::IMREAD_COLOR);
            if (result.empty()) throw std::runtime_error("sc::doc: couldn't read the image's pixels");
            return result;
        }

        image to_image(const cv::Mat &pixels) {
            std::vector<uchar> bytes;
            cv::imencode(".png", pixels, bytes, {cv::IMWRITE_PNG_COMPRESSION, 1});
            return image(std::string(bytes.begin(), bytes.end()));
        }

        double distance(const cv::Point2f &a, const cv::Point2f &b) { return std::hypot(a.x - b.x, a.y - b.y); }

        // Clockwise from the top left, the long side first (a card lies landscape once straightened)
        quad ordered(quad corners) {
            cv::Point2f centre{0, 0};
            for (const auto &c: corners) centre += c * 0.25f;
            std::sort(corners.begin(), corners.end(), [&](const auto &a, const auto &b) {
                return std::atan2(a.y - centre.y, a.x - centre.x) < std::atan2(b.y - centre.y, b.x - centre.x);
            });
            // atan2 with y down runs clockwise from the left; start at the corner nearest the top left
            const auto first = std::min_element(corners.begin(), corners.end(), [](const auto &a, const auto &b) {
                return a.x + a.y < b.x + b.y;
            });
            std::rotate(corners.begin(), first, corners.end());
            if (distance(corners[0], corners[1]) + distance(corners[2], corners[3]) <
                distance(corners[1], corners[2]) + distance(corners[3], corners[0]))
                std::rotate(corners.begin(), corners.begin() + 3, corners.end());
            return corners;
        }

        double interior_angle(const cv::Point2f &before, const cv::Point2f &at, const cv::Point2f &after) {
            const cv::Point2f a = before - at, b = after - at;
            return std::acos(std::clamp(a.dot(b) / (std::hypot(a.x, a.y) * std::hypot(b.x, b.y)), -1.0f, 1.0f)) * 180 /
                   std::numbers::pi;
        }

        // Whether an ordered outline could be a card: its proportions, corners and size. A photo of a
        // card taken for its details is close to square on, so no corner is far from a right angle:
        // that keeps out the parallelograms a patterned background makes against a card's edge.
        bool card_shaped(const quad &q, const cv::Size image_size) {
            std::vector<cv::Point2f> points(q.begin(), q.end());
            if (!cv::isContourConvex(points)) return false;
            const double area = cv::contourArea(points);
            const double whole = static_cast<double>(image_size.area());
            if (area < min_area * whole || area > 0.995 * whole) return false;
            const double aspect = (distance(q[0], q[1]) + distance(q[2], q[3])) / (distance(q[1], q[2]) + distance(q[3], q[0]));
            if (aspect < min_aspect || aspect > max_aspect) return false;
            for (int i = 0; i < 4; ++i)
                if (const double angle = interior_angle(q[(i + 3) % 4], q[i], q[(i + 1) % 4]); angle < 75 || angle > 105) return false;
            return true;
        }

        // How close the proportions are to a card's: 1 exactly, 0 at the limits
        double aspect_fit(const quad &q) {
            const double aspect = (distance(q[0], q[1]) + distance(q[2], q[3])) / (distance(q[1], q[2]) + distance(q[3], q[0]));
            return std::clamp(1 - std::abs(std::log(aspect / card_aspect)) / std::log(max_aspect / card_aspect), 0.0, 1.0);
        }

        struct working {
            cv::Mat colour; // scaled to working_size
            cv::Mat grey;
            cv::Mat even;   // grey with its local contrast evened out (CLAHE): a card's faint edge on a
                            // bright scan as strong as a dark one, for finding outlines
            cv::Mat dx, dy; // brightness gradients of grey, blurred: for scoring outlines
            float strong;   // a gradient this strong may be a card's edge
            double scale;   // working pixels per original pixel
        };

        working prepare(const cv::Mat &original) {
            working w;
            w.scale = std::min(1.0, working_size / std::max(original.cols, original.rows));
            cv::resize(original, w.colour, {}, w.scale, w.scale, cv::INTER_AREA);
            cv::cvtColor(w.colour, w.grey, cv::COLOR_BGR2GRAY);
            cv::createCLAHE(2.0, {8, 8})->apply(w.grey, w.even);
            cv::Mat blurred;
            cv::GaussianBlur(w.grey, blurred, {5, 5}, 0);
            cv::Sobel(blurred, w.dx, CV_32F, 1, 0);
            cv::Sobel(blurred, w.dy, CV_32F, 0, 1);
            cv::Mat magnitude;
            cv::magnitude(w.dx, w.dy, magnitude);
            std::vector<float> values(magnitude.begin<float>(), magnitude.end<float>());
            const auto at = values.begin() + static_cast<std::ptrdiff_t>(0.9 * static_cast<double>(values.size()));
            std::nth_element(values.begin(), at, values.end());
            w.strong = std::max(16.0f, *at * 0.5f);
            return w;
        }

        // The share of each side that runs along a card's edge, sampled: a strong gradient within two
        // pixels, across the side (within 20 degrees) and the same way round all along it (the card
        // brighter, or darker, than what it lies on). A patterned background's edges point every
        // which way, so an outline drawn through it scores little however many edges it crosses.
        std::array<double, 4> edge_support(const quad &q, const working &w) {
            constexpr float across = 0.94f; // cos 20 degrees
            std::array<double, 4> support{};
            for (int side = 0; side < 4; ++side) {
                const auto &from = q[side];
                const auto &to = q[(side + 1) % 4];
                const float length = static_cast<float>(distance(from, to));
                const cv::Point2f normal{(to.y - from.y) / length, -(to.x - from.x) / length};
                const int samples = std::max(8, static_cast<int>(length / 3));
                int brighter = 0, darker = 0;
                // The middle 90%: rounded corners leave the ends off the edge
                for (int i = 0; i < samples; ++i) {
                    const float t = 0.05f + 0.9f * (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
                    const cv::Point2f p = from + (to - from) * t;
                    bool out = false, in = false;
                    for (int k = -2; k <= 2; ++k) {
                        const int x = cvRound(p.x + normal.x * k), y = cvRound(p.y + normal.y * k);
                        if (x < 0 || y < 0 || x >= w.dx.cols || y >= w.dx.rows) continue;
                        const float gx = w.dx.at<float>(y, x), gy = w.dy.at<float>(y, x);
                        const float magnitude = std::hypot(gx, gy);
                        if (magnitude < w.strong) continue;
                        const float along_normal = (gx * normal.x + gy * normal.y) / magnitude;
                        if (along_normal > across) out = true;
                        else if (along_normal < -across) in = true;
                    }
                    if (out) ++brighter;
                    if (in) ++darker;
                }
                support[side] = static_cast<double>(std::max(brighter, darker)) / samples;
            }
            return support;
        }

        double fit_of(const quad &q, const working &w) {
            const auto support = edge_support(q, w);
            const double least = *std::min_element(support.begin(), support.end());
            const double mean = (support[0] + support[1] + support[2] + support[3]) / 4;
            return (0.5 * mean + 0.5 * least) * (0.75 + 0.25 * aspect_fit(q));
        }

        double area_of(const quad &q) { return cv::contourArea(std::vector<cv::Point2f>(q.begin(), q.end())); }

        double overlap(const quad &a, const quad &b) {
            std::vector<cv::Point2f> shared;
            const double common = cv::intersectConvexConvex(std::vector<cv::Point2f>(a.begin(), a.end()),
                                                            std::vector<cv::Point2f>(b.begin(), b.end()), shared, true);
            return common / std::min(area_of(a), area_of(b));
        }

        struct candidate {
            quad corners;
            double fit;
        };

        // Candidates in groups of overlapping ones (a card, the sleeve it's in, the boxes printed on
        // it), each group's best fit first, groups best first; near copies of one outline dropped.
        std::vector<std::vector<candidate> > grouped(std::vector<candidate> found) {
            std::sort(found.begin(), found.end(), [](const auto &a, const auto &b) { return a.fit > b.fit; });
            std::vector<std::vector<candidate> > groups;
            for (const auto &c: found) {
                bool placed = false;
                for (auto &group: groups) {
                    if (overlap(c.corners, group.front().corners) <= 0.5) continue;
                    placed = true;
                    const bool same = std::any_of(group.begin(), group.end(), [&](const candidate &g) {
                        double furthest = 0;
                        for (int i = 0; i < 4; ++i) furthest = std::max(furthest, distance(g.corners[i], c.corners[i]));
                        return furthest < 4;
                    });
                    if (same) break;
                    group.push_back(c);
                    break;
                }
                if (!placed) groups.push_back({c});
            }
            std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b) { return a.front().fit > b.front().fit; });
            return groups;
        }

        // Canny's thresholds from how strong the image's edges are: the strongest tenth of its
        // gradients are edges for sure, those half as strong if they join one. (Thresholds from its
        // brightness fail on a pale card scanned on white paper, where every edge is faint.)
        cv::Mat edges_of(const cv::Mat &grey, const double share = 0.10) {
            cv::Mat dx, dy, magnitude;
            cv::Sobel(grey, dx, CV_32F, 1, 0);
            cv::Sobel(grey, dy, CV_32F, 0, 1);
            cv::magnitude(dx, dy, magnitude);
            std::vector<float> values(magnitude.begin<float>(), magnitude.end<float>());
            const auto at = values.begin() + static_cast<std::ptrdiff_t>((1 - share) * static_cast<double>(values.size()));
            std::nth_element(values.begin(), at, values.end());
            // Canny's thresholds are on the L1 gradient (|dx| + |dy|), about 1.2 times the L2 one
            const double high = std::clamp(static_cast<double>(*at) * 1.2, 20.0, 400.0);
            cv::Mat edges;
            cv::Canny(grey, edges, high / 2, high);
            return edges;
        }

        // The line through points, or none for too few
        std::optional<cv::Vec4f> line_through(const std::vector<cv::Point2f> &points) {
            if (points.size() < 5) return std::nullopt;
            cv::Vec4f line;
            cv::fitLine(points, line, cv::DIST_HUBER, 0, 0.01, 0.01);
            return line;
        }

        std::optional<cv::Point2f> crossing(const cv::Vec4f &a, const cv::Vec4f &b) {
            const cv::Point2f d1{a[0], a[1]}, p1{a[2], a[3]}, d2{b[0], b[1]}, p2{b[2], b[3]};
            const float denominator = d1.x * d2.y - d1.y * d2.x;
            if (std::abs(denominator) < 1e-6f) return std::nullopt;
            const cv::Point2f diff = p2 - p1;
            const float t = (diff.x * d2.y - diff.y * d2.x) / denominator;
            return p1 + d1 * t;
        }

        // Each side refitted to the outline's points along it, away from the rounded corners, and the
        // corners put where the sides cross: a rounded card's corners as its square ones would be.
        quad refined(const quad &q, const std::vector<cv::Point> &outline) {
            std::array<cv::Vec4f, 4> sides;
            for (int side = 0; side < 4; ++side) {
                const auto &from = q[side];
                const auto &to = q[(side + 1) % 4];
                const cv::Point2f along = to - from;
                const float length = std::hypot(along.x, along.y);
                const cv::Point2f unit = along / length;
                std::vector<cv::Point2f> near;
                for (const auto &p: outline) {
                    const cv::Point2f v = cv::Point2f(p) - from;
                    const float t = v.dot(unit) / length;
                    const float off = std::abs(v.x * unit.y - v.y * unit.x);
                    if (t > 0.12f && t < 0.88f && off < std::max(3.0f, 0.03f * length)) near.push_back(p);
                }
                const auto line = line_through(near);
                if (!line) return q;
                sides[side] = *line;
            }
            quad result;
            for (int i = 0; i < 4; ++i) {
                const auto corner = crossing(sides[(i + 3) % 4], sides[i]);
                if (!corner || distance(*corner, q[i]) > 0.15 * distance(q[i], q[(i + 1) % 4])) return q;
                result[i] = *corner;
            }
            return result;
        }

        // Closed outlines with four corners, from several edge maps (brightness, and colour for a
        // card on something as bright as it)
        std::vector<candidate> by_contours(const working &w) {
            std::vector<cv::Mat> maps;
            cv::Mat blurred;
            cv::GaussianBlur(w.even, blurred, {5, 5}, 0);
            maps.push_back(edges_of(blurred));
            cv::Mat hsv;
            cv::cvtColor(w.colour, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            cv::GaussianBlur(channels[1], blurred, {5, 5}, 0);
            maps.push_back(edges_of(blurred));
            cv::Mat gradient;
            cv::morphologyEx(w.even, gradient, cv::MORPH_GRADIENT, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
            cv::threshold(gradient, gradient, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            maps.push_back(gradient);

            std::vector<candidate> found;
            for (auto &edges: maps) {
                cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
                std::vector<std::vector<cv::Point> > contours;
                cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                for (const auto &contour: contours) {
                    if (std::abs(cv::contourArea(contour)) < min_area * w.grey.total() * 0.5) continue;
                    std::vector<cv::Point> hull;
                    cv::convexHull(contour, hull);
                    const double perimeter = cv::arcLength(hull, true);
                    std::vector<cv::Point> corners;
                    for (double share = 0.01; share <= 0.06; share += 0.005) {
                        cv::approxPolyDP(hull, corners, share * perimeter, true);
                        if (corners.size() <= 4) break;
                    }
                    if (corners.size() != 4) continue;
                    quad q;
                    for (int i = 0; i < 4; ++i) q[i] = corners[i];
                    q = ordered(refined(ordered(q), contour));
                    if (!card_shaped(q, w.grey.size())) continue;
                    // The outline must hug its hull: a card, not a blob that happens to have four corners
                    if (std::abs(cv::contourArea(hull)) < 0.85 * area_of(q)) continue;
                    found.push_back({q, fit_of(q, w)});
                }
            }
            return found;
        }

        struct segment {
            cv::Point2f from, to;

            [[nodiscard]] double length() const { return distance(from, to); }
            [[nodiscard]] double angle() const { return std::atan2(to.y - from.y, to.x - from.x); }
            [[nodiscard]] cv::Vec4f line() const {
                const auto d = (to - from) / static_cast<float>(length());
                return {d.x, d.y, from.x, from.y};
            }
        };

        // Angle between two lines' directions, 0 to 90 degrees
        double angle_between(const segment &a, const segment &b) {
            double d = std::abs(a.angle() - b.angle()) * 180 / std::numbers::pi;
            d = std::fmod(d, 180);
            return d > 90 ? 180 - d : d;
        }

        double distance_to_line(const cv::Point2f &p, const segment &s) {
            const cv::Point2f d = (s.to - s.from) / static_cast<float>(s.length());
            const cv::Point2f v = p - s.from;
            return std::abs(v.x * d.y - v.y * d.x);
        }

        // Pieces of one straight edge joined: same direction, on the same line
        std::vector<segment> joined(std::vector<segment> segments, const double tolerance) {
            std::sort(segments.begin(), segments.end(), [](const auto &a, const auto &b) { return a.length() > b.length(); });
            std::vector<segment> lines;
            for (const auto &s: segments) {
                bool absorbed = false;
                for (auto &l: lines) {
                    if (angle_between(s, l) > 3 || distance_to_line(s.from, l) > tolerance || distance_to_line(s.to, l) > tolerance) continue;
                    // Extend l to cover s, along l's direction
                    const cv::Point2f d = (l.to - l.from) / static_cast<float>(l.length());
                    const float a = (s.from - l.from).dot(d), b = (s.to - l.from).dot(d);
                    const float low = std::min({0.0f, a, b}), high = std::max({static_cast<float>(l.length()), a, b});
                    const cv::Point2f origin = l.from;
                    l.from = origin + d * low;
                    l.to = origin + d * high;
                    absorbed = true;
                    break;
                }
                if (!absorbed) lines.push_back(s);
            }
            return lines;
        }

        // Two pairs of nearly parallel straight edges, nearly at right angles to each other
        std::vector<candidate> by_lines(const working &w) {
            auto detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
            std::vector<cv::Vec4f> raw;
            cv::Mat blurred;
            cv::GaussianBlur(w.grey, blurred, {3, 3}, 0);
            detector->detect(blurred, raw);
            const double short_side = std::min(w.grey.cols, w.grey.rows);
            std::vector<segment> segments;
            for (const auto &r: raw)
                if (segment s{{r[0], r[1]}, {r[2], r[3]}}; s.length() > 0.02 * short_side) segments.push_back(s);
            auto lines = joined(segments, 0.006 * short_side + 1.5);
            std::erase_if(lines, [&](const auto &l) { return l.length() < 0.08 * short_side; });
            std::sort(lines.begin(), lines.end(), [](const auto &a, const auto &b) { return a.length() > b.length(); });
            if (lines.size() > 50) lines.resize(50);

            struct pair {
                int a, b;
            };
            std::vector<pair> parallels;
            for (int i = 0; i < static_cast<int>(lines.size()); ++i)
                for (int j = i + 1; j < static_cast<int>(lines.size()); ++j)
                    if (angle_between(lines[i], lines[j]) < 12 &&
                        distance_to_line((lines[j].from + lines[j].to) * 0.5f, lines[i]) > 0.1 * short_side)
                        parallels.push_back({i, j});

            std::vector<candidate> found;
            for (std::size_t p = 0; p < parallels.size(); ++p)
                for (std::size_t r = p + 1; r < parallels.size(); ++r) {
                    const auto &one = parallels[p], &two = parallels[r];
                    if (one.a == two.a || one.a == two.b || one.b == two.a || one.b == two.b) continue;
                    if (angle_between(lines[one.a], lines[two.a]) < 75) continue;
                    const auto c0 = crossing(lines[one.a].line(), lines[two.a].line());
                    const auto c1 = crossing(lines[two.a].line(), lines[one.b].line());
                    const auto c2 = crossing(lines[one.b].line(), lines[two.b].line());
                    const auto c3 = crossing(lines[two.b].line(), lines[one.a].line());
                    if (!c0 || !c1 || !c2 || !c3) continue;
                    const quad q = ordered({*c0, *c1, *c2, *c3});
                    if (!card_shaped(q, w.grey.size())) continue;
                    // Each line must lie along its side, not merely point the same way from elsewhere
                    const double fit = fit_of(q, w);
                    if (fit < 0.3) continue;
                    found.push_back({q, fit});
                }
            return found;
        }

        // OCR, one reader a thread (loading the models takes a while)
        sc::ocr &reader() {
            thread_local sc::ocr ocr;
            return ocr;
        }

        // Upper case letters and digits only: "Lic. No./Ilayisensi" -> "LICNOILAYISENSI"
        std::string normalised(const std::string_view text) {
            std::string result;
            for (const unsigned char c: text)
                if (std::isalnum(c)) result.push_back(static_cast<char>(std::toupper(c)));
            return result;
        }

        // The fewest edits that make word appear somewhere in text, and where that ends (Sellers)
        std::pair<int, std::size_t> fuzzy_find(const std::string &word, const std::string &text) {
            std::vector<int> previous(text.size() + 1, 0), current(text.size() + 1);
            for (std::size_t i = 1; i <= word.size(); ++i) {
                current[0] = static_cast<int>(i);
                for (std::size_t j = 1; j <= text.size(); ++j)
                    current[j] = std::min({previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word[i - 1] == text[j - 1] ? 0 : 1)});
                std::swap(previous, current);
            }
            const auto best = std::min_element(previous.begin(), previous.end());
            return {*best, static_cast<std::size_t>(best - previous.begin())};
        }

        // A document's printed labels: where each starts (or, right aligned, ends) on it straightened, as
        // shares of its width and height; how far from there it may be found (shares too); how much
        // finding it says the document is this kind; and whether it says where the document is (a
        // label whose place depends on what's printed above it doesn't).
        struct label {
            std::string word; // normalised; MRZ1 and MRZ2 are a passport's machine readable lines
            point at;         // the label's left middle (right middle when right_aligned)
            bool right_aligned = false;
            double weight = 1;
            bool anchor = true;
            double x_slack = 0.08;
            double y_slack = 0.06;
        };

        // One kind of document: its size straightened, its labels, the ones of them in its top or
        // bottom band (a quick look there says whether reading the whole of it is worthwhile), those
        // of which one must be read in place for it to be this kind (others share words with it: a
        // passport has Surname and REPUBLIC OF SOUTH AFRICA too), and whether its edges are snapped
        // to (not a border whose inside edge is the sharper one).
        struct layout {
            kind type;
            size_i size;
            std::vector<label> labels;
            std::vector<std::string> quick_words;
            std::vector<std::string> required;
            bool snap = true; // its edges are snapped to an outline found near where its labels place it
        };

        // A label measured in pixels on a straightened document of the given size, as shares
        label at(std::string word, const double x, const double y, const size_i measured, const bool right_aligned = false,
                 const double weight = 1, const bool anchor = true, const double x_slack = 0.08, const double y_slack = 0.06) {
            return {std::move(word), point{x / measured.width(), y / measured.height()}, right_aligned, weight, anchor, x_slack, y_slack};
        }

        // Measured on straightened documents (OCR of the demos' cut-outs)
        const std::vector<layout> &layouts() {
            static const std::vector<layout> all{
                {kind::sa_driving_licence, card_size,
                 {
                     at("DRIVINGLICENCE", 23, 40, card_size, false, 3),
                     at("CARTADECONDUCAO", 24, 104, card_size, false, 2),
                     at("SOUTHAFRICA", 992, 47, card_size, true, 1),
                     at("SADC", 570, 27, card_size),
                     at("IDNO", 24, 186, card_size),
                     at("BIRTH", 24, 223, card_size),
                     at("RESTR", 499, 229, card_size),
                     at("LICNO", 26, 261, card_size, false, 2),
                     at("VALID", 22, 297, card_size),
                     at("ISSUED", 21, 333, card_size),
                     at("CODE", 21, 382, card_size),
                     at("VEHRESTR", 21, 415, card_size),
                     at("FIRSTISSUE", 20, 445, card_size),
                     at("PROFDRIVINGPERMIT", 626, 476, card_size, false, 2),
                 },
                 {"DRIVINGLICENCE", "CARTADECONDUCAO", "SADC", "SOUTHAFRICA"},
                 {"DRIVINGLICENCE", "CARTADECONDUCAO", "LICNO"}},
                // The smart ID card's fields below Names move down when the names take two lines,
                // and Sex is beside Country of birth on some cards: those say what it is, not where
                {kind::sa_id_card, card_size,
                 {
                     at("REPUBLICOFSOUTHAFRICA", 268, 69, card_size, false, 2),
                     at("NATIONALIDENTITYCARD", 271, 104, card_size, false, 3),
                     // OCR sometimes reads NATIONAL and IDENTITY CARD as two lines
                     at("IDENTITYCARD", 538, 104, card_size, false, 2),
                     at("SURNAME", 273, 162, card_size),
                     at("NAMES", 272, 219, card_size),
                     at("SEX", 271, 274, card_size, false, 1, false, 1.0, 0.45),
                     at("NATIONALITY", 272, 330, card_size, false, 1, false, 0.08, 0.3),
                     at("IDENTITYNUMBER", 274, 384, card_size, false, 2, false, 0.08, 0.3),
                     at("DATEOFBIRTH", 276, 439, card_size, false, 1, false, 0.08, 0.3),
                     at("COUNTRYOFBIRTH", 276, 495, card_size, false, 1, false, 0.08, 0.3),
                     at("STATUS", 275, 549, card_size, false, 1, false, 0.08, 0.3),
                     at("SIGNATURE", 795, 510, card_size),
                 },
                 {"REPUBLICOFSOUTHAFRICA", "NATIONALIDENTITYCARD", "IDENTITYCARD"},
                 {"NATIONALIDENTITYCARD", "IDENTITYCARD"}},
                // A passport's data page: any country's by its machine readable lines (ICAO 9303
                // puts them in the same place on every passport; their check digits say it's one),
                // a South African one's printed labels too
                {kind::passport, passport_page_size,
                 {
                     at("MRZ1", 46, 813, {1000, 1000}, false, 2),
                     at("MRZ1", 947, 813, {1000, 1000}, true, 0),
                     at("MRZ2", 45, 887, {1000, 1000}, false, 2),
                     at("MRZ2", 945, 887, {1000, 1000}, true, 0),
                     at("PASSPORTPASSEPORT", 64, 72, {1000, 1000}),
                     at("REPUBLICOFSOUTHAFRICA", 312, 60, {1000, 1000}),
                     at("SURNAMENOM", 315, 174, {1000, 1000}),
                     at("GIVENNAMES", 316, 246, {1000, 1000}),
                     at("NATIONALITY", 316, 320, {1000, 1000}),
                     at("DATEOFBIRTH", 315, 393, {1000, 1000}),
                     at("IDENTITYNO", 657, 388, {1000, 1000}),
                     at("DATEOFISSUE", 316, 543, {1000, 1000}),
                     at("AUTHORITY", 658, 539, {1000, 1000}),
                     at("DATEOFEXPIRY", 316, 616, {1000, 1000}),
                 },
                 {"MRZ1", "MRZ2", "PASSPORTPASSEPORT"},
                 {"MRZ1"}},
                // The temporary driving licence (form TDL): a green security form, its outside the
                // outer edge of its guilloche border, labels in English and Afrikaans all over it
                {kind::sa_temporary_driving_licence, temporary_licence_size,
                 {
                     at("TEMPORARYDRIVING", 498, 81, {1000, 1000}, false, 3),
                     at("TYDELIKEBESTUURS", 750, 75, {1000, 1000}, false, 2),
                     at("TDL5", 470, 67, {1000, 1000}),
                     at("PARTICULARSOFHOLDER", 470, 152, {1000, 1000}, false, 2),
                     at("BESONDERHEDEVANHOUER", 754, 145, {1000, 1000}),
                     at("TYPEOFIDENTIFICATION", 582, 200, {1000, 1000}),
                     at("IDENTIFICATIONNUMBER", 582, 237, {1000, 1000}),
                     at("COUNTRYOFISSUE", 582, 274, {1000, 1000}),
                     at("INITIALSANDSURNAME", 582, 311, {1000, 1000}),
                     at("DATEOFISSUE", 582, 347, {1000, 1000}),
                     at("DRIVINGLICENCETESTINGCENTRE", 583, 382, {1000, 1000}),
                     at("DECLARATION", 126, 429, {1000, 1000}),
                     at("VERKLARING", 322, 425, {1000, 1000}),
                     at("DRIVERRESTRICTIONS", 75, 583, {1000, 1000}),
                     at("VEHICLERESTRICTIONS", 260, 577, {1000, 1000}),
                     at("VALIDFOR6MONTHS", 477, 691, {1000, 1000}, false, 2),
                     at("RECEIPT", 481, 732, {1000, 1000}),
                     at("KWITANSIE", 838, 724, {1000, 1000}),
                     at("RECEIPTNUMBER", 478, 772, {1000, 1000}),
                     at("TOTALAMOUNTRECEIVED", 478, 801, {1000, 1000}),
                     at("LEFTTHUMBIMPRINT", 77, 835, {1000, 1000}),
                     at("LINKERDUIMAFDRUK", 335, 828, {1000, 1000}),
                 },
                 {"TEMPORARYDRIVING", "TYDELIKEBESTUURS", "PARTICULARSOFHOLDER", "TDL5", "RECEIPTNUMBER", "TOTALAMOUNTRECEIVED",
                  "LEFTTHUMBIMPRINT"},
                 {"TEMPORARYDRIVING", "TYDELIKEBESTUURS", "VALIDFOR6MONTHS"},
                 false},
            };
            return all;
        }

        const layout &layout_of(const kind what) {
            for (const auto &l: layouts())
                if (l.type == what) return l;
            throw std::invalid_argument("sc::doc: no layout for " + to_string(what));
        }

        struct label_hit {
            const label *of;
            sc::ocr::line line;
            bool at_start;     // the line starts with it (at its end for a right aligned label)
            double extra = 0;  // weight it adds besides its label's: a machine readable line's good check digits

            // Where it was read: the label's left middle (right middle when right aligned)
            [[nodiscard]] cv::Point2f point() const {
                const auto &box = line.box;
                return {static_cast<float>(of->right_aligned ? box.left() + box.width() : box.left()),
                        static_cast<float>(box.top() + box.height() / 2)};
            }
        };

        // A machine readable line as read: upper case, its letters, digits and <, spaces dropped
        std::string mrz_text(const std::string_view text) {
            std::string result;
            for (const unsigned char c: text) {
                if (std::isalnum(c)) result.push_back(static_cast<char>(std::toupper(c)));
                else if (c == '<' || c == '(' || c == '{' || c == '[') result.push_back('<');
            }
            return result;
        }

        // ICAO 9303's check digit over field (7, 3, 1 weights; letters 10 to 35, < 0); OCR's O read
        // for 0 in a number field forgiven
        bool check_digit_ok(std::string field, const char digit) {
            int total = 0;
            constexpr int weights[] = {7, 3, 1};
            for (std::size_t i = 0; i < field.size(); ++i) {
                const char c = field[i];
                int value = 0;
                if (c >= '0' && c <= '9') value = c - '0';
                else if (c >= 'A' && c <= 'Z') value = c - 'A' + 10;
                total += value * weights[i % 3];
            }
            const char d = digit == 'O' ? '0' : digit;
            return d >= '0' && d <= '9' && total % 10 == d - '0';
        }

        // How many of a passport's second machine readable line's check digits are right: the number,
        // the birth date, the expiry date and the line as a whole (0 to 4)
        int good_check_digits(std::string line) {
            if (line.size() < 44) return 0;
            line.resize(44);
            auto digits = [&](const std::size_t from, const std::size_t length) {
                std::string field = line.substr(from, length);
                for (auto &c: field)
                    if (c == 'O') c = '0';
                return field;
            };
            int good = 0;
            if (check_digit_ok(line.substr(0, 9), line[9])) ++good;
            if (check_digit_ok(digits(13, 6), line[19])) ++good;
            if (check_digit_ok(digits(21, 6), line[27])) ++good;
            const std::string composite = line.substr(0, 10) + digits(13, 7) + digits(21, 22);
            if (check_digit_ok(composite, line[43])) ++good;
            return good;
        }

        // A passport's machine readable lines in the lines read: the first starts P and its type and
        // country, with << between surname and names; the second, under it, is mostly digits
        void add_mrz(const layout &kind_of, const std::vector<sc::ocr::line> &lines, std::vector<label_hit> &hits) {
            const label *first_left = nullptr, *first_right = nullptr, *second_left = nullptr, *second_right = nullptr;
            for (const auto &l: kind_of.labels) {
                if (l.word == "MRZ1") (l.right_aligned ? first_right : first_left) = &l;
                if (l.word == "MRZ2") (l.right_aligned ? second_right : second_left) = &l;
            }
            if (!first_left) return;
            for (const auto &one: lines) {
                const auto first = mrz_text(one.text);
                if (first.size() < 36 || first.size() > 48 || first[0] != 'P' || first.find("<<") == std::string::npos) continue;
                hits.push_back({first_left, one, true});
                hits.push_back({first_right, one, true});
                const double height = one.box.height();
                for (const auto &two: lines) {
                    const auto second = mrz_text(two.text);
                    const auto digits = std::count_if(second.begin(), second.end(), [](const char c) { return c >= '0' && c <= '9'; });
                    if (second.size() < 36 || second.size() > 48 || digits < 12) continue;
                    const double below = two.box.top() - one.box.top();
                    if (below < 0.5 * height || below > 3 * height || std::abs(two.box.left() - one.box.left()) > 2 * height) continue;
                    hits.push_back({second_left, two, true, static_cast<double>(good_check_digits(second))});
                    hits.push_back({second_right, two, true});
                    break;
                }
                return;
            }
        }

        // A layout's labels in the lines read, each once (the best read)
        std::vector<label_hit> words_of(const layout &kind_of, const std::vector<sc::ocr::line> &lines) {
            std::vector<label_hit> hits;
            for (const auto &l: kind_of.labels) {
                if (l.word.starts_with("MRZ")) continue;
                std::optional<label_hit> best;
                int best_edits = 0;
                for (const auto &line: lines) {
                    const std::string text = normalised(line.text);
                    if (text.size() < 3) continue;
                    // The ID card's "REPUBLIC OF SOUTH AFRICA" isn't the licence's
                    if (l.word == "SOUTHAFRICA" && text.find("REPUBLIC") != std::string::npos) continue;
                    const auto [edits, end] = fuzzy_find(l.word, text);
                    const int allowed = l.word.size() <= 4 ? 0 : l.word.size() <= 8 ? 1 : 2;
                    if (edits > allowed) continue;
                    const std::size_t start = end >= l.word.size() ? end - l.word.size() : 0;
                    const bool at_start = l.right_aligned ? text.size() - end <= 1 : start <= 1;
                    // Fewest edits, and where the line starts with it (RESTR also sits inside VEH.RESTR)
                    if (!best || edits < best_edits || (edits == best_edits && at_start && !best->at_start)) {
                        best = label_hit{&l, line, at_start};
                        best_edits = edits;
                    }
                }
                if (best) hits.push_back(*best);
            }
            add_mrz(kind_of, lines, hits);
            return hits;
        }

        // The labels read where they belong, given where each read point lies on the document
        // straightened (as shares of its width and height): a crop of half a card, or of a box
        // printed inside one, reads the words but has them in the wrong places
        template<typename OnDocument>
        std::vector<label_hit> in_place(const std::vector<label_hit> &hits, OnDocument on_document) {
            std::vector<label_hit> placed;
            for (const auto &h: hits) {
                if (!h.at_start) continue;
                const cv::Point2f p = on_document(h.point());
                if (std::abs(p.x - h.of->at.x()) < h.of->x_slack && std::abs(p.y - h.of->at.y()) < h.of->y_slack) placed.push_back(h);
            }
            return placed;
        }

        // On a straightened picture of size
        std::vector<label_hit> in_place(const std::vector<label_hit> &hits, const size_i size) {
            return in_place(hits, [&](const cv::Point2f &p) {
                return cv::Point2f{p.x / static_cast<float>(size.width()), p.y / static_cast<float>(size.height())};
            });
        }

        double weight_of(const std::vector<label_hit> &hits) {
            double total = 0;
            for (const auto &h: hits) total += h.of->weight + h.extra;
            return total;
        }

        bool has_required(const layout &kind_of, const std::vector<label_hit> &hits) {
            return std::any_of(hits.begin(), hits.end(), [&](const label_hit &h) {
                return std::find(kind_of.required.begin(), kind_of.required.end(), h.of->word) != kind_of.required.end();
            });
        }

        // Sure or not from the labels' weight: a title alone (3) isn't enough, a title and two more
        // fields are; and one of the kind's own words must be among them
        percent confidence_of(const layout &kind_of, const std::vector<label_hit> &hits) {
            if (!has_required(kind_of, hits)) return 0;
            return std::clamp((weight_of(hits) - 2) / 6, 0.0, 1.0);
        }

        quad scaled(const quad &q, const double factor) {
            quad result;
            for (int i = 0; i < 4; ++i) result[i] = q[i] * static_cast<float>(factor);
            return result;
        }

        card to_card(const quad &q, const double fit) {
            card c;
            for (int i = 0; i < 4; ++i) c.corners[i] = point{q[i].x, q[i].y};
            c.fit = fit;
            return c;
        }

        quad to_quad(const card &c) {
            quad q;
            for (int i = 0; i < 4; ++i) q[i] = {static_cast<float>(c.corners[i].x()), static_cast<float>(c.corners[i].y())};
            return q;
        }

        cv::Mat straightened(const cv::Mat &original, const quad &corners, const size_i size) {
            const std::array<cv::Point2f, 4> target{
                cv::Point2f{0, 0}, {static_cast<float>(size.width()), 0},
                {static_cast<float>(size.width()), static_cast<float>(size.height())}, {0, static_cast<float>(size.height())}};
            const cv::Mat transform = cv::getPerspectiveTransform(corners.data(), target.data());
            cv::Mat result;
            cv::warpPerspective(original, result, transform, {size.width(), size.height()}, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
            return result;
        }

        // Where a document is from where its anchor labels were read: their places on it straightened
        // (at its layout's size) mapped onto the image by a similarity (turn, scale, shift; the anchors
        // mostly sit in one column, which pins a perspective down poorly, and a photo of a card is close
        // to one), RANSAC dropping misreads. Gives the transform from the document to the image.
        std::optional<cv::Mat> from_labels(const layout &kind_of, const std::vector<label_hit> &hits) {
            std::vector<cv::Point2f> on_document, in_image;
            for (const auto &h: hits) {
                if (!h.at_start || !h.of->anchor) continue;
                on_document.emplace_back(static_cast<float>(h.of->at.x() * kind_of.size.width()),
                                         static_cast<float>(h.of->at.y() * kind_of.size.height()));
                in_image.push_back(h.point());
            }
            if (on_document.size() < 3) return std::nullopt;
            cv::Mat inliers;
            cv::Mat similarity = cv::estimateAffinePartial2D(on_document, in_image, inliers, cv::RANSAC, 12);
            if (similarity.empty() || cv::countNonZero(inliers) < 3) return std::nullopt;
            return similarity;
        }

        std::vector<cv::Point2f> corners_of(const size_i size) {
            const auto w = static_cast<float>(size.width()), h = static_cast<float>(size.height());
            return {{0, 0}, {w, 0}, {w, h}, {0, h}};
        }

        bool wanted(const std::vector<kind> &kinds, const kind what) {
            return kinds.empty() || std::find(kinds.begin(), kinds.end(), what) != kinds.end();
        }

        // Each wanted kind found by its printed words read from the whole image (upright first), its
        // edges snapped to the nearest outline
        std::vector<found_document> by_text(const image &picture, const std::vector<kind> &kinds) {
            std::vector<found_document> result;
            const int rotation = sc::ocr::upright_rotation(picture);
            image upright = picture;
            upright.rotate(rotation);
            auto &ocr = reader();
            ocr.set_auto_rotate(false);
            ocr.detect(upright, 70);
            const auto lines = ocr.lines();

            std::optional<working> w;
            std::vector<candidate> outlines;
            cv::Mat upright_pixels;
            for (const auto &kind_of: layouts()) {
                if (!wanted(kinds, kind_of.type)) continue;
                const auto hits = words_of(kind_of, lines);
                if (confidence_of(kind_of, hits) < 50) continue;
                const auto placed = from_labels(kind_of, hits);
                if (!placed) continue;
                // Sure from the labels read where they belong once the document is placed
                cv::Mat onto_document;
                cv::invertAffineTransform(*placed, onto_document);
                const auto confirmed = in_place(hits, [&](const cv::Point2f &p) {
                    std::vector<cv::Point2f> out;
                    cv::transform(std::vector<cv::Point2f>{p}, out, onto_document);
                    return cv::Point2f{out[0].x / static_cast<float>(kind_of.size.width()),
                                       out[0].y / static_cast<float>(kind_of.size.height())};
                });
                const percent sure = confidence_of(kind_of, confirmed);
                if (sure < 50) continue;

                if (!w) {
                    upright_pixels = to_mat(upright);
                    w = prepare(upright_pixels);
                    outlines = by_lines(*w);
                    const auto more = by_contours(*w);
                    outlines.insert(outlines.end(), more.begin(), more.end());
                }
                std::vector<cv::Point2f> placed_corners;
                cv::transform(corners_of(kind_of.size), placed_corners, *placed);
                quad guess;
                for (int i = 0; i < 4; ++i) guess[i] = placed_corners[i] * static_cast<float>(w->scale);

                // Snap to an outline that matches the labels' document, if there is one: the labels
                // place it to within a few pixels, its edges to within one
                quad best = guess;
                double best_overlap = 0.8;
                for (const auto &c: kind_of.snap ? outlines : std::vector<candidate>{}) {
                    const double o = std::min(overlap(c.corners, guess), std::min(area_of(c.corners), area_of(guess)) /
                                                                         std::max(area_of(c.corners), area_of(guess)));
                    if (o > best_overlap) {
                        best_overlap = o;
                        best = c.corners;
                    }
                }
                // Keep the labels' orientation: start from the outline's corner nearest the guess's top left
                const auto first = std::min_element(best.begin(), best.end(), [&](const auto &a, const auto &b) {
                    return distance(a, guess[0]) < distance(b, guess[0]);
                });
                std::rotate(best.begin(), first, best.end());
                const quad corners = scaled(best, 1 / w->scale);

                found_document doc;
                doc.type = kind_of.type;
                doc.how = method::text;
                doc.confidence = sure;
                for (const auto &h: confirmed)
                    if (h.of->weight > 0) doc.words.push_back(h.line.text);
                doc.picture = to_image(straightened(upright_pixels, corners, kind_of.size));
                // Corners back in the given image's pixels: undo the turn
                quad given = corners;
                const auto size = picture.size();
                for (auto &c: given) {
                    const float x = c.x, y = c.y;
                    switch (((rotation % 360) + 360) % 360) {
                        case 90: c = {y, static_cast<float>(size.height()) - x};
                            break;
                        case 180: c = {static_cast<float>(size.width()) - x, static_cast<float>(size.height()) - y};
                            break;
                        case 270: c = {static_cast<float>(size.width()) - y, x};
                            break;
                        default: break;
                    }
                }
                doc.where = to_card(given, best_overlap > 0.8 ? best_overlap : kind_of.snap ? 0.3 : 0.5);
                result.push_back(std::move(doc));
            }
            return result;
        }
    }

    std::string to_string(const method how) {
        switch (how) {
            case method::contours: return "contours";
            case method::lines: return "lines";
            case method::text: return "text";
        }
        return "unknown";
    }

    std::string to_string(const kind what) {
        switch (what) {
            case kind::sa_driving_licence: return "South African driving licence";
            case kind::sa_id_card: return "South African smart ID card";
            case kind::passport: return "passport";
            case kind::sa_temporary_driving_licence: return "South African temporary driving licence";
            case kind::unknown: break;
        }
        return "unknown";
    }

    std::string short_name(const kind what) {
        switch (what) {
            case kind::sa_driving_licence: return "licence";
            case kind::sa_id_card: return "id-card";
            case kind::passport: return "passport";
            case kind::sa_temporary_driving_licence: return "temporary-licence";
            case kind::unknown: break;
        }
        return "unknown";
    }

    size_i size_of(const kind what) { return what == kind::unknown ? card_size : layout_of(what).size; }

    polygon card::outline() const { return polygon{std::vector<point>(corners.begin(), corners.end())}; }

    std::vector<card> find_cards(const image &picture, const method how) {
        if (how == method::text) {
            std::vector<card> cards;
            for (const auto &doc: by_text(picture, {})) cards.push_back(doc.where);
            return cards;
        }
        const cv::Mat original = to_mat(picture);
        const working w = prepare(original);
        std::vector<card> cards;
        for (const auto &group: grouped(how == method::lines ? by_lines(w) : by_contours(w)))
            cards.push_back(to_card(scaled(group.front().corners, 1 / w.scale), group.front().fit));
        return cards;
    }

    image straightened(const image &picture, const card &where, const size_i size) {
        if (size.width() <= 0 || size.height() <= 0) throw std::invalid_argument("sc::doc::straightened: the size must be positive");
        return to_image(straightened(to_mat(picture), to_quad(where), size));
    }

    identification identify(const image &document_picture) {
        // The way up the orientation classifier says, and if that doesn't read as a known kind, the
        // other (a straightened document is landscape, so it's one of two)
        auto &ocr = reader();
        ocr.set_auto_rotate(false);
        const int first = sc::ocr::upright_rotation(document_picture) == 180 ? 180 : 0;
        identification result;
        std::vector<label_hit> hits;
        const layout *best = nullptr;
        for (const int rotation: {first, 180 - first}) {
            image turned = document_picture;
            turned.rotate(rotation);
            // A quick look first, at its top and bottom bands: the kinds whose words are there are the
            // ones worth reading the whole of it for
            const auto size = turned.size();
            const int band = std::max(1, size.height() * 30 / 100);
            std::vector<sc::ocr::line> quick;
            ocr.detect(turned.cropped(rect_i{0, 0, size.width(), band}), 70);
            quick = ocr.lines();
            ocr.detect(turned.cropped(rect_i{0, size.height() - band, size.width(), band}), 70);
            for (auto line: ocr.lines()) {
                line.box = rect{line.box.left(), line.box.top() + (size.height() - band), line.box.width(), line.box.height()};
                quick.push_back(line);
            }
            std::vector<const layout *> likely;
            for (const auto &kind_of: layouts()) {
                const auto words = words_of(kind_of, quick);
                if (std::any_of(words.begin(), words.end(), [&](const label_hit &h) {
                    return std::find(kind_of.quick_words.begin(), kind_of.quick_words.end(), h.of->word) != kind_of.quick_words.end();
                }))
                    likely.push_back(&kind_of);
            }
            if (likely.empty()) continue;
            ocr.detect(turned, 70);
            for (const auto *kind_of: likely) {
                auto read = in_place(words_of(*kind_of, ocr.lines()), turned.size());
                if (!best || confidence_of(*kind_of, read) > confidence_of(*best, hits) ||
                    (confidence_of(*kind_of, read) == confidence_of(*best, hits) && weight_of(read) > weight_of(hits))) {
                    hits = std::move(read);
                    best = kind_of;
                    result.rotation = rotation;
                }
            }
            if (best && confidence_of(*best, hits) >= 50) break;
        }
        if (!best) return result;
        for (const auto &h: hits)
            if (h.of->weight > 0) result.words.push_back(h.line.text);
        result.confidence = confidence_of(*best, hits);
        result.type = static_cast<double>(result.confidence) >= 50 ? best->type : kind::unknown;
        return result;
    }

    std::vector<found_document> find_documents(const image &picture, const method how, const std::vector<kind> &kinds) {
        std::vector<found_document> result;
        if (how == method::text) result = by_text(picture, kinds);
        else {
            // Best outlines first, and of overlapping ones the best first, then the others (the card
            // in its sleeve, or out of it) until one reads as a wanted kind. Reading takes a while:
            // ten reads at most, three of a group.
            const cv::Mat original = to_mat(picture);
            const working w = prepare(original);
            std::vector<quad> taken;
            int read = 0;
            for (const auto &group: grouped(how == method::lines ? by_lines(w) : by_contours(w))) {
                int tried = 0;
                for (const auto &c: group) {
                    if (++tried > 3 || read >= 10) break;
                    quad corners = scaled(c.corners, 1 / w.scale);
                    // Part of a document already found
                    if (std::any_of(taken.begin(), taken.end(), [&](const quad &t) { return overlap(t, corners) > 0.3; })) break;
                    ++read;
                    // Read at a card's size whatever it is: labels are placed as shares of it
                    const auto what = identify(to_image(straightened(original, corners, card_size)));
                    if (what.type == kind::unknown || !wanted(kinds, what.type)) continue;
                    found_document doc;
                    doc.type = what.type;
                    doc.how = how;
                    doc.confidence = what.confidence;
                    doc.words = what.words;
                    // Upright: turned half round, its top left is the corner that was its bottom right
                    if (what.rotation == 180) std::rotate(corners.begin(), corners.begin() + 2, corners.end());
                    doc.where = to_card(corners, c.fit);
                    doc.picture = to_image(straightened(original, corners, size_of(what.type)));
                    taken.push_back(corners);
                    result.push_back(std::move(doc));
                    break;
                }
            }
        }
        std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return static_cast<double>(a.confidence) > static_cast<double>(b.confidence);
        });
        return result;
    }

    std::vector<found_document> find_driving_licences(const image &picture, const method how) {
        return find_documents(picture, how, {kind::sa_driving_licence});
    }

    std::vector<found_document> find_id_cards(const image &picture, const method how) {
        return find_documents(picture, how, {kind::sa_id_card});
    }

    std::vector<found_document> find_passports(const image &picture, const method how) {
        return find_documents(picture, how, {kind::passport});
    }

    std::vector<found_document> find_temporary_licences(const image &picture, const method how) {
        return find_documents(picture, how, {kind::sa_temporary_driving_licence});
    }
}
