// Driving licences and smart ID cards found by text. --help lists the options.

#include "demo.h"

int main(int argc, char **argv) {
    return demo::run(argc, argv, sc::doc::method::text, "sc-doc-classifier-text",
                     "Finds driving licences and smart ID cards by their printed words (DRIVING LICENCE, Lic. No.; NATIONAL IDENTITY CARD, Surname, ...): where the words are read gives where the card is, then its edges are snapped to.\nEach image is written with what was found drawn on it (licences green, ID cards blue, passports purple, temporary licences orange; their top left "
                     "corner circled in red), and each card found cut out, straight and upright.");
}
