// Driving licences and smart ID cards found by contours. --help lists the options.

#include "demo.h"

int main(int argc, char **argv) {
    return demo::run(argc, argv, sc::doc::method::contours, "sc-doc-classifier-contours",
                     "Finds cards by their outlines (edges closing round four corners in a bank card's proportions), then reads each for a driving licence's or smart ID card's printed words.\nEach image is written with what was found drawn on it (licences green, ID cards blue, passports purple, temporary licences orange; their top left "
                     "corner circled in red), and each card found cut out, straight and upright.");
}
