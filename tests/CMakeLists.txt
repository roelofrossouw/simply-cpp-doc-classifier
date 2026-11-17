# Resources for the tests: made-up specimen documents (resource/samples, as in sc-ai's tests).
set(RESOURCE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resource")
set(RESOURCE_DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/resource")
add_custom_target(copy_sc_doc_classifier_resources ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCE_SOURCE_DIR} ${RESOURCE_DESTINATION_DIR})

# Real identity documents are never in this repository. They are in the simply-cpp suite's private
# area, private/sc-ai (shared with sc-ai's OCR and face tests; deploy.sh copies it to
# /var/www/build/private/sc-ai on the build servers). Where it is found, the tests get it as
# resource/private; without it, the tests that need it are skipped.
set(SC_PRIVATE_DIR "" CACHE PATH "The suite's private test data (default: found beside the module)")
if (NOT SC_PRIVATE_DIR)
    foreach (candidate "${PROJECT_SOURCE_DIR}/../../private" "${PROJECT_SOURCE_DIR}/../private")
        if (IS_DIRECTORY "${candidate}/sc-ai")
            get_filename_component(SC_PRIVATE_DIR "${candidate}" ABSOLUTE)
            break()
        endif ()
    endforeach ()
endif ()
if (SC_PRIVATE_DIR AND IS_DIRECTORY "${SC_PRIVATE_DIR}/sc-ai")
    message(STATUS "Private test documents: ${SC_PRIVATE_DIR}/sc-ai")
    add_custom_command(TARGET copy_sc_doc_classifier_resources POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SC_PRIVATE_DIR}/sc-ai" "${RESOURCE_DESTINATION_DIR}/private")
else ()
    message(STATUS "Private test documents: none (the tests that need them are skipped)")
endif ()

set(SC_TEST_LINK_LIBRARIES sc-doc-classifier)

add_sc_test(geometry LABELS unit)
add_sc_test(identity-fields TIMEOUT 120)
add_dependencies(test-identity-fields copy_sc_doc_classifier_resources)
add_sc_test(identity TIMEOUT 120)
add_dependencies(test-identity copy_sc_doc_classifier_resources)
