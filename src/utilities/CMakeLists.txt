#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

########
# Git setup
########

SET(GITREV_BARE_FILE git-rev.h)
SET(GITREV_FILE ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_FILE})

IF (IS_GIT_PROJECT)
  # Resolve the real git directory (supports linked worktrees where .git is a file).
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --git-dir
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SAFIRE_GIT_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT IS_ABSOLUTE "${SAFIRE_GIT_DIR}")
    set(SAFIRE_GIT_DIR "${CMAKE_SOURCE_DIR}/${SAFIRE_GIT_DIR}")
  endif()

  # Pull the git revision information.
  add_custom_command(
    OUTPUT ${GITREV_FILE}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/gitversion.sh
          --git-executable ${GIT_EXECUTABLE}
          --output ${GITREV_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${SAFIRE_GIT_DIR}/HEAD
            ${SAFIRE_GIT_DIR}/index
    VERBATIM
  )
  add_custom_target(gitversion DEPENDS ${GITREV_FILE})

  add_dependencies(safire_lib gitversion)
ELSE()
  file(WRITE ${GITREV_FILE})
ENDIF()

target_sources(safire_lib
  PRIVATE
    app_version.cpp
    mpi_context.cpp
    mpi_shared_window.cpp
    parser.cpp
    freemem.cpp
    threading.cpp
)

