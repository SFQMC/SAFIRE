#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

########
# Git setup
########

SET(GITREV_BARE_FILE git-rev.h)
SET(GITREV_FILE ${CMAKE_BINARY_DIR}/src/${GITREV_BARE_FILE})

IF (IS_GIT_PROJECT)
  # Resolve the real HEAD and index paths. In a git worktree, ${CMAKE_SOURCE_DIR}/.git
  # is a file (a gitdir pointer), not a directory, so .git/HEAD and .git/index do not
  # exist at those paths; use "git rev-parse --git-path" to find the real locations.
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --git-path HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HEAD_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --git-path index
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_INDEX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # git rev-parse may return paths relative to CMAKE_SOURCE_DIR; make them absolute.
  get_filename_component(GIT_HEAD_PATH "${GIT_HEAD_PATH}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
  get_filename_component(GIT_INDEX_PATH "${GIT_INDEX_PATH}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
  set(GITREV_DEPENDS "")
  if(EXISTS "${GIT_HEAD_PATH}")
    list(APPEND GITREV_DEPENDS "${GIT_HEAD_PATH}")
  endif()
  if(EXISTS "${GIT_INDEX_PATH}")
    list(APPEND GITREV_DEPENDS "${GIT_INDEX_PATH}")
  endif()
  # Pull the git revision information.
  add_custom_command(
    OUTPUT ${GITREV_FILE}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/gitversion.sh
          --git-executable ${GIT_EXECUTABLE}
          --output ${GITREV_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${GITREV_DEPENDS}
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
    Random.cpp
)
