# Contributing Guide

We are excited that you are interested in contributing.
The goals of this guide are to:

1. help you get started with making contributions
2. help you avoid duplicating ongoing efforts
3. describe expectations for contributions to the project

## How to Contribute

Unless otherwise stated, contributions should be made via a pull request (PR) to
the `develop` branch on GitHub.
We encourage opening a PR early in the contribution process so that we can provide guidance.

### Some Preliminaries

We have the following preliminary expectations from contributors.

1. We will not accept PRs from unknown contributors. It must be clear to us who you are before we will engage in the review process. We strongly prefer that your GitHub profile have your name, institutional affiliation, and an accurate image of yourself.

### Review Process

1. **Proposal Review**: Before starting significant work on a contribution, potential contributors should open a draft pull request (PR) to outline what they intend to do. If the contribution is in response to a specific GitHub issue, it should be referenced in the draft PR AND a comment should be left in the issue stating your intent to work on it. This allows core project developers to provide initial feedback and guidance, ensuring that the proposed changes align with the project's goals and standards and that there are no ongoing efforts to make similar contributions.

2. **Initial Review**: Once a pull request (PR) is opened or marked "ready for review", it will be reviewed by one or more core project developers. They will check for:
   - Code quality and adherence to the project's coding standards.
   - Completeness and correctness of the implementation.
   - Adequate test coverage for new or modified code.
   - Proper documentation updates, if applicable.

3. **Feedback and Revisions**: The reviewer(s) may request changes or provide feedback. The contributor should address these comments and push the changes to the same PR. This cycle may repeat until the reviewer(s) are satisfied with the updates.

4. **Approval**: Once the reviewer(s) approve the PR, it will be labeled as `approved`. At this stage, no further changes should be made unless requested by the reviewer(s).

5. **Final Review and Merge**: A core maintainer will perform a final review to ensure everything is in order. If the PR passes this review, it will be merged into the `unstable` branch. The contributor will be notified of the merge.

6. **Post-Merge**: After merging, the changes will be tested in the `unstable` branch. If any issues arise, they will be addressed promptly. Once the changes are deemed stable, they will be merged into the `main` branch during the next release cycle.

## Ensuring CI will pass

Before a PR is merged, it has to pass our CI pipeline. The CI will run the unit tests of the C++ and Python parts of the project. You can run those tests locally as follows.

### Running C++ tests

```sh
make test
```

To run the test suite in parallel, use
```sh
CTEST_PARALLEL_LEVEL=0 make test
```

### Running Python unit tests

Pytest is already configured at the project level.
If `safiretools` was installed via pip according, then pytest should already be installed.
We recommend also installing the optional "testing" dependencies by running the following
in the `python/` directory.

```sh
$ pip install .[testing]
```

Pytest can be run using

```sh
$ pytest
```

from within `python/`.

To target specific categories of test, specify the corresponding pytest for "marks" using.

```sh
$ pytest -m pyscf
```

or

```sh
$ pytest -m "not slow"
```

or 

```sh
$ pytest -m "debug and slow"
```

for example.

If the "TESTING" optional dependencies are installed, then an html report of the test results will be generated
in `.htmlpytest`.
The html report can be viewed in using a web browser by navigating to `file:///path/to/SAFIRE/python/.htmlpytest/pytest.html`.


### Additional Notes

- **Automated Checks**: All PRs must pass automated checks (e.g., CI/CD pipelines) before they can be merged.
- **Communication**: Contributors are encouraged to communicate with the reviewers and maintainers through the PR comments for any clarifications or discussions.
- **Documentation**: Ensure that any new features or significant changes are well-documented in the project's documentation.

By following this review process, we aim to maintain high code quality and ensure that all contributions are thoroughly vetted before being integrated into the project.

## Ways to Contribute

There are several ways to make contributions.
We have provided guides on contributing in several different ways below.
Each guide is listed here with a link to the relevant section.

- [Bug Reports](#bug-reports)
- [Bug and Issue Fixes](#bug-and-issue-fixes)
- [New Features](#new-features)
- [External Code Interfaces](#external-code-interfaces)
- [Documentation](#documentation)
- [Tutorials and Examples](#tutorials-and-examples)
- [Benchmarking](#benchmarking)

### Bug Reports

We are very interested in knowing about any bugs that you may have experienced.
Please report bugs by opening an issue on GitHub.
At a minimum, a bug report issue should include:
1. What you did.
2. What you expected to happen.
3. What actually happened.

To help with identifying the cause of the bug, please include as much of the following information as possible:

- git branch and commit hash. Please ensure that you are running without local changes using `$ git status` before openning an issue
- versions of relevant dependencies
   - for Python: version of Python, version of python packages.
   - for C++: which compiler was used, and at what version
- information about the environment and system setup
- any additional information 

Users who provide a bug report which lead to an identification of an actual bug which is reproducible, as determined by reviewers, will be added
to the "Bug Reporters" section of the CONTRIBUTORS.md file.
See [standards for bugs](#standards-for-bugs)

#### Standards for Bugs

⚠️ **Only bugs that we can reproduce will accepted** Please provide sufficient information to help us reproduce bugs.

⚠️ **Python bugs must occur on builds which are installed via pip**. We do not support manual installations via editing PYTHONPATH, etc.

### Bug and Issue Fixes


We have enabled "GitHub Issues" for this project where we track Issues that the community has with the code base.
For new contributors, some issues are labelled as "good first issue".
These issues have been identified by members of the community as being relatively approachable by new contributors.

When working on an issue, please leave a comment on the issue to let the community know that you are working on it.
This allows us to avoid duplicated efforts.

### New Features

Before adding major new features, please make your intention known to the core developers by opening an issue tagged as a "feature_request" and opening a draft PR which references the feature request issue.
This allows the core development team to provide guidance on successfully implementing the new feature in 
a way which is most consistent with the code base and the design of the code.

### External Code Interfaces

External code interfaces should be written in Python.
Contributing an interface to external quantum chemistry or solid-state physics codes involves several steps to ensure compatibility and maintainability. Follow these guidelines to contribute effectively:

1. **Proposal and Discussion**:
   - Open an issue tagged as `external_interface` to propose the new interface.
   - Provide a detailed description of the external code, its purpose, and how it integrates with the existing project.
   - Discuss the proposal with the core developers and the community to gather feedback and suggestions.

2. **Design and Implementation**:
   - Open a draft pull request (PR) referencing the `external_interface` issue.
   - Design the interface to be modular and maintainable. Ensure it adheres to the project's coding standards and architecture.
   - Implement the interface in a separate module or package to keep the codebase organized.
   - Write comprehensive unit tests to cover the functionality of the new interface.

3. **Documentation**:
   - Update the user documentation to include instructions on how to use the new interface.
   - Provide examples and use cases to help users understand the integration process.
   - Update the developer documentation to explain the design and implementation details of the interface.

4. **Code Review and Testing**:
   - Wark your draft pull request (PR) as "ready for review"
   - Request reviews from core developers and address any feedback or requested changes.
   - Please add tests that demonstrate the the interface is working.
   - Ensure that the new interface passes all automated checks and tests.

5. **Maintenance**:
   - Be prepared to maintain the interface by addressing any bugs or issues that arise.
   - Keep the interface up-to-date with any changes in the external code or the project's codebase.


### Documentation

Writing and maintaining both user documentation and developer documentation are
important ways of contributing to the project.

#### Documenting C++ Code

Doxygen is used to discover and compile the C++ API Documentation for developers.
Documentation should be written in comments and should include at least:
- a brief description of what the class, method, or function does (using the `@brief` tag)
- a list of the parameters and explanation of what the are (using the `@param` tag)
- (highly preferred) a detailed explanation of how the class, method, or function works in context

```c++
namespace sfqmc
{
namespace afqmc
{
/**
 * @brief Factory class for AFQMC. Parses input, performs setup of classes, and executes the driver.
 *
 * @details The AFQMCFactory class is the top-level class for AFQMC. It parses the input file, performs the setup of classes, and executes the driver. 
 It contains instances of the following factories which are used to construct the objects used during AFQMC calculations:
  * - HamiltonianFactory HamFac
  * - WalkerSetFactory WSetFac
  * - WavefunctionFactory WfnFac
  * - PropagatorFactory PropFac
  * - DriverFactory DriverFac

  * It also instances of the following classes which handle MPI communication and task group management:
  * - GlobalTaskGroup gTG
  * - TaskGroupHandler TGHandler
 *
 * @param type std::string describing the type of Driver to be used. Valid choices are "afqmc", "legacy_afqmc", and "csafqmc".
  * @param comm_ boost::mpi3::communicator The MPI communicator.
  * @param pt boost::property_tree::ptree The property tree containing input file parameters
  * @param n_groups int The number of groups to be used in the task group.
 */
class AFQMCFactory
{
   ...

   private:
      /**
       * @brief parses the contents of the boost::propert_tree::ptree instance
       *
       * @param pt boost::propert_tree::ptree is the property tree to be parsed
       */
      bool parse(const ptree pt);

}

...

} //namespace afqmc
} //namespace sfqmc
```

See the official Doxygen ["documenting the code"](https://www.doxygen.nl/manual/docblocks.html) guide
for details on how to write documentation.

#### Documenting Python Code

Python code should be documented by writing a docstring.
We use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style for documenting Python code.
This style is widely used in the scientific Python community and ensures that our documentation is clear and consistent.
At a minimum, docstrings should include:

1. A brief description of what the function does
2. a "Parameters" section which explains each input parameter
3. (if applicable) a "Returns" section

Of course, docstrings should contain any other sections that seem important.

### Tutorials and Examples

#### Contributing Tutorials

Tutorials are detailed, pedagogical explanations designed to help users understand how to use the code and the AFQMC methods. 
They should be comprehensive and cover the topic in depth, providing context, explanations, and examples.

**Guidelines for Contributing Tutorials**

1. **Identify the Topic:** Choose a topic that is relevant and useful for users of the project. Ensure that the topic is not already covered by existing tutorials.

2. **Structure:** Organize the tutorial into clear sections, including:

- **Introduction:** Briefly introduce the topic and explain its importance.
- **Prerequisites:** List any prerequisites, such as required knowledge or software installations.
- **Step-by-Step Instructions:** Provide detailed, step-by-step instructions on how to achieve the tutorial's goal.
- **Examples:** Include code examples and explanations to illustrate key points.
- **Conclusion:** Summarize the key takeaways and provide any additional resources or references.

3. **Code and Output:** Ensure that all code snippets are well-documented and include expected output where applicable. Use comments to explain the purpose of each code block.

4. **Clarity and Readability:** Write in clear, concise language. Explain technical terms when they are first introduced.

5. **Review and Test:** Test the tutorial thoroughly to ensure that all steps work as expected. Ensure that results are reproducible by setting random seeds as needed.

6. **Submission:** Submit the tutorial as a Markdown file in the docs/tutorials directory. Ensure that the file name is descriptive of the tutorial's content.

#### Contributing Examples

Examples are minimal, focused snippets of code that demonstrate how to perform specific tasks with the code.
They should be concise and to the point, providing just enough context to understand the task.

**Guidelines for Contributing Examples**

1. **Identify the Task:** Choose a specific task or feature that you want to demonstrate. Ensure that the task is not already covered by existing examples. Tasks should belong to one of three catagories: setting up an AFQMC calculation, running an AFQMC calculation, or analyzing an AFQMC calculation.

2. **Simplicity:** Keep the example as simple as possible. Focus on demonstrating the task without unnecessary complexity.

3. **Code and Comments:** Include the complete code for the example, along with comments explaining each step. Ensure that the code is well-documented and easy to understand.

4. **Expected Output:** If applicable, include the expected output of the example.

5. **Review and Test:** Test the example to ensure that it works as expected. Have someone else review the example for clarity and completeness.

6. **Submission:** Submit the example as a Python script in the docs/examples directory to the relevant subdirectory. Ensure that the file name is descriptive of the example's content.

