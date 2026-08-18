properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

// --no-deps because dependencies should already be installed in the image
def pythonEnv = {
  stage('python env') {
    sh 'pip install --no-deps $SRC/utils'
  }
}

timeout(time: 1, unit: 'HOURS') {
  parallel cpu: {
    buildPod(context: 'docker', dockerfile: 'Dockerfile_jenkins', tag: 'cpu',
             buildArgs: '--build-arg VARIANT=cpu') {
      withEnv([
        "SRC=$WORKSPACE",
        "BUILD=$WORKSPACE/build",
        "BUILD_CLANG=$WORKSPACE/build-clang"
      ]) {
        pythonEnv()
        parallel python: {
          stage('build docs') {
            sh '''
              cd $SRC/docs
              make html
            '''
          }
          if (env.BRANCH_NAME in ['main', 'develop', 'overhaul'] || env.TAG_NAME) {
            stage('deploy docs') {
              def scm = scmGit(branches: [[name: 'refs/heads/gh-pages']],
                userRemoteConfigs: [[credentialsId: 'github-jenkins', url: 'https://github.com/SFQMC/sfqmc.github.io.git']])
              dir(path: 'sfqmc.github.io') {
                checkout(changelog: false, poll: false, scm: scm)
                sh '''#!/bin/bash -e
                  rm -rf docs/$BRANCH_NAME
                  mkdir -p docs/$BRANCH_NAME
                  cp -a $SRC/docs/_build/html/. docs/$BRANCH_NAME
                  if [ -n "$TAG_NAME" ]; then
                    LATEST=$(git -C $SRC tag --sort=-v:refname | head -n1)
                    ln -sfn "$LATEST" docs/stable
                  fi
                  git add -A docs
                  if git diff --cached --quiet; then
                    echo "No documentation changes to deploy."
                  else
                    GIT_COMMITTER_EMAIL="jenkins@flatironinstitute.org" GIT_COMMITTER_NAME="SAFIRE CI" git commit --author='SAFIRE CI <jenkins@flatironinstitute.org>' --allow-empty -m "Update SAFIRE docs from build $BUILD_NUMBER"
                  fi
                '''
                gitPush(gitScm: scm, targetBranch: 'gh-pages', targetRepo: 'origin')
              }
            }
          }
        },
        cpp_release: {
          stage('release') {
            sh 'mkdir $BUILD'
            sh '''
              cd $BUILD && cmake $SRC \
                -GNinja \
                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DCMAKE_INSTALL_PREFIX="." \
                -DCOMPILE_NDA_TESTS=OFF \
                -DENABLE_CPPTRACE=OFF \
                -DENABLE_SPDLOG=ON
            '''
            sh 'ninja -C $BUILD -j $PARALLEL'
            warnError("Tests failed") {
              sh 'cd $BUILD && ctest --output-on-failure'
            }
            warnError("Snapshot tests failed") {
              sh 'cd $BUILD && AFQMC_EXEC=$(pwd)/bin/safire python3 ../tests/functional/run_functional.py all --snapshot'
            }
          }
        },
        cpp_clang: {
          stage('clang') {
            withEnv([
              "CC=clang-19",
              "CXX=clang++-19"
            ]) {
              sh 'mkdir $BUILD_CLANG'
              sh '''
                cd $BUILD_CLANG && cmake $SRC \
                  -GNinja \
                  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                  -DCMAKE_INSTALL_PREFIX="." \
                  -DCOMPILE_NDA_TESTS=OFF \
                  -DCMAKE_C_COMPILER=clang-19 \
                  -DCMAKE_CXX_COMPILER=clang++-19 \
                  -DENABLE_CPPTRACE=OFF \
                  -DENABLE_SPDLOG=ON
              '''
              sh 'ninja -C $BUILD_CLANG -j $PARALLEL'
              warnError("Clang tests failed") {
                sh 'cd $BUILD_CLANG && ctest --output-on-failure'
              }
            }
          }
        }
      }
    }
  },
  cuda: {
    buildPod(context: 'docker', dockerfile: 'Dockerfile_jenkins', tag: 'cuda', gpus: 1,
             buildArgs: '--build-arg VARIANT=cuda') {
      withEnv([
        "SRC=$WORKSPACE",
        "BUILD=$WORKSPACE/build"
      ]) {
        pythonEnv()
        stage('cuda') {
          sh 'mkdir $BUILD'
          sh '''
            cd $BUILD && cmake $SRC \
              -GNinja \
              -DCMAKE_BUILD_TYPE=RelWithDebInfo \
              -DCMAKE_INSTALL_PREFIX="." \
              -DCMAKE_C_COMPILER=gcc \
              -DCMAKE_CXX_COMPILER=g++ \
              -DCOMPILE_NDA_TESTS=OFF \
              -DENABLE_CPPTRACE=OFF \
              -DENABLE_SPDLOG=ON \
              -DENABLE_CUDA=ON \
              -DCUDA_ARCH=70 \
              -DDEVICE_RNG_FROM_HOST=ON
          '''
          sh 'ninja -C $BUILD -j $PARALLEL'
          warnError("Tests failed") {
            sh 'cd $BUILD && CUDA_LAUNCH_BLOCKING=1 ctest --output-on-failure'
          }
          warnError("Snapshot tests failed") {
            sh 'cd $BUILD && AFQMC_EXEC=$(pwd)/bin/safire python3 ../tests/functional/run_functional.py all --compute gpu --snapshot'
          }
        }
      }
    }
  }
}
