#!/bin/bash
# Shortcuts for common developer tasks

# Setup the virtual environment via Poetry and install pre-commit hooks
run_install() {
   echo "Installing poetry..."

   # Create and update the virtualenv
   poetry install -v
   if [ $? != 0 ]; then
      exit 1
   fi

   # Upgrade embedded packages within the virtualenv
   # This command sometimes returns $?=1 on Windows, even though it succeeds <sigh>
   poetry run pip install --quiet --upgrade pip wheel setuptools 2>/dev/null

   echo "Installing pre-commit hooks"

   # Install the pre-commit hooks
   poetry run pre-commit install
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Activate the current Poetry virtual environment
run_activate() {
   echo "source "$(dirname $(poetry run which python) 2>/dev/null)/activate""
}

# Run autoflake code checker
run_autoflake() {
   echo "Running autoflake checks..."

   poetry run which autoflake > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   poetry run autoflake $* --remove-unused-variables --recursive ./src ./tests
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run tab check
run_tabcheck() {
   echo "Running tab check..."

   bash utils/check-tabs.sh
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run the black code formatter
run_black() {
   echo "Running black formatter..."

   poetry run which black > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   poetry run black $* .
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run the isort import formatter
run_isort() {
   echo "Running isort formatter..."

   poetry run which isort > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   poetry run isort $* .
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run the MyPy code checker
run_mypy() {
   echo "Running mypy checks..."

   poetry run which mypy > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   poetry run mypy src tests
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run the Pylint code checker
run_pylint() {
   echo "Running pylint checks..."

   poetry run which pylint > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   poetry run pylint src tests
   if [ $? != 0 ]; then
      exit 1
   fi

   echo "done"
}

# Run the unit tests, optionally with coverage
run_pytest() {
   coverage="no"
   html="no"

   while getopts ":ch" option; do
     case $option in
       c)
         coverage="yes"
         ;;
       h)
         html="yes"
         ;;
       ?)
         echo "invalid option -$OPTARG"
         exit 1
         ;;
     esac
   done

   poetry run which pytest > /dev/null
   if [ $? != 0 ]; then
      run_install
   fi

   if [ $coverage == "yes" ]; then
      poetry run coverage run -m pytest --testdox tests
      if [ $? != 0 ]; then
         exit 1
      fi

      poetry run coverage report --fail-under=80 --skip-empty --skip-covered
      if [ $? != 0 ]; then
         exit 1
      fi

      if [ $html == "yes" ]; then
         poetry run coverage html -d .htmlcov
         $(which start || which open) .htmlcov/index.html 2>/dev/null  # start on Windows, open on MacOS and Debian (post-bullseye)
      fi
   else
      poetry run pytest --testdox tests
      if [ $? != 0 ]; then
         exit 1
      fi
   fi
}

# Build release artifacts
run_build() {
   echo "Building release artifacts..."

   rm -f dist/*
   poetry version
   poetry build

   echo "done"
}

# Execute one of the developer tasks
case $1 in
   install|setup)
      echo "Executing install"
      run_install
      ;;
   activate)
      echo "Executing activate"
      run_activate
      ;;
   autoflake)
      echo "Executing autoflake"
      run_autoflake --in-place
      ;;
   black)
      echo "Executing black"
      run_black
      ;;
   isort)
      echo "Executing isort"
      run_isort
      ;;
   *lint)
      run_pylint
      ;;
   mypy)
      echo "Executing mypy"
      run_mypy
      ;;
   tabcheck)
      echo "Executing tabcheck"
      run_tabcheck
      ;;
   format)
      echo "Executing format"
      run_autoflake --in-place
      echo ""
      run_tabcheck
      echo ""
      run_black
      echo ""
      run_isort
      ;;
   check*)
      echo "Executing check"
      run_autoflake --check
      echo ""
      run_tabcheck
      echo ""
      run_black --check
      echo ""
      run_isort --check-only
      echo ""
      run_mypy
      echo ""
      run_pylint
      ;;
   pytest|test*)
      shift 1
      run_pytest $*
      ;;
   build)
      echo "Executing build"
      run_build
      ;;
   *)
      echo ""
      echo "------------------------------------"
      echo "Shortcuts for common developer tasks"
      echo "------------------------------------"
      echo ""
      echo "Usage: premerge_script.sh <command>"
      echo ""
      echo "- install: Setup the virtualenv via Poetry and install pre-commit hooks"
      echo "- activate: Print command needed to activate the Poetry virtualenv"
      echo "- format: Run the code formatters"
      echo "- checks: Run the code checkers"
      echo "- build: Build release artifacts into the dist/ directory"
      echo ""
      exit 1
esac
