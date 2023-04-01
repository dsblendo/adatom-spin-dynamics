#!/bin/bash
# Check for tab characters in files controlled by Git
# To exclude a file from this check, list its complete relative path in the .tabignore file

# Attributed to https://github.ecodesamsung.com/iot/snm-support

result=$(grep -l "$(printf '\t')" $(git ls-files | grep -v -x -F --file=.tabignore | grep -v images/| grep -v tests/files/))
if [ $? == 0 ]; then
   echo "*** Error: Some files contain tab characters:"
   echo "${result}"
   exit 1
fi
