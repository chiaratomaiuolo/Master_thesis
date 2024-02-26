#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export LABVIEWDATAPROCESSINGROOT=$SETUP_DIR
echo "LABVIEWDATAPROCESSINGROOT set to " $LABVIEWDATAPROCESSINGROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant (both SWIG and pure-Python) modules.
#
export PYTHONPATH=$LABVIEWDATAPROCESSINGROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH