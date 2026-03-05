#!/bin/bash

DEBUG=false

VENV_PATH="/opt/venvs/mmdet"

MMDET_ROOT="/workspace/mmdetection3d"

APPCONTAINER="$HOME/cuda121-uv.sif"

BIND_PATHS="$HOME/curriculum:/workspace,$HOME/uhome:/opt,/projects:/projects"


for arg in "$@"; do

    [[ "$arg" == "debug=true" ]] && DEBUG=true && shift

done



# --- EXECUTION ---

apptainer exec --nv --bind "$BIND_PATHS" "$APPCONTAINER" bash -c "

    [ -f \"$VENV_PATH/bin/activate\" ] && source \"$VENV_PATH/bin/activate\"

    # 2. Set Python Paths

    export PYTHONPATH=\"$MMDET_ROOT:\$PYTHONPATH\"

    cd $MMDET_ROOT

    if [ \"$DEBUG\" = \"true\" ]; then

        echo \"--- ENV CHECK --- \"

        echo \"Ver:    \$(python --version)\"

        echo \"PPATH:  \$PYTHONPATH\"

        echo \"------------------\"

        [ \$# -eq 0 ] && exit 0

    fi

    if [ \$# -eq 0 ]; then

        python

    else

        python \"\$@\"

    fi

" -- "$@"
