export PROJECT_SOURCE=`pwd`
export PYTHONPATH=`pwd`/src:`pwd`/modules/sinara-config-builder/src:`pwd`/modules/qiskit-terra:`pwd`/modules/qboard-sdk:`pwd`/modules/pylightnix/src:$PYTHONPATH
export MYPYPATH=`pwd`/src:`pwd`/modules/sinara-config-builder/src:`pwd`/modules/qboard-sdk:`pwd`/modules/pylightnix/src:`pwd`/modules/qiskit-terra:`pwd`/modules/qiskit-aer
export PATH=`pwd`/scripts:$PATH
alias ipython="sh `pwd`/ipython.sh"

