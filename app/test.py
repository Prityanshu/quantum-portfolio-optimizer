import sys
import traceback

try:
    from qiskit_ibm_runtime import Sampler

except Exception as e:
    traceback.print_exc()
    print("ERROR IS COMING FROM:", sys.modules.get('qiskit.primitives'))
