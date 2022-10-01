from qiskit import*
from qiskit.extensions import Initialize
import imageio as iio
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from numpy.random import randint
import numpy as np
from qiskit.providers.aer import QasmSimulator
from neqr import NEQR
from qiskit.providers.aer.backends import AerSimulator
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import pyplot as plt
def send_file(path, name,path_destionation):
    
    #Convert the image to grayscale
    qosf = iio.imread('C:/Users/Med Amine Garrach/Desktop/QOSF project/sending files/Mentee file/qosf.jpg')
    gray_qosf = rgb2gray(qosf)
    #Encode the image with a generated key using BB84 protocol
    key = BB84key()
    print("The Generated key is: ",key)
    key=int(key,2)
    ecrypt(path+"/"+name, key)
    
    #Transform the encoded image into quantum circuit using NEQR model.
    image_neqr = NEQR()
    shots = 8192
    backend = AerSimulator()
    
    qubits_idx = QuantumRegister(size=2, name="qubits_idx")
    intensity = QuantumRegister(size=8, name="intensity")
    bits_idx = ClassicalRegister(size=2, name="bits_idx")
    bits_intensity = ClassicalRegister(size=8, name="bits_intensity")
    
    qc_gray = QuantumCircuit(intensity, qubits_idx, bits_intensity, bits_idx)
    qc_gray= image_neqr.image_quantum_circuit(image=gray_qosf, measurements=True)
    
    #Quantum teleportation, teleport the quantum data from the sender to the receiver into quantum channel.
    
    "qc_gray=QTeleport(qc_gray)" 
    
    #Convert the data to the encoded image.
    counts_total = execute(experiments=qc_gray, backend=backend, shots=shots).result().get_counts()

    print("NEQR inverse: from the Qcricuit to image...")
    im=image_neqr.reconstruct_image_from_neqr_result(counts_total, (32,32))
    plt.gray()
    
    #Convert the data to the encoded image.
    decry(path+"/" +name, key)
    
    plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(path_destionation))
    plt.imshow(im)
    plt.savefig("img.png")
    print("The image sent to Mentor file successfully!!")
    
    

def bb84_circuit(state, basis, measurement_basis):
   
    #state: array of 0s and 1s denoting the state to be encoded
    #basis: array of 0s and 1s denoting the basis to be used for encoding
                #0 -> Computational Basis
                #1 -> Hadamard Basis
    #meas_basis: array of 0s and 1s denoting the basis to be used for measurement
                #0 -> Computational Basis
                #1 -> Hadamard Basis
    
    num_qubits = len(state)
    
    bb84_circuit = QuantumCircuit(num_qubits)

    # Sender prepares qubits
    for i in range(len(basis)):
        if state[i] == 1:
            bb84_circuit.x(i)
        if basis[i] == 1:
            bb84_circuit.h(i)
   

    # Measuring action performed by Bob
    for i in range(len(measurement_basis)):
        if measurement_basis[i] == 1:
            bb84_circuit.h(i)

       
    bb84_circuit.measure_all()
    
    return bb84_circuit

def BB84key():
    num_qubits = 14

    alice_basis = np.random.randint(2, size=num_qubits)
    alice_state = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.randint(2, size=num_qubits)


    print(f"Mentee's State:\t {np.array2string(alice_state)}")
    print(f"Mentee's Bases:\t {np.array2string(alice_basis)}")
    print(f"Mentor's Bases:\t {np.array2string(bob_basis)}")
    
    circuit = bb84_circuit(alice_state, alice_basis, bob_basis)
    key = execute(circuit.reverse_bits(),backend=QasmSimulator(),shots=1).result().get_counts().most_frequent()
    encryption_key = ''
    for i in range(num_qubits):
        if alice_basis[i] == bob_basis[i]:
             encryption_key += str(key[i])
    return encryption_key


def decry(path,key):
    try:
        
        # print path of image file and decryption key that we are using
        print('The path of file : ', path)
        print('Note : Encryption key and Decryption key must be same.')
        print('Key for Decryption : ', key)

        # open file for reading purpose
        fin = open(path, 'rb')

        # storing image data in variable "image"
        image = fin.read()
        fin.close()

        # converting image into byte array to perform decryption easily on numeric data
        image = bytearray(image)

        # performing XOR operation on each value of bytearray
        for index, values in enumerate(image):
            image[index] = values ^ key

        # opening file for writting purpose
        fin = open(path, 'wb')

        # writing decryption data in image
        fin.write(image)
        fin.close()
        print('Decryption Done...')


    except Exception:
        print('Error caught : ', Exception.__name__)
        
        
def ecrypt(path,key):
    # try block to handle exception
    try:
        
        # print path of image file and encryption key that
        # we are using
        print('The path of file : ', path)
        print('Key for encryption : ', key)

        # open file for reading purpose
        fin = open(path, 'rb')

        # storing image data in variable "image"
        image = fin.read()
        fin.close()

        # converting image into byte array to
        # perform encryption easily on numeric data
        image = bytearray(image)

        # performing XOR operation on each value of bytearray
        for index, values in enumerate(image):
            image[index] = values ^ key

        # opening file for writing purpose
        fin = open(path, 'wb')

        # writing encrypted data in image
        fin.write(image)
        fin.close()
        print('Encryption Done...')


    except Exception:
        print('Error caught : ', Exception.__name__)
        

        
def entanglement_bell_pair(qc, a, b):
    
    qc.h(a) # Put qubit a into state |+> or |-> using hadamard gate
    qc.cx(a,b) # CNOT with a as control and b as target
def alice_state_qubits(qc, psi, a):
    qc.cx(psi, a) #psi is the state of q0
    qc.h(psi)
def measure_classical_send(qc, a, b):
    
    qc.barrier()
    qc.measure(a,[0,1,2,3,4,5,6,7,8,9])
    qc.measure(b,[10,11,12,13,14,15,16, 17,18,19])
def bob_apply_gates(qc, qubit, cr1, cr2):

    qc.z(qubit).c_if(cr1, 1)  #if cr1 is 1 apply Z gate
    qc.x(qubit).c_if(cr2, 1) #if cr2 is 1 apply x gate, look at table above
    

def QTeleport(qc):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    #print(f"print(job):{job}\n\n")
    #print(job.status())
    result = job.result()
    #print(f"print(result):{result}\n\n")
    output = result.get_statevector(qc)

    init_gate = Initialize(output)

    qr = QuantumRegister(30)   
    cr1 = ClassicalRegister(10) 
    cr2 = ClassicalRegister(10)
    qc = QuantumCircuit(qr, cr1, cr2)

    #let's initialise Alice's q0
    qc.append(init_gate, [0,1,2,3,4,5,6,7,8,9])
    qc.barrier()

    # teleportation protocol
    entanglement_bell_pair(qc, [10,11,12,13,14,15,16, 17,18,19], [20,21,22,23,24,25,26,27,28,29])
    qc.barrier()
    # Send q1 to Alice and q2 to Bob
    alice_state_qubits(qc, [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16, 17,18,19])

    # alice sends to Bob
    measure_classical_send(qc, [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16, 17,18,19])

    # Bob decodes qubits
    bob_apply_gates(qc, [20,21,22,23,24,25,26,27,28,29], cr1, cr2)

    inverse_init_gate = init_gate.gates_to_uncompute()
    qc.append(inverse_init_gate, [20,21,22,23,24,25,26,27,28,29])

    cr_result = ClassicalRegister(10)
    qc.add_register(cr_result)
    qc.measure([20,21,22,23,24,25,26,27,28,29],[20,21,22,23,24,25,26,27,28,29])
    return(qc)