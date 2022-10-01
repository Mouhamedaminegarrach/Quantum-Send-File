# Quantum-Send-File
The project uses the BB84 protocol to generate a key with which you are going to send a multimedia file (image) or video and this can be transferred from the quantum teleportation and deciphered with the same key. send the QOSF logo from folder mentee to folder mentor.

![image](https://user-images.githubusercontent.com/78730355/193431624-101ae1e1-2c33-4d39-bd71-d944d5cdfb01.png)

0.  Convert the image to grayscale
1.	Encode the image with a generated key using BB84 protocol[1]. 
2.	Transform the encoded image into quantum circuit using NEQR model[2-3]. 
3.	Quantum teleportation, teleport the quantum data from the sender to the receiver into quantum channel[4].
4.	Convert the data to the encoded image.
5.	Decode the image using the same generated key.
# References
[1] [Qiskit Textbook](https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html#Novel-Enhanced-Quantum-Representation-(NEQR)-for-Digital-Images)\
[2] [Qiskit Textbook](https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html#Novel-Enhanced-Quantum-Representation-(NEQR)-for-Digital-Images)\
[3] [Anand, Alok, et al. "Quantum Image Processing." arXiv preprint arXiv:2203.01831 (2022).](https://arxiv.org/pdf/2203.01831.pdf)
[4] [Q-munity Textbook](https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html#Novel-Enhanced-Quantum-Representation-(NEQR)-for-Digital-Images)\
