 ***,  ground state
 print,orbitals
symmetry,nosym
  geometry={ang
 O
 H 1 b1
H 1 b1 2 a1
 }

    b1     =     0.9
a1 = 104.5

  set,charge=0
 basis=cc-pVDZ
  {hf;wf,,1,0;orbprint,6;}

{nbo6;save,nbo,2150.2;}  

{multi;start,2150.2}

