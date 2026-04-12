 ***,  ground state
 print,orbitals
  geometry={ang
 O
 H 1 b1
H 1 b1 2 a1
 }

    b1     =     0.9
a1 = 104.5

  set,charge=0

 basis=cc-pVTZ

  {hf;wf,,1,0;orbprint,6}
!{mrcc,METHOD=ccsd(t)}

