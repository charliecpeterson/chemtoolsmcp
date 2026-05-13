<!-- Source: GRASP2018-manual.pdf, pages 229–242 -->
<!-- Part: III Sample runs -->

# Case study II: the Li iso-electronic sequence using scripts

## Chapter 11
## Case study II: the Li iso-electronic
## sequence using scripts
In this case study we use script les to perform systematic calculations for the 1s22s 2S1/2 ground state and the 1s22p 2P1/2,3/2 excited states in the Li iso-electronic sequence. Computing data for an iso-electronic sequence angular data can be reused and need not be recomputed for each member of the sequence. The script les can be found in grasptest/case2/script. We start with a single calculation of the three reference states. After that separate calculations are done for the two parities. Correlation is then included by allowing single-, double-, and triple (SDT) excitations from the reference to active sets up to n = 5 (complete active space calculations). Calculations including hyperne structures and transition rates are performed from Z = 6 to Z = 12. It is convenient to save the results for the dierent ions in directories Z6, Z7, Z8, ..., Z12.
### 11.1
### Running script les
The main script sh_case2 is shown below. This script controls the computational ow and calls several subscripts. #!/bin/sh set -x # Main script for iso-electronic sequence # 1. Generate directories Z6, Z7, .. for the elements # Define nuclear data for each element ./sh_nuc_seq # 2. Generate lists of CSFs in main directory ./sh_files_c # 3. Start by performing rmcdhf calculations for the 1s(2)2s, 1s(2)2p # reference states ./sh_DF 229

230CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS # 4. Perform rmcdhf calculations for all the even expansions # Angular data computed only once and then moved to different directories ./sh_even # 5. Perform rmcdhf calculations for all the odd expansions # Angular data computed only once and then moved to different directories ./sh_odd # 6. Perform rci calculations for the even5 and odd5 expansions # Perform rhfs and transition calculations. # Angular data computed only once and then moved to different directories ./sh_even_odd Each of the subscripts are given below together with some comments. If all script les are available with execute permission (use the command chmod +x) we start the computation by typing the name of the main script ./sh_case2 Please note that these calculations will take several hours!
1. Generate directories and dene nuclear data
The script sh_nuc_seq produces nuclear data for Z = 6, 7, . . . , 12 in the directories Z6, Z7 ,..., Z12. By modifying the script we can produce nuclear data for any sequence of charges. #!/bin/sh set -x # Full loop over all Z # for z in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \ # 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 \ # 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \ # 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 \ # 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \ # 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 \ # 126 127 128 129 130 131 132 133 134 135 136 137 138 # We select Z from 6 to 12 for z in 6 7 8 9 10 11 12 do # Data from Jefferson Lab (http://education.jlab.org/itselemental) case $z in
1) m=0; MM=1.0794;;
# Need to use point nucleus
2) m=4; MM=4.002602;;
3) m=7; MM=6.941;;
4) m=9; MM=9.012182;;
5) m=11; MM=10.811;;

11.1. RUNNING SCRIPT FILES 231
6) m=12; MM=12.0107;;
7) m=14; MM=14.0067;;
8) m=16; MM=15.9994;;
9) m=19; MM=18.9984032;;
10) m=20; MM=20.1797;;
11) m=23; MM=22.98976928;;
12) m=24; MM=24.3050;;
13) m=27; MM=26.9815386;;
14) m=28; MM=29.0855;;
15) m=31; MM=30.973762;;
16) m=32; MM=32.065;;
17) m=35; MM=35.453;;
18) m=40; MM=39.948;;
19) m=39; MM=39.0938;;
20) m=40; MM=40.078;;
21) m=45; MM=44.955912;;
22) m=48; MM=47.867;;
23) m=51; MM=50.9415;;
24) m=52; MM=51.9961;;
25) m=55; MM=54.938045;;
26) m=56; MM=55.845;;
27) m=59; MM=58.933195;;
28) m=59; MM=58.6934;;
29) m=64; MM=63.546;;
30) m=65; MM=65.409;;
31) m=70; MM=69.723;;
32) m=73; MM=72.64;;
33) m=75; MM=74.92160;;
34) m=79; MM=78.96;;
35) m=80; MM=79.904;;
36) m=84; MM=83.798;;
37) m=85; MM=85.4678;;
38) m=88; MM=87.62;;
39) m=89; MM=88.90585;;
40) m=91; MM=91.224;;
41) m=93; MM=92.90638;;
42) m=96; MM=95.94;;
43) m=98; MM=98;;
44) m=101; MM=10.07;;
45) m=103; MM=102.90550;;
46) m=106; MM=106.42;;
47) m=108; MM=107.8682;;
48) m=112; MM=112.411;;
49) m=115; MM=114.818;;
50) m=119; MM=118.710;;
51) m=122; MM=121.760;;
52) m=128; MM=127.60;;
53) m=127; MM=126.90447;;
54) m=131; MM=131.293;;
55) m=133; MM=132.9054519;;
56) m=137; MM=137.327;;
57) m=139; MM=138.90547;;
58) m=140; MM=140.116;;
59) m=141; MM=140.90765;;

232CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS
60) m=144; MM=144.242;;
61) m=145; MM=145;;
62) m=150; MM=150.36;;
63) m=152; MM=151.964;;
64) m=157; MM=157.25;;
65) m=159; MM=158.92535;;
66) m=163; MM=162.5;;
67) m=165; MM=164.93032;;
68) m=167; MM=167.259;;
69) m=169; MM=168.93421;;
70) m=173; MM=173.04;;
71) m=175; MM=174.967;;
72) m=178; MM=178.49;;
73) m=181; MM=180.94788;;
74) m=184; MM=183.84;;
75) m=186; MM=186.207;;
76) m=190; MM=190.23;;
77) m=192; MM=192.217;;
78) m=195; MM=195.084;;
79) m=197; MM=196.966569;;
80) m=201; MM=200.59;;
81) m=204; MM=204.3833;;
82) m=207; MM=207.2;;
83) m=209; MM=208.9804;;
84) m=209; MM=209;;
85) m=210; MM=210;;
86) m=222; MM=222;;
87) m=223; MM=223;;
88) m=226; MM=226;;
89) m=227; MM=227;;
90) m=232; MM=232.03806;;
91) m=231; MM=231.03588;;
92) m=238; MM=238.02891;;
93) m=237; MM=237;;
94) m=244; MM=244;;
95) m=243; MM=243;;
96) m=247; MM=247;;
97) m=247; MM=247;;
98) m=251; MM=251;;
99) m=252; MM=252;;
100) m=257; MM=257;;
101) m=258; MM=258;;
102) m=259; MM=259;;
103) m=262; MM=262;;
104) m=267; MM=267;;
105) m=268; MM=268;;
106) m=271; MM=271;;
107) m=272; MM=272;;
108) m=277; MM=277;;
109) m=276; MM=276;;
110) m=281; MM=281;;
111) m=280; MM=280;;
112) m=285; MM=285;;
113) m=284; MM=284;;

11.1. RUNNING SCRIPT FILES 233
114) m=289; MM=289;;
115) m=288; MM=288;;
116) m=291; MM=291;;
117) m=293; MM=293;;
#Estimated
118) m=294; MM=294;;
119) m=316; MM=316;;
120) m=318; MM=318;;
121) m=322; MM=322;;
122) m=324; MM=324;;
123) m=326; MM=326;;
124) m=330; MM=330;;
125) m=332; MM=332;;
126) m=334; MM=334;;
127) m=338; MM=338;;
128) m=340; MM=340;;
129) m=342; MM=342;;
130) m=346; MM=346;;
131) m=348; MM=348;;
132) m=350; MM=350;;
133) m=354; MM=354;;
134) m=356; MM=356;;
135) m=358; MM=358;;
136) m=362; MM=362;;
137) m=364; MM=364;;
138) m=366; MM=366;;
esac echo "Starting: Z::"${z}, "ZZ::"$ZZ, "mass::"${m}, "Weight::"${MM} rm -r Z$z mkdir Z$z cd Z$z rnucleus <<EOF $z $m n $MM 1 1 1 EOF cd .. done
2. Generate expansions
The expansions are generated by the script sh_files_c. rcsfgenerate << EOF * 0

234CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS 1s(2,i)2s(1,i) 2s,2p 1,1 0 y 1s(2,i)2p(1,i) 2s,2p 1,3 0 n EOF cp rcsf.out DF.c ####################################### rcsfgenerate << EOF * 0 1s(2,*)2s(1,*) 5s,5p,5d,5f,5g 1,1 3 n EOF cp rcsf.out even.c rcsfsplit << EOF even 3 3s,3p,3d 3 4s,4p,4d,4f 4 5s,5p,5d,5f,5g 5 EOF ################################ rcsfgenerate << EOF * 0 1s(2,*)2p(1,*) 5s,5p,5d,5f,5g 1,3 3 n EOF

11.1. RUNNING SCRIPT FILES 235 cp rcsf.out odd.c rcsfsplit << EOF odd 3 3s,3p,3d 3 4s,4p,4d,4f 4 5s,5p,5d,5f,5g 5 EOF
3. Ground and excited reference states
The script sh_DF performs angular integration, gets initial estimates and performs scf calculations for the 1s22s, 1s22p reference states. The angular integration is done only once and the mcp.30, mcp.31 ... les are moved between the directories. for z in 6 7 8 9 10 11 12 do (if test $z -lt 7 then cd Z${z} cp ../DF.c rcsf.inp # Get angular data rangular <<S4 y S4 #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > out_rmcdhf <<S6 y 1 1 1 5 * * 100 S6 rsave DF cp DF.w even2.w cp DF.w odd2.w

236CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS cd .. else cd Z${z} cp ../DF.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > out_rmcdhf <<S6 y 1 1 1 5 * * 100 S6 rsave DF cp DF.w even2.w cp DF.w odd2.w cd .. fi echo) done
4. Perform calculations for the even states
The script sh_even performs angular integration, gets initial estimates and performs rmcdhf calculations for the even states. The script loops over both the active set and the atomic number Z. Angular les are reused and moved between the directories. for n in 3 4 5 do ( for z in 6 7 8 9 10 11 12 do (if test $z -lt 7 then cd Z${z} cp ../even${n}.c rcsf.inp

11.1. RUNNING SCRIPT FILES 237 # Get angular data rangular <<S4 y S4 k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 even${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf_${n} <<S6 y 1 ${n}* 100 S6 rsave even${n} cd .. else cd Z${z} cp ../even${n}.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 even${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf_${n} <<S6 y 1 ${n}* 100 S6

238CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS rsave even${n} cd .. fi echo) done ) done
5. Perform calculations for the odd states
The script sh_odd performs angular integration, gets initial estimates and performs rmcdhf cal- culations for the odd states. The script loops over both the active set and the atomic number Z. Angular les are reused and moved between the directories. for n in 3 4 5 do ( for z in 6 7 8 9 10 11 12 do (if test $z -lt 7 then cd Z${z} cp ../odd${n}.c rcsf.inp # Get angular data rangular <<S4 y S4 k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 odd${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outodd_rmcdhf_${n} <<S6 y 1 1 5 ${n}* 100 S6 rsave odd${n}

11.1. RUNNING SCRIPT FILES 239 cd .. else cd Z${z} cp ../odd${n}.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 odd${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > out_rmcdhf_${n} <<S6 y 1 1 5 ${n}* 100 S6 rsave odd${n} cd .. fi echo) done ) done
6. Conguration interaction and transition calculations
The script sh_even_odd performs conguration interaction and transition calculations for even5 and odd5. Angular les are reused and moved between the directories. for z in 6 7 8 9 10 11 12 do (cd Z${z} # RCI calculations for even5 rci > outeven_rci <<S6 y even5 y

240CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS y 1.d-6 y n n y 3 1 S6 # RCI calculations for odd5 rci > outodd_rci <<S6 y odd5 y y 1.d-6 y n n y 3 1 1 S6 if test $z -lt 7 then # Run rbiotransform ans save angular data rbiotransform <<S4 y y even5 odd5 y S4 # Run rtransition save angular data rtransition <<S4 y y even5 odd5 E1 S4 else #Move angular files from previous directory m=`expr $z - 1` mv ../Z${m}/even5.TB . mv ../Z${m}/odd5.TB . mv ../Z${m}/even5.odd5.-1T .

11.2. COMPARISON WITH EXPERIMENT 241 # Run rbiotransform using available angular data rbiotransform <<S4 y y even5 odd5 y S4 # Run rtransition using available angular data rtransition <<S4 y y even5 odd5 E1 S4 fi cd .. echo) done
### 11.2
### Comparison with experiment
To display the computed energies for Z = 6 we enter the Z6 directory and we give the command rlevels even5.cm odd5.cm The computer returns the energies together with labels in LSJ-coupling for all the states. Energy levels for ... Rydberg constant is 109737.31569 No - Serial number of the state; Pos - Position of the state within the J/P block; Splitting is the energy difference with the lower neighbor ------------------------------------------------------------------------- No Pos J Parity Energy Total Levels Splitting (a.u.) (cm^-1) (cm^-1) ------------------------------------------------------------------------- 1 1 1/2 + -34.7859395 2 1 1/2 - -34.4919396 64525.53 64525.53 3 1 3/2 - -34.4914500 64632.98 107.45 These energies should be compared with NIST that give 64484.0 cm−1, 64591.7 cm−1. Increasing the active set further will improve the agreement with experiment. The transition parameters are given in even5.odd5.ct. There is a good agreement between length (B) and velocity (C) forms of the parameters. The gf values in the length form are in good agreement with the values 0.1895 and 0.3789 from large scale MCHF calculations [36]. Again, an increased active set will improve the agreement. Transition between files: f1 = even5 f2 = odd5

242CHAPTER 11. CASE STUDY II: THE LI ISO-ELECTRONIC SEQUENCE USING SCRIPTS Electric 2**( 1)-pole transitions ================================= Upper Lower File Lev J P File Lev J P E (Kays) A (s-1) gf S f2 1 1/2 - f1 1 1/2 + 64525.53 C 2.68596D+08 1.93430D-01 9.86890D-01 B 2.64473D+08 1.90461D-01 9.71741D-01 f2 1 3/2 - f1 1 1/2 + 64632.98 C 2.69981D+08 3.87564D-01 1.97408D+00 B 2.65907D+08 3.81715D-01 1.94429D+00
### 11.3
### Scripts for MPI codes
The scripts above can, with very small modications, also be used for performing the calculations using the MPI codes. The most important change is that the user needs to prepare the les disks6, disks7, ..., disks12 with paths to the working directory and to the directory containing temporary data. The disks les are copied to the Z6, Z7, .., Z12 directories by sh_nuc_seq. In the dierent scripts the calls to the MPI programs amount to changes of the type rangular --> mpirun -np 8 rangular_mpi rmcdhf --> mpirun -np 8 rmcdhf_mpi rci --> mpirun -np 8 rci_mpi etc. For the MPI runs the saved angular les reside in the tmp_mpi directory and thus they need not be copied from Z6 to Z7 etc. Scripts for the MPI cases are included under case2_mpi in the test data set (see section 2.11). Consult the README le in the working directory for more details on setting up the le disks.