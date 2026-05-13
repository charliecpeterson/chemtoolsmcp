<!-- Source: GRASP2018-manual.pdf, pages 243–270 -->
<!-- Part: III Sample runs -->

# Case study III: graphical analysis of the Mg iso-electronic sequence

## Chapter 12
## Case study III: graphical analysis of
## the Mg iso-electronic sequence
In this case study we use script les to perform systematic calculations for states belonging to the 3s2, 3p2, 3d2, 3s3d even congurations and to the 3s3p, 3p3d odd congurations in the Mg iso- electronic sequence. Angular data are reused from one ion to another. The script les can be found in grasptest/case3/script. Calculations are done by parity and valence-valence correlation is accounted for by allowing for SD excitations from the valence orbitals to active sets up to n = 5. Calculations for transition rates are performed for Z = 26, 27, . . . , 60. It is convenient to save the results for the dierent ions in directories named Z26, Z27, ..., Z60. After all calculations are nished the energies from rlevels, the hyperne data and the transition data are collected from the dierent directories and saved in les energy26, energy27, ..., energy60, hfs26, hfs27, . . . , hfs60, trans26, trans27, ..., trans60. These les are read by the rseqenergy rseqtrans programs to produce GNU Octave/Matlab M-les that plot computed properties as functions of the nuclear charge Z of the ions. The M-les include some tting capabilities as well.
### 12.1
### Iso-electronic sequences
Properties of states, as specied by parity, J quantum number and order number within the symmetry (e.g. the second eigenvalue), are smoothly varying functions of the nuclear charge Z along the iso-electronic sequence. Based on hydrogenic approximations, scaling with Z can be derived for dierent properties (see for example [37], chapter 19). Using spline methods or least- squares ts to scaling expressions, atomic data along a sequence can be reconstructed with high accuracy from a limited set of calculations. When reconstructing data attention must be paid to label changes. These changes are consequences of the transition from LSJ to jj-coupling, which introduces a label change between the low Z and high Z regions. In the Mg-sequence a label change occurs for the 3l3l′, J = 2 even parity states. At low Z the ordering is 3p2 1D2, 3p2 3P2, 3s3d 3P2, 3s3d 1D2. When the spin-orbit coupling becomes dominant the ordering in jj-coupling is (1/2, 1/2), (1/2, 3/2), (1/2, 5/2), (3/2, 3/2). Since in the high-Z limit the 3s1/23d3/2 state is lower than the 3p2 1/2 state there must be a label change for some Z. A label change corresponds to an energy level anti-crossing, where two energy levels with the same J and parity will be very close to each other and there will, in the multiconguration approximation, be strong interactions between CSFs over a range of Z values. These interactions may result in a decrease or increase of transition probabilities due to negative or positive interference between terms in the expressions for the transition matrix element. For Mg such interference eects can be seen around Z = 45. In section 12.2 we will generate atomic data for the 3l3l′ states in the Mg iso-electronic sequence. In section 12.3 we will explore the energy level anti-crossings and the interference eects using the 243

244CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE graphical tools.
### 12.2
### Running script les
The main script sh_case3 is shown below. This script controls the computational ow and calls several subscripts. #!/bin/sh set -x # Main script for iso-electronic sequence # 1. Generate directories for the elements and nuclear data ./sh_nuc_seq # 2. Generate lists of CSFs in main directory ./sh_files_c # 3. Perform MCDHF calculations for the even reference states ./sh_DF_even # 4. Perform MCDHF calculations for the odd reference states ./sh_DF_odd # 5. Perform MCDHF calculations for even states ./sh_even # 6. Perform MCDHF calculations for odd states ./sh_odd # 7. Perform RCI, transition calculations ./sh_even_odd # 8. Transformation to LSJ, run rlevels and pipe to energyZ ./sh_rlevels # 9. Collect all data files and copy to the main directory ./sh_collect
1. Generate directories and dene nuclear data
The script sh_nuc_seq produces nuclear data for Z = 26, 27, . . . , 60 in the directories Z26, Z27 ,..., Z60. By modifying the script we can produce nuclear data for any sequence of charges. #!/bin/sh

12.2. RUNNING SCRIPT FILES 245 set -x # Full loop over all Z # for z in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \ # 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 \ # 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \ # 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 \ # 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 \ # 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 \ # 126 127 128 129 130 131 132 133 134 135 136 137 138 # We select Z from 26 to 60 for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do # Data from Jefferson Lab (http://education.jlab.org/itselemental) case $z in
1) m=0; MM=1.0794;;
# Need to use point nucleus
2) m=4; MM=4.002602;;
3) m=7; MM=6.941;;
4) m=9; MM=9.012182;;
5) m=11; MM=10.811;;
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

246CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE
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

12.2. RUNNING SCRIPT FILES 247
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
esac echo "Starting: Z::"${z}, "ZZ::"$ZZ, "mass::"${m}, "Weight::"${MM} rm -r Z$z

248CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE mkdir Z$z cd Z$z rnucleus <<EOF $z $m n $MM 1 1 1 EOF cd .. done
2. Generate expansions
The expansions are generated by the script sh_files_c. rcsfgenerate << EOF * 2 3s(2,i) 3p(2,i) 3d(2,i) 3s(1,i)3d(1,i) 3s,3p,3d 0,8 0 n EOF cp rcsf.out DFeven.c ####################################### rcsfgenerate << EOF * 2 3s(1,i)3p(1,i) 3p(1,i)3d(1,i) 3s,3p,3d 0,8 0 n EOF cp rcsf.out DFodd.c #######################################

12.2. RUNNING SCRIPT FILES 249 rcsfgenerate << EOF * 2 3s(2,*) 5s,5p,5d,5f,5g 0,8 2 n EOF cp rcsf.out even.c rcsfsplit << EOF even 2 4s,4p,4d,4f 4 5s,5p,5d,5f,5g 5 EOF ################################ rcsfgenerate << EOF * 2 3s(1,*)3p(1,*) 5s,5p,5d,5f,5g 0,8 2 n EOF cp rcsf.out odd.c rcsfsplit << EOF odd 2 4s,4p,4d,4f 4 5s,5p,5d,5f,5g 5 EOF
3. Even parity reference states
The script sh_DF_even performs angular integration, gets initial estimates and performs rmcdhf calculations for the even reference states. The angular integration is done only once and the mcp.30, mcp.31 ... les are moved between the directories. for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60

250CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE do (if test $z -lt 27 then cd Z${z} cp ../DFeven.c rcsf.inp # Get angular data rangular <<S4 y S4 #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf <<S6 y 1-5 1-3 1-7 1-2 1-2 5 * * 100 S6 rsave DFeven cp DFeven.w even3.w cd .. else cd Z${z} cp ../DFeven.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf <<S6 y

12.2. RUNNING SCRIPT FILES 251 1-5 1-3 1-7 1-2 1-2 5 * * 100 S6 rsave DFeven cp DFeven.w even3.w cd .. fi echo) done
4. Odd parity reference states
The script sh_DF_odd performs angular integration, gets initial estimates and performs rmcdhf calculations for the odd reference states. The angular integration is done only once and the mcp.30, mcp.31 ... les are moved between the directories. for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (if test $z -lt 27 then cd Z${z} cp ../DFodd.c rcsf.inp # Get angular data rangular <<S4 y S4 #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > outodd_rmcdhf <<S6 y 1-2 1-5 1-5 1-3 1 5

252CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE * * 100 S6 rsave DFodd cp DFodd.w odd3.w cd .. else cd Z${z} cp ../DFodd.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . #Get initial estimates of wave functions rwfnestimate <<S5 y 2 * S5 # Perform self-consistent field calculations rmcdhf > outodd_rmcdhf <<S6 y 1-2 1-5 1-5 1-3 1 5 * * 100 S6 rsave DFodd cp DFodd.w odd3.w cd .. fi echo) done
5. Perform calculations for even states
The script sh_even performs angular integration, gets initial estimates and performs rmcdhf calculations for the odd states. The script loops over both the active set and the atomic number Z. Angular les are reused and moved between the directories.

12.2. RUNNING SCRIPT FILES 253 for n in 4 5 do ( for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (if test $z -lt 27 then cd Z${z} cp ../even${n}.c rcsf.inp # Get angular data rangular <<S4 y S4 k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 even${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf_${n} <<S6 y 1-5 1-3 1-7 1-2 1-2 5 ${n}* 100 S6 rsave even${n} cd .. else cd Z${z} cp ../even${n}.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y

254CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE 1 even${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outeven_rmcdhf_${n} <<S6 y 1-5 1-3 1-7 1-2 1-2 5 ${n}* 100 S6 rsave even${n} cd .. fi echo) done ) done
6. Perform calculations for odd states
The script sh_odd performs angular integration, gets initial estimates and performs rmcdhf cal- culations for the odd states. The script loops over both the active set and the atomic number Z. Angular les are reused and moved between the directories. for n in 4 5 do ( for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (if test $z -lt 27 then cd Z${z} cp ../odd${n}.c rcsf.inp # Get angular data rangular <<S4 y S4 k=`expr $n - 1` #Get initial estimates of wave functions

12.2. RUNNING SCRIPT FILES 255 rwfnestimate <<S5 y 1 odd${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outodd_rmcdhf_${n} <<S6 y 1-2 1-5 1-5 1-3 1 5 ${n}* 100 S6 rsave odd${n} cd .. else cd Z${z} cp ../odd${n}.c rcsf.inp #Move mcp files from previous directory m=`expr $z - 1` mv ../Z${m}/mcp* . k=`expr $n - 1` #Get initial estimates of wave functions rwfnestimate <<S5 y 1 odd${k}.w * 2 * S5 # Perform self-consistent field calculations rmcdhf > outodd_rmcdhf_${n} <<S6 y 1-2 1-5 1-5 1-3 1 5 ${n}*

256CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE 100 S6 rsave odd${n} cd .. fi echo) done ) done
7. Perform RCI and transition calculations
The script sh_even_odd performs conguration interaction and transition calculations for even5 and odd5. Angular les are reused and moved between the directories. for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (cd Z${z} # RCI calculations for even5 rci > outeven_rci <<S6 y even5 y y 1.d-6 y n n y 3 1-5 1-3 1-7 1-2 1-2 S6 # RCI calculations for odd5 rci > outodd_rci <<S6 y odd5 y y 1.d-6 y n n y 3

12.2. RUNNING SCRIPT FILES 257 1-2 1-5 1-5 1-3 1 S6 if test $z -lt 27 then # Run rbiotransform ans save angular data rbiotransform <<S4 y y even5 odd5 y S4 # Run rtransition save angular data rtransition <<S4 y y even5 odd5 E1 S4 else #Move angular files from previous directory m=`expr $z - 1` mv ../Z${m}/even5.TB . mv ../Z${m}/odd5.TB . mv ../Z${m}/even5.odd5.-1T . # Run rbiotransform using available angular data rbiotransform <<S4 y y even5 odd5 y S4 # Run rtransition using available angular data rtransition <<S4 y y even5 odd5 E1 S4

258CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE fi cd .. echo) done Transformation to LSJ, run rlevels and pipe to energyZ This script runs jj2lsj to transform to LSJ-coupling. The energy les energyZ are created by redirecting the output from rlevels. for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (cd Z${z} jj2lsj <<S1 even5 y y y S1 jj2lsj <<S2 odd5 y y y S2 rlevels even5.cm odd5.cm > energy${z} cd .. echo) done Collect data to prepare for the runs of the iso-electronic plotting tools This script collects, in one directory, all the energy, hfs and transition les that are needed to run the tools that create the iso-electronic plots. for z in 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 \ 49 50 51 52 53 54 55 56 57 58 59 60 do (cd Z${z} cp energy${z} ../. cp even5.odd5.ct ../trans${z} cd .. echo) done

12.3. GENERATING PLOTS OF PROPERTIES ALONG THE SEQUENCE 259
### 12.3
### Generating plots of properties along the sequence
After the script sh_case3 has been executed the energy les energy26, energy27, ..., energy60, as obtained from rlevels, the hyperne structure les hfs26, hfs27 ,..., hfs60 and the transition les trans26, trans27, ..., trans60 all reside in one directory. The energy le energy26 is shown below nblock = 5 ncftot = 327 nw = 25 nelec = 12 nblock = 5 ncftot = 320 nw = 25 nelec = 12 Energy levels for ... Rydberg constant is 109737.31569 Splitting is the energy difference with the lower neighbor ------------------------------------------------------------------------------------------ No Pos J Parity Energy Total Levels Splitting Configuration (a.u.) (cm^-1) (cm^-1) ------------------------------------------------------------------------------------------ 1 1 0 + -1182.3727090 0.00 0.00 3s(2)_1S0 2 1 0 - -1181.3113166 232948.71 232948.71 3s.3p_3P 3 1 1 - -1181.2846597 238799.23 5850.52 3s.3p_3P 4 1 2 - -1181.2204883 252883.23 14084.01 3s.3p_3P 5 2 1 - -1180.7554032 354957.60 102074.37 3s.3p_1P 6 2 0 + -1179.8372639 556465.90 201508.29 3p(2)_3P2 7 1 2 + -1179.8216886 559884.26 3418.37 3p(2)_1D2 8 1 1 + -1179.7920264 566394.37 6510.11 3p(2)_3P2 9 2 2 + -1179.7153461 583223.75 16829.37 3p(2)_3P2 10 3 0 + -1179.3515090 663076.77 79853.02 3p(2)_1S0 11 2 1 + -1179.2726052 680394.14 17317.37 3s.3d_3D 12 3 2 + -1179.2680609 681391.51 997.37 3s.3d_3D 13 1 3 + -1179.2609531 682951.48 1559.97 3s.3d_3D 14 4 2 + -1178.8797415 766617.76 83666.28 3s.3d_1D 15 2 2 - -1178.1399645 928980.06 162362.29 3p.3d_3F 16 1 3 - -1178.0955852 938720.18 9740.12 3p.3d_3F 17 3 2 - -1178.0442285 949991.68 11271.50 3p.3d_1D 18 1 4 - -1178.0435428 950142.16 150.48 3p.3d_3F 19 3 1 - -1177.8804977 985926.43 35784.26 3p.3d_3D 20 4 2 - -1177.8791112 986230.73 304.30 3p.3d_3P 21 2 3 - -1177.8254894 997999.35 11768.62 3p.3d_3D 22 2 0 - -1177.8237420 998382.86 383.51 3p.3d_3P 23 4 1 - -1177.8212951 998919.91 537.05 3p.3d_3P 24 5 2 - -1177.8187683 999474.46 554.55 3p.3d_3D 25 3 3 - -1177.5115780 1066894.94 67420.48 3p.3d_1F 26 5 1 - -1177.4559591 1079101.88 12206.94 3p.3d_1P 27 5 2 + -1176.1164738 1373084.92 293983.05 3d(2)_3F2 28 2 3 + -1176.1091088 1374701.36 1616.44 3d(2)_3F2 29 1 4 + -1176.1001073 1376676.96 1975.59 3d(2)_3F2 30 6 2 + -1175.9682840 1405608.82 28931.87 3d(2)_1D2 31 4 0 + -1175.9538651 1408773.40 3164.58 3d(2)_3P2 32 3 1 + -1175.9510570 1409389.72 616.31 3d(2)_3P2 33 2 4 + -1175.9496227 1409704.50 314.78 3d(2)_1G2 34 7 2 + -1175.9446189 1410802.71 1098.21 3d(2)_3P2 35 5 0 + -1175.5824289 1490294.23 79491.51 3d(2)_1S0 ------------------------------------------------------------------------------------------

260CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE Each state is specied by the position within the symmetry, the J quantum number and the parity. For example, the four states 3p2 1D2, 3p2 3P2, 3s3d 3P2, 3s3d 1D2 with even parity and J = 2 are specied as 1 2 +, 2 2 +, 3 2 +, 4 2 +. These specications remain valid over the iso-electronic sequence, although the LSJ designation may change. Thus, to follow states along the iso-electronic sequence the above specications should be used. To generate a GNU Octave/Matlab M-le that plots the 3p2 1D2, 3p2 3P2, 3s3d 3P2, 3s3d 1D2 states along the sequence, the rseqenergy program should be run. The program looks for all energy les in a given range of Z. Then after having specied the states to be plotted there is an option to perform least squares ts to obtain analytical expressions of the trends. If no ts are done the data are instead interpolated using cubic splines. The rseqenergy program outputs an M-le with name seqenergyplot.m. The M-le contains all data needed for the plot and the le can also very easily be edited to comply with the desires of the user. The input session for rseqenergy is shown below. Please note that you should input 2J and not J and the sequence 1 2 +, 2 2 +, 3 2 +, 4 2 + above should thus be inserted in the program as 1 4 +, 2 4 +, 3 4 +, 4 4 +. >>rseqenergy RSEQENERGY This program reads output from rlevels for several ions and produces a Matlab/Octave file that plots energy as a function of Z Input files: energyZ1, energyZ2, .., energyZn Output file: seqenergyplot.m Give the first Z and last Z of the sequence >>26,60 How many states do you want to plot? >>4 Give number within symmetry,2*J and parity (+/-) >>1,4,+ Give number within symmetry,2*J and parity (+/-) >>2,4,+ Give number within symmetry,2*J and parity (+/-) >>3,4,+ Give number within symmetry,2*J and parity (+/-) >>4,4,+ Least-squares fit (y/n) ? >>n rseqenergy produces the le seqenergyplot.m. To run this le open GNU Octave (or Matlab) and issue the command octave:1>seqenergyplot on the GNU Octave command line and the plot shown in gure 12.1 will appear. There is an energy level anti-crossing around Z = 44. We now turn to the hyperne structure. The hyperne structure le hfs26 is shown below Nuclear spin 1.000000000000000D+00 au Nuclear magnetic dipole moment 1.000000000000000D+00 n.m. Nuclear electric quadrupole moment 1.000000000000000D+00 barns

12.3. GENERATING PLOTS OF PROPERTIES ALONG THE SEQUENCE 261 25 30 35 40 45 50 55 60 0 1e+06 2e+06 3e+06 4e+06 5e+06 Z E (cm-1) E in cm-1 as a function of Z Figure 12.1: Plot of the energy of the four lowest even parity states with J = 2 as function of the nuclear charge Z. There is an energy level anti-crossing around Z = 44. Interaction constants: Level1 J Parity A (MHz) B (MHz) g_J 1 1 + -3.4234387290D+02 4.4613609751D+03 1.5001883281D+00 2 1 + -1.9243940191D+04 6.2528207125D+02 4.9806452455D-01 3 1 + -1.1351221472D+01 6.2649343730D+02 1.5002882611D+00 1 2 + 8.7390039737D+03 1.1076010824D+04 1.0772839773D+00 2 2 + 4.4795463061D+03 -5.4746853597D+03 1.4215360611D+00 3 2 + 8.8227565268D+03 8.9310423785D+02 1.1660563087D+00 4 2 + 2.4077941743D+03 5.0250914053D+03 9.9939681256D-01 5 2 + 1.8500822031D+03 5.8565537496D+02 6.6595057455D-01 6 2 + 1.2714709452D+03 -7.1108880021D+02 1.0473515693D+00 7 2 + 7.3078966195D+02 -1.2063797282D+03 1.4511063532D+00 1 3 + 1.5225231973D+04 1.7639256412D+03 1.3331477513D+00 2 3 + 1.1924106280D+03 6.4316601614D+02 1.0826447519D+00 1 4 + 9.0356372098D+02 8.6788621658D+02 1.2493914009D+00 2 4 + 1.2426420188D+03 3.5362685851D+03 9.9942996698D-01 States are specied in the same way as in the energy le by giving position (level1) within the symmetry, the J quantum number and the parity. To plot the hyperne interaction constants or the Landé gJ factor as a function of the nuclear charge we use the program rseqhfs. The input session for plotting the magnetic dipole interaction constant for the states 2 2 + and 3 2 + is shown below (again please note that you should input 2J and not J) >>rseqhfs RSEQHFS This program reads output from rhfs for several ions and produces a Matlab/Octave file that plots hfs parameters as functions of Z Input files: hfsZ1, hfsZ2, .., hfsZn or Output file: seqhfsplot.m Give the first Z and last Z of the sequence

262CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE >>26,60 How many states do you want to plot? >>2 Give number within symmetry,2*J and parity (+/-) >>2,4,+ Give number within symmetry,2*J and parity (+/-) >>3,4,+ Plot A (1), B (2) or gJ (3) ? >>1 Least-squares fit (y/n) ? >>n rseqhfs produces the le seqhfsplot.m. To run this le open GNU Octave (or Matlab) and issue the command octave:1>seqhfsplot at the GNU Octave command line and the plot in gure 12.2 will now be displayed. The strong mixing of the CSFs around the level anti-crossing at Z = 44 causes interference eects that have large inuence on the hyperne structure constants of the two states. 25 30 35 40 45 50 55 60 -400000 -200000 0 200000 400000 600000 Z A_J (MHz) Figure 12.2: Plot of the hyperne interaction constants AJ for the two interfering even parity states with J = 2 as function of the nuclear charge Z. The transition le trans26 is shown below. A transition is specied by giving the multipolarity along with the position (Lev) within the symmetry, the J quantum number and the parity for the upper and lower states. Transition between files: f1 = even5 f2 = odd5 Electric 2**( 1)-pole transitions =================================

12.3. GENERATING PLOTS OF PROPERTIES ALONG THE SEQUENCE 263 Upper Lower Lev J P Lev J P E (Kays) A (s-1) gf S f2 1 1 - f1 1 0 + 238799.23 C 4.30267D+07 3.39353D-03 4.67836D-03 B 4.08058D+07 3.21836D-03 4.43688D-03 f2 2 1 - f1 1 0 + 354957.60 C 2.33297D+10 8.32789D-01 7.72385D-01 B 2.28391D+10 8.15275D-01 7.56142D-01 f2 3 1 - f1 1 0 + 985926.43 C 2.38994D+05 1.10580D-06 3.69239D-07 B 9.80859D+04 4.53833D-07 1.51540D-07 f2 4 1 - f1 1 0 + 998919.91 C 6.50659D+04 2.93272D-07 9.66531D-08 B 4.71470D+04 2.12506D-07 7.00353D-08 f2 5 1 - f1 1 0 + 1079101.88 C 3.67085D+08 1.41782D-03 4.32548D-04 B 3.05390D+08 1.17953D-03 3.59850D-04 f1 2 0 + f2 1 1 - 317666.67 C 1.88797D+10 2.80485D-01 2.90679D-01 B 1.82388D+10 2.70963D-01 2.80811D-01 ............... To plot A, gf or S as a function of the nuclear charge we use the program rseqtrans. The input session for plotting the transition rate A from the states 2 2 + and 3 2 + to 1 1 - is shown below. Please observe that we should input 2J. >>rseqtrans RSEQTRANS This program reads output from rtransition for several ions and produces a Matlab/Octave file that plots A, gf, or S as a function of Z Input files: transZ1, transZ2, .., transZn Output file: seqtransplot.m Give the first Z and last Z of the sequence >>26,60 Give multipolarity of transition: E1, M1, E2, M2 >>E1 How many transitions do you want to plot? >>2 Give number within symmetry,2*J and parity (+/-) for upper and lower state >>2,4,+,1,2,- Give number within symmetry,2*J and parity (+/-) for upper and lower state >>3,4,+,1,2,- Plot A (1), gf (2) or S (3) ? >>1 Least-squares fit (y/n) ? >>n The rseqtrans program produces the le seqtransplot.m. To run this le open GNU Octave or Matlab and issue the command octave:1>seqtransplot and the plot in gure 12.3 will now be shown. The strong mixing of the CSFs around the level anti-crossing at Z = 44 causes interference eects that inuence the rates.

264CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE 25 30 35 40 45 50 55 60 -1e+11 0 1e+11 2e+11 3e+11 4e+11 5e+11 6e+11 Z A (s-1) transition parameters as functions of Z Figure 12.3: Plot of transition rates involving two interfering states.
### 12.4
### Least-squares ts to data
If deemed important least-squares ts can be done for atomic data that are not aected by inter- ference eects from level anti-crossings. Below we t a polynomial to the energies for the 1 0 -, 1 1 -, 2 1 -, and 1 2 - states. >>rseqenergy RSEQENERGY This program reads output from rlevels for several ions and produces a Matlab/Octave file that plots energy as a function of Z Input files: energyZ1, energyZ2, .., energyZn Output file: seqenergyplot.m Give the first Z and last Z of the sequence >>26,60 How many states do you want to plot? >>4 Give number within symmetry,2*J and parity (+/-) >>1,0,- Give number within symmetry,2*J and parity (+/-) >>1,2,- Give number within symmetry,2*J and parity (+/-) >>2,2,- Give number within symmetry,2*J and parity (+/-) >>1,4,- Least-squares fit (y/n) ? >>y Type of fitting: a1 Z^-2 + a2 Z^-1 + ...+ a6 Z^3 (1) a1 + a2 Z + a3 Z^2 + a4 Z^3 (2) >>2 Starting GNU Octave (or Matlab) and giving the command octave:1>seqenergyplot

12.4. LEAST-SQUARES FITS TO DATA 265 at the GNU Octave command line gives the tting coecients for the four states a = -3.8990e+00 9.3243e-02 -3.2068e-04 5.5596e-06 a = -3.3884e+00 5.7768e-02 4.6734e-04 -7.9666e-08 a = -3.4271e+00 1.4637e-01 -3.6443e-03 4.5791e-05 a = -3.1674e+00 1.3634e-01 -3.6161e-03 4.7051e-05 along with the plot in gure 12.4. 25 30 35 40 45 50 55 60 0 500000 1e+06 1.5e+06 2e+06 2.5e+06 Z E (cm-1) E in cm-1 as a function of Z Figure 12.4: Polynomial tted to the energies of the 1 0 -, 1 1 -, 2 1 -, and 1 2 - states We can do ts to transition data as well. Below we t a Laurent series to the line strength S for the transition from 1 1 -, 2 1 - down to 1 0 +.

266CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE >>rseqtrans RSEQTRANS This program reads output from rtransition for several ions and produces a Matlab/Octave file that plots A, gf, or S as a function of Z Input files: transZ1, transZ2, .., transZn Output file: seqtransplot.m Give the first Z and last Z of the sequence >>26,60 Give multipolarity of transition: E1, M1, E2, M2 >>E1 How many transitions do you want to plot? >>2 Give number within symmetry,2*J and parity (+/-) for upper and lower state >>1,2,-,1,0,+ Give number within symmetry,2*J and parity (+/-) for upper and lower state >>2,2,-,1,0,+ Plot A (1), gf (2) or S (3) ? >>3 Least-squares fit (y/n) ? >>y Type of fitting: a1 Z^-2 + a2 Z^-1 + ...+ a6 Z^3 (1) a1 + a2 Z + a3 Z^2 + a4 Z^3 (2) >>1 Starting GNUOctave (or Matlab) and giving the command octave:1>seqtransplot at the GNU Octave command line gives the tting coecients a = -2.9422e+04 4.7527e+03 -2.9700e+02 8.5307e+00 -1.1226e-01 5.4992e-04 a = 1.4276e+04 -1.1873e+03 4.8383e+01 -1.0910e+00 1.2034e-02 -5.2794e-05 The produced plot in displayed in gure 12.5. The tted function describes the data very well.

12.5. MODIFYING THE GNU OCTAVE/MATLAB M-FILES 267 25 30 35 40 45 50 55 60 0 0.2 0.4 0.6 0.8 Z S transition parameters as functions of Z Figure 12.5: Fitted function to the line strength S for the transitions of 1 1 -, 2 1 - down to the 1 0 + groundstate.
### 12.5
### Modifying the GNU Octave/MATLAB M-les
The M-les produced by rseqenergy, rseqhfs and rseqtrans are very easy to modify to include legends, change captions etc. Also other types of modications should be considered. If, for example, calculations are done for even Z in an iso-electronic sequence then the user can easily modify seqenergyplot.m to output interpolated values of the energies for odd Z. Away from level anti-crossings the accuracy of the interpolated values should be quite high. In many cases data for a full iso-electronic sequence can be interpolated from a comparatively small number of ions. The M-les can be concatenated (some minor editing is needed) and it is then possible to overlay several plots. The seqtransplot.m le from the last run is shown below. The data are organized in a matrix A where the rst column contains the nuclear charge Z. The atomic data are stored in columns 2 and 3. Standard commands are used for plotting and least-squares ts. A = [ 26 4.4368799999999998E-003 0.75614199999999998 27 5.1946300000000004E-003 0.68008400000000002 28 6.0161700000000004E-003 0.61473800000000001 29 6.8959800000000003E-003 0.55813599999999997 30 7.8269499999999992E-003 0.50873800000000002 31 8.8007499999999995E-003 0.46534399999999998 32 9.8050199999999994E-003 0.42698700000000001 33 1.0828300000000001E-002 0.39289900000000000 34 1.1857500000000000E-002 0.36245699999999997 35 1.2878700000000000E-002 0.33515299999999998 36 1.3878400000000001E-002 0.31056699999999998 37 1.4843300000000000E-002 0.28835200000000000 38 1.5761100000000000E-002 0.26822099999999999 39 1.6621100000000000E-002 0.24992900000000001 40 1.7414099999999998E-002 0.23326900000000000 41 1.8133000000000000E-002 0.21806200000000001

268CHAPTER 12. CASE STUDY III: GRAPHICAL ANALYSIS OF THE MG ISO-ELECTRONIC SEQUENCE 42 1.8772700000000000E-002 0.20415900000000001 43 1.9330000000000000E-002 0.19142600000000001 44 1.9803700000000000E-002 0.17974599999999999 45 2.0194199999999999E-002 0.16901600000000000 46 2.0503600000000000E-002 0.15914600000000001 47 2.0734900000000001E-002 0.15005599999999999 48 2.0892299999999999E-002 0.14167199999999999 49 2.0980599999999999E-002 0.13392999999999999 50 2.1004800000000001E-002 0.12677099999999999 51 2.0970699999999998E-002 0.12014300000000000 52 2.0883800000000001E-002 0.11399800000000000 53 2.0749699999999999E-002 0.10829400000000000 54 2.0573700000000000E-002 0.10299200000000000 55 2.0360799999999998E-002 9.8056500000000005E-002 56 2.0116100000000001E-002 9.3456800000000007E-002 57 1.9844100000000000E-002 8.9163800000000001E-002 58 1.9548800000000002E-002 8.5151699999999997E-002 59 1.9234100000000001E-002 8.1397200000000003E-002 60 1.8903300000000001E-002 7.7879100000000007E-002 ]; clf, hold on zip = linspace( 26, 60); title('transition parameters as functions of Z') xlabel('Z') ylabel('S') plot(A(:,1),A(:, 2),'+') z = A(:,1); AD = [z.^(-2) z.^(-1) z.^0 z.^1 z.^2 z.^3]; y = A(:, 2); m = mean(y); s = std(y); a = AD\(y-m)/s aiplsq = a(1)./zip.^2 + a(2)./zip + a(3) + a(4)*zip + a(5)*zip.^2 + a(6)*zip.^3; aiplsq = s*aiplsq + m; plot(zip,aiplsq,'r') plot(A(:,1),A(:, 3),'+') z = A(:,1); AD = [z.^(-2) z.^(-1) z.^0 z.^1 z.^2 z.^3]; y = A(:, 3); m = mean(y); s = std(y); a = AD\(y-m)/s aiplsq = a(1)./zip.^2 + a(2)./zip + a(3) + a(4)*zip + a(5)*zip.^2 + a(6)*zip.^3; aiplsq = s*aiplsq + m; plot(zip,aiplsq,'r')

## Part IV
## Issues of convergence, trouble
## shooting and non-default options
269