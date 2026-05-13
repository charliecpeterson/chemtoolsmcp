orphan  

# LUCITA:Sample inputs

Perform SDCI calculations:

    *LUCITA
    .INIWFC
      DHFSCF
    .CITYPE
      SDCI
    .MULTIP
      3
    *END OF

Perform GASCI calculations:

    *LUCITA
    .TITLE
      Aluminum atom in C2h symmetry, 7s5p2d (L) basis, SDT/SD CI (1s inactive).
    .INIWFC
      OSHSCF
    .CITYPE
      GASCI
    .MULTIP
      2
    .NACTEL
      13
    .GASSHE
     5
     1,0,0,0   ! 1s
     2,0,0,0   ! 2s3s
     0,0,2,4   ! 2p3p
     5,2,1,2   ! 4s5s 4p 3d
     5,2,2,4   ! 6s7s 5p6p 4d
    .GASSPC
     1
      2  2
      3  6
     11 13
     11 13
     13 13
    *END OF

In addition, for we provide three LUCITA input templates with increasing
complexity (lucita.novice,lucita.expert,lucita.wizard)

The user can download and modify them according to his needs.
