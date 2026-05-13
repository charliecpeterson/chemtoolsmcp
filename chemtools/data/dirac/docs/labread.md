orphan  

# LABREAD

LABREAD is a utility program for getting information about integrals
stored on the one-electron integral file AOPROPER. Usage is
straightforward, for instance:

    dirac 55>labread.x AOPROPER

     MOLECULE labels found on file :AOPROPER

             1     ********  25Jul12   SY 1TFFT  OVERLAP 
             3     ********  25Jul12   SY 1TFFT  MOLFIELD
             5     ********  25Jul12   SY 1FFFT  BETAMAT 
             7     ********  25Jul12   AN 1FTTF  XDIPVEL 
             9     ********  25Jul12   AN 1FTTF  YDIPVEL 
            11     ********  25Jul12   AN 1FTTF  ZDIPVEL 
            13     ********  25Jul12   SY 1TFFT  PVCF1 01
            15     ********  25Jul12   SY 1TFFT  PVCF1 02
            17     ********  25Jul12   SY 1TFFT  FC F1 01
            19     ********  25Jul12   SY 1TFFT  FC F1 02
            21     ********  25Jul12   SY 1FTTF  PVCF1 01
            23     ********  25Jul12   SY 1FTTF  PVCF1 02
            25     ********  25Jul12   SY 1TFFT  ED F1 01
            27     ********  25Jul12   SY 1TFFT  ED F1 02
            29     ********  25Jul12   SY 1TFFT  EOFLABEL

            29 records read before EOF on file.
    STOP  

The notation *SY* and *AN* indicate whether the integral matrix is
symmetric or not. In general only certain combinations of large and
small component blocks have been calculated. An even operator has
*TFFT*, meaning that only *LL* and *SS* blocks have been generated,
whereas an odd operator has *FTTF*, meaning that only *LS* and *SL*
blocks were generated. For information about integral labels, see the
`one_electron_operators` section.
